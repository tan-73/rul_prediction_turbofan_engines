"""Physics-Informed LightGBM backend.

Wraps the existing LightGBM artifact backend and applies a post-prediction
physics constraint layer inspired by the Node-RED cVAE-RUL pipeline.

Physics checks (derived from C-MAPSS thermodynamic bounds):
  - HPC outlet overtemp (T30 / s_3)
  - LPT outlet overtemp (T50 / s_4)
  - HPC static pressure drop (Ps30 / s_9)
  - HPT coolant bleed depletion (W31 / s_18)
  - LPT coolant bleed depletion (W32 / s_19)

The CPC (Counterfactual Physical Consistency) score penalizes predictions
that occur in physically inconsistent operating regimes.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from backend.artifact_backend import NotebookArtifactRunner, build_artifact_result
from inference.attention_model import EARLY_RUL, RAW_COLUMN_NAMES, WINDOW_LENGTH, read_input_dataframe
from inference.reliability import compute_reliability_index, gate_prediction, reason_codes


# ---------------------------------------------------------------------------
# Physics thresholds — from C-MAPSS FD001 domain knowledge + Node-RED logic
# ---------------------------------------------------------------------------

PHYSICS_THRESHOLDS = {
    "s_3":  {"op": "gt", "value": 1620.0, "penalty": 0.25, "label": "HPC outlet overtemp (T30)"},
    "s_4":  {"op": "gt", "value": 1450.0, "penalty": 0.25, "label": "LPT overtemp (T50)"},
    "s_9":  {"op": "lt", "value": 8500.0, "penalty": 0.20, "label": "HPC pressure drop (Ps30)"},
    "s_18": {"op": "lt", "value": 88.0,   "penalty": 0.15, "label": "HPT coolant bleed depletion (W31)"},
    "s_19": {"op": "lt", "value": 32.0,   "penalty": 0.15, "label": "LPT coolant bleed depletion (W32)"},
}


def compute_physics_risk(row: pd.Series) -> Dict[str, object]:
    """Compute per-row physics violation risk and CPC score."""
    total_risk = 0.0
    violations: List[str] = []

    for sensor, cfg in PHYSICS_THRESHOLDS.items():
        if sensor not in row.index:
            continue
        val = float(row[sensor])
        triggered = False
        if cfg["op"] == "gt" and val > cfg["value"]:
            triggered = True
        elif cfg["op"] == "lt" and val < cfg["value"]:
            triggered = True
        if triggered:
            total_risk += cfg["penalty"]
            violations.append(cfg["label"])

    total_risk = min(1.0, total_risk)
    cpc = 1.0 - total_risk

    return {
        "physics_risk": round(total_risk, 4),
        "cpc": round(cpc, 4),
        "violations": violations,
    }


def compute_engine_physics_summary(engine_df: pd.DataFrame) -> Dict[str, object]:
    """Aggregate physics metrics across all rows of an engine."""
    risks = []
    all_violations: List[str] = []
    for _, row in engine_df.iterrows():
        pr = compute_physics_risk(row)
        risks.append(pr["physics_risk"])
        all_violations.extend(pr["violations"])

    mean_risk = float(np.mean(risks)) if risks else 0.0
    max_risk = float(np.max(risks)) if risks else 0.0
    mean_cpc = 1.0 - mean_risk
    unique_violations = sorted(set(all_violations))

    return {
        "mean_physics_risk": round(mean_risk, 4),
        "max_physics_risk": round(max_risk, 4),
        "mean_cpc": round(mean_cpc, 4),
        "violation_types": unique_violations,
        "violation_count": len(all_violations),
    }


def apply_physics_adjustment(
    raw_rul: float,
    physics_summary: Dict[str, object],
    penalty_scale: float = 0.15,
) -> float:
    """Adjust raw RUL downward based on physics risk.

    Higher physics risk → more conservative (lower) RUL.
    The adjustment is capped at penalty_scale * raw_rul to avoid
    over-correction.
    """
    risk = float(physics_summary["mean_physics_risk"])
    adjustment = raw_rul * risk * penalty_scale
    adjusted = max(0.0, raw_rul - adjustment)
    return round(adjusted, 2)


# ---------------------------------------------------------------------------
# PI-LightGBM Adapter
# ---------------------------------------------------------------------------

@dataclass
class PILightGBMAdapter:
    """Physics-Informed LightGBM adapter.

    Runs standard LightGBM prediction from model_artifacts.zip,
    then applies physics constraint post-processing to adjust
    confidence and RUL estimates.
    """
    model_mode: str
    penalty_scale: float = 0.15

    def __post_init__(self) -> None:
        self._runner = NotebookArtifactRunner()

    def predict_detailed_from_csv_bytes(self, csv_bytes: bytes) -> Dict[str, object]:
        raw_df = read_input_dataframe(io.BytesIO(csv_bytes))
        preds = self._runner.predict_per_row(raw_df)

        # Build the standard artifact result first
        rows = raw_df.copy()
        rows["__predicted_rul"] = np.asarray(preds, dtype=float)
        rows["unit_nr"] = rows["unit_nr"].astype(int)
        rows["time_cycles"] = rows["time_cycles"].astype(int)
        rows = rows.sort_values(["unit_nr", "time_cycles"]).reset_index(drop=True)

        engine_ids = sorted(rows["unit_nr"].astype(int).unique().tolist())
        per_engine_windows: Dict[int, List[float]] = {}
        per_engine_mean: Dict[int, float] = {}
        per_engine_attention_last: Dict[int, List[float]] = {}
        per_engine_reliability: Dict[int, Dict[str, object]] = {}
        per_engine_physics: Dict[int, Dict[str, object]] = {}
        decisions_numeric: List[float] = []
        num_test_windows_list: List[int] = []

        for engine_id in engine_ids:
            engine_df = rows[rows["unit_nr"] == int(engine_id)]
            window_preds = engine_df["__predicted_rul"].astype(float).tolist()
            num_test_windows_list.append(len(window_preds))
            per_engine_windows[int(engine_id)] = [float(v) for v in window_preds]

            # Compute physics summary for this engine
            physics = compute_engine_physics_summary(engine_df)
            per_engine_physics[int(engine_id)] = physics

            # Raw mean RUL
            raw_mean = float(np.mean(window_preds))
            # Apply physics adjustment
            adjusted_mean = apply_physics_adjustment(raw_mean, physics, self.penalty_scale)
            per_engine_mean[int(engine_id)] = adjusted_mean

            # Also adjust window predictions for RI computation
            risk = float(physics["mean_physics_risk"])
            adjusted_windows = [
                max(0.0, v - v * risk * self.penalty_scale) for v in window_preds
            ]

            tail = min(len(window_preds), WINDOW_LENGTH)
            per_engine_attention_last[int(engine_id)] = [float(1.0 / tail)] * tail if tail > 0 else [1.0]

            # Compute reliability on adjusted predictions
            metrics = compute_reliability_index(adjusted_windows, early_rul=float(EARLY_RUL))
            gating = gate_prediction(adjusted_mean, metrics["ri"], adjusted_windows)
            combined = {**metrics, **gating, "reason_codes": reason_codes(metrics)}

            # Inject physics-specific fields
            combined["physics_risk"] = physics["mean_physics_risk"]
            combined["cpc"] = physics["mean_cpc"]
            combined["physics_violations"] = physics["violation_types"]

            if physics["mean_physics_risk"] > 0.3:
                combined["reason_codes"] = combined["reason_codes"] + ["PHYSICS_RISK_HIGH"]

            per_engine_reliability[int(engine_id)] = combined
            decisions_numeric.append({"ACCEPT": 1.0, "WARN": 0.5, "REJECT": 0.0}[combined["decision"]])

        overall_ri = float(np.mean([v["ri"] for v in per_engine_reliability.values()]))
        overall_decision_score = float(np.mean(decisions_numeric)) if decisions_numeric else 0.0
        overall_cpc = float(np.mean([v["mean_cpc"] for v in per_engine_physics.values()]))

        return {
            "raw_df": raw_df,
            "engine_ids": engine_ids,
            "num_test_windows_list": num_test_windows_list,
            "per_engine_mean_rul": per_engine_mean,
            "per_engine_window_rul": per_engine_windows,
            "per_engine_last_attention": per_engine_attention_last,
            "per_engine_reliability": per_engine_reliability,
            "per_engine_physics": per_engine_physics,
            "overall_reliability_index": overall_ri,
            "overall_decision_score": overall_decision_score,
            "overall_mean_rul": float(np.mean(list(per_engine_mean.values()))),
            "overall_cpc": overall_cpc,
            "artifact_runtime": self._runner.metadata(),
            "backend_type": "pi-lightgbm",
        }

    def replay_from_csv_bytes(self, csv_bytes: bytes, engine_id: int, step: int) -> pd.DataFrame:
        result = self.predict_detailed_from_csv_bytes(csv_bytes)
        raw_df = pd.DataFrame(result["raw_df"])
        engine = int(engine_id)
        if engine not in set(raw_df["unit_nr"].astype(int).unique()):
            raise ValueError(f"Engine {engine} not found in input data.")

        engine_df = raw_df[raw_df["unit_nr"].astype(int) == engine].sort_values("time_cycles")
        if len(engine_df) < WINDOW_LENGTH:
            raise ValueError(f"Engine {engine} has only {len(engine_df)} rows. Need at least {WINDOW_LENGTH}.")

        replay_rows = []
        for end_idx in range(WINDOW_LENGTH, len(engine_df) + 1, max(int(step), 1)):
            partial = engine_df.iloc[:end_idx].copy()
            payload = io.StringIO()
            partial.to_csv(payload, index=False)
            partial_result = self.predict_detailed_from_csv_bytes(payload.getvalue().encode("utf-8"))
            rel_map = partial_result["per_engine_reliability"]
            mean_map = partial_result["per_engine_mean_rul"]
            physics_map = partial_result.get("per_engine_physics", {})
            rel = rel_map.get(engine, rel_map.get(str(engine)))
            pred_rul = mean_map.get(engine, mean_map.get(str(engine)))
            phys = physics_map.get(engine, physics_map.get(str(engine), {}))
            if rel is None or pred_rul is None:
                raise ValueError(f"Engine {engine} missing from replay output.")
            replay_rows.append(
                {
                    "time_cycles": int(partial["time_cycles"].max()),
                    "predicted_rul": float(pred_rul),
                    "trusted_rul": float(rel["trusted_rul"]),
                    "reliability_index": float(rel["ri"]),
                    "decision": rel["decision"],
                    "window_std": float(rel["window_std"]),
                    "monotonic_violation_rate": float(rel["monotonic_violation_rate"]),
                    "physics_risk": float(phys.get("mean_physics_risk", 0.0)),
                    "cpc": float(phys.get("mean_cpc", 1.0)),
                }
            )
        return pd.DataFrame(replay_rows)
