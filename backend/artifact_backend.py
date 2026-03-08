from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from inference.attention_model import COLUMNS_TO_BE_DROPPED, EARLY_RUL, RAW_COLUMN_NAMES, WINDOW_LENGTH
from inference.reliability import compute_reliability_index, gate_prediction, reason_codes


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ZIP = REPO_ROOT / "model_artifacts.zip"


class ArtifactLoadError(RuntimeError):
    pass


@dataclass
class NotebookArtifactRunner:
    artifact_zip: Path = DEFAULT_ARTIFACT_ZIP

    def __post_init__(self) -> None:
        self._regressor = None
        self._metadata: Dict[str, object] = {}
        self._feature_strategy = "unknown"
        self._feature_names: List[str] = []
        self._load()

    def _load(self) -> None:
        if not self.artifact_zip.exists():
            raise ArtifactLoadError(
                f"Artifact backend requested but archive was not found: {self.artifact_zip}. "
                "Place model_artifacts.zip in repo root."
            )
        try:
            import joblib  # type: ignore
        except Exception as exc:  # pragma: no cover - environment guard.
            raise ArtifactLoadError("joblib is required for artifact backend. Install `joblib`.") from exc

        try:
            with zipfile.ZipFile(self.artifact_zip, "r") as zf:
                names = set(zf.namelist())
                if "metadata.json" in names:
                    with zf.open("metadata.json") as fp:
                        self._metadata = json.loads(fp.read().decode("utf-8"))
                reg_name = self._pick_regressor_file(names)
                with zf.open(reg_name) as fp:
                    reg_bytes = io.BytesIO(fp.read())
                self._regressor = joblib.load(reg_bytes)
        except ArtifactLoadError:
            raise
        except Exception as exc:  # pragma: no cover - runtime load guard.
            raise ArtifactLoadError(
                "Failed to load model artifact. Ensure compatible lightgbm/sklearn versions are installed."
            ) from exc

    @staticmethod
    def _pick_regressor_file(names: set[str]) -> str:
        for candidate in ["lgb_reg.joblib", "lgb_reg2.joblib", "regressor.joblib"]:
            if candidate in names:
                return candidate
        reg_candidates = sorted([n for n in names if "reg" in n.lower() and n.lower().endswith(".joblib")])
        if reg_candidates:
            return reg_candidates[0]
        raise ArtifactLoadError("No regressor joblib found in model_artifacts.zip.")

    def _build_feature_candidates(self, raw_df: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
        ordered = raw_df[RAW_COLUMN_NAMES].copy()
        ordered = ordered.apply(pd.to_numeric, errors="raise")

        by_index = ordered.copy()
        by_index.columns = list(range(26))
        attention_like = by_index.drop(columns=COLUMNS_TO_BE_DROPPED, errors="ignore")
        attention_like.columns = [f"f_{i}" for i in range(attention_like.shape[1])]

        sensor_cols = [c for c in RAW_COLUMN_NAMES if c.startswith("s_")]
        return [
            ("drop_unit_time", ordered.drop(columns=["unit_nr", "time_cycles"], errors="ignore")),
            ("attention_drop_columns", attention_like),
            ("drop_unit", ordered.drop(columns=["unit_nr"], errors="ignore")),
            ("sensors_only", ordered[sensor_cols]),
            ("raw_all_26", ordered),
        ]

    def _select_features(self, raw_df: pd.DataFrame) -> np.ndarray:
        if self._regressor is None:
            raise ArtifactLoadError("Artifact regressor is not loaded.")

        expected = int(getattr(self._regressor, "n_features_in_", 0) or 0)
        candidates = self._build_feature_candidates(raw_df)

        if expected > 0:
            for name, candidate in candidates:
                if candidate.shape[1] == expected:
                    self._feature_strategy = name
                    self._feature_names = [str(v) for v in candidate.columns]
                    return candidate.to_numpy(dtype=float)

        for name, candidate in candidates:
            try:
                trial = candidate.to_numpy(dtype=float)
                _ = self._regressor.predict(trial[:1])  # type: ignore[union-attr]
                self._feature_strategy = name
                self._feature_names = [str(v) for v in candidate.columns]
                return trial
            except Exception:
                continue

        raise ArtifactLoadError(
            "Could not align runtime features with artifact regressor input shape. "
            "Re-export artifact including feature transformer metadata."
        )

    def predict_per_row(self, raw_df: pd.DataFrame) -> np.ndarray:
        features = self._select_features(raw_df)
        pred = np.asarray(self._regressor.predict(features), dtype=float).reshape(-1)  # type: ignore[union-attr]
        return np.clip(pred, 0.0, float(EARLY_RUL))

    def metadata(self) -> Dict[str, object]:
        return {
            "artifact_zip": str(self.artifact_zip),
            "feature_strategy": self._feature_strategy,
            "feature_count": len(self._feature_names),
            "metadata": self._metadata,
        }


def build_artifact_result(raw_df: pd.DataFrame, predicted_rul: np.ndarray) -> Dict[str, object]:
    rows = raw_df.copy()
    rows["__predicted_rul"] = np.asarray(predicted_rul, dtype=float)
    rows["unit_nr"] = rows["unit_nr"].astype(int)
    rows["time_cycles"] = rows["time_cycles"].astype(int)
    rows = rows.sort_values(["unit_nr", "time_cycles"]).reset_index(drop=True)

    engine_ids = sorted(rows["unit_nr"].astype(int).unique().tolist())
    per_engine_windows: Dict[int, List[float]] = {}
    per_engine_mean: Dict[int, float] = {}
    per_engine_attention_last: Dict[int, List[float]] = {}
    per_engine_reliability: Dict[int, Dict[str, object]] = {}
    decisions_numeric: List[float] = []
    num_test_windows_list: List[int] = []

    for engine_id in engine_ids:
        engine_df = rows[rows["unit_nr"] == int(engine_id)]
        window_preds = engine_df["__predicted_rul"].astype(float).tolist()
        num_test_windows_list.append(len(window_preds))
        per_engine_windows[int(engine_id)] = [float(v) for v in window_preds]
        per_engine_mean[int(engine_id)] = float(np.mean(window_preds))

        tail = min(len(window_preds), WINDOW_LENGTH)
        per_engine_attention_last[int(engine_id)] = [float(1.0 / tail)] * tail if tail > 0 else [1.0]

        metrics = compute_reliability_index(window_preds, early_rul=float(EARLY_RUL))
        gating = gate_prediction(per_engine_mean[int(engine_id)], metrics["ri"], window_preds)
        combined = {**metrics, **gating, "reason_codes": reason_codes(metrics)}
        per_engine_reliability[int(engine_id)] = combined
        decisions_numeric.append({"ACCEPT": 1.0, "WARN": 0.5, "REJECT": 0.0}[combined["decision"]])

    overall_reliability_index = float(np.mean([v["ri"] for v in per_engine_reliability.values()]))
    overall_decision_score = float(np.mean(decisions_numeric)) if decisions_numeric else 0.0

    return {
        "raw_df": raw_df,
        "engine_ids": engine_ids,
        "num_test_windows_list": num_test_windows_list,
        "per_engine_mean_rul": per_engine_mean,
        "per_engine_window_rul": per_engine_windows,
        "per_engine_last_attention": per_engine_attention_last,
        "per_engine_reliability": per_engine_reliability,
        "overall_reliability_index": overall_reliability_index,
        "overall_decision_score": overall_decision_score,
        "overall_mean_rul": float(np.mean(list(per_engine_mean.values()))),
    }
