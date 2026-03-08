from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from typing import Dict, Protocol, Tuple

import pandas as pd

from backend.artifact_backend import NotebookArtifactRunner, build_artifact_result
from inference.attention_model import (
    get_model_weights_path,
    load_attention_model,
    predict_rul_detailed_from_csv,
    read_input_dataframe,
    simulate_realtime_engine_from_df,
    WINDOW_LENGTH,
)


class ModelAdapter(Protocol):
    def predict_detailed_from_csv_bytes(self, csv_bytes: bytes) -> Dict[str, object]:
        ...

    def replay_from_csv_bytes(self, csv_bytes: bytes, engine_id: int, step: int) -> pd.DataFrame:
        ...


def _sanitize_payload(value):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = k
            if hasattr(k, "item"):
                try:
                    key = k.item()
                except Exception:
                    key = k
            out[key] = _sanitize_payload(v)
        return out
    if isinstance(value, list):
        return [_sanitize_payload(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_payload(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict(orient="records")
        except TypeError:
            return value.to_dict()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


@dataclass
class AttentionModelAdapter:
    model_mode: str

    def __post_init__(self) -> None:
        self._model = load_attention_model(get_model_weights_path(self.model_mode))

    def predict_detailed_from_csv_bytes(self, csv_bytes: bytes) -> Dict[str, object]:
        result = predict_rul_detailed_from_csv(io.BytesIO(csv_bytes), model=self._model)
        return _sanitize_payload(result)

    def replay_from_csv_bytes(self, csv_bytes: bytes, engine_id: int, step: int) -> pd.DataFrame:
        result = predict_rul_detailed_from_csv(io.BytesIO(csv_bytes), model=self._model)
        raw_df = result["raw_df"]
        return simulate_realtime_engine_from_df(
            raw_df=raw_df,
            engine_id=int(engine_id),
            model=self._model,
            step=max(int(step), 1),
        )


@dataclass
class ArtifactModelAdapter:
    model_mode: str

    def __post_init__(self) -> None:
        normalized_mode = self.model_mode.strip().lower()
        if normalized_mode != "baseline":
            raise ValueError("Artifact backend currently supports Baseline mode semantics only.")
        self._runner = NotebookArtifactRunner()

    def predict_detailed_from_csv_bytes(self, csv_bytes: bytes) -> Dict[str, object]:
        raw_df = read_input_dataframe(io.BytesIO(csv_bytes))
        preds = self._runner.predict_per_row(raw_df)
        result = build_artifact_result(raw_df, preds)
        result["artifact_runtime"] = self._runner.metadata()
        return _sanitize_payload(result)

    def replay_from_csv_bytes(self, csv_bytes: bytes, engine_id: int, step: int) -> pd.DataFrame:
        result = self.predict_detailed_from_csv_bytes(csv_bytes)
        raw_df = pd.DataFrame(result["raw_df"])
        engine = int(engine_id)
        if engine not in set(raw_df["unit_nr"].astype(int).unique()):
            raise ValueError(f"Engine {engine} not found in input data.")

        engine_df = raw_df[raw_df["unit_nr"].astype(int) == engine].sort_values("time_cycles")
        if len(engine_df) < WINDOW_LENGTH:
            raise ValueError(f"Engine {engine} has only {len(engine_df)} rows. Need at least {WINDOW_LENGTH}.")

        rows = []
        for end_idx in range(WINDOW_LENGTH, len(engine_df) + 1, max(int(step), 1)):
            partial = engine_df.iloc[:end_idx].copy()
            payload = io.StringIO()
            partial.to_csv(payload, index=False)
            partial_result = self.predict_detailed_from_csv_bytes(payload.getvalue().encode("utf-8"))
            rel_map = partial_result["per_engine_reliability"]
            mean_map = partial_result["per_engine_mean_rul"]
            rel = rel_map.get(engine, rel_map.get(str(engine)))
            pred_rul = mean_map.get(engine, mean_map.get(str(engine)))
            if rel is None or pred_rul is None:
                raise ValueError(f"Engine {engine} missing from replay output.")
            rows.append(
                {
                    "time_cycles": int(partial["time_cycles"].max()),
                    "predicted_rul": float(pred_rul),
                    "trusted_rul": float(rel["trusted_rul"]),
                    "reliability_index": float(rel["ri"]),
                    "decision": rel["decision"],
                    "window_std": float(rel["window_std"]),
                    "monotonic_violation_rate": float(rel["monotonic_violation_rate"]),
                }
            )
        return pd.DataFrame(rows)


class ModelService:
    def __init__(self) -> None:
        self._adapters: Dict[Tuple[str, str], ModelAdapter] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _normalize_mode(model_mode: str) -> str:
        mode = model_mode.strip().lower()
        if mode in {"baseline"}:
            return "Baseline"
        if mode in {"physics-informed", "physics_informed", "pi"}:
            return "Physics-Informed"
        raise ValueError(f"Unsupported model_mode: {model_mode}")

    @staticmethod
    def _normalize_backend(model_backend: str) -> str:
        backend = model_backend.strip().lower()
        if backend in {"attention", "default"}:
            return "attention"
        if backend in {"artifact", "model-artifact", "model_artifact", "lightgbm", "notebook-artifact"}:
            return "artifact"
        raise ValueError(f"Unsupported model_backend: {model_backend}")

    def _get_adapter(self, model_mode: str, model_backend: str) -> ModelAdapter:
        mode = self._normalize_mode(model_mode)
        backend = self._normalize_backend(model_backend)
        key = (backend, mode)
        with self._lock:
            if key not in self._adapters:
                if backend == "attention":
                    self._adapters[key] = AttentionModelAdapter(mode)
                elif backend == "artifact":
                    self._adapters[key] = ArtifactModelAdapter(mode)
                else:  # pragma: no cover - backend normalization guard.
                    raise ValueError(f"Unsupported model_backend: {model_backend}")
            return self._adapters[key]

    def infer(self, csv_bytes: bytes, model_mode: str, model_backend: str = "attention") -> Dict[str, object]:
        adapter = self._get_adapter(model_mode, model_backend)
        return adapter.predict_detailed_from_csv_bytes(csv_bytes)

    def compare(self, csv_bytes: bytes, model_backend: str = "attention") -> Dict[str, object]:
        baseline = self.infer(csv_bytes, "Baseline", model_backend=model_backend)
        pi = self.infer(csv_bytes, "Physics-Informed", model_backend=model_backend)

        baseline_engines = set(baseline["engine_ids"])
        pi_engines = set(pi["engine_ids"])
        shared_engines = sorted(baseline_engines & pi_engines)

        rows = []
        for engine_id in shared_engines:
            b_rel = baseline["per_engine_reliability"][engine_id]
            p_rel = pi["per_engine_reliability"][engine_id]
            b_pred = float(baseline["per_engine_mean_rul"][engine_id])
            p_pred = float(pi["per_engine_mean_rul"][engine_id])
            rows.append(
                {
                    "engine_id": int(engine_id),
                    "baseline_pred_rul": b_pred,
                    "pi_pred_rul": p_pred,
                    "pred_rul_delta_pi_minus_baseline": p_pred - b_pred,
                    "baseline_ri": float(b_rel["ri"]),
                    "pi_ri": float(p_rel["ri"]),
                    "ri_delta_pi_minus_baseline": float(p_rel["ri"]) - float(b_rel["ri"]),
                    "baseline_decision": b_rel["decision"],
                    "pi_decision": p_rel["decision"],
                    "decision_delta": "CHANGED" if b_rel["decision"] != p_rel["decision"] else "SAME",
                }
            )

        return {
            "baseline": baseline,
            "physics_informed": pi,
            "engine_deltas": rows,
        }

    def replay(
        self,
        csv_bytes: bytes,
        model_mode: str,
        engine_id: int,
        step: int,
        model_backend: str = "attention",
    ) -> Dict[str, object]:
        adapter = self._get_adapter(model_mode, model_backend)
        replay_df = adapter.replay_from_csv_bytes(csv_bytes, engine_id=int(engine_id), step=int(step))
        return {"rows": replay_df.to_dict(orient="records")}
