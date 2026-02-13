from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from typing import Dict, Protocol

import pandas as pd

from inference.attention_model import (
    get_model_weights_path,
    load_attention_model,
    predict_rul_detailed_from_csv,
    simulate_realtime_engine_from_df,
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


class ModelService:
    def __init__(self) -> None:
        self._adapters: Dict[str, ModelAdapter] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _normalize_mode(model_mode: str) -> str:
        mode = model_mode.strip().lower()
        if mode in {"baseline"}:
            return "Baseline"
        if mode in {"physics-informed", "physics_informed", "pi"}:
            return "Physics-Informed"
        raise ValueError(f"Unsupported model_mode: {model_mode}")

    def _get_adapter(self, model_mode: str) -> ModelAdapter:
        mode = self._normalize_mode(model_mode)
        with self._lock:
            if mode not in self._adapters:
                self._adapters[mode] = AttentionModelAdapter(mode)
            return self._adapters[mode]

    def infer(self, csv_bytes: bytes, model_mode: str) -> Dict[str, object]:
        adapter = self._get_adapter(model_mode)
        return adapter.predict_detailed_from_csv_bytes(csv_bytes)

    def compare(self, csv_bytes: bytes) -> Dict[str, object]:
        baseline = self.infer(csv_bytes, "Baseline")
        pi = self.infer(csv_bytes, "Physics-Informed")

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

    def replay(self, csv_bytes: bytes, model_mode: str, engine_id: int, step: int) -> Dict[str, object]:
        adapter = self._get_adapter(model_mode)
        replay_df = adapter.replay_from_csv_bytes(csv_bytes, engine_id=int(engine_id), step=int(step))
        return {"rows": replay_df.to_dict(orient="records")}
