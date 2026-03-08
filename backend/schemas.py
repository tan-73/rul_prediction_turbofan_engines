from __future__ import annotations

from typing import Dict, List, Literal, Union

from pydantic import BaseModel, Field


ModelMode = Literal["Baseline", "Physics-Informed", "baseline", "physics-informed", "physics_informed", "pi"]
ModelBackend = Literal["attention", "artifact", "default", "lightgbm", "model_artifact", "model-artifact"]
NumericValue = Union[float, int]


class RowBatchRequest(BaseModel):
    rows: List[Dict[str, NumericValue]]
    model_mode: ModelMode = "Baseline"
    model_backend: ModelBackend = "attention"


class CompareBatchRequest(BaseModel):
    rows: List[Dict[str, NumericValue]]


class ReplayRequest(BaseModel):
    rows: List[Dict[str, NumericValue]]
    model_mode: ModelMode = "Baseline"
    model_backend: ModelBackend = "attention"
    engine_id: int = Field(..., ge=1)
    step: int = Field(1, ge=1, le=10)
