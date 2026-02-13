from __future__ import annotations

from typing import Dict, List, Literal, Union

from pydantic import BaseModel, Field


ModelMode = Literal["Baseline", "Physics-Informed", "baseline", "physics-informed", "physics_informed", "pi"]
NumericValue = Union[float, int]


class RowBatchRequest(BaseModel):
    rows: List[Dict[str, NumericValue]]
    model_mode: ModelMode = "Baseline"


class CompareBatchRequest(BaseModel):
    rows: List[Dict[str, NumericValue]]


class ReplayRequest(BaseModel):
    rows: List[Dict[str, NumericValue]]
    model_mode: ModelMode = "Baseline"
    engine_id: int = Field(..., ge=1)
    step: int = Field(1, ge=1, le=10)
