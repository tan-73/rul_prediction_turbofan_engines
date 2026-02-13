from __future__ import annotations

import io
from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from backend.model_service import ModelService
from backend.schemas import CompareBatchRequest, ReplayRequest, RowBatchRequest


app = FastAPI(title="RUL Backend API", version="1.0.0")
model_service = ModelService()


def _rows_to_csv_bytes(rows: List[Dict[str, float | int]]) -> bytes:
    if not rows:
        raise ValueError("rows cannot be empty.")
    df = pd.DataFrame(rows)
    payload = io.StringIO()
    df.to_csv(payload, index=False)
    return payload.getvalue().encode("utf-8")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/infer/file")
async def infer_from_file(
    file: UploadFile = File(...),
    model_mode: str = "Baseline",
) -> Dict[str, object]:
    try:
        payload = await file.read()
        return model_service.infer(payload, model_mode=model_mode)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/infer/json")
def infer_from_rows(req: RowBatchRequest) -> Dict[str, object]:
    try:
        payload = _rows_to_csv_bytes(req.rows)
        return model_service.infer(payload, model_mode=req.model_mode)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/compare/json")
def compare_from_rows(req: CompareBatchRequest) -> Dict[str, object]:
    try:
        payload = _rows_to_csv_bytes(req.rows)
        return model_service.compare(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/replay/json")
def replay_from_rows(req: ReplayRequest) -> Dict[str, object]:
    try:
        payload = _rows_to_csv_bytes(req.rows)
        return model_service.replay(
            payload,
            model_mode=req.model_mode,
            engine_id=int(req.engine_id),
            step=int(req.step),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

