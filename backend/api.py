from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.model_service import ModelService
from backend.schemas import CompareBatchRequest, ReplayRequest, RowBatchRequest


app = FastAPI(title="RUL Backend API", version="1.0.0")
model_service = ModelService()
REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="ui-assets")


def _rows_to_csv_bytes(rows: List[Dict[str, Union[float, int]]]) -> bytes:
    if not rows:
        raise ValueError("rows cannot be empty.")
    df = pd.DataFrame(rows)
    payload = io.StringIO()
    df.to_csv(payload, index=False)
    return payload.getvalue().encode("utf-8")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def ui_home():
    if FRONTEND_INDEX.exists():
        return FileResponse(str(FRONTEND_INDEX))
    return JSONResponse(
        {
            "status": "ok",
            "message": "Frontend build not found. Build React UI in frontend/ and rerun API.",
            "api_docs": "/docs",
        }
    )


@app.post("/v1/infer/file")
async def infer_from_file(
    file: UploadFile = File(...),
    model_mode: str = "Baseline",
    model_backend: str = "attention",
) -> Dict[str, object]:
    try:
        payload = await file.read()
        return model_service.infer(payload, model_mode=model_mode, model_backend=model_backend)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/infer/json")
def infer_from_rows(req: RowBatchRequest) -> Dict[str, object]:
    try:
        payload = _rows_to_csv_bytes(req.rows)
        return model_service.infer(payload, model_mode=req.model_mode, model_backend=req.model_backend)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/compare/json")
def compare_from_rows(req: CompareBatchRequest, model_backend: str = "attention") -> Dict[str, object]:
    try:
        payload = _rows_to_csv_bytes(req.rows)
        return model_service.compare(payload, model_backend=model_backend)
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
            model_backend=req.model_backend,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

