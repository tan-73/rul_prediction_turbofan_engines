# Architecture Context

## High-Level Flow

1. Input data (CSV rows or streaming rows) enters the system.
2. Preprocessing/windowing runs in `inference/attention_model.py`.
3. Model predicts per-window RUL (Baseline or Physics-Informed).
4. Aggregation computes per-engine mean RUL.
5. Reliability metrics are computed in `inference/reliability.py`.
6. Gating produces `ACCEPT` / `WARN` / `REJECT` + trusted RUL.
7. Outputs are exposed via:
   - Streamlit UI (`app.py`)
   - FastAPI (`backend/api.py`)
   - CLI scripts (`scripts/run_headless_inference.py`)
   - MQTT ingestion pipeline (`ingestion/mqtt_secure_ingest.py`)

## Important Invariants

- Gating is usage control after prediction.
- Raw model prediction should remain available for analysis.
- Baseline and PI should remain comparable under same preprocessing.

## Main Extension Points

- New model backend: add/replace adapter in `backend/model_service.py`.
- New API behavior: extend `backend/api.py` with versioned routes.
- New streaming source: add ingestion adapters under `ingestion/`.
- New reports: add scripts under `scripts/` and write artifacts under `reports/`.

## Operational Modes

- `Baseline`
- `Physics-Informed`
- `Compare (Baseline vs PI)` in UI or compare endpoint

## Current Runtime Interfaces

- Streamlit:
  `streamlit run app.py`
- FastAPI:
  `python scripts\run_api.py --host 0.0.0.0 --port 8000`
- Headless CLI:
  `python scripts\run_headless_inference.py --csv <file> --mode Baseline`
- MQTT ingest:
  `python ingestion\mqtt_secure_ingest.py ...`
- MQTT simulator:
  `python ingestion\mqtt_simulator.py ...`

