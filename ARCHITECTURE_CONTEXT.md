# Architecture Context

## High-Level Flow

1. Input arrives either from uploaded CSV (batch) or MQTT stream (live).
2. Preprocessing and windowing happen in `inference/attention_model.py`.
3. Model predicts per-window RUL using `Baseline` or `Physics-Informed` weights.
4. Per-engine aggregation computes mean raw RUL.
5. Reliability metrics are computed in `inference/reliability.py`.
6. Gating produces `ACCEPT/WARN/REJECT` and trusted RUL.
7. Outputs are surfaced in:
   - Streamlit (`app.py`) for batch + live MQTT monitoring
   - FastAPI (`backend/api.py`) for API/headless usage
   - CLI scripts under `scripts/`

## Important Invariants

- Gating is post-prediction usage control.
- Raw model prediction remains available for analysis.
- Baseline and PI must stay comparable under shared preprocessing.
- Baseline mode must never be broken.

## Main Extension Points

- New backend model: add adapter/service logic in `backend/model_service.py`.
- API behavior changes: extend/version routes in `backend/api.py`.
- New stream source: add under `ingestion/`.
- New reports: add under `scripts/` and emit to `reports/`.

## Operational Modes

- `Baseline`
- `Physics-Informed`
- `Compare (Baseline vs PI)`

## Runtime Interfaces

- Streamlit:
  `streamlit run app.py`
- FastAPI:
  `python scripts\run_api.py --host 0.0.0.0 --port 8000`
- Headless CLI:
  `python scripts\run_headless_inference.py --csv <file> --mode Baseline`
- MQTT ingest:
  `python ingestion\mqtt_secure_ingest.py ...`
- Digital twin publisher:
  `python ingestion\digital_twin_streamer.py ...`
- MQTT CSV simulator:
  `python ingestion\mqtt_simulator.py ...`

## Notebook Model Artifact Context

The notebook `notebooks/cmapss_notebooks/predicting-remaining-useful-life-turbofan-engine.ipynb`
exports Kaggle artifacts to `model_artifacts.zip`. This is a secondary model pipeline and should
be integrated behind adapter boundaries when used for runtime inference.
