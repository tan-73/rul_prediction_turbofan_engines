# AGENTS.md

Repo context and guardrails for coding agents.

## Project Purpose

Aircraft engine RUL prediction system for NASA C-MAPSS FD001 with:

- Baseline attention model inference
- Physics-Informed (PI) model mode
- Reliability Index (RI)
- Post-prediction gating (`ACCEPT` / `WARN` / `REJECT`)
- Live MQTT digital twin monitoring in Streamlit
- FastAPI + headless runtime scripts

## Source of Truth

- Runtime inference core: `inference/attention_model.py`
- Reliability/gating logic: `inference/reliability.py`
- Streamlit UI: `app.py`
- Backend API: `backend/api.py`
- Backend model adapter/service: `backend/model_service.py`
- MQTT ingestion/simulator: `ingestion/mqtt_secure_ingest.py`, `ingestion/mqtt_simulator.py`
- Digital twin publisher: `ingestion/digital_twin_streamer.py`
- Ablation report generation: `scripts/generate_ablation_report.py`
- Validation suite: `scripts/run_validation_suite.py`, `tests/test_inference_regression.py`

## Hard Constraints (Do Not Break)

1. Do not redesign the attention model architecture in runtime modules.
2. Do not break Baseline mode.
3. Keep preprocessing/inference compatibility.
4. Reliability gating must remain post-prediction usage gating.

## Model Swapping Contract

- Swap model backends behind `backend/model_service.py` adapters.
- Keep API request/response contracts stable.
- Preserve mode names (`Baseline`, `Physics-Informed`) and semantics.

## API Contracts

- `GET /health`
- `POST /v1/infer/file`
- `POST /v1/infer/json`
- `POST /v1/compare/json`
- `POST /v1/replay/json`

Any contract-breaking change must be versioned and documented.

## Real-Time Ingestion Contracts

- MQTT payload columns:
  `unit_nr,time_cycles,op_setting_1..3,s_1..s_21`
- TLS/auth supported for secure environments.
- Local no-TLS mode supported via `--insecure-no-tls`.
- Logs under `logs/` are operational artifacts.

## Notebook Context

Notebook `notebooks/cmapss_notebooks/predicting-remaining-useful-life-turbofan-engine.ipynb`
can produce `model_artifacts.zip` (LightGBM artifacts). Treat it as optional alternate model output,
not default runtime source, until adapter integration is done.

## Validation Baseline

Before finalizing major changes, run:

```powershell
python scripts\run_api.py --help
python scripts\run_headless_inference.py --help
python ingestion\mqtt_secure_ingest.py --help
python ingestion\mqtt_simulator.py --help
python ingestion\digital_twin_streamer.py --help
python scripts\export_tflite_edge.py --help
python scripts\benchmark_edge_inference.py --help
python scripts\run_validation_suite.py
streamlit run app.py
```
