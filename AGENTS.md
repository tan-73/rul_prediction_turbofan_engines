# AGENTS.md

Repo context and guardrails for coding agents.

## Project Purpose

Aircraft engine RUL prediction system for NASA C-MAPSS FD001 with:

- Plug-and-play multi-model backend (Attention GRU, LightGBM, PI-LightGBM)
- Baseline and Physics-Informed model modes
- Reliability Index (RI) and CPC (Counterfactual Physical Consistency) scoring
- Post-prediction gating (`ACCEPT` / `WARN` / `REJECT`)
- Premium Streamlit dashboard with 5 navigation pages
- Live MQTT digital twin monitoring (Node-RED + Mosquitto)
- FastAPI + headless runtime scripts
- Edge deployment support (TFLite export)

## Source of Truth

- Runtime inference core: `inference/attention_model.py`
- Reliability/gating logic: `inference/reliability.py`
- Streamlit UI: `app.py`
- Theme/styling: `assets/theme.css`, `.streamlit/config.toml`
- Backend API: `backend/api.py`
- Backend registry + model service: `backend/model_service.py`
- LightGBM artifact adapter: `backend/artifact_backend.py`
- Physics-Informed LightGBM adapter: `backend/pi_lightgbm_backend.py`
- MQTT ingestion/simulator: `ingestion/mqtt_secure_ingest.py`, `ingestion/mqtt_simulator.py`
- Digital twin publisher: `ingestion/digital_twin_streamer.py`
- Node-RED subscriber: `ingestion/nodered_subscriber.py`
- Node-RED flow definition: `flows.json`
- Ablation report generation: `scripts/generate_ablation_report.py`
- Validation suite: `scripts/run_validation_suite.py`, `tests/test_inference_regression.py`
- Academic report: `PROJECT_REPORT.md`

## Hard Constraints (Do Not Break)

1. Do not redesign the attention model architecture in runtime modules.
2. Do not break Baseline mode.
3. Keep preprocessing/inference compatibility.
4. Reliability gating must remain post-prediction usage gating.
5. BackendRegistry must remain the single source of truth for model adapters.
6. Do not modify physics thresholds in `pi_lightgbm_backend.py` without domain justification.

## Model Swapping Contract

- All model backends are registered via `BackendRegistry` in `backend/model_service.py`.
- New backends plug in with `registry.register(name, factory, description, aliases)`.
- Keep API request/response contracts stable.
- Preserve mode names (`Baseline`, `Physics-Informed`) and semantics.
- The `pi-lightgbm` backend accepts only `Baseline` mode (physics applied post-prediction).

## API Contracts

- `GET /health`
- `GET /v1/backends`
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
- Node-RED publishes to `engine/rul` and `engine/sensors/all` topics.
- Logs under `logs/` are operational artifacts.

## Notebook Context

Notebook `notebooks/cmapss_notebooks/predicting-remaining-useful-life-turbofan-engine.ipynb`
produces `model_artifacts.zip` (LightGBM artifacts). This is integrated via:
- `backend/artifact_backend.py` → `artifact` backend
- `backend/pi_lightgbm_backend.py` → `pi-lightgbm` backend (adds physics constraints)

Note: `model_artifacts.zip` has sklearn version dependencies that must match the runtime.

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
