# AGENTS.md

This file gives AI coding agents repo-specific context and guardrails.

## Project Purpose

Aircraft engine RUL prediction system for NASA C-MAPSS FD001 with:
- Baseline attention model inference
- Physics-Informed (PI) model mode
- Reliability Index (RI) computation
- Post-prediction gating (`ACCEPT` / `WARN` / `REJECT`)
- Streaming replay and ablation tooling
- Modular FastAPI backend and secure MQTT ingestion scripts
- Raspberry Pi edge export/benchmark scripts
- Lightweight React UI (API-first) for Pi/browser usage

## Source of Truth

- Runtime inference core: `inference/attention_model.py`
- Reliability and gating logic: `inference/reliability.py`
- Streamlit app: `app.py`
- React UI source: `frontend/src/App.jsx`
- Backend API: `backend/api.py`
- Backend model adapter/service: `backend/model_service.py`
- MQTT ingestion/simulator: `ingestion/mqtt_secure_ingest.py`, `ingestion/mqtt_simulator.py`
- Digital twin publisher: `ingestion/digital_twin_streamer.py`
- Ablation report generation: `scripts/generate_ablation_report.py`
- Edge export/benchmark: `scripts/export_tflite_edge.py`, `scripts/benchmark_edge_inference.py`

## Hard Constraints (Do Not Break)

1. Do not redesign the model architecture.
2. Do not break Baseline mode.
3. Keep preprocessing/inference compatibility.
4. Reliability gating must remain post-prediction usage gating (not output replacement).

## Model Swapping Contract

- Swapping model implementations should happen behind `backend/model_service.py` adapters.
- Keep API request/response contracts stable when introducing new model backends.
- Preserve existing mode names (`Baseline`, `Physics-Informed`) and semantics.

## API Contracts

- `GET /health`
- `POST /v1/infer/file`
- `POST /v1/infer/json`
- `POST /v1/compare/json`
- `POST /v1/replay/json`

Any contract-breaking change must be versioned and documented.

## UI Contracts

- React UI must consume existing `/v1/*` endpoints without changing their payload semantics.
- Keep Baseline, PI, and compare behaviors equivalent to Streamlit outputs.
- UI should remain responsive on Raspberry Pi-class browsers (avoid heavy client-side dependencies).

## Real-Time Ingestion Contracts

- MQTT payload must include C-MAPSS columns:
  `unit_nr,time_cycles,op_setting_1..3,s_1..s_21`
- TLS/auth is expected for secure ingestion.
- Event/prediction logs under `logs/` are operational artifacts.

## Testing and Validation Baseline

Before finalizing major changes, run:

```powershell
python scripts\run_api.py --help
python scripts\run_headless_inference.py --help
python ingestion\mqtt_secure_ingest.py --help
python ingestion\mqtt_simulator.py --help
python ingestion\digital_twin_streamer.py --help
python scripts\export_tflite_edge.py --help
python scripts\benchmark_edge_inference.py --help
streamlit run app.py
```

## Notes About Notebooks

- `notebooks/cmapss_notebooks/nasa-eda-rul-prediction.ipynb` is included for EDA context.
- Do not treat EDA notebook cells as implementation source of truth.
- Production logic should be edited in Python modules under `inference/`, `backend/`, `scripts/`, `ingestion/`.

