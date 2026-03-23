# Architecture Context

## High-Level Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  INPUT: CSV Upload │ MQTT Stream │ Node-RED │ API JSON           │
└───────┬────────────┴──────┬──────┴────┬─────┴────────┬───────────┘
        │                   │           │              │
        ▼                   ▼           ▼              ▼
┌──────────────────────────────────────────────────────────────────┐
│  PREPROCESSING: CSV Parse → Standardize → Scale → Window (30,1) │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  BACKEND REGISTRY                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐         │
│  │ attention    │  │ artifact    │  │ pi-lightgbm      │         │
│  │ GRU+Attn    │  │ LightGBM   │  │ LightGBM+Physics │         │
│  │ Base / PI   │  │ Base only  │  │ Base + CPC       │         │
│  └─────────────┘  └─────────────┘  └──────────────────┘         │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  POST-PREDICTION: RI → Gating → Trusted RUL → Reason Codes      │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  OUTPUT: Streamlit │ FastAPI │ CLI │ TFLite Edge │ Logs          │
└──────────────────────────────────────────────────────────────────┘
```

## Important Invariants

1. Gating is post-prediction usage control — never modifies raw model output.
2. Raw model prediction remains available for analysis alongside trusted RUL.
3. Baseline and PI must stay comparable under shared preprocessing.
4. Baseline mode must never be broken.
5. BackendRegistry is the single source of truth for model adapters.
6. cVAE, SHAP, and LLM modules remain independent — usable without each other.
7. LLM explainer always works without API key (template fallback).

## Backend Registry

All model backends are registered via `BackendRegistry` in `backend/model_service.py`:

```python
registry.register(name, factory, description, aliases)
```

Currently registered:
- `attention` (default) — `AttentionModelAdapter` — Baseline + Physics-Informed
- `artifact` — `ArtifactModelAdapter` — LightGBM from `model_artifacts.zip`
- `pi-lightgbm` — `PILightGBMAdapter` — LightGBM + physics post-processing + CPC

## Main Extension Points

- **New model backend**: implement adapter with `predict_detailed_from_csv_bytes()` and `replay_from_csv_bytes()`, then register in `model_service.py`.
- **API behavior changes**: extend/version routes in `backend/api.py`.
- **New stream source**: add under `ingestion/`.
- **New reports**: add under `scripts/` and emit to `reports/`.
- **Physics thresholds**: update `PHYSICS_THRESHOLDS` in `backend/pi_lightgbm_backend.py`.

## Operational Modes

- `Baseline` — standard model inference
- `Physics-Informed` — physics-constrained model inference (attention backend) or physics post-processing (pi-lightgbm backend)
- `Compare (Baseline vs PI)` — side-by-side comparison

## Runtime Interfaces

| Interface | Command |
|-----------|---------|
| Streamlit | `streamlit run app.py` |
| FastAPI | `python scripts\run_api.py --host 0.0.0.0 --port 8000` |
| Headless CLI | `python scripts\run_headless_inference.py --csv <file> --mode Baseline --backend attention` |
| MQTT ingest | `python ingestion\mqtt_secure_ingest.py --broker 127.0.0.1 --port 1883 --topic engines/fd001/raw --model-mode Baseline --insecure-no-tls` |
| Digital twin | `python ingestion\digital_twin_streamer.py --broker 127.0.0.1 --port 1883 --topic engines/fd001/raw --interval-sec 0.5` |
| Node-RED sub | `python ingestion\nodered_subscriber.py --broker 127.0.0.1 --port 1883` |
| MQTT simulator | `python ingestion\mqtt_simulator.py ...` |

## Node-RED Digital Twin

`flows.json` defines a Node-RED flow simulating realistic engine sensors:
- 6 sensor groups: vibration, thermal, pressure, flow, mechanical, RUL model
- cVAE-RUL pipeline with physics risk scoring and CPC
- MQTT output to `engine/rul` and `engine/sensors/all` topics
- Dashboard controls: reset, force failure, inject anomaly, degradation speed
- Broker config: `mqtt_broker_local` → `localhost:1883`

## Notebook Model Artifact Context

The notebook `notebooks/cmapss_notebooks/predicting-remaining-useful-life-turbofan-engine.ipynb`
exports Kaggle artifacts to `model_artifacts.zip`. This is integrated via:
- `backend/artifact_backend.py` → `artifact` backend
- `backend/pi_lightgbm_backend.py` → `pi-lightgbm` backend

Note: `model_artifacts.zip` has sklearn version dependencies that must match the runtime environment.
