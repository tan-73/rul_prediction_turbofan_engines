# PhysGen-RUL — Aircraft Engine Remaining Useful Life Prediction

Physics-Integrated Generative Edge-AI for Aero-Engine Prognostics  
**IEEE IES GenAI Challenge 2026 · NASA C-MAPSS FD001**

---

## Features

- **Multi-model inference** — Attention GRU (Baseline / Physics-Informed), LightGBM, and Physics-Informed LightGBM via plug-and-play backend registry
- **cVAE Trajectory Generator** — probabilistic RUL degradation paths with confidence intervals (the core Generative AI component)
- **LLM Maintenance Briefs** — AI-generated natural language reports with optional Gemini API integration
- **SHAP-style Sensor Attribution** — per-sensor contribution analysis with anomaly detection
- **Reliability Index (RI)** — calibrated prediction trustworthiness scoring
- **Post-prediction gating** — `ACCEPT` / `WARN` / `REJECT` decisions with reason codes
- **CPC scoring** — Counterfactual Physical Consistency checks on predictions
- **Premium Streamlit dashboard** — dark glassmorphism theme, 6 navigation pages, rich Plotly charts, interactive model internals
- **Model Internals visual explainer** — animated pipeline flow, clickable model core, engine cross-section sensor map, and Reliability Gate Simulator
- **Live MQTT digital twin** — real-time sensor streaming with Node-RED integration
- **FastAPI backend** — RESTful endpoints with model backend discovery
- **Edge deployment** — TFLite export and benchmark scripts

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Model Backends

The system uses a **BackendRegistry** pattern for plug-and-play model switching:

| Backend | Description | Modes |
|---------|-------------|-------|
| `attention` | Attention GRU encoder-decoder (default) | Baseline, Physics-Informed |
| `artifact` | LightGBM from `model_artifacts.zip` | Baseline |
| `pi-lightgbm` | LightGBM + physics constraint post-processing | Baseline |

Select backends in the Streamlit sidebar or via CLI:

```powershell
python scripts\run_headless_inference.py --csv examples\sample_cmapss_engine.csv --mode Baseline --backend attention
python scripts\run_headless_inference.py --csv examples\sample_cmapss_engine.csv --mode Baseline --backend pi-lightgbm
```

## Streamlit Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Fleet Overview | Upload CSV, fleet inference, engine table, RUL charts, per-engine deep dive with GenAI panels |
| 📡 Live Digital Twin | Real-time MQTT feed, auto-refresh, decision tracking |
| 📈 RUL Trajectories | Streaming replay — cycle-by-cycle RUL evolution |
| 🧬 Model Internals | Interactive visual explainer for pipeline flow, sensor groups, attention, RI gating, and RUL futures |
| 🔬 Batch Inference | Detailed analytics, sensor correlation, ground-truth evaluation |
| ⚙️ Settings | Backend discovery, validation commands, MQTT setup |

### Model Internals Visual Explainer

The **🧬 Model Internals** page turns a single engine prediction into a visual systems map:

- **Inference Pipeline Flow** — animated path from raw `CSV/MQTT` telemetry through preprocessing, backend inference, RUL prediction, RI gating, and trusted maintenance action
- **Immersive Model Core** — clickable canvas with sensor nodes, attention arcs, RI gate ring, backend/model core, physics field, and future RUL fan
- **Engine Cross-Section Sensor Map** — turbofan-style layout where sensor nodes are grouped by thermal, pressure, mechanical, and flow categories
- **Reliability Gate Simulator** — interactive sliders for prediction spread, monotonicity violations, smoothness noise, and base RUL; uses the same RI/gating functions as runtime

If a selected artifact backend cannot build a compatible internals payload, the page falls back to the `attention` backend visualization while preserving the rest of the dashboard.

### Per-Engine GenAI Panels (Fleet Overview Deep Dive)

- **🔮 Probabilistic RUL Trajectories** — cVAE / Monte Carlo fan plot with 95% and 50% confidence bands, median/mean lines, and sample trajectories
- **🔬 Sensor Attribution (SHAP-style)** — horizontal bar chart showing per-sensor contribution to RUL risk, grouped by physical category (thermal, pressure, mechanical, flow)
- **🤖 AI Maintenance Brief** — structured natural language report with urgency level, decision rationale, diagnostic flags, and recommended actions

## Local Digital Twin (MQTT)

Requires [Mosquitto](https://mosquitto.org/) running on `localhost:1883`.  
Run in separate terminals:

```powershell
# 1. MQTT ingest + inference subscriber
python ingestion\mqtt_secure_ingest.py --broker 127.0.0.1 --port 1883 --topic engines/fd001/raw --model-mode Baseline --insecure-no-tls

# 2. Digital twin publisher
python ingestion\digital_twin_streamer.py --broker 127.0.0.1 --port 1883 --topic engines/fd001/raw --interval-sec 0.5 --cycles 3000

# 3. Streamlit app
streamlit run app.py
```

### Node-RED Digital Twin

A Node-RED flow (`flows.json`) simulates realistic engine sensor data with:
- 6 sensor sections: vibration, thermal, pressure, flow, mechanical, RUL model
- cVAE-RUL pipeline with physics risk and CPC scoring
- MQTT publishing to `engine/rul` and `engine/sensors/all`
- Dashboard controls: reset engine, force failure, inject anomaly, degradation speed slider

Import `flows.json` into Node-RED and configure the broker to `localhost:1883`.

## API

```powershell
python scripts\run_api.py --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/backends` | GET | List available model backends |
| `/v1/infer/file` | POST | Inference from uploaded CSV |
| `/v1/infer/json` | POST | Inference from JSON row batch |
| `/v1/compare/json` | POST | Baseline vs PI side-by-side comparison |
| `/v1/replay/json` | POST | Streaming cycle-by-cycle replay |

## Input Format (CSV)

Accepted formats:
- Named C-MAPSS columns: `unit_nr,time_cycles,op_setting_1..3,s_1..s_21`
- Or at least 26 raw columns in original C-MAPSS order

Example files:
- `examples/sample_cmapss_engine.csv`
- `examples/sample_cmapss_engine_dual.csv`
- `examples/scenarios/scenario_stable_behavior.csv`
- `examples/scenarios/scenario_rapid_degradation.csv`

### Gate Demo Fixtures

Verified demo fixtures are provided for each post-prediction gate decision using `Baseline` + `attention`:

| Decision | Files |
|----------|-------|
| `ACCEPT` | `examples/scenarios/gate_accept_demo.csv`, `gate_accept_demo_2.csv`, `gate_accept_demo_3.csv` |
| `WARN` | `examples/scenarios/gate_warn_demo.csv`, `gate_warn_demo_2.csv`, `gate_warn_demo_3.csv` |
| `REJECT` | `examples/scenarios/gate_reject_demo.csv`, `gate_reject_demo_2.csv`, `gate_reject_demo_3.csv` |

Use these CSVs to demo the Fleet Overview, Model Internals, and Reliability Gate Simulator behavior.

## Project Structure

```
├── app.py                           # Streamlit dashboard (primary UI + Model Internals visuals)
├── assets/theme.css                 # Premium dark theme CSS
├── .streamlit/config.toml           # Streamlit dark theme config
├── backend/
│   ├── api.py                       # FastAPI endpoints
│   ├── model_service.py             # BackendRegistry + ModelService
│   ├── artifact_backend.py          # LightGBM artifact adapter
│   ├── pi_lightgbm_backend.py       # Physics-Informed LightGBM adapter
│   └── schemas.py                   # Pydantic request schemas
├── inference/
│   ├── attention_model.py           # Attention GRU model + preprocessing
│   ├── reliability.py               # RI, gating, reason codes, evaluation
│   ├── cvae_trajectory.py           # cVAE trajectory generator (GenAI)
│   ├── llm_explainer.py             # LLM maintenance brief engine (GenAI)
│   └── shap_explainer.py            # SHAP sensor attribution (GenAI)
├── ingestion/
│   ├── mqtt_secure_ingest.py        # MQTT subscriber + inference + logging
│   ├── mqtt_simulator.py            # CSV-to-MQTT publisher
│   ├── digital_twin_streamer.py     # Synthetic digital twin publisher
│   └── nodered_subscriber.py        # Node-RED MQTT subscriber for Streamlit
├── scripts/
│   ├── run_api.py                   # FastAPI launcher
│   ├── run_headless_inference.py    # CLI inference runner
│   ├── run_validation_suite.py      # Validation + golden output generation
│   ├── generate_ablation_report.py  # Ablation study report generator
│   ├── export_tflite_edge.py        # TFLite edge model export
│   └── benchmark_edge_inference.py  # Edge inference benchmarking
├── tests/
│   └── test_inference_regression.py # Golden output regression tests
├── examples/scenarios/gate_*_demo*.csv # ACCEPT/WARN/REJECT demo fixtures
├── flows.json                       # Node-RED digital twin flow
├── model_artifacts.zip              # LightGBM exported model
└── PROJECT_REPORT.md                # Full academic project report
```

## Validation

```powershell
python scripts\run_validation_suite.py
python -m pytest tests\test_inference_regression.py -q
```

## License

MIT — See `LICENSE.md`
