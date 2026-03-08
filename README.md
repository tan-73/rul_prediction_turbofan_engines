# Aircraft Engine RUL System

Streamlit-first aircraft engine RUL project for NASA C-MAPSS FD001 with:

- Baseline attention model inference
- Physics-Informed model inference
- Reliability Index (RI) and post-prediction gating (`ACCEPT` / `WARN` / `REJECT`)
- Live MQTT digital twin ingestion and monitoring
- Headless/API scripts and edge export/benchmark utilities

## Primary Runtime

- Main UI: `streamlit run app.py`
- Live MQTT panel is integrated inside Streamlit (`Live Digital Twin Feed`)

The React app is kept in `frontend/`, but Streamlit is the primary operational UI.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If TensorFlow import shows protobuf descriptor errors:

```powershell
pip install "protobuf<=3.20.3"
```

## Local Digital Twin (Mosquitto, No TLS)

Run in separate terminals:

1. MQTT ingest + inference subscriber

```powershell
python ingestion\mqtt_secure_ingest.py ^
  --broker 127.0.0.1 ^
  --port 1883 ^
  --topic engines/fd001/raw ^
  --model-mode Baseline ^
  --insecure-no-tls
```

2. Digital twin publisher

```powershell
python ingestion\digital_twin_streamer.py ^
  --broker 127.0.0.1 ^
  --port 1883 ^
  --topic engines/fd001/raw ^
  --interval-sec 0.5 ^
  --cycles 3000
```

3. Streamlit app

```powershell
streamlit run app.py
```

Streamlit reads:

- `logs/mqtt_events.ndjson`
- `logs/mqtt_predictions.csv`
- `logs/live_state.json`

## Input Format (CSV Inference)

Accepted:

- Named C-MAPSS columns:
  `unit_nr,time_cycles,op_setting_1,op_setting_2,op_setting_3,s_1..s_21`
- Or at least 26 raw columns in original C-MAPSS order

Examples:

- `examples/sample_cmapss_engine.csv`
- `examples/sample_cmapss_engine_dual.csv`
- `examples/scenarios/scenario_stable_behavior.csv`
- `examples/scenarios/scenario_noisy_behavior.csv`
- `examples/scenarios/scenario_rapid_degradation.csv`

## Core Modules

- `app.py`: Streamlit dashboard (batch + live MQTT)
- `inference/attention_model.py`: model loading/preprocessing/inference/replay
- `inference/reliability.py`: RI, gating, reason codes, evaluation utilities
- `backend/model_service.py`: model adapter/service boundary
- `backend/api.py`: FastAPI endpoints
- `ingestion/mqtt_secure_ingest.py`: MQTT subscriber + model inference + logs
- `ingestion/mqtt_simulator.py`: CSV-to-MQTT publisher
- `ingestion/digital_twin_streamer.py`: synthetic digital twin publisher

## Notebook Training and Export

Notebook:

- `notebooks/cmapss_notebooks/predicting-remaining-useful-life-turbofan-engine.ipynb`

It is prepared for Kaggle `Run All` and includes final cells to export trained artifacts.

Expected Kaggle outputs:

- `/kaggle/working/model_artifacts/`
- `/kaggle/working/model_artifacts.zip`

Local archive currently present in project root:

- `model_artifacts.zip`

This artifact comes from notebook training (LightGBM workflow). Validate it against project RUL metrics before replacing default attention-model inference.

## Validation and Regression

Generate validation summaries and golden outputs:

```powershell
python scripts\run_validation_suite.py
```

Run regression test:

```powershell
python -m pytest tests\test_inference_regression.py -q
```

## Other Useful Commands

Headless inference:

```powershell
python scripts\run_headless_inference.py --csv examples\sample_cmapss_engine.csv --mode Baseline
```

API:

```powershell
python scripts\run_api.py --host 0.0.0.0 --port 8000
```

Ablation report:

```powershell
python scripts\generate_ablation_report.py --freeze-config reproducibility\fd001_ablation_freeze.json --test-path <test_FD001.txt> --rul-path <RUL_FD001.txt>
```
