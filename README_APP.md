# Aircraft Engine RUL Predictor (Streamlit)

This app is the primary UI for the project and supports:

- CSV-based inference
- Baseline / Physics-Informed / Compare modes
- Reliability gating and diagnostics
- Streaming replay
- Live MQTT digital twin monitoring

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Live MQTT Panel

Inside Streamlit, use **Live Digital Twin Feed (MQTT)**.

Run these in separate terminals:

```powershell
python ingestion\mqtt_secure_ingest.py --broker 127.0.0.1 --port 1883 --topic engines/fd001/raw --model-mode Baseline --insecure-no-tls
python ingestion\digital_twin_streamer.py --broker 127.0.0.1 --port 1883 --topic engines/fd001/raw --interval-sec 0.5 --cycles 3000
```

Live panel reads:

- `logs/live_state.json`
- `logs/mqtt_predictions.csv`
- `logs/mqtt_events.ndjson`

## Notebook Training Artifact

The notebook
`notebooks/cmapss_notebooks/predicting-remaining-useful-life-turbofan-engine.ipynb`
can export model files on Kaggle to `model_artifacts.zip`.

A local archive currently exists at project root:

- `model_artifacts.zip`

This artifact is not yet wired as default runtime inference; integrate through adapter layer first.
