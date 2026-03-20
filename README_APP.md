# PhysGen-RUL — Streamlit Dashboard

Premium dark-theme Streamlit dashboard for aircraft engine RUL prediction.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Dashboard Pages

### 🏠 Fleet Overview
Upload a C-MAPSS CSV and run fleet inference. Displays:
- Top-level metrics: fleet size, mean RUL, RI, accept/warn/reject counts
- Fleet engine table with per-unit RUL, RI, CPC, and gate decision
- RUL distribution bar chart color-coded by gate decision
- Reliability gating pie chart and RI vs RUL scatter plot
- Per-engine deep dive: window-level RUL, attention weights, sensor telemetry

### 📡 Live Digital Twin
Real-time MQTT monitoring with auto-refresh. Shows:
- Current engine cycle, predicted/trusted RUL, RI, gate decision
- Live RUL trajectory chart
- Reliability index over time
- Decision distribution

### 📈 RUL Trajectories
Streaming replay — simulates cycle-by-cycle RUL evolution for a selected engine.

### 🔬 Batch Inference
Detailed analytics with sensor correlation matrices, distributions, and optional ground-truth evaluation.

### ⚙️ Settings
Lists available model backends, validation commands, and MQTT pipeline setup.

## Model Backend Selection

Select a backend in the sidebar:
- **attention** — Attention GRU (Baseline / Physics-Informed)
- **artifact** — LightGBM from model_artifacts.zip
- **pi-lightgbm** — LightGBM + physics constraint post-processing

## Live MQTT Pipeline

Run in separate terminals:

```powershell
# 1. MQTT subscriber + inference
python ingestion\mqtt_secure_ingest.py --broker 127.0.0.1 --port 1883 --topic engines/fd001/raw --model-mode Baseline --insecure-no-tls

# 2. Digital twin publisher
python ingestion\digital_twin_streamer.py --broker 127.0.0.1 --port 1883 --topic engines/fd001/raw --interval-sec 0.5 --cycles 3000
```

The dashboard reads from:
- `logs/live_state.json` — current engine state
- `logs/mqtt_predictions.csv` — prediction history
- `logs/mqtt_events.ndjson` — raw MQTT events

## Theme & Styling

- Dark glassmorphism theme: `assets/theme.css`
- Streamlit config: `.streamlit/config.toml`
- Inter font (Google Fonts)
- Custom decision badges (ACCEPT green / WARN amber / REJECT red)
- Plotly charts with `plotly_dark` template
