# Aircraft Engine RUL System (Attention + Reliability Gating + PI Mode)

This repository contains an end-to-end aircraft engine prognostics prototype for NASA C-MAPSS:

- attention-based RUL inference (baseline checkpoint)
- physics-informed checkpoint support (PI mode)
- reliability-aware gating (`ACCEPT` / `WARN` / `REJECT`)
- streaming-style replay for cycle-by-cycle behavior inspection
- modular FastAPI backend for headless/API-first deployment
- secure MQTT ingestion and simulator scripts for real-time flows

The focus is operational trustworthiness and explainability, not only raw RMSE.

## Current Scope

- **Inference model family**: attention-based seq2seq GRU (unchanged architecture)
- **Modes in app**:
  - `Baseline`
  - `Physics-Informed`
  - `Compare (Baseline vs PI)`
- **Reliability layer**:
  - temporal stability checks
  - physics-consistency checks
  - combined Reliability Index (RI)
  - gated decision policy

## Repository Layout

- `app.py` — Streamlit UI
- `inference/attention_model.py` — model loading, preprocessing, inference, replay
- `inference/reliability.py` — RI computation and gating logic
- `scripts/evaluate_reliability.py` — CLI reliability evaluation
- `training/train_physics_informed.py` — baseline + PI retraining pipeline
- `examples/sample_cmapss_engine.csv` — single-engine demo input
- `examples/sample_cmapss_engine_dual.csv` — multi-engine demo input
- `saved_models/cmapss/` — PI checkpoint files from retraining
- `notebooks/cmapss_notebooks/attention_based_RUL/Colab_AttnPINN_RUL_FD001.ipynb` — run-all Colab training notebook
- `notebooks/cmapss_notebooks/attention_based_RUL/saved_weights/FD001/` — baseline FD001 checkpoint

## Quick Start (Local App)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install streamlit tensorflow pandas numpy scikit-learn
streamlit run app.py
```

For report plotting and colorful interactive dashboards, also install:

```powershell
pip install matplotlib plotly
```

For API + MQTT + headless runtime:

```powershell
pip install fastapi uvicorn paho-mqtt
```

For Raspberry Pi edge export + benchmark tooling:

```powershell
pip install psutil
```

If TensorFlow import fails with protobuf descriptor errors, pin protobuf:

```powershell
pip install "protobuf<=3.20.3"
```

### Input CSV format

Upload either:

- named C-MAPSS-style columns:
  - `unit_nr`, `time_cycles`, `op_setting_1`, `op_setting_2`, `op_setting_3`, `s_1` ... `s_21`
- or at least 26 raw columns in original C-MAPSS ordering

Use:

- `examples/sample_cmapss_engine.csv`
- `examples/sample_cmapss_engine_dual.csv`
- `examples/scenarios/scenario_stable_behavior.csv`
- `examples/scenarios/scenario_noisy_behavior.csv`
- `examples/scenarios/scenario_rapid_degradation.csv`

Generate/refresh curated replay scenarios:

```powershell
python scripts\generate_demo_scenarios.py
```

## Reliability Gating Workflow

The app computes per-engine RI and decision:

- `ACCEPT`: prediction trusted
- `WARN`: degraded confidence
- `REJECT`: conservative fallback policy

The app also supports:

- prediction log download (`engine_id,predicted_rul,ri,decision,trusted_rul`)
- optional ground-truth upload (`engine_id,true_rul`) for on-the-fly evaluation
- streaming replay mode for cycle-by-cycle behavior

## Reliability Evaluation CLI

```powershell
python scripts/evaluate_reliability.py --csv path\to\predictions.csv --cat-threshold 20
```

Expected CSV columns:

- required: `true_rul`, `predicted_rul`, `ri`
- optional: `decision`

## Physics-Informed Retraining

Local retraining script:

```powershell
python training\train_physics_informed.py ^
  --train-path D:\path\to\train_FD001.txt ^
  --test-path D:\path\to\test_FD001.txt ^
  --rul-path D:\path\to\RUL_FD001.txt ^
  --out-dir saved_models\cmapss\attn_pi_fd001
```

Outputs:

- `train_history.csv`
- `test_metrics.csv`
- `config.json`
- TensorFlow checkpoint files

Colab run-all notebook:

- `notebooks/cmapss_notebooks/attention_based_RUL/Colab_AttnPINN_RUL_FD001.ipynb`

## Automated Ablation Report (Baseline / Baseline+RI / PI / PI+RI)

Generate publication-ready tables and plots under `reports/`:

```powershell
python scripts\generate_ablation_report.py ^
  --test-path D:\path\to\test_FD001.txt ^
  --rul-path D:\path\to\RUL_FD001.txt ^
  --reports-dir reports ^
  --cat-threshold 20
```

Outputs:

- `reports/tables/ablation_metrics.csv`
- `reports/tables/ablation_per_engine_predictions.csv`
- `reports/figures/ablation_summary_metrics.png`
- `reports/figures/ri_error_scatter_baseline.png`
- `reports/figures/ri_error_scatter_physics_informed.png`
- `reports/run_manifest.json`

## Side-by-Side Comparison Mode in App

Run app:

```powershell
streamlit run app.py
```

In the UI, set **Model Mode** to `Compare (Baseline vs PI)` and run inference once.
The app will display direct per-engine deltas:

- predicted RUL delta (`PI - Baseline`)
- RI delta (`PI - Baseline`)
- decision delta (`SAME` / `CHANGED`)

The dashboard now includes additional colorful visual diagnostics (Plotly-enabled):

- grouped Baseline vs PI bar comparison
- delta scatter (`RI delta` vs `RUL delta`) with decision-change coloring
- decision donut charts
- RI vs raw prediction scatter with uncertainty sizing
- interactive streaming replay trajectories

## FastAPI Backend (Modular Model Service)

Run API server:

```powershell
python scripts\run_api.py --host 0.0.0.0 --port 8000
```

Key endpoints:

- `GET /health`
- `POST /v1/infer/file` (multipart file + `model_mode`)
- `POST /v1/infer/json` (JSON rows + `model_mode`)
- `POST /v1/compare/json` (JSON rows for Baseline vs PI deltas)
- `POST /v1/replay/json` (JSON rows + engine replay settings)

Modularity note:

- model inference is behind `backend/model_service.py` adapters
- model swapping can be done by replacing adapter implementation while preserving API contract
- reliability gating remains post-prediction usage gating

## Secure MQTT Real-Time Ingestion

Subscriber (TLS + auth) with prediction/event logs:

```powershell
python ingestion\mqtt_secure_ingest.py ^
  --broker your-broker-host ^
  --port 8883 ^
  --topic engines/fd001/raw ^
  --ca-cert D:\path\to\ca.crt ^
  --username your_user ^
  --password your_pass ^
  --model-mode Baseline
```

Simulator publisher from scenario CSV:

```powershell
python ingestion\mqtt_simulator.py ^
  --csv examples\scenarios\scenario_noisy_behavior.csv ^
  --broker your-broker-host ^
  --port 8883 ^
  --topic engines/fd001/raw ^
  --ca-cert D:\path\to\ca.crt ^
  --username your_user ^
  --password your_pass ^
  --delay-sec 0.2
```

Logs produced by subscriber:

- `logs/mqtt_events.ndjson`
- `logs/mqtt_predictions.csv`

## Terminal-Only Runtime (Raspberry Pi Friendly)

Single-mode inference:

```powershell
python scripts\run_headless_inference.py --csv examples\sample_cmapss_engine.csv --mode Baseline
```

Side-by-side compare:

```powershell
python scripts\run_headless_inference.py --csv examples\sample_cmapss_engine_dual.csv --compare
```

Replay output from terminal:

```powershell
python scripts\run_headless_inference.py --csv examples\sample_cmapss_engine.csv --mode Baseline --replay-engine 1 --replay-step 1
```

## Raspberry Pi Phase (Edge Export + Benchmark)

Export one-step TFLite model (Baseline example):

```powershell
python scripts\export_tflite_edge.py --mode Baseline --out-dir edge_models --quantization float16
```

Export PI model:

```powershell
python scripts\export_tflite_edge.py --mode "Physics-Informed" --out-dir edge_models --quantization float16
```

Run edge parity + latency benchmark:

```powershell
python scripts\benchmark_edge_inference.py ^
  --csv examples\sample_cmapss_engine_dual.csv ^
  --mode Baseline ^
  --tflite edge_models\baseline_one_step_fp16.tflite ^
  --reports-dir reports\edge
```

Artifacts produced:

- `reports/edge/edge_benchmark_metrics_baseline.json`
- `reports/edge/edge_benchmark_parity_baseline.csv`
- `reports/edge/edge_benchmark_summary_baseline.csv`

Notes:

- Benchmark checks prediction parity, RI parity, and decision-match rate (`ACCEPT/WARN/REJECT`) between TensorFlow and TFLite.
- Gating remains post-prediction usage gating.

## Reproducibility Freeze

Frozen artifact:

- `reproducibility/fd001_ablation_freeze.json`

Run with freeze config:

```powershell
python scripts\generate_ablation_report.py ^
  --freeze-config reproducibility\fd001_ablation_freeze.json ^
  --test-path D:\path\to\test_FD001.txt ^
  --rul-path D:\path\to\RUL_FD001.txt
```

One-command convention (if dataset is placed at freeze-default paths):

```powershell
python scripts\generate_ablation_report.py --freeze-config reproducibility\fd001_ablation_freeze.json
```

Note: the freeze config locks default experiment settings and checkpoint references.
CLI flags override freeze defaults when both are provided.

## Notes

- Intended for research prototyping and demonstration.
- Not a certified maintenance decision system.
- Baseline and PI modes are both retained for ablation and operational comparison.
