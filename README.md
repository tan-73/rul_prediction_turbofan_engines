# Aircraft Engine RUL System (Attention + Reliability Gating + PI Mode)

This repository contains an end-to-end aircraft engine prognostics prototype for NASA C-MAPSS:

- attention-based RUL inference (baseline checkpoint)
- physics-informed checkpoint support (PI mode)
- reliability-aware gating (`ACCEPT` / `WARN` / `REJECT`)
- streaming-style replay for cycle-by-cycle behavior inspection

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
