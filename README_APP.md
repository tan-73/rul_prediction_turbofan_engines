# Aircraft Engine RUL Predictor (Demo App)

This app loads the existing pretrained attention-based GRU model from this repository and predicts Remaining Useful Life (RUL) from uploaded sensor CSV data.

## Run the app

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install streamlit tensorflow pandas numpy scikit-learn
```

3. Run:

```bash
streamlit run app.py
```

On the app page, choose **Model Mode**:

- `Baseline` (original FD001 attention checkpoint)
- `Physics-Informed` (uploaded PI checkpoint from retraining)

## Expected CSV format

The app supports either of these input styles:

- **Named C-MAPSS columns**:
  - `unit_nr`, `time_cycles`, `op_setting_1`, `op_setting_2`, `op_setting_3`, `s_1` ... `s_21`
- **Raw numeric columns**:
  - At least 26 columns in C-MAPSS original order (`0..25` style).

Notes:
- Each row is one cycle record.
- One file can contain one or multiple engines (`unit_nr`).
- Each engine must have at least 30 rows (window length used in the original notebook).

## Example use case

An aircraft maintenance engineer uploads recent engine run-to-date sensor readings.
The app runs the pretrained attention model and returns:

- Predicted RUL
- A simple status:
  - `Healthy`
  - `Maintenance Required Soon`

This is for demonstration/prototyping and not intended for production maintenance decisions.

## Dashboard visuals included

After inference, the app also shows an **Advanced Insights** section with:

- window-level RUL trend and spread (mean/min/max/std)
- attention weight profile for the latest prediction window
- selected sensor trend lines over cycles
- cycle continuity and data-quality checks
- sensor correlation matrix and histogram snapshot (EDA-style)

## Reliability-aware gating

The app now adds an operational reliability layer on top of model predictions:

- computes a **Reliability Index (RI)** from:
  - temporal stability across overlapping window predictions
  - physics-consistency checks (monotonicity, smoothness, boundary sanity)
- gates usage of predictions as:
  - `ACCEPT`
  - `WARN`
  - `REJECT` (with conservative fallback RUL)

This gating layer does not change model architecture or retrain weights. It controls how predictions are trusted operationally.

## Streaming-style demonstration

In the **Reliability Gating** tab, a cycle-by-cycle replay simulates real-time operation:

- runs rolling inference as cycles arrive
- updates RI and decision state each step
- visualizes raw vs trusted RUL trends over time

## Reliability evaluation workflow

In the **Reliability Gating** tab:

- Download model output log as CSV (`engine_id,predicted_rul,ri,decision,trusted_rul`)
- Upload optional ground-truth CSV (`engine_id,true_rul`)
- App computes:
  - mean absolute error
  - RI-error correlation
  - catastrophic error rate (|error| > 20)
  - accept-rate under gating
