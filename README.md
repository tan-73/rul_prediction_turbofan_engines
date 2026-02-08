# Data-Driven Remaining Useful Life (RUL) Prediction

This repository contains reproducible experiments for Remaining Useful Life (RUL) prediction on C-MAPSS, including classical ML and deep learning approaches.

Project page: [biswajitsahoo1111.github.io/rul_codes_open](https://biswajitsahoo1111.github.io/rul_codes_open/)

## Repository Structure

- `notebooks/` — original notebooks for preprocessing, training, and result reproduction
- `saved_models/` — pretrained model artifacts used by notebooks
- `inference/attention_model.py` — inference-focused module for pretrained attention-based GRU model
- `app.py` — Streamlit demo app for aircraft-engine RUL prediction
- `examples/sample_cmapss_engine.csv` — tiny demo CSV input
- `README_APP.md` — focused app usage notes

## Streamlit Demo App (Inference Only)

The demo app:
- loads the existing pretrained attention-based model weights
- reuses the same preprocessing approach used in the repository notebooks
- runs inference on uploaded sensor CSV files
- shows prediction + lightweight analytics/EDA visuals

### Run Locally

From project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install streamlit tensorflow pandas numpy scikit-learn
streamlit run app.py
```

### Input Format

Upload either:
- named C-MAPSS-like columns:
  - `unit_nr`, `time_cycles`, `op_setting_1`, `op_setting_2`, `op_setting_3`, `s_1` ... `s_21`
- or at least 26 raw columns in original C-MAPSS ordering

Quick test file:
- `examples/sample_cmapss_engine.csv`

## Notes

- Existing notebooks are kept intact.
- This app is for demo/prototyping; not for production maintenance decisions.

## Citation

For attribution, cite this project as:

```bibtex
@misc{Sahoo_Data-Driven_Remaining_Useful_2020,
  author = {Sahoo, Biswajit},
  doi = {10.5281/zenodo.5890595},
  month = {9},
  title = {Data-Driven Remaining Useful Life (RUL) Prediction},
  url = {https://biswajitsahoo1111.github.io/rul_codes_open/},
  year = {2020}
}
```

Please cite original datasets separately.
