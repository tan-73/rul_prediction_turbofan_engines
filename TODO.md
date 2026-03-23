# TODO

## Completed ✅

- [x] Plug-and-play BackendRegistry in `model_service.py`
- [x] Attention GRU adapter (Baseline + Physics-Informed)
- [x] LightGBM artifact adapter (`model_artifacts.zip`)
- [x] Physics-Informed LightGBM adapter with CPC post-processing
- [x] `/v1/backends` API endpoint for backend discovery
- [x] Premium dark Streamlit UI with 5-page sidebar navigation
- [x] Node-RED digital twin flow with MQTT broker config
- [x] Node-RED MQTT subscriber (`ingestion/nodered_subscriber.py`)
- [x] Academic project report (`PROJECT_REPORT.md`)
- [x] cVAE trajectory generator (`inference/cvae_trajectory.py`) — Monte Carlo + cVAE modes
- [x] LLM explanation engine (`inference/llm_explainer.py`) — Template + Gemini API
- [x] SHAP sensor attribution (`inference/shap_explainer.py`) — Attention + statistical modes
- [x] GenAI panels integrated into Streamlit Fleet Overview deep dive
- [x] Fix `model_artifacts.zip` sklearn compatibility (installed LightGBM 4.6.0 in venv)

## Current Priority

1. Train cVAE model on FD001 training data (currently using Monte Carlo fallback).
2. Test LLM mode with Gemini API key (`GEMINI_API_KEY` env var).
3. Run fair benchmark: attention vs LightGBM vs PI-LightGBM on same FD001 test split.

## Validation Tasks

- Run `python scripts\run_validation_suite.py`
- Run `python -m pytest tests\test_inference_regression.py -q`
- Verify Streamlit GenAI panels (trajectory fan, SHAP chart, AI brief)
- Test all 3 backends via headless CLI

## Later Tasks

- Dockerize entire application (Streamlit + API + Mosquitto + Node-RED)
- Add probabilistic predictions (Monte Carlo Dropout / Deep Ensembles)
- Add integration tests for MQTT-to-Streamlit live flow
- Add Kubernetes deployment manifests for cloud scaling
- Extend to FD002/FD003/FD004 datasets
- Federated learning across multiple fleet operators
- Mobile companion app for field engineers
