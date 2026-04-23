# Physics-Integrated RUL Prediction for Aircraft Turbofan Engines

## 1. Title

**Physics-Integrated Generative Edge-AI for Aero-Engine Prognostics: Remaining Useful Life Prediction on NASA C-MAPSS FD001**

Short presentation title:

**Aircraft Engine Remaining Useful Life Prediction Using Attention GRU, LightGBM, Reliability Gating, and Explainable Model Internals**

---

## 2. Abstract

This project is a data-science-driven predictive maintenance system for estimating the **Remaining Useful Life (RUL)** of aircraft turbofan engines using the NASA **C-MAPSS FD001** dataset. The main objective is to predict how many operational cycles remain before an engine reaches failure, while also quantifying the trustworthiness of the prediction and explaining the result in an operator-friendly way.

The project uses multivariate time-series sensor data containing engine identifiers, cycle numbers, 3 operational settings, and 21 sensor measurements. The primary predictive model is an **Attention-based GRU encoder-decoder**, which learns temporal degradation behavior from 30-cycle windows. A Physics-Informed variant of the attention model is also supported, and the backend architecture allows switching between Attention GRU, LightGBM artifact inference, and Physics-Informed LightGBM. This makes the project more than a single-model notebook: it is a modular predictive maintenance platform.

A key contribution is the addition of a **Reliability Index (RI)** and post-prediction operational gating. Instead of blindly trusting every RUL estimate, the system evaluates prediction stability, spread, monotonicity, smoothness, and boundary consistency. Based on RI, each prediction is classified as `ACCEPT`, `WARN`, or `REJECT`. This is important in real-world maintenance because a wrong prediction can lead either to unsafe operation or unnecessary maintenance cost.

The project also includes explainability and decision-support features. Sensor attribution identifies which sensors contribute most to risk, trajectory generation produces probabilistic future RUL paths, and a maintenance explainer creates readable summaries. The Streamlit dashboard includes fleet-level inference, live MQTT digital twin monitoring, replay analysis, batch analytics, and a **Model Internals** page that visually explains the pipeline, sensor groups, model core, Reliability Gate, and future RUL behavior.

Experimental results show that the Attention GRU Baseline model achieves approximately **RMSE 14.21 cycles** on FD001, while the Physics-Informed Attention variant achieves approximately **RMSE 16.46 cycles** with improved physical consistency. Demo and validation fixtures show successful behavior across `ACCEPT`, `WARN`, and `REJECT` gate decisions. Overall, the project demonstrates a complete data science workflow: dataset understanding, preprocessing, model development, reliability analysis, explainability, deployment, and visual communication.

---

## 3. Novelty

The novelty of this project is not only in predicting RUL, but in converting a raw ML prediction into a **reliable, explainable, and deployable data science system**.

### 3.1 Reliability-Aware Prediction Instead of Only Point Prediction

Most RUL projects stop after predicting a numeric RUL value. In this project, every prediction is evaluated using a **Reliability Index (RI)**. The RI combines:

- prediction stability across recent model windows
- coefficient of variation
- spread ratio
- monotonic degradation behavior
- smoothness of the prediction trajectory
- boundary compliance between 0 and early RUL limit

This makes the system more practical because maintenance decisions should not depend only on the predicted value, but also on how trustworthy that value is.

### 3.2 Post-Prediction Operational Gating

The project introduces a decision layer:

| RI Range | Gate | Meaning |
|----------|------|---------|
| RI >= 0.75 | `ACCEPT` | Prediction is reliable enough for operational use |
| 0.45 <= RI < 0.75 | `WARN` | Prediction is usable but should be monitored carefully |
| RI < 0.45 | `REJECT` | Prediction is not reliable; conservative fallback is applied |

This is novel from a data science deployment point of view because it treats model output as a risk-managed decision, not just a number.

### 3.3 Plug-and-Play Model Backend

The system supports multiple model backends using a `BackendRegistry` pattern:

- Attention GRU
- LightGBM artifact backend
- Physics-Informed LightGBM

This allows fair comparison and future model swapping without rewriting the API or dashboard. From a software and data science engineering perspective, this is important because production ML systems often need model versioning and model replacement.

### 3.4 Physics-Informed Reasoning

The project includes physics-aware logic through:

- Physics-Informed Attention model weights
- Physics-Informed LightGBM post-processing
- Counterfactual Physical Consistency (CPC)
- threshold-based checks on relevant turbofan sensors

This brings domain knowledge into the ML pipeline. For example, predictions are evaluated not only statistically but also based on whether engine sensor states are physically plausible.

### 3.5 Explainability and Visual Model Internals

The project includes an interactive **Model Internals** page. This visually explains:

- raw data flow into preprocessing
- backend model inference
- attention and sensor behavior
- RI gating
- RUL future trajectory
- engine cross-section sensor grouping

This is valuable for presentation and evaluation because it helps non-technical users understand what the model is doing internally.

### 3.6 Generative AI Components

The project includes:

- cVAE / Monte Carlo trajectory generator for future RUL paths
- LLM maintenance explanation engine with template fallback
- SHAP-style sensor attribution

Even when an external API key is unavailable, the system still generates explanations using deterministic templates.

---

## 4. Dataset Details

### 4.1 Dataset Name

The dataset used is the **NASA C-MAPSS FD001** dataset.

C-MAPSS stands for:

**Commercial Modular Aero-Propulsion System Simulation**

It is a widely used benchmark dataset for aircraft engine prognostics and Remaining Useful Life prediction.

### 4.2 Data Source

The dataset comes from the **NASA Prognostics Center of Excellence (PCoE)**. It is a simulated turbofan engine degradation dataset.

### 4.3 Selected Subset: FD001

The project focuses on **FD001**.

FD001 characteristics:

- single operating condition
- single fault mode
- high-pressure compressor degradation
- 100 training engines
- 100 testing engines
- run-to-failure trajectories

FD001 was selected because it is the most suitable starting point for RUL prediction. It avoids the extra complexity of multiple operating conditions and multiple fault modes, which are present in FD002, FD003, and FD004.

### 4.4 Dataset Shape and Structure

Each row represents one engine at one time cycle.

Columns:

| Column Type | Count | Description |
|-------------|-------|-------------|
| Engine ID | 1 | `unit_nr` |
| Time index | 1 | `time_cycles` |
| Operational settings | 3 | `op_setting_1`, `op_setting_2`, `op_setting_3` |
| Sensor readings | 21 | `s_1` to `s_21` |

Total input columns:

```text
unit_nr,time_cycles,op_setting_1,op_setting_2,op_setting_3,s_1,...,s_21
```

So each row contains **26 columns**.

### 4.5 Dataset Splits

The standard FD001 split is:

| Split | Number of Engines | Approximate Rows |
|-------|-------------------|------------------|
| Training | 100 | 20,631 |
| Testing | 100 | 13,096 |

Training engines run from healthy condition to failure. Testing engines are truncated before failure, and the model must estimate how many cycles remain.

### 4.6 Target Variable: Remaining Useful Life

The target variable is **RUL**, meaning:

```text
RUL = failure_cycle - current_cycle
```

The project uses an **early RUL cap of 125 cycles**. This means that very early in an engine's life, RUL is clipped to 125 instead of using very large values.

Why this is done:

- early-life degradation is weak and noisy
- exact large RUL values are difficult to learn
- capped RUL improves training stability
- this is common practice in C-MAPSS RUL research

### 4.7 Important Sensor Groups

The 21 sensors represent temperatures, pressures, speeds, fuel flow, and cooling flows.

Examples:

| Sensor | Meaning | Group |
|--------|---------|-------|
| `s_2` | LPC outlet temperature | Thermal |
| `s_3` | HPC outlet temperature | Thermal |
| `s_4` | LPT outlet temperature | Thermal |
| `s_7` | HPC outlet pressure | Pressure |
| `s_8` | Physical fan speed | Mechanical |
| `s_9` | Physical core speed | Mechanical |
| `s_11` | HPC static pressure | Pressure |
| `s_12` | Fuel flow ratio | Flow |
| `s_15` | Bypass ratio | Flow |
| `s_20` | HPT coolant bleed | Flow |
| `s_21` | LPT coolant bleed | Flow |

### 4.8 Data Science Challenges

The dataset is challenging because:

1. **Time-series structure**  
   Each engine is a sequence, not an independent row.

2. **Variable engine lifetimes**  
   Engines fail after different numbers of cycles.

3. **No explicit fault onset label**  
   The model must infer degradation from sensor trends.

4. **Sensor redundancy**  
   Some sensors are highly correlated.

5. **Near-constant sensors**  
   Some sensors have very little information.

6. **Noise**  
   Sensor readings contain simulation noise.

7. **Operational decision risk**  
   Incorrect RUL prediction can lead to unsafe or costly maintenance decisions.

### 4.9 Data Preprocessing

The preprocessing pipeline includes:

1. **CSV parsing**
   - accepts named C-MAPSS columns
   - accepts raw 26-column C-MAPSS format
   - accepts fallback space-separated format

2. **Column standardization**
   - converts all inputs into standard column order

3. **Numeric validation**
   - non-numeric values are rejected

4. **Feature selection**
   - low-information columns are dropped for the Attention GRU path

5. **Scaling**
   - sensor features are standardized using `StandardScaler`

6. **Windowing**
   - fixed-length windows of 30 cycles are created
   - last few windows are used for inference

7. **Target inverse scaling**
   - model outputs are converted back into RUL cycles

---

## 5. System Architecture

The system is designed as a full data science product, not only a training notebook. It supports dashboard usage, API usage, CLI usage, and real-time MQTT streaming.

### 5.1 High-Level Architecture

```text
┌────────────────────────────────────────────────────────────────────┐
│                           Input Sources                            │
│                                                                    │
│      CSV Upload        API JSON        MQTT Stream       Demo CSV   │
└───────────────┬───────────────┬───────────────┬────────────────────┘
                │               │               │
                ▼               ▼               ▼
┌────────────────────────────────────────────────────────────────────┐
│                        Data Preprocessing                          │
│                                                                    │
│  Column validation → Numeric conversion → Scaling → 30-cycle window │
└──────────────────────────────────┬─────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│                        Backend Registry                            │
│                                                                    │
│   ┌──────────────────┐   ┌────────────────┐   ┌─────────────────┐ │
│   │ Attention GRU     │   │ LightGBM        │   │ PI-LightGBM      │ │
│   │ Baseline / PI     │   │ Artifact        │   │ Physics + CPC    │ │
│   └──────────────────┘   └────────────────┘   └─────────────────┘ │
└──────────────────────────────────┬─────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│                         Prediction Layer                           │
│                                                                    │
│       Per-engine RUL → Window-level RUL → Attention Weights         │
└──────────────────────────────────┬─────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│                    Reliability and Physics Layer                    │
│                                                                    │
│  Reliability Index → Reason Codes → ACCEPT / WARN / REJECT Gate     │
│  CPC / Physics Risk → Trusted RUL → Conservative fallback if needed │
└──────────────────────────────────┬─────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│                         Output Interfaces                          │
│                                                                    │
│ Streamlit Dashboard │ FastAPI │ CLI Scripts │ MQTT Logs │ Visuals   │
└────────────────────────────────────────────────────────────────────┘
```

### 5.2 Runtime Components

| Component | File | Role |
|----------|------|------|
| Streamlit UI | `app.py` | Main dashboard and visualization layer |
| Attention model | `inference/attention_model.py` | Deep learning inference and preprocessing |
| Reliability engine | `inference/reliability.py` | RI, reason codes, gating |
| Model service | `backend/model_service.py` | Backend registry and unified inference interface |
| FastAPI API | `backend/api.py` | REST endpoints for inference |
| Artifact backend | `backend/artifact_backend.py` | LightGBM artifact inference |
| PI-LightGBM backend | `backend/pi_lightgbm_backend.py` | Physics-informed LightGBM post-processing |
| cVAE trajectory | `inference/cvae_trajectory.py` | Probabilistic future RUL paths |
| SHAP explainer | `inference/shap_explainer.py` | Sensor attribution |
| LLM explainer | `inference/llm_explainer.py` | Maintenance explanation |
| MQTT ingestion | `ingestion/mqtt_secure_ingest.py` | Real-time stream inference |

### 5.3 Dashboard Architecture

The Streamlit dashboard has six pages:

| Page | Data Science Purpose |
|------|----------------------|
| Fleet Overview | Batch inference and fleet-level RUL comparison |
| Live Digital Twin | Real-time monitoring of streamed sensor data |
| RUL Trajectories | Cycle-by-cycle replay and degradation trajectory visualization |
| Model Internals | Visual explanation of model pipeline, sensors, RI gate, and trajectory |
| Batch Inference | Detailed analytics, sensor correlation, optional ground-truth evaluation |
| Settings | Backend discovery, validation commands, MQTT setup |

### 5.4 Model Internals Visuals

The Model Internals page is useful for explaining the project during evaluation. It includes:

1. **Inference Pipeline Flow**
   - shows raw telemetry moving through preprocessing, backend inference, RUL prediction, RI gate, and final action

2. **Immersive Model Core**
   - sensor nodes orbit around the model core
   - attention arcs show model focus
   - RI gate ring shows prediction acceptance
   - future RUL fan shows probabilistic degradation

3. **Engine Cross-Section Sensor Map**
   - maps sensors onto simplified turbofan sections
   - groups sensors by thermal, pressure, mechanical, and flow categories

4. **Reliability Gate Simulator**
   - interactively demonstrates how spread, monotonicity violations, and smoothness affect RI
   - uses the same functions as the actual runtime gating engine

---

## 6. Models Used

### 6.1 Attention GRU Encoder-Decoder

This is the primary model.

Architecture:

- Encoder: stacked GRU
- Decoder: GRU with additive attention
- Hidden units: 64
- GRU layers: 2
- Attention size: 32
- Input window length: 30 cycles
- Output: scaled RUL prediction

Why GRU?

- GRUs are good for sequence modeling.
- They are simpler than LSTMs and train faster.
- They can capture temporal dependencies in degradation data.
- They are suitable for multivariate time-series data.

Why attention?

- Not every cycle is equally important.
- Attention allows the model to focus on the most informative timesteps.
- Attention weights improve interpretability.

### 6.2 Physics-Informed Attention Model

The project includes a Physics-Informed variant of the Attention GRU.

The purpose is to improve physical consistency by encouraging:

- monotonic RUL degradation
- smoother predictions
- boundary-aware predictions
- thermodynamically reasonable behavior

In this project, the Physics-Informed model trades a small loss in RMSE for improved consistency and reliability.

### 6.3 LightGBM Artifact Backend

LightGBM is used as a secondary model backend.

Why LightGBM?

- strong performance on tabular data
- efficient inference
- handles nonlinear feature interactions
- useful as a comparison against deep sequence models

However, LightGBM does not naturally model temporal sequence structure unless features are engineered. It also does not support differentiable physics-informed training in the same way as neural networks.

### 6.4 Physics-Informed LightGBM

The PI-LightGBM backend wraps the LightGBM artifact and adds post-prediction physics checks.

It computes:

- physics risk
- CPC score
- physics violations
- adjusted RUL

Physics checks include conditions such as:

- high-pressure compressor outlet overtemperature
- low-pressure turbine overtemperature
- pressure drop
- coolant bleed depletion

### 6.5 Reliability Index and Gating Model

Although not a predictive ML model, the RI engine is a major data science component.

RI combines:

```text
RI = 0.6 × stability_score + 0.4 × physics_score
```

Stability score includes:

- coefficient of variation
- spread ratio

Physics score includes:

- monotonicity score
- smoothness score
- boundary score

Gating:

```text
if RI >= 0.75:
    ACCEPT
elif RI >= 0.45:
    WARN
else:
    REJECT
```

If a prediction is rejected, the system applies a conservative fallback:

```text
trusted_rul = max(0, predicted_rul - 2 × prediction_std)
```

### 6.6 cVAE / Monte Carlo Trajectory Generator

The trajectory generator produces possible future RUL paths.

Currently, when trained cVAE weights are unavailable, the system uses a Monte Carlo fallback:

- samples degradation rates
- adds uncertainty based on RI
- generates multiple future paths
- computes confidence intervals

This helps answer:

> "What are the possible future degradation trajectories, not just the current RUL?"

### 6.7 SHAP-Style Sensor Attribution

The sensor attribution module estimates which sensors contribute most to RUL risk.

It supports:

- attention-weighted sensor importance
- statistical fallback
- group-level importance
- anomaly flags

### 6.8 LLM Maintenance Explainer

The explanation module generates a maintenance brief.

It supports:

- Gemini API mode when an API key is available
- deterministic template fallback when no API key is available

This is important because the project remains functional even without external APIs.

---

## 7. Results

### 7.1 Model Accuracy Results

The main reported model results on FD001 are:

| Model | RMSE | Notes |
|-------|------|-------|
| Attention GRU Baseline | ~14.21 cycles | Best point prediction performance |
| Attention PINN Physics-Informed | ~16.46 cycles | Slightly higher RMSE but improved physical consistency |
| LightGBM Artifact | Runtime artifact backend | Used for model swapping and comparison |
| PI-LightGBM | Runtime artifact + physics layer | Adds CPC and physics risk post-processing |

Interpretation:

- The Baseline Attention GRU gives the strongest numeric accuracy.
- The Physics-Informed model is more conservative and physically consistent.
- LightGBM provides a tabular ML comparison backend.
- PI-LightGBM demonstrates how domain constraints can be added after prediction.

### 7.2 Validation Fixture Results

The project includes golden validation outputs. Example results:

| Fixture | Mode | Engines | Mean RUL | Mean RI | Gate Counts |
|---------|------|---------|----------|---------|-------------|
| `sample_cmapss_engine.csv` | Baseline | 1 | 0.31 | 0.572 | 1 WARN |
| `sample_cmapss_engine.csv` | Physics-Informed | 1 | 8.21 | 0.789 | 1 ACCEPT |
| `sample_cmapss_engine_dual.csv` | Baseline | 2 | 5.77 | 0.902 | 2 ACCEPT |
| `sample_cmapss_engine_dual.csv` | Physics-Informed | 2 | 9.77 | 0.930 | 2 ACCEPT |
| `scenario_stable_behavior.csv` | Baseline | 1 | 0.71 | 0.569 | 1 WARN |
| `scenario_noisy_behavior.csv` | Baseline | 1 | 0.26 | 0.522 | 1 WARN |
| `scenario_rapid_degradation.csv` | Baseline | 1 | 0.24 | 0.537 | 1 WARN |

These validation outputs show that RI and gating vary depending on prediction consistency and model mode.

### 7.3 Gate Demo Fixture Results

The project provides verified demo CSVs for all three operational gate classes.

| Decision | Files | Verified Behavior |
|----------|-------|-------------------|
| ACCEPT | `gate_accept_demo.csv`, `gate_accept_demo_2.csv`, `gate_accept_demo_3.csv` | All engines accepted |
| WARN | `gate_warn_demo.csv`, `gate_warn_demo_2.csv`, `gate_warn_demo_3.csv` | Warning behavior |
| REJECT | `gate_reject_demo.csv`, `gate_reject_demo_2.csv`, `gate_reject_demo_3.csv` | Rejected due to unstable prediction windows |

Example verified RI values:

| File | Decision | RI |
|------|----------|----|
| `gate_accept_demo.csv` | ACCEPT | 0.805 and 1.000 |
| `gate_warn_demo.csv` | WARN | 0.572 |
| `gate_reject_demo.csv` | REJECT | 0.374 |
| `gate_reject_demo_2.csv` | REJECT | 0.380 |
| `gate_reject_demo_3.csv` | REJECT | 0.387 |

The REJECT cases are rejected because of reason codes such as:

- `HIGH_WINDOW_VARIANCE`
- `HIGH_WINDOW_SPREAD`
- `MONOTONICITY_WARN`
- `SMOOTHNESS_WARN`

### 7.4 Visual Results

The dashboard produces several data science visual outputs:

1. RUL distribution by engine
2. Reliability vs prediction scatter plot
3. Gate decision pie chart
4. Window-level RUL area chart
5. Attention weight chart
6. Probabilistic RUL fan plot
7. SHAP-style sensor attribution bar chart
8. Sensor group importance chart
9. Engine cross-section sensor map
10. Reliability Gate Simulator
11. Live digital twin RUL timeline
12. Replay trajectory of raw vs trusted RUL

### 7.5 Discussion of Results

From a data science perspective, the most important result is not only the RMSE but the complete decision pipeline.

The Attention GRU Baseline model performs well because it directly learns sequence behavior from engine cycles. The Physics-Informed variant shows that adding domain constraints can improve trustworthiness even if pure RMSE becomes slightly worse. This is a common trade-off in applied data science: the most accurate model is not always the safest or most useful model.

The RI and gating layer makes the system operationally meaningful. For example, a low RUL prediction with unstable windows should not automatically trigger a confident maintenance decision. Instead, the system can mark it as `WARN` or `REJECT`, allowing engineers to inspect uncertainty and sensor behavior.

The visual internals improve interpretability. A teacher or evaluator can see the full data pipeline, how sensors relate to engine sections, how RI changes with unstable predictions, and why the model output becomes an operational gate.

---

## 8. Limitations

1. **Only FD001 is used**
   - FD001 has one operating condition and one fault mode.
   - Real-world aircraft engines are more complex.

2. **Synthetic dataset**
   - C-MAPSS is simulated, not real airline sensor data.

3. **LightGBM artifact dependency**
   - The LightGBM artifact depends on compatible sklearn / LightGBM versions.

4. **cVAE fallback**
   - The trajectory generator currently uses Monte Carlo fallback if trained cVAE weights are unavailable.

5. **No certification**
   - This is an academic / prototype system, not FAA/EASA certified software.

6. **No multi-fault evaluation**
   - FD002, FD003, and FD004 are not currently evaluated.

---

## 9. Future Improvements

1. Extend evaluation to FD002, FD003, and FD004.
2. Train the cVAE on complete FD001 degradation trajectories.
3. Add uncertainty calibration metrics.
4. Add model comparison reports for LightGBM vs Attention GRU.
5. Add Docker deployment for Streamlit + API + Mosquitto + Node-RED.
6. Add stronger integration tests for MQTT streaming.
7. Improve artifact backend compatibility by exporting full preprocessing metadata.
8. Add true SHAP values for LightGBM backend.
9. Add confidence calibration curves.
10. Add automatic model monitoring for data drift.

---

## 10. Viva Questions and Suggested Answers

### Q1. What is the main objective of your project?

The main objective is to predict the Remaining Useful Life of aircraft turbofan engines using sensor time-series data, and to provide reliability-aware decisions through `ACCEPT`, `WARN`, and `REJECT` gating.

### Q2. Why did you choose the NASA C-MAPSS FD001 dataset?

FD001 is a standard benchmark dataset for turbofan RUL prediction. It has one operating condition and one fault mode, which makes it suitable for focused modeling and evaluation before moving to more complex subsets.

### Q3. What is RUL?

RUL stands for Remaining Useful Life. It is the number of operational cycles left before an engine reaches failure.

### Q4. Why is RUL capped at 125 cycles?

Early in engine life, degradation is weak and RUL values are large and noisy. Capping RUL at 125 stabilizes training and is common practice in C-MAPSS research.

### Q5. Why did you use GRU instead of a normal regression model?

The dataset is sequential. A GRU can learn temporal degradation patterns across cycles, whereas a simple regression model treats rows more independently.

### Q6. Why use attention?

Attention helps the model focus on the most informative timesteps in the 30-cycle window. It also improves interpretability by showing which cycles influenced the prediction.

### Q7. What is the input to the Attention GRU model?

The input is a fixed-length 30-cycle window of scaled sensor and operational features for each engine.

### Q8. What is the output of the model?

The model outputs a scaled RUL prediction, which is inverse-transformed back into RUL cycles.

### Q9. What is the Reliability Index?

RI is a trust score between 0 and 1. It measures prediction stability and physical consistency using spread, variance, monotonicity, smoothness, and boundary compliance.

### Q10. Why do you need `ACCEPT`, `WARN`, and `REJECT`?

Because in safety-critical systems, not all model predictions should be trusted equally. Gating converts a raw ML output into an operational decision.

### Q11. What happens when a prediction is rejected?

The system applies a conservative fallback RUL:

```text
trusted_rul = max(0, predicted_rul - 2 × std)
```

This reduces operational risk.

### Q12. What is the difference between predicted RUL and trusted RUL?

Predicted RUL is the raw model output. Trusted RUL is the post-gating value used for decision support.

### Q13. What is CPC?

CPC means Counterfactual Physical Consistency. It measures whether the prediction and sensor state are physically consistent with expected engine behavior.

### Q14. What are the models used?

The project uses:

- Attention GRU Baseline
- Physics-Informed Attention GRU
- LightGBM artifact backend
- Physics-Informed LightGBM backend
- Monte Carlo / cVAE trajectory generator
- SHAP-style explainer
- LLM/template maintenance explainer

### Q15. Which model gave the best RMSE?

The Attention GRU Baseline achieved approximately RMSE 14.21 cycles on FD001.

### Q16. Why is the Physics-Informed model RMSE slightly worse?

Physics-informed models may trade pure accuracy for physical consistency. The model becomes more constrained, which can slightly increase RMSE but improve trustworthiness.

### Q17. What is LightGBM doing in the project?

LightGBM provides a tabular ML backend for comparison and model-swapping. It is integrated through the backend registry.

### Q18. What is the BackendRegistry?

It is a central registry that maps backend names to model adapters. It allows switching models without changing the API or dashboard logic.

### Q19. How do you handle missing values?

C-MAPSS has no missing values, but the pipeline validates numeric columns and rejects inputs with missing or non-numeric required values.

### Q20. What preprocessing is done?

The pipeline standardizes columns, converts values to numeric, drops low-information columns, scales features, creates 30-cycle windows, and inverse-scales model outputs.

### Q21. Why do you use StandardScaler?

Sensors have different numeric ranges. Scaling prevents large-range sensors from dominating the model.

### Q22. What is the difference between batch inference and replay?

Batch inference predicts from the full uploaded dataset. Replay simulates cycle-by-cycle inference to show how predictions evolve over time.

### Q23. What is the purpose of the digital twin?

The digital twin simulates live engine sensor streaming through MQTT, allowing real-time inference and monitoring.

### Q24. What is the purpose of the Model Internals page?

It visually explains how data flows through preprocessing, model inference, reliability gating, sensor groups, and future RUL trajectories.

### Q25. How does sensor attribution work?

It estimates each sensor's contribution using attention-weighted importance or statistical fallback, then groups sensors by physical category.

### Q26. Is this system production-ready?

It is a strong prototype with API, UI, CLI, and streaming support, but it is not certified for aircraft use and would require real-data validation, security hardening, and certification.

### Q27. What are the limitations?

The project is limited to FD001, uses simulated data, has artifact compatibility constraints, and does not yet evaluate multi-condition or multi-fault datasets.

### Q28. What would you improve next?

I would extend to FD002-FD004, train the cVAE, add calibration curves, improve LightGBM artifact metadata, and add drift monitoring.

### Q29. How is this project data science oriented?

It includes dataset analysis, preprocessing, feature handling, time-series modeling, model comparison, reliability scoring, explainability, validation, visualization, and deployment.

### Q30. What is the business value?

It can reduce unscheduled maintenance, improve safety, help plan maintenance windows, reduce downtime, and provide interpretable decision support for fleet operators.

---

## 11. Short Presentation Flow

If asked to explain the project in 2-3 minutes:

1. Start with the problem: aircraft engines need predictive maintenance.
2. Explain the dataset: NASA C-MAPSS FD001, 100 train engines, 100 test engines, 21 sensors.
3. Explain preprocessing: scaling and 30-cycle windows.
4. Explain the primary model: Attention GRU for sequence prediction.
5. Explain reliability: RI and `ACCEPT/WARN/REJECT` gates.
6. Explain novelty: physics-informed backend, explainability, model internals, digital twin.
7. Explain results: Baseline RMSE ~14.21, Physics-Informed RMSE ~16.46, verified gate fixtures.
8. End with practical value: safer and more interpretable predictive maintenance.

---

## 12. Key Takeaway

This project demonstrates a complete applied data science workflow for predictive maintenance. It does not only train a model; it builds a reliability-aware system around the model. The strongest part of the project is the combination of **time-series deep learning**, **physics-informed validation**, **model explainability**, **operational gating**, and **interactive visualization**.

