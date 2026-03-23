# Physics-Integrated Generative Edge-AI for Aero-Engine Prognostics

## Remaining Useful Life Prediction of Turbofan Engines Using Attention-Based Deep Learning and Physics-Informed Machine Learning

**IEEE IES GenAI Challenge 2026 · NASA C-MAPSS FD001**

---

# 1. Introduction

## 1.1 Background of Study

Modern aircraft engines operate under extreme thermomechanical loads, making predictive maintenance critical for flight safety, operational efficiency, and cost reduction. Remaining Useful Life (RUL) prediction — estimating how many operational cycles an engine can sustain before failure — is a cornerstone of Prognostics and Health Management (PHM).

Traditionally, engine maintenance follows time-based or usage-based schedules, leading to either premature part replacements (increasing cost) or missed failure precursors (increasing risk). Condition-Based Maintenance (CBM) using sensor data and machine learning offers a paradigm shift: maintenance is triggered by the actual degradation state of the engine rather than fixed intervals.

The NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset provides a standardized benchmark for RUL prediction research. It simulates the degradation of turbofan engines from healthy state to failure using 21 sensor channels and 3 operational settings across multiple flight regimes.

This project develops a production-grade RUL prediction system that combines:
- **Attention-based GRU encoder-decoder** networks for sequence modeling
- **Physics-Informed Neural Networks (PINNs)** that embed thermodynamic constraints
- **LightGBM gradient-boosted trees** with physics-informed post-processing
- **Conditional VAE trajectory generator** for probabilistic RUL paths (Generative AI)
- **LLM-powered explanation engine** for natural language maintenance briefs (Generative AI)
- **SHAP-style sensor attribution** for interpretable per-sensor contribution analysis
- **Real-time digital twin integration** via MQTT and Node-RED
- A **Streamlit dashboard** for fleet-level monitoring and decision support

## 1.2 Motivation

1. **Safety-critical application**: Unscheduled engine failures can lead to catastrophic events. Accurate RUL prediction with calibrated confidence bounds is essential.
2. **Economic impact**: Airlines spend approximately 40-50% of their total maintenance budget on engine overhaul. Predictive maintenance can reduce unscheduled downtime by 30-50%.
3. **Regulatory requirements**: Aviation authorities (FAA, EASA) increasingly mandate data-driven health monitoring for next-generation engines.
4. **Physics-AI integration gap**: Most ML-based prognostics treat engines as black boxes. Incorporating physics constraints (thermodynamic laws, degradation models) improves prediction reliability and interpretability.
5. **Edge deployment need**: Real-time inference on edge devices (e.g., Raspberry Pi, embedded systems) enables in-flight prognostics without relying on cloud connectivity.

## 1.3 Problem Statement

Given multivariate sensor time-series data from turbofan engines operating under varying conditions, predict the Remaining Useful Life (RUL) of each engine with:

1. **Accurate point estimates** with RMSE ≤ 20 cycles on the FD001 test set
2. **Calibrated confidence** via a Reliability Index (RI) that measures prediction trustworthiness
3. **Operational gating** that classifies predictions as `ACCEPT`, `WARN`, or `REJECT` for safe operational use
4. **Physics consistency** ensuring predictions respect thermodynamic bounds and degradation monotonicity
5. **Real-time capability** for live digital twin monitoring with sub-second inference latency

## 1.4 Objectives

1. Develop an attention-based encoder-decoder model (GRU backbone) for multivariate RUL prediction on C-MAPSS FD001
2. Train a Physics-Informed variant incorporating thermodynamic loss penalties during training
3. Integrate a LightGBM gradient-boosted tree model with physics-informed post-processing for a hybrid PI-LightGBM approach
4. Implement a post-prediction Reliability Index (RI) and gating framework for operational decision support
5. Build a plug-and-play model backend system for seamless switching between model architectures
6. Implement a Conditional VAE trajectory generator for probabilistic RUL predictions with confidence intervals
7. Develop an LLM-powered explanation engine for natural language maintenance briefs
8. Build SHAP-style sensor attribution analysis for per-sensor contribution to RUL
9. Create a premium Streamlit dashboard with fleet-level monitoring, per-engine deep-dive, GenAI panels, and live digital twin feed
10. Develop a Node-RED-based digital twin simulation for realistic sensor data generation
11. Provide FastAPI endpoints and headless CLI tools for production deployment

## 1.5 Scope

**In Scope:**
- NASA C-MAPSS FD001 dataset (single operating condition, single fault mode: HPC degradation)
- Attention GRU Baseline model (RMSE ~14.21)
- Attention PINN Physics-Informed model (RMSE ~16.46)
- LightGBM artifact model with physics post-processing
- Real-time MQTT ingestion and digital twin streaming
- Streamlit dashboard, FastAPI backend, and CLI tooling
- Edge inference benchmarking (TFLite export)

**Out of Scope:**
- FD002, FD003, FD004 datasets (multiple operating conditions, multiple fault modes)
- Hardware deployment on actual aircraft systems
- FAA/EASA certification compliance
- Production-grade security hardening beyond TLS/auth MQTT support

## 1.6 Organization of the Report

| Chapter | Content |
|---------|---------|
| 1 | Introduction, motivation, problem statement, objectives |
| 2 | Literature review of RUL prediction methods and research gaps |
| 3 | Dataset description and characteristics |
| 4 | Data preprocessing pipeline |
| 5 | Proposed methodology and system architecture |
| 6 | Model development and training |
| 7 | Experimental results and evaluation |
| 8 | System implementation (backend, frontend, API) |
| 9 | Conclusion, limitations, and future work |

---

# 2. Literature Review

## 2.1 Existing Systems

| System / Method | Approach | Limitations |
|----------------|----------|-------------|
| **GE Predix** | Cloud-based industrial IoT platform for engine health monitoring | Proprietary, cloud-dependent, high cost |
| **Pratt & Whitney EngineWise** | Fleet analytics using operational data | Limited to P&W engines, no physics integration |
| **NASA PCOE Tools** | Open-source prognostics algorithms (particle filters, Kalman) | Single-engine focus, no fleet-level orchestration |
| **LSTM-based RUL** (Zheng et al., 2017) | Vanilla LSTM on C-MAPSS | No attention mechanism, poor long-range dependency capture |
| **CNN-LSTM Hybrid** (Li et al., 2018) | Spatial-temporal feature extraction | No physics constraints, black-box predictions |

## 2.2 Related Work

### Deep Learning for RUL Prediction

**Recurrent Neural Networks (RNNs):** Heimes (2008) first applied RNNs to C-MAPSS, establishing the deep learning paradigm for RUL. GRU variants (Cho et al., 2014) later showed faster convergence and comparable accuracy to LSTMs with fewer parameters.

**Attention Mechanisms:** Bahdanau attention (2014) was adapted for sequence-to-sequence RUL prediction by enabling the model to focus on critical degradation timesteps. Chen et al. (2020) demonstrated that attention-augmented GRU models achieve state-of-the-art RMSE on FD001 (13-16 cycles).

**Physics-Informed Neural Networks (PINNs):** Raissi et al. (2019) introduced PINNs for embedding physical laws into neural network training via custom loss functions. For turbofan engines, physics constraints include:
- Monotonic RUL decrease over time
- Thermodynamic bounds on sensor readings (temperature, pressure)
- Conservation laws (mass flow, energy balance)

### Gradient-Boosted Trees for Prognostics

**LightGBM** (Ke et al., 2017) has shown competitive performance on tabular prognostics tasks due to its ability to handle heterogeneous features and missing data. However, tree-based models lack native support for physics-loss training, motivating post-prediction physics correction approaches.

### Reliability and Gating

**Prediction reliability** has received less attention than accuracy in PHM literature. This project introduces a Reliability Index (RI) that combines:
- Window prediction stability (coefficient of variation, spread ratio)
- Physics consistency (monotonicity, smoothness, boundary compliance)

The RI enables post-prediction gating (`ACCEPT`/`WARN`/`REJECT`) for safe operational deployment.

## 2.3 Research Gap

| Gap | How This Project Addresses It |
|-----|-------------------------------|
| Most models are black-box without reliability estimates | RI + gating framework provides calibrated trustworthiness |
| Physics constraints are rarely integrated into ML predictions | PINN training + post-prediction CPC scoring |
| No plug-and-play model switching in existing systems | BackendRegistry pattern enables seamless model swaps |
| Limited real-time digital twin integration | MQTT + Node-RED + Streamlit live dashboard |
| No probabilistic RUL trajectories | cVAE / Monte Carlo trajectory generator with confidence intervals |
| No natural language explanations for operators | LLM-powered maintenance briefs (Gemini API + template fallback) |
| Limited sensor-level interpretability | SHAP-style per-sensor attribution with anomaly detection |
| Edge deployment not considered | TFLite export + benchmark scripts for Raspberry Pi |

---

# 3. Dataset Description

## 3.1 Data Source

**NASA C-MAPSS FD001** (Turbofan Engine Degradation Simulation)

- **Source:** NASA Prognostics Center of Excellence (PCoE)
- **Simulation:** Commercial Modular Aero-Propulsion System Simulation
- **Subset:** FD001 — single operating condition, single fault mode (HPC degradation)

| Split | Engines | Total Rows |
|-------|---------|------------|
| Training | 100 | ~20,631 |
| Testing | 100 | ~13,096 |

Each engine starts from a healthy state with normal manufacturing variations and develops a fault at an unknown point, progressing to failure.

## 3.2 Attributes Description

| Column | Name | Description | Unit |
|--------|------|-------------|------|
| 0 | `unit_nr` | Engine unit identifier | - |
| 1 | `time_cycles` | Operational cycle number | cycles |
| 2 | `op_setting_1` | Operational setting 1 (altitude) | - |
| 3 | `op_setting_2` | Operational setting 2 (Mach number) | - |
| 4 | `op_setting_3` | Operational setting 3 (throttle resolver angle) | - |
| 5 | `s_1` | Fan inlet temperature (T2) | °R |
| 6 | `s_2` | LPC outlet temperature (T24) | °R |
| 7 | `s_3` | HPC outlet temperature (T30) | °R |
| 8 | `s_4` | LPT outlet temperature (T50) | °R |
| 9 | `s_5` | Fan inlet pressure (P2) | psia |
| 10 | `s_6` | Bypass-duct pressure (P15) | psia |
| 11 | `s_7` | HPC outlet pressure (P30) | psia |
| 12 | `s_8` | Physical fan speed (Nf) | rpm |
| 13 | `s_9` | Physical core speed (Nc) | rpm |
| 14 | `s_10` | Engine pressure ratio (epr) | - |
| 15 | `s_11` | HPC outlet static pressure (Ps30) | psia |
| 16 | `s_12` | Ratio of fuel flow to Ps30 (phi) | pps/psi |
| 17 | `s_13` | Corrected fan speed (NRf) | rpm |
| 18 | `s_14` | Corrected core speed (NRc) | rpm |
| 19 | `s_15` | Bypass ratio (BPR) | - |
| 20 | `s_16` | Burner fuel-air ratio (farB) | - |
| 21 | `s_17` | Bleed enthalpy (htBleed) | - |
| 22 | `s_18` | Demanded fan speed (Nf_dmd) | rpm |
| 23 | `s_19` | Demanded corrected fan speed (PCNfR_dmd) | rpm |
| 24 | `s_20` | HPT coolant bleed (W31) | lbm/s |
| 25 | `s_21` | LPT coolant bleed (W32) | lbm/s |

## 3.3 Data Distribution

- **Cycle lengths** range from 128 to 362 cycles per engine (FD001 training set)
- **Mean engine lifetime:** ~206 cycles
- **RUL target:** Capped at 125 cycles (early RUL clipping) — engines with more than 125 remaining cycles are all assigned RUL = 125, since degradation in the early phase is noisy and difficult to predict
- **Operational regime:** Single operating condition (FD001) — operational settings are approximately constant

## 3.4 Challenges in Dataset

1. **Piecewise-linear RUL target:** The early_rul=125 clipping creates a flat region followed by linear decline, making the prediction task heterogeneous across the engine lifetime.
2. **Noisy sensors:** Several sensors (`s_1`, `s_5`, `s_10`, `s_16`, `s_18`, `s_19`) show near-constant readings with noise, providing minimal degradation information.
3. **Varying engine lifetimes:** Engines fail at different cycle counts (128-362), requiring models to handle variable-length sequences.
4. **No explicit fault labels:** The dataset provides run-to-failure data without explicit fault onset markers, meaning the model must implicitly learn the degradation trajectory.
5. **Sensor correlation:** Many sensors are highly correlated (e.g., temperatures T24, T30, T50), requiring careful feature selection to avoid redundancy.

---

# 4. Data Preprocessing

## 4.1 Handling Missing Values

The C-MAPSS dataset is synthetically generated and contains no missing values. However, the preprocessing pipeline includes defensive checks:

```python
standardized = standardized.apply(pd.to_numeric, errors="coerce")
if standardized.isna().any().any():
    raise ValueError("Input CSV has non-numeric values in required sensor columns.")
```

For real-world deployment, the MQTT ingestion pipeline handles partial sensor packets gracefully by buffering until a minimum window length is reached.

## 4.2 Data Cleaning

Data cleaning involves:
1. **Column standardization:** Input CSVs are accepted in multiple formats (named columns, raw numeric columns, space-separated), with automatic detection and normalization to the 26-column C-MAPSS format.
2. **All-NaN column removal:** `df.dropna(axis=1, how="all")` removes completely empty columns.
3. **Numeric coercion:** All sensor values are forced to numeric type.

## 4.3 Encoding Categorical Variables

The C-MAPSS dataset is entirely numeric. The only categorical-like variable is `unit_nr` (engine identifier), which is used for grouping, not as a model feature. It is excluded during feature scaling and windowing.

## 4.4 Feature Scaling

Two scaling strategies are employed:

1. **StandardScaler** — for input features (zero mean, unit variance):
```python
scaler = StandardScaler()
scaled = scaler.fit_transform(indexed_raw_df.drop(columns=COLUMNS_TO_BE_DROPPED))
```

2. **MinMaxScaler** — for RUL target inverse scaling (0-1 range mapped to 0-125):
```python
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(np.arange(0, EARLY_RUL + 1).reshape(-1, 1))
rul = target_scaler.inverse_transform(rul_pred_scaled.reshape(-1, 1))
```

## 4.5 Feature Selection

The following 12 columns are dropped (indices into the 26-column frame), based on domain analysis showing they carry minimal degradation information:

```python
COLUMNS_TO_BE_DROPPED = [0, 1, 2, 3, 4, 5, 9, 10, 14, 20, 22, 23]
```

| Dropped Index | Column | Reason |
|---------------|--------|--------|
| 0 | unit_nr | Identifier, not a feature |
| 1 | time_cycles | Used for ordering, not as sensor input |
| 2-4 | op_setting_1-3 | Near-constant in FD001 (single operating condition) |
| 5 | s_1 (T2) | Near-constant, minimal degradation signal |
| 9 | s_5 (P2) | Near-constant |
| 10 | s_6 (P15) | Near-constant |
| 14 | s_10 (epr) | Near-constant |
| 20 | s_16 (farB) | Near-constant |
| 22 | s_18 (Nf_dmd) | Demanded (not measured), near-constant |
| 23 | s_19 (PCNfR_dmd) | Demanded (not measured), near-constant |

This leaves **14 informative sensor features** for model input.

---

# 5. Proposed Methodology

## 5.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA ACQUISITION                             │
│  CSV Upload │ MQTT Stream │ Node-RED Digital Twin │ API JSON        │
└──────┬──────┴──────┬───────┴───────────┬──────────┴────────┬────────┘
       │             │                   │                   │
       ▼             ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING LAYER                            │
│  CSV Parser → Column Standardization → Feature Selection →          │
│  StandardScaler → Sliding Window (W=30, S=1)                        │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MODEL BACKEND REGISTRY                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ Attention GRU     │  │ LightGBM         │  │ PI-LightGBM      │  │
│  │ (Baseline / PI)   │  │ (Artifact)       │  │ (Physics+LGB)    │  │
│  │ RMSE: 14.21/16.46 │  │ from zip         │  │ + CPC scoring    │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   POST-PREDICTION LAYER                             │
│  Reliability Index (RI) → Gating (ACCEPT/WARN/REJECT)              │
│  → Trusted RUL → Reason Codes → CPC Score                          │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                             │
│  Streamlit Dashboard │ FastAPI │ CLI Scripts │ TFLite Edge          │
└─────────────────────────────────────────────────────────────────────┘
```

## 5.2 Algorithm Selection

### 5.2.1 Attention-Based GRU Encoder-Decoder (Primary Model)

**Why GRU over LSTM?**
- Fewer parameters (2 gates vs 3 gates) → faster training and inference
- Comparable accuracy on C-MAPSS benchmarks
- Better suited for edge deployment due to lower memory footprint

**Why Attention?**
- Standard seq2seq models compress the entire input sequence into a fixed-length context vector, losing fine-grained temporal information
- Additive attention (Bahdanau-style) allows the decoder to selectively focus on critical degradation timesteps
- Attention weights provide interpretability — revealing which sensor readings at which timesteps drove the prediction

### 5.2.2 LightGBM (Secondary Model)

**Why LightGBM?**
- Handles tabular features efficiently without sequence windowing overhead
- Robust to noisy features and does not require extensive hyperparameter tuning
- Fast inference suitable for edge deployment
- Complements deep learning approaches by providing a fundamentally different modeling paradigm

### 5.2.3 Physics-Informed Post-Processing

**Why post-prediction rather than in-training physics?**
- LightGBM (tree-based) cannot incorporate differentiable physics loss during training
- Post-prediction physics checks are model-agnostic and can be applied to any backend
- Separating physics validation from prediction enables clearer debugging

## 5.3 Mathematical Model

### 5.3.1 Encoder

The GRU encoder processes the windowed sensor input sequence **X** = {**x**₁, **x**₂, ..., **x**_T} where T=30 (window length) and each **x**_t ∈ ℝ¹⁴ (14 selected features):

**h**_t = GRU(**x**_t, **h**_{t-1})

With 2 stacked GRU layers (64 hidden units each):

**h**_t^(l) = GRU^(l)(**h**_t^(l-1), **h**_{t-1}^(l))

### 5.3.2 Additive Attention

The attention mechanism computes a weighted sum of encoder outputs:

Score: **e**_t = tanh(**W**_a · [**h**_{enc_t} ; **h**_{dec}])

Weights: **α**_t = softmax(Σ **e**_t)

Context: **c** = Σ_t **α**_t · **h**_{enc_t}

### 5.3.3 Decoder

The decoder produces a single RUL prediction:

**h**_{dec} = GRU(**x**_T, **c**)

ŷ = **W**_out · **h**_{dec} + **b**_out

### 5.3.4 RUL Clipping

RUL values are clipped to [0, 125]:

RUL_final = clip(MinMaxScaler⁻¹(ŷ), 0, 125)

### 5.3.5 Reliability Index

The RI combines stability and physics scores:

RI = w_stability · S_stability + w_physics · S_physics

Where:
- **S_stability** = mean(S_cv, S_spread) — measures window prediction consistency
- **S_physics** = 0.5·S_monotonic + 0.3·S_smoothness + 0.2·S_boundary
- **w_stability** = 0.6, **w_physics** = 0.4

### 5.3.6 CPC (Counterfactual Physical Consistency)

For the PI-LightGBM backend:

CPC = 1 − PhysicsRisk

PhysicsRisk = Σ (penalty_i × violation_i)

Where violations are checked against thermodynamic thresholds:
- HPC outlet overtemp (T30 > 1620K): penalty = 0.25
- LPT overtemp (T50 > 1450K): penalty = 0.25
- HPC pressure drop (Ps30 < 8500): penalty = 0.20
- HPT coolant bleed depletion (W31 < 88): penalty = 0.15
- LPT coolant bleed depletion (W32 < 32): penalty = 0.15

## 5.4 Model Training Process

### Attention GRU (Baseline)

1. Parse training data → per-engine DataFrames
2. Compute piecewise-linear RUL targets (capped at 125)
3. Apply StandardScaler to 14 selected features
4. Create sliding windows (W=30, stride=1)
5. Train encoder-decoder with MSE loss
6. Save best weights by validation RMSE

### Attention PINN (Physics-Informed)

Same pipeline as Baseline, with additional physics-penalty terms in the loss:

L_total = L_MSE + λ_mono · L_monotonicity + λ_bound · L_boundary

Where:
- L_monotonicity penalizes non-decreasing RUL predictions across consecutive windows
- L_boundary penalizes predictions outside [0, 125]

### LightGBM

Trained separately via Kaggle notebook on flattened tabular features (no windowing), exported as `model_artifacts.zip` containing the serialized regressor and metadata.

## 5.5 Hyperparameter Tuning

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Window length | 30 | Captures sufficient degradation trajectory |
| Stride | 1 | Maximum temporal resolution |
| GRU hidden units | 64 | Balance between capacity and edge deployability |
| Num GRU layers | 2 | Sufficient depth for C-MAPSS complexity |
| Attention size | 32 | Half of hidden units — standard practice |
| Early RUL cap | 125 | Reduces noisy early-life predictions |
| Num test windows | 5 | Multiple windows for reliability estimation |
| RI stability weight | 0.6 | Stability more important than physics for gating |
| RI physics weight | 0.4 | Physics provides additional safety net |
| Accept threshold | 0.75 | High bar for operational acceptance |
| Warning threshold | 0.45 | Moderate bar before rejection |

---

# 6. Model Development

## 6.1 Training Setup

| Aspect | Configuration |
|--------|--------------|
| Framework | TensorFlow 2.20 (Attention GRU), LightGBM 4.6 (tree model) |
| Hardware | GPU-accelerated training (CUDA); CPU inference supported |
| Random seed | 42 (reproducibility) |
| Loss function | MSE (Baseline), MSE + physics penalties (PI) |
| Optimizer | Adam with default learning rate |
| Batch processing | Full-batch per engine during inference |

## 6.2 Testing Strategy

1. **Holdout test set:** 100 engines reserved for evaluation (standard C-MAPSS split)
2. **Golden regression outputs:** Reference predictions stored in `reproducibility/golden_inference_outputs.json` for regression testing
3. **Tolerance bounds:** RMSE delta ≤ 1.0 cycle, RI delta ≤ 0.08 between runs

## 6.3 Cross Validation

The C-MAPSS benchmark uses a fixed train/test split (100/100 engines) to ensure comparability with published results. Within the training set, validation is performed via:
- Monitoring validation RMSE during training (early stopping)
- Best model checkpoint selection by validation loss

## 6.4 Implementation Tools

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Core runtime |
| TensorFlow 2.20 | Attention GRU model |
| LightGBM 4.6 | Gradient-boosted tree model |
| scikit-learn 1.6 | Preprocessing (scalers) |
| pandas 2.3 | Data manipulation |
| NumPy 2.0 | Numerical computation |
| Streamlit 1.50 | Dashboard frontend |
| FastAPI 0.128 | REST API backend |
| Plotly 6.5 | Interactive visualizations |
| paho-mqtt 2.1 | MQTT client for digital twin |
| Node-RED | Visual flow editor for sensor simulation |
| Mosquitto | MQTT broker |
| joblib 1.5 | Model serialization |
| psutil 7.1 | Edge benchmarking |

---

# 7. Experimental Results and Evaluation

## 7.1 Evaluation Metrics

Since RUL prediction is a **regression** task, the primary metrics are:

### RMSE (Root Mean Squared Error)
RMSE = √(1/N · Σ(ŷᵢ − yᵢ)²)

Penalizes large errors more heavily, critical for safety applications.

### MAE (Mean Absolute Error)
MAE = 1/N · Σ|ŷᵢ − yᵢ|

Provides intuitive average error magnitude.

### Scoring Function (NASA)
The asymmetric scoring function penalizes late predictions (under-estimation) more than early predictions:

s = Σ sᵢ, where:
- sᵢ = e^(-d/13) − 1, if d < 0 (early prediction)
- sᵢ = e^(d/10) − 1, if d ≥ 0 (late prediction)
- d = ŷ − y

### Reliability Index (RI)
RI ∈ [0, 1] — measures prediction trustworthiness based on window stability and physics compliance.

### Catastrophic Error Rate
Percentage of predictions with |error| > 20 cycles — a safety metric.

## 7.2 Performance Analysis

| Model | RMSE | Architecture | Physics |
|-------|------|--------------|---------|
| Attention GRU (Baseline) | ~14.21 | 2×GRU(64) + Attention | No |
| Attention PINN (Physics-Informed) | ~16.46 | 2×GRU(64) + Attention + Physics Loss | Yes (training) |
| LightGBM (Artifact) | TBD* | Gradient-boosted trees | No |
| PI-LightGBM | TBD* | LightGBM + CPC post-processing | Yes (post-prediction) |

*Note: The LightGBM artifact backend requires compatible sklearn/LightGBM versions for runtime evaluation.

**Key observations:**
1. The Baseline model achieves the lowest RMSE (~14.21), indicating strong pattern recognition without physics constraints
2. The Physics-Informed model trades ~2 RMSE points for improved physical consistency — predictions are more monotonic and respect thermodynamic bounds
3. The RI framework successfully identifies unreliable predictions: engines with `REJECT` decisions consistently have higher absolute errors

## 7.3 Model Comparison

The system supports side-by-side comparison of Baseline vs Physics-Informed models via the `compare` endpoint:

| Comparison Metric | Baseline | Physics-Informed | Delta |
|-------------------|----------|------------------|-------|
| Mean RUL prediction | Available per-engine | Available per-engine | PI − Baseline computed |
| Mean RI | Higher (no physics penalty) | Slightly lower (physics checks tighter) | Computed |
| Decision agreement | Reference | May differ for borderline engines | CHANGED/SAME flag |

The comparison reveals that physics-informed predictions tend to be slightly more conservative (lower RUL) for engines near failure, which is desirable from a safety perspective.

## 7.4 Graphical Results

The Streamlit dashboard provides the following visualizations:
1. **RUL Distribution Bar Chart** — per-engine RUL color-coded by gate decision
2. **Reliability Gating Pie Chart** — fleet-wide ACCEPT/WARN/REJECT distribution
3. **RI vs RUL Scatter Plot** — reliability against prediction magnitude
4. **Window-Level RUL Area Chart** — per-engine prediction across test windows
5. **Attention Weight Heatmap** — which timesteps the model focused on
6. **Sensor Telemetry Line Chart** — raw sensor trends over engine lifecycle
7. **Sensor Correlation Matrix** — inter-sensor relationships
8. **Streaming Replay Trajectory** — cycle-by-cycle RUL evolution
9. **Live MQTT RUL Feed** — real-time prediction trajectory from digital twin
10. **Probabilistic RUL Fan Plot** — cVAE / Monte Carlo trajectory fan with 95% and 50% confidence bands, median/mean lines, and sample traces
11. **Sensor Attribution Bar Chart** — SHAP-style horizontal bar chart showing per-sensor contribution to RUL risk (red = risk, green = healthy)
12. **Sensor Group Importance** — aggregated importance by physical category (thermal, pressure, mechanical, flow)
13. **AI Maintenance Brief** — structured natural language report with urgency level, decision rationale, diagnostic flags, and recommended actions

---

# 8. System Implementation

## 8.1 Backend

### Model Service (`backend/model_service.py`)

The core backend uses a **BackendRegistry** pattern for plug-and-play model management:

```python
class BackendRegistry:
    def register(self, name, factory, description, aliases):
        ...
    def get_factory(self, name) -> AdapterFactory:
        ...
    def list_backends(self) -> List[Dict[str, str]]:
        ...
```

Three adapters are registered:
1. **AttentionModelAdapter** — loads TensorFlow GRU weights, runs seq2seq inference
2. **ArtifactModelAdapter** — loads LightGBM from `model_artifacts.zip`
3. **PILightGBMAdapter** — extends artifact with physics constraint post-processing

### Reliability Engine (`inference/reliability.py`)

Post-prediction reliability computation:
- `compute_reliability_index()` — calculates RI from window predictions
- `gate_prediction()` — classifies as ACCEPT/WARN/REJECT
- `reason_codes()` — generates diagnostic codes (HIGH_WINDOW_VARIANCE, MONOTONICITY_WARN, etc.)

### FastAPI (`backend/api.py`)

RESTful endpoints:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/backends` | GET | List available model backends |
| `/v1/infer/file` | POST | Inference from uploaded CSV |
| `/v1/infer/json` | POST | Inference from JSON row batch |
| `/v1/compare/json` | POST | Side-by-side Baseline vs PI comparison |
| `/v1/replay/json` | POST | Streaming cycle-by-cycle replay |

## 8.2 Frontend

### Streamlit Dashboard (`app.py`)

Premium dark-theme dashboard with 5 navigation pages:

1. **🏠 Fleet Overview** — Upload CSV, run fleet inference, view engine table with RI/CPC/gate, RUL distribution charts, per-engine deep dive with GenAI panels:
   - **🔮 Probabilistic RUL Trajectories** — cVAE / Monte Carlo fan plot with 95%/50% CI bands
   - **🔬 Sensor Attribution (SHAP)** — per-sensor contribution bar chart with group importance
   - **🤖 AI Maintenance Brief** — LLM/template-generated maintenance report with urgency level

2. **📡 Live Digital Twin** — Real-time MQTT feed showing predicted/trusted RUL, RI trajectory, decision distribution, auto-refresh capability

3. **📈 RUL Trajectories** — Streaming replay simulation showing cycle-by-cycle RUL evolution with reliability tracking

4. **🔬 Batch Inference** — Detailed analytics with sensor correlation matrices, distributions, and optional ground-truth evaluation

5. **⚙️ Settings** — Backend discovery, validation commands, MQTT pipeline instructions

### Design Features
- **Dark glassmorphism theme** with gradient backgrounds and backdrop blur
- **Inter font** (Google Fonts) for modern typography
- **Decision badges** with color-coded pills (green/amber/red)
- **Plotly dark template** for all charts
- **Responsive sidebar** with categorized navigation

## 8.3 API Integration

### MQTT Digital Twin Pipeline

```
Node-RED (flows.json) → Mosquitto (localhost:1883) → mqtt_secure_ingest.py → logs/ → Streamlit
```

The Node-RED flow simulates 6 sensor categories:
- **Vibration** — fan, compressor, turbine vibration
- **Thermal** — T24, T30, T50 temperatures
- **Pressure** — P30, Ps30 pressures
- **Flow** — fuel flow, bypass ratio
- **Mechanical** — fan speed, core speed, bleed flow
- **RUL Model** — cVAE-style RUL estimation with physics risk and CPC scoring

### CLI Tools

```powershell
# Headless inference
python scripts\run_headless_inference.py --csv <file> --mode Baseline --backend attention

# API server
python scripts\run_api.py --host 0.0.0.0 --port 8000

# Digital twin streaming
python ingestion\digital_twin_streamer.py --broker 127.0.0.1 --port 1883 --topic engines/fd001/raw

# MQTT ingestion with live inference
python ingestion\mqtt_secure_ingest.py --broker 127.0.0.1 --port 1883 --topic engines/fd001/raw --model-mode Baseline --insecure-no-tls

# Edge export and benchmarking
python scripts\export_tflite_edge.py
python scripts\benchmark_edge_inference.py
```

---

# 9. Conclusion

## 9.1 Summary

This project developed a comprehensive, production-grade system for predicting the Remaining Useful Life of turbofan engines using the NASA C-MAPSS FD001 dataset. Key achievements include:

1. **Attention-based GRU model** achieving RMSE ~14.21 on FD001, competitive with state-of-the-art approaches
2. **Physics-Informed variant** incorporating thermodynamic constraints during training (RMSE ~16.46 with improved physical consistency)
3. **PI-LightGBM hybrid** combining gradient-boosted trees with post-prediction physics constraint validation and CPC scoring
4. **Reliability Index framework** providing calibrated prediction trustworthiness with operational gating (ACCEPT/WARN/REJECT)
5. **Plug-and-play model backend** enabling seamless switching between 3 model architectures without code changes
6. **Premium Streamlit dashboard** with fleet monitoring, per-engine deep dive, live digital twin feed, and streaming replay
7. **Real-time digital twin integration** via Node-RED sensor simulation, MQTT streaming, and live inference pipeline
8. **Production tooling** including FastAPI endpoints, headless CLI, edge export (TFLite), and comprehensive validation suite

## 9.2 Limitations

1. **Single dataset:** Evaluated only on FD001 (single operating condition, single fault mode). Performance on FD002-FD004 (multi-condition, multi-fault) is not assessed.
2. **Synthetic data:** C-MAPSS is a simulation; real engine sensor data may exhibit different noise characteristics and failure modes.
3. **Early RUL clipping:** The 125-cycle cap limits the model's ability to predict very long RULs, which may be relevant for low-usage engines.
4. **LightGBM compatibility:** The `model_artifacts.zip` has sklearn version dependencies that must match the runtime environment.
5. **cVAE training data:** The cVAE trajectory generator currently uses Monte Carlo fallback; training on FD001 data would produce more accurate degradation-pattern-driven trajectories.
6. **Edge latency:** The full attention GRU model requires ~200-500ms per inference on CPU, which may be insufficient for very high-frequency monitoring.

## 9.3 Future Work

1. **Multi-dataset evaluation:** Extend to FD002, FD003, FD004 datasets and cross-dataset transfer learning
2. **cVAE training:** Train the Conditional VAE on full FD001 training data for learned degradation pattern generation
3. **Federated learning:** Enable multi-fleet training without sharing raw sensor data across operators
4. **Reinforcement learning:** Optimize maintenance scheduling by treating RUL predictions as state inputs to an RL agent
5. **Diffusion model augmentation:** Use diffusion models to generate synthetic rare failure patterns for data augmentation
6. **Real sensor data validation:** Partner with MRO providers for validation on actual fleet maintenance records
7. **Dockerization:** Containerize the full stack (API + Dashboard + MQTT + Node-RED) for portable deployment
8. **Kubernetes deployment:** Orchestrate containers for scalable cloud deployment
9. **Mobile dashboard:** Develop a companion mobile app for field engineers

---

# References

1. Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). "Damage propagation modeling for aircraft engine run-to-failure simulation." *International Conference on Prognostics and Health Management (PHM)*.

2. Heimes, F. O. (2008). "Recurrent neural networks for remaining useful life estimation." *International Conference on PHM*.

3. Cho, K., Van Merriënboer, B., Gulcehre, C., et al. (2014). "Learning phrase representations using RNN encoder-decoder for statistical machine translation." *EMNLP*.

4. Bahdanau, D., Cho, K., & Bengio, Y. (2014). "Neural machine translation by jointly learning to align and translate." *arXiv:1409.0473*.

5. Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017). "Long short-term memory network for remaining useful life estimation." *IEEE ICPHM*.

6. Ke, G., Meng, Q., Finley, T., et al. (2017). "LightGBM: A highly efficient gradient boosting decision tree." *NeurIPS*.

7. Li, X., Ding, Q., & Sun, J. Q. (2018). "Remaining useful life estimation in prognostics using deep convolution neural networks." *Reliability Engineering & System Safety*.

8. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*.

9. Chen, Z., Wu, M., Zhao, R., et al. (2020). "Machine remaining useful life prediction via an attention-based deep learning approach." *IEEE Transactions on Industrial Electronics*.

10. TensorFlow. (2024). *TensorFlow: An End-to-End Open Source Machine Learning Platform.* https://www.tensorflow.org

11. Streamlit. (2024). *Streamlit: A faster way to build and share data apps.* https://streamlit.io

12. Node-RED. (2024). *Node-RED: Low-code programming for event-driven applications.* https://nodered.org

---

# Appendix: Code Snippets

## A.1 Attention Layer Implementation

```python
class AdditiveAttentionForSeq(tf.keras.layers.Layer):
    def __init__(self, attention_size: int, **kwargs):
        super().__init__(**kwargs)
        self.attention = tf.keras.layers.Dense(attention_size)
        self.last_attention_weights = None

    def call(self, state, encoder_outputs):
        flat_state = []
        for item in state:
            if isinstance(item, (list, tuple)):
                if len(item) == 0:
                    continue
                flat_state.append(item[0])
            else:
                flat_state.append(item)

        seq_len = encoder_outputs.shape[1]
        averaged_state = tf.reduce_mean(tf.stack(flat_state, axis=1), axis=1)
        state_rep = tf.repeat(tf.expand_dims(averaged_state, axis=1),
                              repeats=seq_len, axis=1)
        concat = tf.concat((state_rep, encoder_outputs), axis=-1)
        scores = tf.nn.tanh(self.attention(concat))
        attention_weights = tf.nn.softmax(tf.reduce_sum(scores, axis=-1), axis=-1)
        self.last_attention_weights = attention_weights
        return tf.matmul(tf.expand_dims(attention_weights, axis=1), encoder_outputs)
```

## A.2 Reliability Index Computation

```python
def compute_reliability_index(window_predictions, *, early_rul=125.0,
                              w_stability=0.6, w_physics=0.4):
    preds = np.asarray(window_predictions, dtype=float).reshape(-1)
    pred_mean, pred_std = float(np.mean(preds)), float(np.std(preds))

    cv = pred_std / max(abs(pred_mean), 1e-6)
    spread = (np.max(preds) - np.min(preds)) / max(abs(pred_mean), 1e-6)
    stability_score = (1/(1+cv) + 1/(1+spread)) / 2

    diffs = np.diff(preds)
    monotonic_score = max(0, 1 - float(np.mean(diffs > 1.0)))
    smoothness_score = 1 / (1 + np.mean(np.abs(np.diff(preds, n=2))))
    boundary_score = max(0, 1 - np.mean((preds < 0) | (preds > early_rul)))

    physics_score = 0.5*monotonic_score + 0.3*smoothness_score + 0.2*boundary_score
    ri = w_stability * stability_score + w_physics * physics_score
    return min(max(ri, 0.0), 1.0)
```

## A.3 Backend Registry Pattern

```python
class BackendRegistry:
    def __init__(self):
        self._factories = {}

    def register(self, name, factory, description="", aliases=None):
        self._factories[name.lower()] = (factory, description, aliases or [])
        for alias in (aliases or []):
            self._factories[alias.lower()] = (factory, description, [])

    def get_factory(self, name):
        entry = self._factories.get(name.lower())
        if entry is None:
            raise ValueError(f"Unknown backend: {name}")
        return entry[0]

    def list_backends(self):
        return [{"name": k, "description": v[1]}
                for k, (_, _, aliases) in self._factories.items()
                if k not in {a.lower() for a in aliases}]

# Registration
registry = BackendRegistry()
registry.register("attention", lambda m: AttentionModelAdapter(m),
                   "Attention GRU (Baseline / Physics-Informed)")
registry.register("artifact", lambda m: ArtifactModelAdapter(m),
                   "LightGBM from model_artifacts.zip")
registry.register("pi-lightgbm", lambda m: PILightGBMAdapter(m),
                   "Physics-Informed LightGBM")
```

## A.4 Physics Constraint Layer (CPC)

```python
PHYSICS_THRESHOLDS = {
    "s_3":  {"op": "gt", "value": 1620.0, "penalty": 0.25, "label": "HPC overtemp"},
    "s_4":  {"op": "gt", "value": 1450.0, "penalty": 0.25, "label": "LPT overtemp"},
    "s_9":  {"op": "lt", "value": 8500.0, "penalty": 0.20, "label": "HPC pressure drop"},
    "s_18": {"op": "lt", "value": 88.0,   "penalty": 0.15, "label": "HPT coolant depletion"},
    "s_19": {"op": "lt", "value": 32.0,   "penalty": 0.15, "label": "LPT coolant depletion"},
}

def compute_physics_risk(row):
    total_risk = 0.0
    for sensor, cfg in PHYSICS_THRESHOLDS.items():
        val = float(row[sensor])
        if (cfg["op"] == "gt" and val > cfg["value"]) or \
           (cfg["op"] == "lt" and val < cfg["value"]):
            total_risk += cfg["penalty"]
    cpc = 1.0 - min(1.0, total_risk)
    return {"physics_risk": total_risk, "cpc": cpc}
```

## A.5 Gating Logic

```python
def gate_prediction(predicted_rul, reliability_index, window_predictions,
                    accept_threshold=0.75, warning_threshold=0.45):
    pred_std = float(np.std(window_predictions))
    conservative_fallback = max(0, predicted_rul - 2 * pred_std)

    if reliability_index >= accept_threshold:
        return {"decision": "ACCEPT", "trusted_rul": predicted_rul}
    elif reliability_index >= warning_threshold:
        return {"decision": "WARN", "trusted_rul": predicted_rul}
    else:
        return {"decision": "REJECT", "trusted_rul": conservative_fallback}
```

## A.6 cVAE Trajectory Generator

```python
class ConditionalVAE(tf.keras.Model):
    def __init__(self, latent_dim=16, trajectory_length=30):
        super().__init__()
        self.encoder = CVAEEncoder(latent_dim)
        self.decoder = CVAEDecoder(trajectory_length)

    def reparameterize(self, mu, log_var):
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps

    def generate_trajectories(self, condition, n_samples=50):
        z = tf.random.normal(shape=(n_samples, self.latent_dim))
        decoder_input = tf.concat([z, condition_batch], axis=-1)
        return self.decoder(decoder_input).numpy()
```

## A.7 LLM Maintenance Explainer

```python
class MaintenanceExplainer:
    def __init__(self, api_key=None):
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def explain(self, ctx: ExplanationContext) -> Dict[str, str]:
        if self.has_llm:
            llm_text = generate_llm_explanation(ctx, self._api_key)
            if llm_text:
                return {"text": llm_text, "mode": "llm", ...}
        return {"text": generate_template_explanation(ctx), "mode": "template", ...}
```

## A.8 SHAP Sensor Attribution

```python
def compute_attention_sensor_importance(attention_weights, sensor_window):
    importance = np.zeros(n_sensors)
    for s in range(n_sensors):
        weighted_mean = np.average(sensor_vals, weights=attn)
        weighted_var = np.average((sensor_vals - weighted_mean)**2, weights=attn)
        trend = np.polyfit(np.arange(T), sensor_vals, deg=1, w=attn)[0]
        importance[s] = np.sqrt(weighted_var) * np.sign(trend)
    return _build_contributions(importance)
```
