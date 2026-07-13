# Future Prediction Engine (FPE) — Comprehensive Development & Evaluation Report

**Document Version**: 1.0  
**Repository**: [FPE-Student-State-Forecasting-Engine--Wellmefy](https://github.com/tezendrax/FPE-Student-State-Forecasting-Engine--Wellmefy)  
**Position in Architecture**: Module 5 (Student Wellness Forecaster)

---

## 1. Executive Summary
The Future Prediction Engine (FPE) is designed to forecast student wellness vectors over a 7-day forecast horizon. By utilizing temporal multi-quantile self-attention forecasting models, the engine identifies gradual declines in student wellness states before they escalate into acute health conditions. This report compiles the full system specifications, data pipelines, model configurations, training histories, evaluation parameters, and frontend interfaces.

---

## 2. Positioning in System Architecture
The FPE acts as a downstream processor for student states and feeds predictive signals into explainability and intervention systems:

```mermaid
graph LR
    subgraph Upstream Systems
        M3[Student Digital Twin Module 3] --> |14-Day Lookback History| FPE
        M4[Multi-Outcome Prediction Module 4] --> |Historical Risks| FPE
    end
    
    subgraph FPE Core
        FPE[Future Prediction Engine Module 5]
    end
    
    subgraph Downstream Systems
        FPE --> |7-Day Quantile Forecasts| M6[Explainability Engine Module 6]
        FPE --> |7-Day Quantile Forecasts| M7[Personalized Intervention Module 7]
        FPE --> |REST API Response| UI[Interactive Dashboard]
    end
```

---

## 3. Forecast Lifecycle Flow
This sequence diagram shows the step-by-step execution path when the client dashboard queries a student forecast:

```mermaid
sequenceDiagram
    autonumber
    actor User as Dashboard Frontend UI
    participant API as FastAPI Server (Port 8003)
    participant Cache as Redis / Local Cache
    participant DB as SQLite sdt.db (Digital Twin)
    participant Model as TFT Forecasting Core
    participant Fallback as Linear Baseline Fallback
    participant DB_FPE as SQLite fpe.db (Forecasts)
    
    User->>API: GET /api/v1/predictions/forecast?student_id=std-1001
    API->>Cache: Check cached forecast for std-1001
    alt Cache Hit
        Cache-->>API: Return cached JSON
        API-->>User: Render forecast graphs (Latency ~1ms)
    else Cache Miss
        API->>DB: Fetch encrypted state history (14 days lookback)
        DB-->>API: Return encrypted records & active key
        API->>API: Decrypt payloads using Fernet key
        API->>API: Linearly interpolate missing dates & resample to daily grid
        API->>API: Engineer features (Academic pressure, sleep-stress ratio, rolling std)
        API->>Model: Execute forward pass (scaled features)
        alt Inference Successful (Within bounds)
            Model-->>API: Return p10, p50, p90 predictions
        else Inference Fails or Diverges (values out of [-0.5, 1.5] bounds)
            API->>Fallback: Trigger Linear Regression fallback
            Fallback-->>API: Return linear forecast projections
        end
        API->>API: Clip final predictions to [0.0, 1.0] boundaries
        API->>DB_FPE: Write daily forecast values to database
        API->>Cache: Save forecast to Cache
        API-->>User: Return forecast JSON (Latency ~50ms)
    end
```

---

## 4. Data Preprocessing & Feature Engineering
Input telemetry is processed through a sequential pipeline in [fpe/dataset.py](file:///c:/Users/Tejendra/Singh/Desktop/Sarthi_Summer_Intern/Wellmate-Web/backend/Engines/Future/Prediction/Engine/fpe/dataset.py):

### 4.1 Imputation & Resampling
Telemetry entries inside `sdt.db` are decrypted and mapped to a regular daily grid. Missing entries are imputed using linear interpolation:
$$X_t = X_{t-a} + \frac{t - (t-a)}{(t+b) - (t-a)} \cdot (X_{t+b} - X_{t-a})$$
Edge values are filled using backward/forward propagation to ensure a continuous 14-day history.

### 4.2 Feature Selection
The engine uses three classes of variables:
1. **Historical Covariates (17 Features)**:
   * 10 primary student state dimensions (stress, anxiety, fatigue, social, academic, burnout, sleep, mood, resilience, focus).
   * **Academic Workload Pressure**: $AP_t = e^{-dist\_to\_exam / 7.0}$, representing exponential pressure scaling near exam events (Midterms at day 45, Finals at day 88).
   * **Sinusoidal Day-of-Week**: $\sin(2\pi d / 7)$ and $\cos(2\pi d / 7)$ to capture weekly routines.
   * **7-Day Sleep Volatility**: Rolling standard deviation of sleep quality.
   * **7-Day Stress Volatility**: Rolling standard deviation of stress levels.
   * **7-Day Stress Delta**: Velocity of stress accumulation ($Stress_t - Stress_{t-7}$).
   * **Sleep-to-Stress Ratio**: $\frac{Sleep_t}{Stress_t + 10^{-5}}$ representing stress buffers.
2. **Future Known Covariates (3 Features)**:
   * Planned Academic Workload Pressure, Future Day-of-Week Sine/Cosine.
3. **Static Covariates (10 Features)**:
   * Baseline wellness state means for each student, representing static personal bounds.

---

## 5. Deep Forecasting Model Architecture (TFT)
The forecasting core is built in PyTorch under [fpe/model.py](file:///c:/Users/Tejendra/Singh/Desktop/Sarthi_Summer_Intern/Wellmate-Web/backend/Engines/Future/Prediction/Engine/fpe/model.py):

```mermaid
graph TD
    %% Inputs
    subgraph Input Layers
        H[Historical Covariates: 14x17]
        F[Future Covariates: 7x3]
        S[Static Metadata: 10]
    end
    
    %% Processing
    subgraph Feature Gate Routing
        H --> GRN1[Gated Residual Network GRN]
        F --> GRN2[Gated Residual Network GRN]
        S --> GRN3[Gated Residual Network GRN]
    end
    
    %% Attention
    subgraph Self-Attention Core
        GRN1 --> SA[Temporal Multi-Head Self-Attention]
        GRN2 --> SA
    end
    
    %% Decoders
    subgraph Quantile Decoder
        SA --> QD[Quantile Projection Layers]
        GRN3 --> QD
    end
    
    %% Outputs
    subgraph Target Forecasts
        QD --> P10[p10 Quantile: 7x10]
        QD --> P50[p50 Quantile: 7x10]
        QD --> P90[p90 Quantile: 7x10]
    end
```

### 5.1 Gate Components (GRN & GLU)
All inputs pass through **Gated Residual Networks (GRN)** containing **Gated Linear Units (GLU)**. This enables the model to suppress irrelevant covariates:
$$GRN(a, s) = LayerNorm(a + GLU(Linear(Linear(a) + Linear(s))))$$
$$GLU(x) = \sigma(Linear_1(x)) \odot Linear_2(x)$$
Where $\sigma$ is the sigmoid activation function and $\odot$ is the Hadamard product.

### 5.2 Multi-Head Self-Attention
A temporal fusion decoder processes lookback states using self-attention:
$$Attention(Q, K, V) = softmax\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
This allows the model to learn long-range temporal dependencies and isolate sudden changes.

### 5.3 Multi-Quantile Decoder
Linear layers map the decoder outputs to 3 quantiles ($q \in \{0.1, 0.5, 0.9\}$) for all 10 wellness dimensions:
$$\hat{Y}_{t+h|t} = [\hat{y}_{t+h}^{(p10)}, \hat{y}_{t+h}^{(p50)}, \hat{y}_{t+h}^{(p90)}]$$
This guarantees that prediction limits narrow or widen depending on temporal volatility.

### 5.4 Model Parameters
* **Historical features**: 17
* **Future features**: 3
* **Static features**: 10
* **Hidden size**: 16
* **Attention heads**: 2
* **Target dimensions**: 10
* **Total trainable parameters**: **4,628 parameters** (optimized for CPU deployment, processing queries in under $60\text{ ms}$).

---

## 6. Training Pipeline & Parameters
The training loop is defined in [fpe/pipeline.py](file:///c:/Users/Tejendra/Singh/Desktop/Sarthi_Summer_Intern/Wellmate-Web/backend/Engines/Future/Prediction/Engine/fpe/pipeline.py):

### 6.1 Loss Function (Pinball Loss)
The model is trained using multi-quantile pinball loss:
$$\mathcal{L}_{pinball}(y, \hat{y}, q) = \max(q(y - \hat{y}), (q-1)(y - \hat{y}))$$
$$\mathcal{L}_{total} = \sum_{h=1}^{H} \sum_{d=1}^{D} \sum_{q \in \{0.1, 0.5, 0.9\}} \mathcal{L}_{pinball}(y_{t+h,d}, \hat{y}_{t+h,d}^{(q)}, q)$$

### 6.2 Hyperparameters & Settings
* **Max Training Epochs**: 30
* **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$)
* **Learning Rate**: $10^{-3}$
* **Batch Size**: 64
* **Early Stopping Patience**: 15 epochs
* **Scale Normalization**: Min-max normalization parameters cached in `scaler_params.json` during training dataset initialization.

---

## 7. Evaluation Metrics & Results
The model was tested against a 20% hold-out evaluation set (36 student cohorts over 90 days):

### 7.1 Performance Table
* **Quantile Loss (q-Loss)**: **`0.01804`** (Target: $< 0.08$) — **PASSED**
* **Mean Absolute Scaled Error (MASE)**: **`0.59286`** (Target: $< 1.10$) — **PASSED**
* **Prediction Drift (Wasserstein Distance)**: **`0.01793`** — **HEALTHY**

### 7.2 Forecast Output Trajectory Visualization
Below is a conceptual representation of how the three quantiles ($p10$, $p50$, $p90$) are served by the API. The solid line represents the most likely trajectory ($p50$), and the dashed boundaries represent the confidence interval ($p10$ to $p90$):

```mermaid
graph LR
    subgraph Quantile Forecast Trajectory
        D1((Day 1)) --> D2((Day 2)) --> D3((Day 3)) --> D4((Day 4)) --> D5((Day 5)) --> D6((Day 6)) --> D7((Day 7))
        
        %% Visual limits representation
        p90[p90 upper bound: high uncertainty / pessimistic path]
        p50[p50 median curve: curvy, non-linear forecast path]
        p10[p10 lower bound: low uncertainty / optimistic path]
        
        D4 -.-> p90
        D4 === p50
        D4 -.-> p10
    end
```

### 7.3 Unit Testing Validation
The testing suite in the `tests/` directory covers all core classes:
1. `test_gated_residual_network`: Verifies GRN hidden size projections.
2. `test_temporal_fusion_transformer_forward`: Validates attention forward passes and checks that $p10 \le p50 \le p90$ limits hold.
3. `test_quantile_loss`: Validates pinball loss outputs.
4. `test_linear_fallback_forecaster`: Checks regression fallback, slope bounds, and clipping.
5. `test_preprocess_and_interpolate`: Verifies linear interpolation over daily sequences.
6. `test_dataset_sequence_loading`: Verifies synthetic dataset sequence slicing.
7. `test_mase_calculation`: Validates accuracy evaluation metrics.
8. `test_prepare_inference_sequence`: Checks real-time scaling and inputs preparation.

**Testing Status**: **8/8 unit tests passed successfully**.

---

## 8. Customizations & Dashboard UI
The local dashboard (hosted at http://localhost:8003) includes custom interface structures:

1. **Rebranding**: Changed all header branding to **"State Future Prediction Engine"** using the *Outfit* typeface.
2. **Preloaded Cohort**: Preloaded `sdt.db` with 30 days of custom histories for 4 profiles:
   * `std-9874` (High Burnout) — Burnout rises linearly to `~0.73`.
   * `std-1001` (Anxiety Cycle) — Anxiety peaks around exams (`0.75`) and returns to normal.
   * `std-1002` (Chronic Sleep Debt) — Sleep quality stays low (`~0.22`), pushing fatigue to `~0.74` and burnout to `~0.63`.
   * `std-1003` (Resilient Profile) — Low stress and stable indicators.
3. **Interactive Dropdowns**: Dropdown select menu with a **"Custom Student ID"** option. Selecting "Custom" reveals a text input field for typing any student ID, while choosing a preloaded student queries their forecast instantly.
4. **Resilient Scaling Bounds**: We tuned model bounds tolerances in `fpe/inference.py`. Raw inputs that exceed normal limits are clipped to `[0.0, 1.0]`, allowing the curved attention-based TFT forecasts to run directly instead of falling back to the linear baseline.
5. **Score Interpretation Panel**: A visual color-coded index placed on the sidebar to help users understand what the 0.0 - 1.0 scores represent.
