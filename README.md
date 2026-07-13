# State Future Prediction Engine (FPE) — Student Wellness Forecasting

Welcome to the official repository of the **State Future Prediction Engine (FPE)**. This engine forecasts the trajectory of student wellness states over a 7-day forecast horizon (with extensibility to 14 and 30 days) to identify gradual declines in wellness before they manifest as critical health issues.

FPE operates as **Module 5** of the Wellmate wellness platform, utilizing deep temporal self-attention models to deliver quantile forecasts.

---

## 1. System & Architecture Workflows

### 1.1 Complete System Integration Flow
The FPE interacts with the Student Digital Twin database, processes temporal sequences, executes real-time inference, and feeds results downstream:

```mermaid
graph TD
    %% Upstream Data Sources
    U1[Multi-Outcome Prediction Engine Module 4] --> |Historical Risks| FPE[Future Prediction Engine]
    U2[Student Digital Twin Module 3] --> |Encrypted State History| FPE
    
    %% FPE Internal Workflow
    subgraph FPE [Future Prediction Engine Core]
        DB_SDT[(Digital Twin sdt.db)] --> |Telemetry Decryption| DP[Data Preprocessing]
        DP --> |Rolling Feature Engineering| TF[TFT Feature Sequence]
        TF --> |Forward Pass Inference| TFT[Temporal Fusion Transformer]
        TFT --> |Quantiles: p10, p50, p90| BC{Divergence Check}
        BC --> |Normal Predictions| OUT[Clip to 0.0 - 1.0]
        BC --> |Anomalous / NaNs| LRF[Linear Regression Fallback]
        OUT --> API[FastAPI REST Endpoints]
        LRF --> API
        API --> DB_FPE[(FPE Database fpe.db)]
        API --> Cache[(Redis / Local Memory Cache)]
    end
    
    %% Downstream Components
    API --> |JSON Forecasts| FE[Interactive Dashboard UI]
    API --> |Predictive Trajectories| D1[Explainability Engine Module 6]
    API --> |Predictive Trajectories| D2[Personalized Intervention Engine Module 7]

    %% Styling
    classDef default fill:#1e1e2e,stroke:#3b3b4f,color:#cdd6f4;
    classDef highlight fill:#11111b,stroke:#a6e3a1,color:#a6e3a1;
    class FPE,TFT highlight;
```

---

### 1.2 Forecast Request Lifecycle
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

## 2. Core Model Architecture (Temporal Fusion Transformer)

The deep learning model is built using a custom light-weight **Temporal Fusion Transformer (TFT)** designed to run efficiently on low-resource environments (CPUs):

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

* **Inputs & Features**:
  * **17 Historical Covariates**: 10 primary dimensions (stress, anxiety, fatigue, social, academic, burnout, sleep, mood, resilience, focus) + 7 engineered rolling metrics (Academic Workload Pressure, Day of Week Sine/Cosine, 7-Day Sleep Volatility, 7-Day Stress Volatility, 7-Day Stress Delta, Sleep-to-Stress Ratio).
  * **3 Future Known Covariates**: Workload Academic Pressure, Day of Week Sine/Cosine.
  * **10 Static Covariates**: Student average state baselines.
* **Quantile Outputs**: Decoder maps hidden states to 3 target quantiles:
  * **10th Percentile (p10)**: Optimistic/Lower bound forecast.
  * **50th Percentile (p50)**: Median/Most likely forecast trajectory.
  * **90th Percentile (p90)**: Pessimistic/Upper bound forecast.
* **Gate mechanisms**: Utilizing **Gated Residual Networks (GRN)** and **Gated Linear Units (GLU)** to filter out redundant features.
* **Self-Attention**: Multi-head self-attention layers to identify temporal dependencies.
* **Complexity**: **4,628 parameters** (extremely lightweight, CPU latency $< 60\text{ ms}$).

### Linear Regression Fallback
If the TFT model encounters extreme anomalies (severe drift, values exceeding the $[-0.5, 1.5]$ threshold, or NaNs), it triggers a `LinearBaselineFallback` model leveraging `scikit-learn` to project linear trajectories while raising backend warnings.

---

## 3. Pre-Configured Test Cohort

To demonstrate different student wellness trajectories in real-time, the database comes pre-populated with **30 days of custom daily history** for 4 distinct student profiles inside `sdt.db`:

1. **`std-9874` (High Burnout Trajectory)**: Exhibits a steady, linear rise in burnout (starting at `0.15` and climbing to `0.73` by day 30).
2. **`std-1001` (Midterm Anxiety Cycle)**: Shows a classic stress/anxiety spike centered around the midterm exam date (Day 18), which successfully recovers back to normal.
3. **`std-1002` (Chronic Sleep Debt)**: Displays high fatigue and extremely low sleep quality (`~0.22`) due to cumulative sleep debt, leading to moderate-high burnout (`~0.63`).
4. **`std-1003` (Stable/Resilient Profile)**: Balanced wellness indices (low stress, low anxiety, high resilience, and stable mood).

---

## 4. Evaluation Results

The TFT model has been evaluated on a hold-out test set (20% of the cohort):

### 4.1 Evaluation Summary Metrics
| Metric | Target Threshold | Evaluated Value | Status |
|---|---|---|---|
| **Quantile Loss (Pinball Loss)** | $< 0.08$ | **0.01804** | **PASSED** |
| **Mean Absolute Scaled Error (MASE)** | $< 1.10$ | **0.59286** | **PASSED** |
| **Prediction Drift (Wasserstein Distance)** | Info Only | **0.01793** | **HEALTHY** |

### 4.2 Forecast Output Trajectory Visualization
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

---

## 5. Local Setup & Deployment

### Prerequisites
* Python 3.10+
* Virtual Environment

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/tezendrax/FPE-Student-State-Forecasting-Engine--Wellmefy.git
   cd FPE-Student-State-Forecasting-Engine--Wellmefy
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline
* **Generate Synthetic Data**:
  ```bash
  python scripts/generate_data.py
  ```
* **Train the TFT Model**:
  *(Trains for 30 epochs with early stopping patience of 15 epochs and Adam optimizer $lr = 10^{-3}$)*
  ```bash
  python scripts/train_model.py
  ```
* **Evaluate the Trained Checkpoint**:
  ```bash
  python scripts/evaluate_model.py
  ```
* **Populate Test DB with Cohorts**:
  ```bash
  python scripts/populate_test_history.py
  ```

### Launching the Dashboard and API Server
1. Start the FastAPI server:
   ```bash
   python run_server.py
   ```
   *The server runs on http://localhost:8003.*
2. Open http://localhost:8003/ in your browser. The dashboard interface includes:
   * Rebranded **"State Future Prediction Engine"** logo.
   * Student selector dropdown for the 4 preloaded profiles.
   * **Custom Student ID** input field that reveals itself dynamically.
   * Responsive interactive line charts (solid p50, dashed p10/p90 confidence limits).
   * Diagnostic telemetry indicators (Inference Latency, MASE, Quantile Loss).

---

## 6. Score Interpretation Guide

All wellness parameters are normalized on a scale from **0.0 to 1.0**:

* **🟢 Low Range (0.0 – 0.3)**: 
  * Good for risk factors (e.g. Stress, Anxiety, Burnout, Fatigue).
  * Deficit for protective boosters (e.g. Sleep quality, Mood, Focus, Resilience).
* **🟡 Moderate Range (0.3 – 0.7)**: 
  * Transition/Typical activity levels. Requires baseline observation and monitoring.
* **🔴 High Range (0.7 – 1.0)**: 
  * Critical alert for risk factors (requires preventive interventions).
  * Optimal/Healthy for protective boosters.

---

## 7. License

This project is licensed under the MIT License - see the LICENSE file for details.
