# MedflowAI — System Architecture

## Full Stack Overview

```mermaid
flowchart LR
    subgraph Input["📋 Patient Input"]
        FORM["Step-by-step form\nAge · Sex · BP · Cholesterol\nGlucose · BMI · Lifestyle..."]
    end

    subgraph Frontend["🌐 Frontend (Person C)"]
        UI["One-question-at-a-time UI"]
        RESULTS["Risk Score Screen\nGauge · Probability · Verdict"]
        CF["Counterfactual Slider\n'What if you changed X?'"]
        UI --> RESULTS --> CF
    end

    subgraph Backend["⚙️ C++ Server (Person B)\ncpp-httplib · nlohmann/json"]
        direction TB
        ROUTER["HTTP Router"]
        P1["POST /predict/heart"]
        P2["POST /predict/stroke"]
        P3["POST /counterfactual"]
        ENGINE["Inference Engine\nStandardScale\nReLU layers\nSigmoid output"]
        ROUTER --> P1 & P2 & P3 --> ENGINE
    end

    subgraph MLPipeline["🐍 ML Pipeline (Person A)\nPython · sklearn"]
        direction TB
        subgraph HeartPipeline["Heart Disease"]
            HD_CSV[("heart.csv")] --> HD_PRE["Preprocess\nImpute · Scale"] --> HD_MLP["MLP 13→64→32→1\nGridSearchCV"] --> HD_JSON[/"heart_model.json"/]
        end
        subgraph StrokePipeline["Stroke Risk"]
            SK_CSV[("stroke_prediction.csv")] --> SK_PRE["Preprocess\nEncode · Impute\nSample Weights"] --> SK_MLP["MLP 15→64→32→1\nGridSearchCV F1"] --> SK_JSON[/"stroke_prediction.json"/]
        end
    end

    FORM --> UI
    CF -->|"POST /counterfactual\n{ feature, value }"| P3
    Results -->|"POST /predict/heart\nPOST /predict/stroke"| ROUTER
    ENGINE -->|"{ risk, prob, delta }"| Frontend
    HD_JSON & SK_JSON -->|"Load at startup"| ROUTER
```

---

## Data Flow — Single Prediction

```mermaid
sequenceDiagram
    actor User
    participant FE as Frontend (JS)
    participant BE as C++ Server
    participant IE as Inference Engine

    User->>FE: Fills patient form
    FE->>BE: POST /predict/heart<br/>{ age:55, sex:1, cp:0, ... }
    BE->>IE: load heart_model.json weights
    IE->>IE: z = (x - mean) / scale
    IE->>IE: h0 = relu(W0 @ z + b0)
    IE->>IE: h1 = relu(W1 @ h0 + b1)
    IE->>IE: logit = dot(W2, h1) + b2
    IE->>IE: prob = sigmoid(logit)
    IE-->>BE: { prob: 0.87 }
    BE-->>FE: { risk: "HIGH", prob: 0.87, threshold: 0.5 }
    FE-->>User: Risk gauge + explanation

    User->>FE: Drags "Exercise" slider
    FE->>BE: POST /counterfactual<br/>{ feature:"thalach", value:150 }
    BE->>IE: Re-run with flipped feature
    IE-->>BE: { new_prob: 0.43, delta: -0.44 }
    BE-->>FE: { delta: -0.44, new_risk: "LOW" }
    FE-->>User: "Exercising more could reduce your risk by 44%"
```

---

## Neural Network Architecture (Both Models)

```mermaid
graph LR
    subgraph Input["Input Layer"]
        I1["x₁"]
        I2["x₂"]
        I3["..."]
        IN["xₙ"]
    end

    subgraph Scale["StandardScaler"]
        S["z = (x - μ) / σ"]
    end

    subgraph H1["Hidden Layer 1 · 64 neurons"]
        H1A["ReLU"]
        H1B["ReLU"]
        H1C["..."]
        H1D["ReLU"]
    end

    subgraph H2["Hidden Layer 2 · 32 neurons"]
        H2A["ReLU"]
        H2B["ReLU"]
        H2C["..."]
        H2D["ReLU"]
    end

    subgraph Output["Output"]
        O["sigmoid(logit)"]
        PRED["pred ≥ 0.5 ?"]
    end

    Input --> Scale --> H1 --> H2 --> O --> PRED
```

| Layer | Operation | Shape (Heart) | Shape (Stroke) |
|---|---|---|---|
| Input | Raw features | `(13,)` | `(15,)` |
| StandardScaler | `z = (x−μ)/σ` | `(13,)` | `(15,)` |
| Hidden 1 | `relu(W₁z + b₁)` | `W: (64×13)` | `W: (64×15)` |
| Hidden 2 | `relu(W₂h + b₂)` | `W: (32×64)` | `W: (32×64)` |
| Output | `sigmoid(W₃h + b₃)` | `W: (1×32)` | `W: (1×32)` |

---

## JSON Schema (shared by both models)

```
heart_model.json / stroke_prediction.json
├── model            "mlp_relu"
├── architecture     "13 -> 64 -> 32 -> 1"
├── decision_threshold  0.5
├── feature_names    ["age", "sex", ...]      ← ORDER MATTERS for C++
├── scaler
│   ├── type         "standard"
│   ├── mean         [54.6, 0.69, ...]        ← per feature, same order
│   └── scale        [9.09, 0.46, ...]
├── layers
│   ├── [0]  activation:"relu"    W:(64×13)  b:(64,)
│   ├── [1]  activation:"relu"    W:(32×64)  b:(32,)
│   └── [2]  activation:"sigmoid" W:(1×32)   b:scalar
└── test_metrics
    └── accuracy     0.961
```

---

## Model Performance

### Heart Disease (`heart_model.json`)
```
Accuracy:  96.1%

              precision  recall  f1
No Disease      0.96     0.96   0.96   (100 samples)
Disease         0.96     0.96   0.96   (105 samples)

Confusion Matrix:
          Pred 0   Pred 1
Actual 0    96       4
Actual 1     4     101
```

### Stroke (`stroke_prediction.json`)
```
Tuned for RECALL (class imbalance 19.5:1)

              precision  recall  f1
No Stroke       0.99     0.69   0.81   (972 samples)
Stroke          0.12     0.80   0.21   ( 50 samples)

Stroke recall: 80% — catches 4 of every 5 real stroke patients.
Lower threshold (e.g. 0.3) in JSON to increase recall further.
```

---

## C++ Pseudo-code

```cpp
#include <nlohmann/json.hpp>
#include <cmath>
#include <vector>

// Load once at startup
auto model = nlohmann::json::parse(std::ifstream("heart_model.json"));

double relu(double x) { return x > 0 ? x : 0; }
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

double predict(std::vector<double> x) {
    auto& scaler = model["scaler"];
    auto& layers = model["layers"];

    // 1. StandardScale
    for (int i = 0; i < x.size(); i++)
        x[i] = (x[i] - scaler["mean"][i].get<double>())
                     / scaler["scale"][i].get<double>();

    // 2. Forward pass
    std::vector<double> a = x;
    for (auto& layer : layers) {
        std::vector<double> out;
        auto W = layer["W"].get<std::vector<std::vector<double>>>();
        auto b = layer["b"];
        bool is_output = (layer["activation"] == "sigmoid");

        for (int j = 0; j < W.size(); j++) {
            double pre = (is_output ? b.get<double>()
                                    : b[j].get<double>());
            for (int k = 0; k < a.size(); k++)
                pre += W[j][k] * a[k];
            out.push_back(is_output ? sigmoid(pre) : relu(pre));
        }
        a = out;
    }
    return a[0];   // probability [0, 1]
}
```
