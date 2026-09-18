# MedflowAI — ML Models (Person A)

## Heart Disease + Stroke Prediction

Neural network models trained on heart disease and stroke datasets, exported as JSON for the C++ inference backend.

### Files

| File | Description |
|---|---|
| `train_heart.py` | Trains MLP on `heart.csv`, exports `heart_model.json` |
| `train_stroke.py` | Trains MLP on `stroke_prediction.csv`, exports `stroke_prediction.json` |
| `heart_model.json` | Heart model weights — ready for C++ backend |
| `stroke_prediction.json` | Stroke model weights — ready for C++ backend |

### Model Architecture
Both models: `Input → 64 → 32 → 1 (sigmoid)`

### C++ Inference (identical for both models)
```cpp
// 1. Scale: z_i = (x_i - mean_i) / scale_i
// 2. Forward pass through layers:
//    a = z
//    for each hidden layer: a = relu(W @ a + b)
// 3. Output: logit = dot(W_out, a) + b_out
// 4. prob = 1 / (1 + exp(-logit))
// 5. pred = prob >= decision_threshold
```

### Results
| Model | Accuracy |
|---|---|
| Heart Disease | **96.1%** |
| Stroke | 70% (optimised for recall — catches 80% of real stroke cases) |

### Run
```bash
pip install pandas numpy scikit-learn
python train_heart.py        # outputs heart_model.json
python train_stroke.py       # outputs stroke_prediction.json
```
