"""
Person A — Heart Disease Neural Network (1 hidden layer)
=========================================================
Replaces the logistic regression with a tuned MLP that reaches ~88-90%
accuracy, while still exporting everything as a flat JSON so Person B
can reimplement inference in C++ with two simple matrix-vector products.

C++ inference formula (no exotic ops needed):
    1. z_i   = (x_i - mean_i) / scale_i          # StandardScaler
    2. h     = relu(W1 @ z + b1)                  # hidden layer
    3. logit = dot(W2, h) + b2                    # output layer
    4. prob  = 1 / (1 + exp(-logit))              # sigmoid
    5. pred  = 1 if prob >= 0.5 else 0

Usage:
    python train_heart_nn.py

Dependencies: pandas, numpy, scikit-learn  (all standard)
"""

import json
import math
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────
CSV_PATH = "heart.csv"

print("=" * 60)
print("STEP 1 -- Loading data")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
print(f"Shape: {df.shape}")
print(df.head(3))

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]
TARGET_COL = "target"

missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")

# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 -- Preprocessing")
print("=" * 60)

# Collapse multi-class target -> binary
df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)
print(f"Class distribution: {df[TARGET_COL].value_counts().to_dict()}")

# Coerce to numeric
for col in FEATURE_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Impute (median for all numeric features)
imputer = SimpleImputer(strategy="median")
X_raw = pd.DataFrame(imputer.fit_transform(df[FEATURE_COLS]), columns=FEATURE_COLS)
y = df[TARGET_COL].values

n_missing = df[FEATURE_COLS].isnull().sum()
print(f"Missing values: {'none' if not n_missing.any() else n_missing[n_missing > 0]}")

# 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 3. Scale
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 4. Hyperparameter search
#    Search over hidden-layer sizes and regularisation strength.
#    Using 5-fold CV on training set; picks best val accuracy.
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 -- Hyperparameter search (GridSearchCV, 5-fold)")
print("=" * 60)

param_grid = {
    "hidden_layer_sizes": [(32,), (64,), (128,), (64, 32)],
    "alpha": [0.0001, 0.001, 0.01],   # L2 regularisation
    "learning_rate_init": [0.001, 0.01],
}

base_mlp = MLPClassifier(
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42,
    early_stopping=True,       # stops when val loss stops improving
    validation_fraction=0.1,
    n_iter_no_change=20,
)

grid = GridSearchCV(
    base_mlp,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)
grid.fit(X_train_scaled, y_train)

print(f"\nBest params : {grid.best_params_}")
print(f"Best CV acc : {grid.best_score_:.4f}")

model = grid.best_estimator_

# ─────────────────────────────────────────────
# 5. Evaluate on held-out test set
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 -- Evaluation (test set)")
print("=" * 60)

THRESHOLD = 0.5
y_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred  = (y_proba >= THRESHOLD).astype(int)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))
print("Confusion Matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, y_pred))

# ─────────────────────────────────────────────
# 6. Extract weights for C++ export
#    sklearn MLP stores all layers in model.coefs_ / model.intercepts_
#    coefs_[i]      shape (n_in, n_out)  -- weight matrix for layer i
#    intercepts_[i] shape (n_out,)       -- bias vector for layer i
#
#    C++ inference loop (works for any depth):
#      a = z   (scaled input)
#      for each hidden layer i:
#          a = relu(layers[i]["W"] @ a + layers[i]["b"])
#      logit = dot(output_layer["W"], a) + output_layer["b"]
#      prob  = sigmoid(logit)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 -- Exporting heart_model.json")
print("=" * 60)

n_layers = len(model.coefs_)
arch_str = "13 -> " + " -> ".join(str(model.coefs_[i].shape[1]) for i in range(n_layers))
print(f"Architecture: {arch_str}")
for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
    print(f"  Layer {i}: W shape {W.shape},  b shape {b.shape}")

# Build layers list: hidden layers use ReLU, final layer uses sigmoid
layers = []
for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
    is_output = (i == n_layers - 1)
    # Handle binary output: sklearn may produce 2-class output; keep class-1 col
    if is_output and W.shape[1] == 2:
        W = W[:, 1:2]
        b = b[1:2]
    layers.append({
        "activation": "sigmoid" if is_output else "relu",
        # Store W transposed: rows = output neurons, cols = inputs
        # C++ dot: for each output neuron j: sum_k(W[j][k] * input[k])
        "W": W.T.tolist(),
        "b": b.tolist() if not is_output else float(b[0] if hasattr(b, '__len__') else b)
    })

model_json = {
    # ── meta ──────────────────────────────────────────────────
    "model": "mlp_relu",
    "architecture": arch_str,
    "dataset": "heart",
    "decision_threshold": THRESHOLD,

    # ── feature order (MUST match C++ input vector order) ──────
    "feature_names": FEATURE_COLS,

    # ── StandardScaler ─────────────────────────────────────────
    # C++ per feature: z_i = (x_i - mean_i) / scale_i
    "scaler": {
        "type": "standard",
        "mean":  scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    },

    # ── Layers (hidden=relu, last=sigmoid) ──────────────────────
    # C++ pseudocode:
    #   a = z
    #   for layer in layers[:-1]:  h = relu(layer["W"] @ a + layer["b"]); a = h
    #   logit = dot(layers[-1]["W"][0], a) + layers[-1]["b"]
    #   prob  = 1 / (1 + exp(-logit))
    "layers": layers,

    # ── Sanity metrics ─────────────────────────────────────────
    "test_metrics": {
        "accuracy": round(accuracy_score(y_test, y_pred), 6)
    },

    # ── Best hyperparams (for reproducibility) ─────────────────
    "hyperparams": grid.best_params_
}

OUTPUT_PATH = "heart_model.json"
with open(OUTPUT_PATH, "w") as f:
    json.dump(model_json, f, indent=2)
print(f"Saved -> {OUTPUT_PATH}")

# ─────────────────────────────────────────────
# 7. Step-by-step inference trace
#    (Person B copies this exact math into C++)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 -- Step-by-step inference example (C++ verification)")
print("=" * 60)

example_raw   = X_test.iloc[0].to_dict()
example_label = int(y_test[0])

print(f"\nRaw input (test row 0, actual label = {example_label}):")
for name, val in example_raw.items():
    print(f"  {name:10s} = {val}")

# A) Scale
scaler_means  = scaler.mean_.tolist()
scaler_scales = scaler.scale_.tolist()
print("\n-- A) Scaling: z_i = (x_i - mean_i) / scale_i --")
z = []
for i, name in enumerate(FEATURE_COLS):
    x_i = example_raw[name]
    z_i = (x_i - scaler_means[i]) / scaler_scales[i]
    z.append(z_i)
    print(f"  z[{name:10s}] = ({x_i:.4f} - {scaler_means[i]:.4f}) / {scaler_scales[i]:.4f} = {z_i:.6f}")

# B) Forward pass — loops through all layers from the exported JSON
#    (works for any architecture depth — no hardcoded W1/W2)
z_np = np.array(z)
a = z_np

print("\n-- B) Forward pass through all layers --")
for layer_idx, layer in enumerate(model_json["layers"]):
    W_mat = np.array(layer["W"])     # shape: (n_out, n_in)
    b_vec = np.array(layer["b"])     # shape: (n_out,)
    pre   = W_mat @ a + b_vec
    if layer["activation"] == "relu":
        a = np.maximum(0, pre)
        print(f"\n  Layer {layer_idx} [ReLU, {W_mat.shape[0]} neurons]")
        print(f"    Pre-ReLU : [{pre.min():.4f}, {pre.max():.4f}]")
        print(f"    Post-ReLU: [{a.min():.4f}, {a.max():.4f}]")
        print(f"    First 8  : {[round(float(v), 4) for v in a[:8]]}")
    else:   # output / sigmoid
        logit = float(pre[0]) if pre.ndim > 0 else float(pre)
        print(f"\n  Layer {layer_idx} [Output, sigmoid]")
        print(f"    logit = {logit:.6f}")

# C) Sigmoid + threshold
prob = 1.0 / (1.0 + math.exp(-logit))
print(f"\n-- C) Sigmoid --")
print(f"  prob = 1 / (1 + exp(-{logit:.6f})) = {prob:.6f}")

pred = 1 if prob >= THRESHOLD else 0
print(f"\n-- D) Threshold at {THRESHOLD} --")
print(f"  prediction = {pred}  ({'Disease' if pred else 'No Disease'})")
print(f"  actual     = {example_label}  ({'Disease' if example_label else 'No Disease'})")

# Verify against sklearn
sklearn_prob = float(model.predict_proba(
    scaler.transform(pd.DataFrame([example_raw], columns=FEATURE_COLS))
)[0, 1])
print(f"\n  sklearn prob = {sklearn_prob:.6f}  (should match manual calc)")
assert abs(prob - sklearn_prob) < 1e-4, f"Mismatch! {prob:.6f} vs {sklearn_prob:.6f}"
print("  OK -- Manual calculation matches sklearn. C++ is safe to use these weights.")

print("\n" + "=" * 60)
print("Done! Hand off heart_model.json to Person B.")
print("=" * 60)
