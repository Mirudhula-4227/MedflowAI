"""
Person A — Stroke Prediction Neural Network (1-2 hidden layers)
================================================================
Trains an MLP on stroke.csv (Kaggle Healthcare Dataset), exports
all inference parameters as stroke_prediction.json so Person B
can plug it straight into the C++ backend alongside heart_model.json.

Expected CSV columns:
    id, gender, age, hypertension, heart_disease, ever_married,
    work_type, Residence_type, avg_glucose_level, bmi,
    smoking_status, stroke

C++ inference (identical pattern to heart_model.json):
    z_i   = (x_i - mean_i) / scale_i          # StandardScaler
    a = z
    for each hidden layer:  a = relu(W @ a + b)
    logit = dot(W_out, a) + b_out
    prob  = 1 / (1 + exp(-logit))
    pred  = 1 if prob >= threshold else 0

Usage:
    python train_stroke.py

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
CSV_PATH = "stroke_prediction.csv"

print("=" * 60)
print("STEP 1 -- Loading data")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
print(f"Shape: {df.shape}")
print(df.head(3))
print(f"\nColumns: {list(df.columns)}")

# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 -- Preprocessing")
print("=" * 60)

# Drop id column if present (not a feature)
if "id" in df.columns:
    df = df.drop(columns=["id"])
    print("Dropped 'id' column.")

TARGET_COL = "stroke"
print(f"Class distribution (raw): {df[TARGET_COL].value_counts().to_dict()}")

# --- Categorical columns: label-encode or one-hot ---
# Binary cats -> 0/1 directly; multi-class -> one-hot
# gender: Male/Female/Other
df["gender"] = df["gender"].map({"Male": 1, "Female": 0, "Other": 0})

# ever_married: Yes/No
df["ever_married"] = df["ever_married"].map({"Yes": 1, "No": 0})

# Residence_type: Urban/Rural
df["Residence_type"] = df["Residence_type"].map({"Urban": 1, "Rural": 0})

# work_type: Private, Self-employed, Govt_job, children, Never_worked
# One-hot encode (drop first to avoid multicollinearity)
df = pd.get_dummies(df, columns=["work_type"], drop_first=True, dtype=float)

# smoking_status: formerly smoked, never smoked, smokes, Unknown
df = pd.get_dummies(df, columns=["smoking_status"], drop_first=True, dtype=float)

print(f"Columns after encoding: {list(df.columns)}")

# Separate target
y = df[TARGET_COL].astype(int).values
df = df.drop(columns=[TARGET_COL])

# Record final feature order (MUST be passed to C++ in this exact order)
FEATURE_COLS = list(df.columns)
print(f"\nFinal feature count: {len(FEATURE_COLS)}")
print(f"Features: {FEATURE_COLS}")

# Coerce everything to numeric
for col in FEATURE_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Impute missing values (bmi commonly has NaNs in this dataset)
n_missing = df.isnull().sum()
if n_missing.any():
    print(f"\nMissing values before imputation:")
    print(n_missing[n_missing > 0])
imputer = SimpleImputer(strategy="median")
X_raw = pd.DataFrame(imputer.fit_transform(df), columns=FEATURE_COLS)

print(f"\nClass distribution: No Stroke={int((y==0).sum())}, Stroke={int((y==1).sum())}")
print(f"Class imbalance ratio: {(y==0).sum() / (y==1).sum():.1f}:1")

# 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 3. Scale
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 4. Hyperparameter search
#    Stroke data is heavily imbalanced (~20:1), so we pass
#    class_weight handling via balanced sampling in GridSearchCV.
#    MLPClassifier doesn't have class_weight, so we use
#    sample_weight in the fit call — but GridSearchCV handles
#    this via scoring. We use 'f1' instead of 'accuracy' as
#    the search metric since the dataset is imbalanced.
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 -- Hyperparameter search (GridSearchCV, 5-fold, scoring=f1)")
print("=" * 60)

# Compute sample weights to handle class imbalance
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
weight_for_0 = 1.0
weight_for_1 = n_neg / n_pos   # up-weight the minority (stroke) class
sample_weights = np.where(y_train == 1, weight_for_1, weight_for_0)
print(f"Stroke class up-weight: {weight_for_1:.1f}x")

param_grid = {
    "hidden_layer_sizes": [(64, 32), (128, 64), (64,)],
    "alpha": [0.001, 0.01],
    "learning_rate_init": [0.001, 0.01],
}

base_mlp = MLPClassifier(
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)

grid = GridSearchCV(
    base_mlp,
    param_grid,
    cv=5,
    scoring="f1",     # F1 more informative than accuracy for imbalanced data
    n_jobs=-1,
    verbose=1,
)
# GridSearchCV pass 1: find best architecture (no sample_weight — MLP CV doesn't support it)
grid.fit(X_train_scaled, y_train)

print(f"\nBest params (architecture search): {grid.best_params_}")
print(f"Best CV F1 : {grid.best_score_:.4f}")

# Pass 2: refit the best estimator on full training set WITH sample_weight
# This is the model we'll actually export
best_params = grid.best_params_
model = MLPClassifier(
    activation="relu",
    solver="adam",
    max_iter=2000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    **best_params
)
model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
print("Refitted best model with class-imbalance sample weights.")

# ─────────────────────────────────────────────
# 5. Evaluate on test set
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 -- Evaluation (test set)")
print("=" * 60)

THRESHOLD = 0.5
y_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred  = (y_proba >= THRESHOLD).astype(int)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Stroke", "Stroke"]))
print("Confusion Matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, y_pred))

# ─────────────────────────────────────────────
# 6. Export JSON for C++ backend
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 -- Exporting stroke_prediction.json")
print("=" * 60)

n_layers = len(model.coefs_)
arch_str = f"{len(FEATURE_COLS)} -> " + " -> ".join(
    str(model.coefs_[i].shape[1]) for i in range(n_layers)
)
print(f"Architecture: {arch_str}")
for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
    print(f"  Layer {i}: W shape {W.shape},  b shape {b.shape}")

# Build layers list (same format as heart_model.json for consistency)
layers = []
for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
    is_output = (i == n_layers - 1)
    if is_output and W.shape[1] == 2:
        W = W[:, 1:2]
        b = b[1:2]
    layers.append({
        "activation": "sigmoid" if is_output else "relu",
        # W transposed: rows=output neurons, cols=input neurons
        "W": W.T.tolist(),
        "b": b.tolist() if not is_output else float(b[0] if hasattr(b, '__len__') else b)
    })

model_json = {
    # -- meta -----------------------------------------------
    "model": "mlp_relu",
    "architecture": arch_str,
    "dataset": "stroke",
    "decision_threshold": THRESHOLD,

    # -- feature order (MUST match C++ input vector order) --
    "feature_names": FEATURE_COLS,

    # -- StandardScaler -------------------------------------
    # C++ per feature: z_i = (x_i - mean_i) / scale_i
    "scaler": {
        "type": "standard",
        "mean":  scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    },

    # -- Layers (hidden=relu, last=sigmoid) -----------------
    # C++ pseudocode (same as heart_model.json):
    #   a = z
    #   for layer in layers[:-1]: a = relu(W @ a + b)
    #   logit = dot(layers[-1]["W"][0], a) + layers[-1]["b"]
    #   prob  = 1 / (1 + exp(-logit))
    "layers": layers,

    # -- Sanity metrics -------------------------------------
    "test_metrics": {
        "accuracy": round(accuracy_score(y_test, y_pred), 6)
    },
    "hyperparams": grid.best_params_
}

OUTPUT_PATH = "stroke_prediction.json"
with open(OUTPUT_PATH, "w") as f:
    json.dump(model_json, f, indent=2)
print(f"Saved -> {OUTPUT_PATH}")

# ─────────────────────────────────────────────
# 7. Step-by-step inference trace for C++ verification
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 -- Step-by-step inference example (C++ verification)")
print("=" * 60)

example_raw   = X_test.iloc[0].to_dict()
example_label = int(y_test[0])
print(f"\nRaw input (test row 0, actual label = {example_label} "
      f"[{'Stroke' if example_label else 'No Stroke'}]):")
for name, val in example_raw.items():
    print(f"  {name:35s} = {val}")

# A) Scale
scaler_means  = scaler.mean_.tolist()
scaler_scales = scaler.scale_.tolist()
print("\n-- A) Scaling: z_i = (x_i - mean_i) / scale_i --")
z = []
for i, name in enumerate(FEATURE_COLS):
    x_i = example_raw[name]
    z_i = (x_i - scaler_means[i]) / scaler_scales[i]
    z.append(z_i)
    print(f"  z[{name:35s}] = {z_i:.6f}")

# B) Generic forward pass (reads from exported JSON)
z_np = np.array(z)
a = z_np
print("\n-- B) Forward pass through all layers --")
for layer_idx, layer in enumerate(model_json["layers"]):
    W_mat = np.array(layer["W"])     # (n_out, n_in)
    b_vec = np.array(layer["b"])     # (n_out,)
    pre   = W_mat @ a + b_vec
    if layer["activation"] == "relu":
        a = np.maximum(0, pre)
        print(f"\n  Layer {layer_idx} [ReLU, {W_mat.shape[0]} neurons]")
        print(f"    Pre-ReLU : [{pre.min():.4f}, {pre.max():.4f}]")
        print(f"    Post-ReLU: [{a.min():.4f}, {a.max():.4f}]")
    else:
        logit = float(pre[0]) if pre.ndim > 0 else float(pre)
        print(f"\n  Layer {layer_idx} [Output, sigmoid]")
        print(f"    logit = {logit:.6f}")

# C) Sigmoid + threshold
prob = 1.0 / (1.0 + math.exp(-logit))
print(f"\n-- C) Sigmoid --")
print(f"  prob = 1 / (1 + exp(-{logit:.6f})) = {prob:.6f}")

pred = 1 if prob >= THRESHOLD else 0
print(f"\n-- D) Threshold at {THRESHOLD} --")
print(f"  prediction = {pred}  ({'Stroke' if pred else 'No Stroke'})")
print(f"  actual     = {example_label}  ({'Stroke' if example_label else 'No Stroke'})")

# Verify against sklearn
sklearn_prob = float(model.predict_proba(
    scaler.transform(pd.DataFrame([example_raw], columns=FEATURE_COLS))
)[0, 1])
print(f"\n  sklearn prob = {sklearn_prob:.6f}  (should match manual calc)")
assert abs(prob - sklearn_prob) < 1e-4, f"Mismatch! {prob:.6f} vs {sklearn_prob:.6f}"
print("  OK -- Manual calculation matches sklearn. C++ is safe to use these weights.")

print("\n" + "=" * 60)
print("Done! Both heart_model.json and stroke_prediction.json are ready for Person B.")
print("=" * 60)
