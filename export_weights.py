"""
Person A runs this. Trains a simple logistic regression on your cleaned
heart / stroke dataframes and dumps weights + feature order to JSON,
which Person B's C++ server loads directly. No pickling, no XGBoost,
no model files to port — just numbers in a JSON file.

Usage:
    python export_weights.py

Edit the two train_and_export(...) calls at the bottom to point at your
actual cleaned dataframes and target column names.
"""

import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_and_export(df: pd.DataFrame, feature_cols: list[str], target_col: str, out_path: str):
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Simple, fast scaling -- store mean/std too, since C++ needs to
    # apply the SAME scaling to incoming user answers before scoring.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_scaled, y)

    export = {
        "feature_order": feature_cols,
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
        "weights": clf.coef_[0].tolist(),
        "bias": float(clf.intercept_[0]),
    }

    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"  train accuracy: {clf.score(X_scaled, y):.3f}")


if __name__ == "__main__":
    # --- EDIT THESE THREE LINES PER MODEL ---
    # heart_df = pd.read_csv("heart.csv")
    # heart_features = ["age", "sex", "cp", "trestbps", "chol", ...]  # your actual columns
    # train_and_export(heart_df, heart_features, "target", "heart_weights.json")

    # stroke_df = pd.read_csv("stroke.csv")
    # stroke_features = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", ...]
    # train_and_export(stroke_df, stroke_features, "stroke", "stroke_weights.json")
    pass
