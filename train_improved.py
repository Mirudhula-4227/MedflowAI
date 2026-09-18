"""Leakage-safe, server-compatible retraining for MedFlowAI.

Example:
  python train_improved.py --heart-csv heart.csv --stroke-csv stroke_prediction.csv --output-dir models-v2

The source CSVs are intentionally not committed. This script leaves the deployed
heart_model.json and stroke_prediction.json unchanged unless its output directory
is explicitly copied over after review.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, fbeta_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
HEART_FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
STROKE_FEATURES = [
    "gender", "age", "hypertension", "heart_disease", "ever_married", "Residence_type",
    "avg_glucose_level", "bmi", "work_type_Never_worked", "work_type_Private",
    "work_type_Self-employed", "work_type_children", "smoking_status_formerly smoked",
    "smoking_status_never smoked", "smoking_status_smokes",
]


class BalancedMLPClassifier(ClassifierMixin, BaseEstimator):
    """MLP with deterministic, fold-local random oversampling.

    sklearn versions before 1.7 do not accept MLPClassifier.sample_weight.
    Keeping resampling in fit means GridSearchCV applies it only to each
    training fold, rather than leaking copied minority examples into validation.
    """
    def __init__(self, hidden_layer_sizes=(32, 16), alpha=0.01, learning_rate_init=0.001,
                 max_iter=2000, early_stopping=True, validation_fraction=0.15,
                 n_iter_no_change=30, random_state=RANDOM_STATE):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        positive = np.flatnonzero(y == 1)
        negative = np.flatnonzero(y == 0)
        if not len(positive) or not len(negative):
            raise ValueError("Both outcome classes are required for balanced stroke training.")
        rng = np.random.RandomState(self.random_state)
        extra_positive = rng.choice(positive, size=max(0, len(negative) - len(positive)), replace=True)
        indices = np.concatenate([np.arange(len(y)), extra_positive])
        rng.shuffle(indices)
        self.model_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
            learning_rate_init=self.learning_rate_init, max_iter=self.max_iter,
            early_stopping=self.early_stopping, validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change, random_state=self.random_state,
        ).fit(X[indices], y[indices])
        self.classes_ = self.model_.classes_
        self.coefs_ = self.model_.coefs_
        self.intercepts_ = self.model_.intercepts_
        self.n_features_in_ = self.model_.n_features_in_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_heart(path):
    df = pd.read_csv(path)
    required = HEART_FEATURES + ["target"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Heart CSV is missing columns: {sorted(missing)}")
    frame = df[required].copy()
    frame[HEART_FEATURES] = frame[HEART_FEATURES].apply(pd.to_numeric, errors="coerce")
    # The common 1,025-row heart CSV repeats records from a much smaller cohort.
    before = len(frame)
    frame = frame.drop_duplicates(subset=required).reset_index(drop=True)
    y = (pd.to_numeric(frame.pop("target"), errors="coerce") == 0).astype(int)
    return frame, y, {"input_rows": before, "unique_rows": len(frame), "target_definition": "target == 0 (disease)"}


def prepare_stroke(path):
    df = pd.read_csv(path)
    required = {"gender", "age", "hypertension", "heart_disease", "ever_married", "Residence_type", "avg_glucose_level", "bmi", "work_type", "smoking_status", "stroke"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Stroke CSV is missing columns: {sorted(missing)}")
    # Explicit encodings keep the feature names identical to frontend/src/lib/encoder.js.
    out = pd.DataFrame(index=df.index)
    out["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    out["age"] = pd.to_numeric(df["age"], errors="coerce")
    for name, source, values in [
        ("hypertension", "hypertension", None), ("heart_disease", "heart_disease", None),
        ("ever_married", "ever_married", {"Yes": 1, "No": 0}),
        ("Residence_type", "Residence_type", {"Urban": 1, "Rural": 0}),
    ]:
        out[name] = pd.to_numeric(df[source], errors="coerce") if values is None else df[source].map(values)
    out["avg_glucose_level"] = pd.to_numeric(df["avg_glucose_level"], errors="coerce")
    out["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    work_map = {"Never_worked": "work_type_Never_worked", "Private": "work_type_Private", "Self-employed": "work_type_Self-employed", "children": "work_type_children"}
    smoke_map = {"formerly smoked": "smoking_status_formerly smoked", "never smoked": "smoking_status_never smoked", "smokes": "smoking_status_smokes"}
    for column in list(work_map.values()) + list(smoke_map.values()):
        out[column] = 0.0
    for raw, column in {**work_map, **smoke_map}.items():
        out.loc[df["work_type"].eq(raw) | df["smoking_status"].eq(raw), column] = 1.0
    y = pd.to_numeric(df["stroke"], errors="coerce").fillna(0).astype(int)
    return out[STROKE_FEATURES], y, {"input_rows": len(df), "unique_rows": len(df), "target_definition": "stroke == 1"}


def choose_threshold(y, probability, beta):
    candidates = np.linspace(0.05, 0.95, 181)
    scores = [fbeta_score(y, probability >= t, beta=beta, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def metric_report(y, probability, threshold, subgroup=None):
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    report = {
        "threshold": round(float(threshold), 4), "roc_auc": round(float(roc_auc_score(y, probability)), 4),
        "pr_auc": round(float(average_precision_score(y, probability)), 4),
        "brier": round(float(brier_score_loss(y, probability)), 4),
        "precision": round(float(precision_score(y, prediction, zero_division=0)), 4),
        "recall_sensitivity": round(float(recall_score(y, prediction, zero_division=0)), 4),
        "specificity": round(float(tn / (tn + fp)) if tn + fp else 0.0, 4),
        "f1": round(float(f1_score(y, prediction, zero_division=0)), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if subgroup is not None:
        report["subgroups"] = {}
        for value in sorted(pd.Series(subgroup).dropna().unique()):
            mask = np.asarray(subgroup) == value
            if mask.sum() >= 10 and len(np.unique(np.asarray(y)[mask])) == 2:
                report["subgroups"][str(value)] = metric_report(np.asarray(y)[mask], probability[mask], threshold)
    return report


def export_model(path, pipeline, platt, feature_names, dataset, threshold, metadata, X_fit):
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    mlp = pipeline.named_steps["mlp"]
    layers = []
    for index, (weight, bias) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        output = index == len(mlp.coefs_) - 1
        if output:
            # Platt calibration: sigmoid(a * raw_logit + b), folded into the final C++ layer.
            weight = weight * platt.coef_[0][0]
            bias = bias * platt.coef_[0][0] + platt.intercept_[0]
        layers.append({"activation": "sigmoid" if output else "relu", "W": weight.T.tolist(), "b": float(bias[0]) if output else bias.tolist()})
    scaled = scaler.transform(imputer.transform(X_fit))
    ood_threshold = float(np.quantile(np.linalg.norm(scaled, axis=1), 0.995))
    model = {
        "model": "mlp_relu", "architecture": " -> ".join([str(len(feature_names))] + [str(w.shape[1]) for w in mlp.coefs_]),
        "dataset": dataset, "decision_threshold": threshold, "feature_names": feature_names,
        "scaler": {"type": "standard", "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "layers": layers, "ood_threshold": ood_threshold, "training_metadata": metadata,
    }
    path.write_text(json.dumps(model, indent=2))


def train(dataset, X, y, provenance, source_path, destination, stroke_beta=2.0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    X_fit, X_cal, y_fit, y_cal = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train)
    estimator = BalancedMLPClassifier() if dataset == "stroke" else MLPClassifier(max_iter=2000, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=RANDOM_STATE)
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("mlp", estimator)])
    scoring = "f1" if dataset == "stroke" else "roc_auc"
    grid = GridSearchCV(pipe, {"mlp__hidden_layer_sizes": [(16,), (32, 16), (64, 32)], "mlp__alpha": [0.001, 0.01, 0.1], "mlp__learning_rate_init": [0.001, 0.005]}, scoring=scoring, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), n_jobs=1, refit=True)
    grid.fit(X_fit, y_fit)
    pipeline = grid.best_estimator_
    cal_prob = pipeline.predict_proba(X_cal)[:, 1]
    cal_logit = np.log(np.clip(cal_prob, 1e-6, 1 - 1e-6) / np.clip(1 - cal_prob, 1e-6, 1))
    platt = LogisticRegression(C=1e6, solver="lbfgs").fit(cal_logit.reshape(-1, 1), y_cal)
    cal_probability = platt.predict_proba(cal_logit.reshape(-1, 1))[:, 1]
    threshold_beta = stroke_beta if dataset == "stroke" else 1.0
    threshold = choose_threshold(y_cal, cal_probability, beta=threshold_beta)
    test_prob = pipeline.predict_proba(X_test)[:, 1]
    test_logit = np.log(np.clip(test_prob, 1e-6, 1 - 1e-6) / np.clip(1 - test_prob, 1e-6, 1))
    test_probability = platt.predict_proba(test_logit.reshape(-1, 1))[:, 1]
    subgroup = X_test["gender"] if dataset == "stroke" else X_test["sex"]
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "source_sha256": file_sha256(source_path), "provenance": provenance, "selection_metric": scoring, "threshold_selection": f"F-beta, beta={threshold_beta}", "best_params": grid.best_params_, "calibration": "Platt scaling on a held-out calibration split", "calibration_brier": round(float(brier_score_loss(y_cal, cal_probability)), 4), "test_metrics": metric_report(y_test, test_probability, threshold, subgroup)}
    export_model(destination, pipeline, platt, list(X.columns), dataset, threshold, metadata, X_fit)
    print(f"{dataset}: wrote {destination}")
    print(json.dumps(metadata["test_metrics"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heart-csv", type=Path, required=True)
    parser.add_argument("--stroke-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models-v2"))
    parser.add_argument("--stroke-beta", type=float, default=2.0, help="Threshold F-beta weight for stroke: >1 favors recall; 1 balances recall and precision.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heart_X, heart_y, heart_info = prepare_heart(args.heart_csv)
    stroke_X, stroke_y, stroke_info = prepare_stroke(args.stroke_csv)
    train("heart", heart_X, heart_y, heart_info, args.heart_csv, args.output_dir / "heart_model.json")
    train("stroke", stroke_X, stroke_y, stroke_info, args.stroke_csv, args.output_dir / "stroke_prediction.json", args.stroke_beta)


if __name__ == "__main__":
    main()
