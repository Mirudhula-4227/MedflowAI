"""Train server-compatible Gradient Boosted Decision Tree candidates.

Uses only scikit-learn. Outputs are separate from deployed model files:
  python train_gbdt.py --heart-csv heart.csv --stroke-csv stroke_prediction.csv
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from train_improved import (RANDOM_STATE, choose_threshold, file_sha256, metric_report,
                            prepare_heart, prepare_stroke)


def logit(probability):
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def export(path, model, imputer, platt, features, dataset, threshold, metadata, X_fit):
    trees = []
    for stage in model.estimators_:
        tree = stage[0].tree_
        trees.append({
            "left": tree.children_left.tolist(), "right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(),
            "value": [float(item[0][0]) for item in tree.value],
        })
    base_probability = float(model.init_.class_prior_[1])
    calibration_scale = float(platt.coef_[0][0])
    calibration_bias = float(platt.intercept_[0])
    fit_array = imputer.transform(X_fit)
    ood_scale = np.std(fit_array, axis=0)
    ood_scale[ood_scale == 0] = 1.0
    payload = {
        "model": "gradient_boosted_trees", "dataset": dataset, "feature_names": list(features),
        "decision_threshold": threshold,
        # Fold Platt scaling directly into the raw GBDT score for the C++ sigmoid.
        "base_logit": calibration_scale * float(logit(base_probability)) + calibration_bias,
        "learning_rate": calibration_scale * float(model.learning_rate), "trees": trees,
        "ood": {"mean": np.mean(fit_array, axis=0).tolist(), "scale": ood_scale.tolist(),
                "threshold": float(np.quantile(np.linalg.norm((fit_array - np.mean(fit_array, axis=0)) / ood_scale, axis=1), .995))},
        "imputer": {"strategy": "median", "values": imputer.statistics_.tolist()},
        "training_metadata": metadata,
    }
    path.write_text(json.dumps(payload, indent=2))


def train(dataset, X, y, info, source, destination, stroke_beta=2.0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=RANDOM_STATE)
    X_fit, X_cal, y_fit, y_cal = train_test_split(X_train, y_train, test_size=.25, stratify=y_train, random_state=RANDOM_STATE)
    imputer = SimpleImputer(strategy="median").fit(X_fit)
    fit = imputer.transform(X_fit)
    cal = imputer.transform(X_cal)
    test = imputer.transform(X_test)
    grid = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [100, 200], "learning_rate": [.03, .05], "max_depth": [2, 3], "subsample": [.8]},
        scoring="average_precision" if dataset == "stroke" else "roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), n_jobs=1,
    )
    weights = None
    if dataset == "stroke":
        ratio = (y_fit == 0).sum() / (y_fit == 1).sum()
        weights = np.where(np.asarray(y_fit) == 1, ratio, 1.0)
    grid.fit(fit, y_fit, sample_weight=weights)
    model = grid.best_estimator_
    calibration = LogisticRegression(C=1e6, solver="lbfgs").fit(logit(model.predict_proba(cal)[:, 1]).reshape(-1, 1), y_cal)
    cal_probability = calibration.predict_proba(logit(model.predict_proba(cal)[:, 1]).reshape(-1, 1))[:, 1]
    beta = stroke_beta if dataset == "stroke" else 1.0
    threshold = choose_threshold(y_cal, cal_probability, beta)
    test_probability = calibration.predict_proba(logit(model.predict_proba(test)[:, 1]).reshape(-1, 1))[:, 1]
    subgroup = X_test["gender"] if dataset == "stroke" else X_test["sex"]
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "algorithm": "GradientBoostingClassifier", "source_sha256": file_sha256(source), "provenance": info, "best_params": grid.best_params_, "calibration": "Platt scaling on held-out calibration split", "threshold_selection": f"F-beta, beta={beta}", "test_metrics": metric_report(y_test, test_probability, threshold, subgroup)}
    export(destination, model, imputer, calibration, X.columns, dataset, threshold, metadata, X_fit)
    print(f"{dataset}: wrote {destination}\n{json.dumps(metadata['test_metrics'], indent=2)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heart-csv", type=Path, required=True)
    parser.add_argument("--stroke-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models-gbdt"))
    parser.add_argument("--stroke-beta", type=float, default=1.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heart_X, heart_y, heart_info = prepare_heart(args.heart_csv)
    stroke_X, stroke_y, stroke_info = prepare_stroke(args.stroke_csv)
    train("heart", heart_X, heart_y, heart_info, args.heart_csv, args.output_dir / "heart_model.json")
    train("stroke", stroke_X, stroke_y, stroke_info, args.stroke_csv, args.output_dir / "stroke_prediction.json", args.stroke_beta)


if __name__ == "__main__":
    main()
