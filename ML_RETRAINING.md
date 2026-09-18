# Improved model retraining

The deployed model files remain unchanged. When the source datasets are available, run:

```bash
python train_improved.py --heart-csv heart.csv --stroke-csv stroke_prediction.csv --output-dir models-v2
```

The default stroke threshold uses F2, favoring recall for screening. To choose a more balanced precision/recall threshold, use F1:

```bash
python train_improved.py --heart-csv heart.csv --stroke-csv stroke_prediction.csv --output-dir models-balanced --stroke-beta 1
```

Review the printed held-out metrics and the `training_metadata` inside each generated JSON before replacing either deployed model file. The script uses deduplication for heart records, leakage-safe preprocessing pipelines, cross-validated model selection, a held-out calibration split, threshold selection, subgroup reporting, and a data-derived OOD threshold.

`models-v2` is intentionally separate from the live artifacts. Copy a reviewed artifact to `heart_model.json` or `stroke_prediction.json`, restart the C++ server, and run an API smoke test only after deciding it is clinically appropriate. These models are still a prototype, not a diagnostic device.
