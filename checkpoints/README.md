# Checkpoints of the trained components

The detector is frozen. YOLOv8x's COCO weights come from Ultralytics:
- file `yolov8x.pt`, 136,890,692 bytes;
- SHA-256 `3df4ada6b4dad6d657868f2fdf7faecfb34dcfccf3a25c4b82079064718524c8`;
- placed in `weights/`, which git ignores.

Everything we fitted is small enough to keep here. All of it was fitted on BDD100K train only (21,831
frames).

| file | what | made by |
|---|---|---|
| `per_detection_yolo.joblib` | **model 2, the adopted per-detection model**: sklearn `HistGradientBoostingClassifier` (107 boosting iterations, 68 features) plus the PCA of the neck vectors, the column list and settings; it keeps the training and validation log-loss per iteration (`model.train_score_`, `model.validation_score_`) | `python -m wad.per_detection --features yolo` |
| `per_detection.joblib` | model 1, the 20 context features only (the ablation baseline) | `python -m wad.per_detection --features context` |
| `one_threshold_table.csv` | option A: the exact switch points λ → threshold (258 rows) | `python -m wad.one_threshold` |
| `fbeta_table.csv` | the F-score map β → λ_eq for both options | `python -m wad.fbeta_dial` |

**Using them:**
1. Copy the files into `weights/`, which is where the code looks.
2. Call `wad.dial.keep(...)` (doc 08 §8). For new images, call `wad.yolo_box_features.Detector()(image)`: it
   returns the detections and the per-box outputs model 2 needs.

**Run logs:**
- `results/threshold/per_detection_yolo.log`: training model 2.
- `results/ablation/dial_ablation.log`: the Phase 4 refits.
- `results/crc_lambda/crc_lambda.log`: the Phase 4 CRC comparison.
- `logs/`: the GPU passes (detection and per-box outputs).

The caches the logs refer to are rebuilt by the commands in `README.md`.
