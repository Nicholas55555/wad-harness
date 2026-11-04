# wad-harness

Evaluation harness for **Weather Adaptive Autonomous Driving**: can a weather-conditional
confidence threshold recover the recall a detector loses in adverse weather, at a bounded
false-positive cost, without retraining the detector?

A stock COCO YOLOv8s runs once over DAWN; every threshold, weather bin and calibration split is
then computed from the cached detections. Thresholds are chosen per weather bin with conformal
risk control (Angelopoulos et al., ICLR 2024; the first step of Andéol et al.'s SeqCRC, 2025) so
that the expected per-image miss fraction stays under a budget α. That is run 1, the project's
baseline; run 2 moved to YOLOv8x (primary at 1280 px) and added BDD100K. The detector is never
trained or fine-tuned: the models that are trained (the per-detection dial's gradient-boosted trees,
the O3 hazard and visibility estimators, the exploratory per-image threshold networks) read only the
detector's outputs or the pixels.

**Method (CAI 6930 Phase 3): the calibrated trade-off dial.** One setting λ, the false alarms
accepted per avoided miss, picks the expected-cost-best rule: one confidence threshold, or a
per-detection cut on a calibrated chance that each box is real. See the section "Threshold: the
dial" below, the write-up [`Docs/08_method_lambda_dial.md`](Docs/08_method_lambda_dial.md), the
Phase 3 kit [`Docs/10_phase3_methodology_kit.md`](Docs/10_phase3_methodology_kit.md) and the index
of all project docs, [`Docs/README.md`](Docs/README.md).

**Experiments (CAI 6930 Phase 4).** The dial against the Phase-1 baseline on the same test data:
- **The pre-registered comparison:** CRC over the dial λ vs per-weather CRC over the score.
  - Pre-registration: [`PREREG_crc_lambda.md`](PREREG_crc_lambda.md).
  - Results: [`results/crc_lambda/crc_lambda.md`](results/crc_lambda/crc_lambda.md).
- **Ablations of the per-detection model:**
  - feature groups and training-set size (a, c):
    [`results/ablation/dial_ablation.md`](results/ablation/dial_ablation.md);
  - per-condition recalibration (b):
    [`results/ablation/dial_recalibration.md`](results/ablation/dial_recalibration.md);
  - other detectors and resolutions (d), YOLOv8x @640 and YOLOv8s @640 / @1280:
    [`results/ablation/detector_ablation.md`](results/ablation/detector_ablation.md).
- **The paper's tables and figures:** made by `python -m wad.phase4_paper` into `Docs/latex/phase4/`.
- **Write-up:** [`Docs/14_phase4_experiments.md`](Docs/14_phase4_experiments.md).
- **Trained components and run logs:** [`checkpoints/`](checkpoints/) and [`logs/`](logs/).

```bash
.venv/Scripts/python.exe -m wad.crc_lambda           # CRC over lambda vs the per-weather baseline, 200 splits (CPU)
.venv/Scripts/python.exe -m wad.dial_ablation        # refits of the per-detection model (CPU, ~45 min)
.venv/Scripts/python.exe -m wad.dial_recalibration   # ablation (b): per-condition isotonic maps, 200 splits (CPU, ~15 min)
bash scripts/run_detector_ablation.sh                # ablation (d): detection + per-box passes (GPU), fits, report
.venv/Scripts/python.exe -m wad.phase4_paper         # LaTeX tables and figures of the Experiments section
```

**Results:**

- Run 1 (O1/O2 on DAWN, YOLOv8s, CPU): [`results/results.md`](results/results.md), figures in
  `results/figures/`.
- Run 2 (O2: YOLOv8x, grouped vehicle loss, clean labels, confirmed tier) and the clear-weather
  control on BDD100K: [`results/run2/run2.md`](results/run2/run2.md), figures in
  `results/run2/figures/`. Pre-registration: [`PREREG_run2.md`](PREREG_run2.md).
- O3 (learned hazard estimator): [`results/o3/o3.md`](results/o3/o3.md). Pre-registration:
  [`PREREG_o3.md`](PREREG_o3.md).
- O3 version C (trained visibility estimator): v2 [`results/vis2/vis2.md`](results/vis2/vis2.md)
  (pre-registered) and v3 [`results/vis3/vis3.md`](results/vis3/vis3.md) (night relighting, not
  blind). Pre-registration: [`PREREG_visibility.md`](PREREG_visibility.md). Per-condition
  statistics and recoverable recall: `results/run2/stats_*.csv`, `recoverable_by_condition.csv`.
- The method, the λ dial (one threshold, per detection, Fβ units): `results/threshold/`
  (`one_threshold_*.csv`, `per_detection_yolo_*.csv`, `fbeta_*.csv`, `label_free_*.csv`); the
  decisions are the "DECISION" sections at the end of `NOTES.md`.
- Decisions and data surprises: [`NOTES.md`](NOTES.md).

## Install

Python 3.10+ (3.11.15 was used). With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

(On Linux/macOS the interpreter is `.venv/bin/python`. With plain pip: create a venv, install
the two torch wheels from the CPU index as above, then `pip install -r requirements.txt`.)
The CPU build is enough for run 1: detection takes about two minutes.

**GPU (run 2, BDD100K, O3).** This machine has an NVIDIA RTX 3050 Ti Laptop GPU (4 GB) with
driver 566.07, which supports CUDA 12.7; no driver change was needed. Install the CUDA 12.6
build of the same torch release (about 2.5 GB) in place of the CPU one:

```bash
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cu126 --reinstall-package torch --reinstall-package torchvision torch==2.14.1+cu126 torchvision==0.29.1+cu126
```

`detector.device` in each config chooses the device (`cpu`, `cuda:0` or `auto`). Run 1 is pinned
to `cpu`, and the CUDA build reproduces its CPU detections bit for bit. The cu130 wheels need a
newer driver (580+); cu126 is the right build for driver 566.

## Data

DAWN (Kenk & Hassaballah 2020, Mendeley Data, CC BY-NC 3.0) goes in `data/DAWN/`, one folder per
weather class exactly as the zips unpack (`data/DAWN/Fog/foggy-001.jpg`,
`data/DAWN/Fog/Fog_PASCAL_VOC/foggy-001.xml`, ...). Either download the four zips from
<https://data.mendeley.com/datasets/766ygrbt8y/3> and unzip them there, or:

```bash
.venv/Scripts/python.exe scripts/get_dawn.py
```

which fetches the 139 MB, checks each zip's SHA-256 and unpacks it. The YOLOv8s weights
(`weights/yolov8s.pt`, 22 MB) download on first use.

## Run

Everything, from the raw data (phases whose outputs exist are skipped; `--force` reruns them):

```bash
.venv/Scripts/python.exe scripts/run_all.py
```

Everything downstream of detection, from the cache (this is the reproducibility check: two runs
give byte-identical CSVs):

```bash
.venv/Scripts/python.exe scripts/run_all.py --from-cache
```

The 20-image smoke test (writes only to `cache/smoke/` and `results/smoke/`):

```bash
.venv/Scripts/python.exe scripts/run_all.py --smoke
```

Tests (154, all passing on 2026-10-29 in about 2 minutes):

```bash
.venv/Scripts/python.exe -m pytest tests
```

Each phase also runs on its own: `python -m wad.data`, `wad.detect`, `wad.match`, `wad.curves`,
`wad.crc`, `wad.hazard`, `wad.hazard_crc`, `wad.extras`, `wad.figures`, `wad.report` (each takes
`--smoke`). The detector-resolution sensitivity run is separate:
`python -m wad.sensitivity` (YOLOv8s at imgsz 1280, about 7 minutes on CPU).

## Run 2, the clear-weather control and O3

```bash
.venv/Scripts/python.exe scripts/get_bdd100k.py
```

fetches the BDD100K labels (190 MB) and the 10,000 val images (about 0.6 GB) from the official
host, reading only those members out of the 5.7 GB image archive (range requests, CRC-checked).

```bash
.venv/Scripts/python.exe -m wad.data --config config_bdd.yaml
```

```bash
.venv/Scripts/python.exe -m wad.run2 detect
```

```bash
.venv/Scripts/python.exe -m wad.run2 analyse
```

```bash
.venv/Scripts/python.exe -m wad.run2 report
```

`detect` runs YOLOv8x at 640 and 1280 on DAWN and YOLOv8s at 640 / YOLOv8x at 1280 on BDD100K
(GPU, about 35 minutes in all); `analyse` runs every detector × loss × label analysis from the
cache, each with the pre-registered 200-split rerun (about 55 minutes on this laptop: ~1.5 min per
DAWN analysis, ~12 min per BDD100K analysis); `report` writes `results/run2/run2.md` and the
figures. O3 timings: `select` ~3 min (reads 70k label files), the 6,048-image download ~5 min,
`train` ~17 min on the GPU (8 epochs), `predict` ~3 min, `o3_eval` ~1 min.

O3 (learned hazard estimator, GPU):

```bash
.venv/Scripts/python.exe -m wad.hazard_cnn select
```

```bash
.venv/Scripts/python.exe scripts/get_bdd100k.py --no-val --train-list cache/o3/train_subset_names.txt
```

```bash
.venv/Scripts/python.exe -m wad.hazard_cnn train
```

```bash
.venv/Scripts/python.exe -m wad.hazard_cnn predict
```

```bash
.venv/Scripts/python.exe -m wad.o3_eval
```

## Per-condition statistics and visibility (2026-10-11/12)

Analyses on the run 2 caches (CPU, a few minutes each):

```bash
.venv/Scripts/python.exe -m wad.recoverable
```

```bash
.venv/Scripts/python.exe -m wad.visibility
```

```bash
.venv/Scripts/python.exe -m wad.stats_tables
```

`recoverable` writes recall recoverable by lowering the threshold, per weather × time of day
(`results/run2/recoverable_by_condition.csv`, fig9); `visibility` computes the exploratory pixel
visibility score v1 (contrast + fine detail); `stats_tables` writes recall, precision, F1, FPPI,
AP and per-class numbers per condition and per policy (`results/run2/stats_*.csv`).

O3 version C, the trained visibility estimator (`PREREG_visibility.md`, GPU). `select` writes the
frame lists (3,000 extra clear-daytime train frames, seed 1; the 400 version-3 development night
frames, seed 2; it needs the O3 `hazard_cnn select` step first), then the images are fetched
(~210 MB and ~17 MB):

```bash
.venv/Scripts/python.exe -m wad.vis_estimator select
```

```bash
.venv/Scripts/python.exe scripts/get_bdd100k.py --no-val --train-list cache/vis2/extra_clear_names.txt
```

```bash
.venv/Scripts/python.exe scripts/get_bdd100k.py --no-val --train-list cache/vis3/dev_night_names.txt
```

```bash
.venv/Scripts/python.exe -m wad.vis_estimator prepare
```

```bash
.venv/Scripts/python.exe -m wad.vis_estimator train
```

```bash
.venv/Scripts/python.exe -m wad.vis_estimator predict
```

```bash
.venv/Scripts/python.exe -m wad.vis_eval
```

`prepare` caches 4,500 base frames with metric depth (Depth Anything V2 Metric Outdoor Small) and
vehicle boxes (~10 min); `train` synthesises fog/haze, rain, wheel spray and darkness on the fly
and trains a ResNet-18 for 20 epochs (~30 min on the RTX 3050 Ti); `predict` scores BDD100K val
and DAWN (~3 min); `vis_eval` writes `results/vis2/` (criteria, correlations, policies, bins,
`vis2.md`, fig10, fig11; ~10 min).

Version 3 (night relighting, designed after v2 read real night as daylight; see the 'Version 3'
section of `PREREG_visibility.md`) uses 400 BDD100K *train* night frames as a development set
(`cache/vis3/dev_night.csv`, ~17 MB of images) and writes `weights/visibility_v3_resnet18.pt`, `cache/vis3/`,
`results/vis3/`:

```bash
.venv/Scripts/python.exe -m wad.vis_estimator train --version 3
```

```bash
.venv/Scripts/python.exe -m wad.vis_estimator dev_check --version 3
```

```bash
.venv/Scripts/python.exe -m wad.vis_estimator predict --version 3
```

```bash
.venv/Scripts/python.exe -m wad.vis_eval --version 3
```

Diagnostics and exploratory follow-ups (labelled as such in the reports): `dev_check --version 2`
(v2's light level on the same train frames, run after the v2 evaluation as v3's baseline),
`python -m wad.vis_dev gap` (the night diagnosis that motivated v3; it views 24 BDD100K val images,
no detection results), `python -m wad.vis_dev luma` (the final v3 rendering check against the
development frames), `python -m wad.vis_combo` (v1 + v3 rank mean; within-day/within-night
correlations).

```bash
.venv/Scripts/python.exe -m wad.vis_estimator dev_check --version 2
```

```bash
.venv/Scripts/python.exe -m wad.vis_dev gap
```

```bash
.venv/Scripts/python.exe -m wad.vis_dev luma
```

```bash
.venv/Scripts/python.exe -m wad.vis_combo
```

Results: `results/vis2/vis2.md` (v2, pre-registered), `results/vis3/vis3.md` (v3, not blind),
`results/vis3/exploratory_combo_*.csv`; the narrative and verdicts are at the end of `NOTES.md`.

## Threshold: the dial, one threshold or per detection (the method, 2026-10-13/14)

Instead of a fixed, arbitrary confidence cut-off (YOLO's default 0.25), a dial λ: the false alarms
you accept to avoid one missed object. Each setting is the expected-cost-best choice for
λ·misses + false alarms, fitted on BDD100K train and only graded on BDD100K val and DAWN, and it
can be applied two ways:

- **one threshold**: one confidence threshold per λ, the same for every box (`wad/one_threshold.py`);
- **per detection**: keep a box iff a model's chance that it is real is ≥ 1/(1+λ); the model reads
  only YOLOv8x's outputs (score, class, box, rank, overlaps, crowding, and since 2026-10-15 YOLOv8x's
  own per-box head outputs: class probabilities, box-side uncertainty, neck features) and its chances
  are calibrated, which is what makes that cut right (`wad/per_detection.py`).

Presets are λ values (precision first 1, balanced 10, recall first 30, catch everything 100). One
entry point for both (`wad/dial.py`): `dial.keep(detections, λ, mode="one" | "per detection",
sizes=...)`, with detections from YOLOv8x as it was run here (imgsz 1280, confidence floor 0.001, NMS
IoU 0.5, at most 300 boxes, six classes) and sizes the image width and height by image_id. Decision and results: the "DECISION" sections at the end of `NOTES.md`.

Training detections (shared by all threshold work): `select` lists 21,831 BDD100K train frames
(12,383 to fetch, ~0.85 GB; if the script's label check is slow, the images can be fetched with its
`fetch()` directly); `scan` ~10 min; `detect` ~60 min on the GPU; `labels` matches them
(`cache/thresh/v8x_1280__group/`):

```bash
.venv/Scripts/python.exe -m wad.thresh_net select
```

```bash
.venv/Scripts/python.exe scripts/get_bdd100k.py --no-val --train-list cache/thresh/fetch_names.txt
```

```bash
.venv/Scripts/python.exe -m wad.thresh_net scan
```

```bash
.venv/Scripts/python.exe -m wad.thresh_net detect
```

```bash
.venv/Scripts/python.exe -m wad.thresh_net labels
```

Fit the dial's thresholds, check their stability (image bootstrap, 5-fold) and grade them against no
tuning (0.25) and per-weather thresholds (`results/threshold/one_threshold_*.csv`, fig20; ~1–2 min):

```bash
.venv/Scripts/python.exe -m wad.one_threshold
```

False alarms estimated without the image's labels (score → chance real learned on BDD100K train;
val labels used only to check; `results/threshold/label_free_*.csv`):

```bash
.venv/Scripts/python.exe -m wad.label_free_fp
```

The per-detection option. YOLOv8x's own per-box features first (re-runs YOLOv8x on all 30.9k images
and keeps, for each cached detection, the head outputs at its anchor; ~85 min GPU, resumable, ~4.3 GB
in `cache/thresh/yolo_box/`):

```bash
.venv/Scripts/python.exe -m wad.yolo_box_features
```

Then the models with their statistical backing: calibration on BDD100K val / DAWN per condition and
around each preset's cut, 5-fold stability inside train, grading against one threshold and no tuning
with 95% bootstrap intervals. Model 1 (context features; `results/threshold/per_detection_*.csv`,
fig21, ~5 min CPU) and model 2 (+ per-box features, the adopted one; `per_detection_yolo_*.csv` incl.
the pre-registered comparison with model 1, fig23, ~11 min CPU):

```bash
.venv/Scripts/python.exe -m wad.per_detection --features context
```

```bash
.venv/Scripts/python.exe -m wad.per_detection --features yolo
```

The same λ dial set in F-score units: β (how many times more recall counts than precision, as in Fβ)
is turned into its equivalent λ = (1+β²)/F* − 1, with F* the best Fβ of that option on BDD100K train
(per detection on 5-fold out-of-fold chances), and the λ dial is applied there. Table
`weights/fbeta_table.csv` (`dial.keep_f(detections, β, mode=...)`), stability, grading with 95%
intervals and an exploratory check of models trained for one β (`results/threshold/fbeta_*.csv`, fig22;
needs `one_threshold` and `per_detection` first):

```bash
.venv/Scripts/python.exe -m wad.fbeta_dial
```

Interactive dial page (`results/threshold/dial_demo/index.html`), both dials (λ or β) and both options
on example frames and condition-wide, with F1 and Fβ:

```bash
.venv/Scripts/python.exe -m wad.dial_demo
```

Background on the cached detections (CPU, ~7 min each): what lowering the threshold recovers and
costs per condition, and a graded stop / slow-down response from confidence
(`results/threshold/budget_*.csv`, `graded_response.csv`, `graded_response.md`, fig14, fig15):

```bash
.venv/Scripts/python.exe -m wad.threshold_budget
```

```bash
.venv/Scripts/python.exe -m wad.graded_response
```

## Threshold exploration (kept so every NOTES number reproduces)

Model comparisons behind the per-detection option (baseline on score and class, ablation without the
overlap features, softmax / noisy-OR box combinations, permutation importance;
`results/threshold/rescore_*.csv`):

```bash
.venv/Scripts/python.exe -m wad.detection_rescore
```

Per-image threshold models (`PREREG_threshold.md` versions 1–3): a network gives each image its
cheapest threshold for the dial. Version 1 reads pixels (ResNet-18; the input-frame caches take
~14 GB in `cache/thresh/`, moved out on 2026-10-13 and rebuilt automatically when missing; `train` ~30 min; `thresh_eval` writes `threshold.md`, criteria, policies,
fig13). The in-sample ceiling is `python -m wad.threshold_headroom` (fig12):

```bash
.venv/Scripts/python.exe -m wad.thresh_net train
```

```bash
.venv/Scripts/python.exe -m wad.thresh_net predict
```

```bash
.venv/Scripts/python.exe -m wad.thresh_eval
```

```bash
.venv/Scripts/python.exe -m wad.threshold_headroom
```

Version 2 reads the detector's confidence profile; version 3 reads YOLOv8x's own feature maps of
the image (extraction ~70-85 min on the GPU, resumable; ~0.7 GB). The full run (2026-10-14) uses the
structured head, `train --curves` and `check --curves` (`results/threshold/yolo_curves/`); without
`--curves` the head as first registered (`results/threshold/yolo/`); the pilot is `--pilot
--steps-matched --curves`:

```bash
.venv/Scripts/python.exe -m wad.thresh_dfl features
```

```bash
.venv/Scripts/python.exe -m wad.thresh_dfl train
```

```bash
.venv/Scripts/python.exe -m wad.thresh_dfl check
```

```bash
.venv/Scripts/python.exe -m wad.thresh_yolo extract
```

```bash
.venv/Scripts/python.exe -m wad.thresh_yolo train --curves
```

```bash
.venv/Scripts/python.exe -m wad.thresh_yolo check --curves
```

How much of the per-image oracle's edge is luck (`noise_floor.csv`), and the summed-confidence map
vs ordinary thresholding at region level (`summed_map_*.csv`, fig18):

```bash
.venv/Scripts/python.exe -m wad.threshold_noise
```

```bash
.venv/Scripts/python.exe -m wad.summed_map
```

Example figures of every box down to confidence ~0 (re-runs YOLOv8x on a few images; fig16, fig17;
`--low` only fig17; `--summed` fig19, after `summed_map`):

```bash
.venv/Scripts/python.exe -m wad.show_boxes
```

## Runtime on this machine

Windows 11, AMD Ryzen 7 4800H, CPU-only torch:

| command / step | time |
| :--- | ---: |
| `run_all.py --force` (everything from the raw images) | 5.7 min (341 s) |
| of which phase 1, data scan, hashes and pixel checks of duplicate pairs | 22 s |
| of which phase 2, detection (1,005 images, YOLOv8s, imgsz 640) + EXIF frame check | 2.5 min (0.13 s/image) |
| of which hazard features (pixels only) | 50 s |
| of which extras (bootstrap, 200-split rerun, alpha sweep, miss diagnosis) | 55 s |
| `run_all.py --from-cache` | 69 s |
| `run_all.py --smoke` (20 images) | 76 s |
| `python -m wad.sensitivity` (YOLOv8s at imgsz 1280, detection included) | ~7 min |
| `pytest tests` (91 tests at run 1; 167 now, about 2 min) | 3 s |

Per-phase timings of the last run are in `results/run_timing.json`. Detection is
deterministic: a forced rerun reproduced `detections.parquet` bit for bit, and two
`--from-cache` runs produce byte-identical CSVs, JSON and `results.md`.

## Layout

```
config.yaml          every fixed choice (fixed before any split)
wad/config.py        config loading, the tau grid
wad/imageio.py       the single image reader (EXIF orientation applied, CMYK/grey -> RGB)
wad/data.py          DAWN loader -> cache/gt.parquet, cache/images.parquet; duplicate checks
wad/detect.py        YOLOv8 once -> cache/detections.parquet (+ meta, sanity, EXIF check, overlays)
wad/match.py         greedy IoU matching -> cache/matches_gt.parquet, matches_pred.parquet
wad/curves.py        miss rate vs FPPI, max recall, LAMR, mAP, go/no-go
wad/crc.py           splits, conformal calibration, evaluation (phase 4)
wad/hazard.py        pixel-only hazard features and scalar h (imports no detection code)
wad/hazard_crc.py    phase 4 with predicted hazard bins (phase 6a)
wad/extras.py        bootstrap of the verdict, miss diagnosis, exploratory alpha sweep, robustness
wad/sensitivity.py   same weights at a different inference size
wad/figures.py       fig1-fig3 (fig4 in hazard_crc, fig5 in extras)
wad/report.py        results/results.md
wad/dial.py          the dial: λ (false alarms per avoided miss) -> keep mask, one threshold or per detection
wad/one_threshold.py the dial's one-threshold option, fitted on BDD100K train (+ weights/one_threshold_table.csv)
wad/per_detection.py the dial's per-detection option (model 1 weights/per_detection.joblib; adopted model 2 per_detection_yolo.joblib)
wad/fbeta_dial.py    the dial in F-score units: β -> equivalent λ fitted on train (+ weights/fbeta_table.csv)
wad/yolo_box_features.py YOLOv8x's own per-box features (head outputs at each box's anchor); Detector for new images
wad/dial_demo.py     the interactive dial page (template wad/assets/dial_demo.html)
wad/crc_lambda.py    Phase 4: CRC over the dial λ vs per-weather CRC over the score, same 200 splits (PREREG_crc_lambda.md)
wad/dial_ablation.py Phase 4: refits of the per-detection model (feature groups, training-set size)
wad/dial_recalibration.py  Phase 4 ablation (b): per-condition isotonic recalibration of the chance, 200 splits
wad/detector_ablation.py   Phase 4 ablation (d): the whole dial pipeline for YOLOv8x @640, YOLOv8s @640 and @1280
wad/dial_runtime.py  Phase 4: per-frame timing of the per-detection rule
wad/phase4_paper.py  Phase 4: the Experiments section's LaTeX tables and figures (Docs/latex/phase4/)
checkpoints/         the trained per-detection models and dial tables (copies of weights/), with a README
logs/                logs of the GPU passes (BDD100K train detection, per-box outputs, the detector ablation)
wad/run2.py, run2_analysis.py, run2_report.py   run 2 (O2) and the BDD100K clear-weather control (PREREG_run2.md)
wad/hazard_cnn.py, o3_eval.py                   O3 version B, a learned camera hazard estimator, and its evaluation (PREREG_o3.md)
wad/visibility.py                               exploratory pixel-only visibility score (v1)
wad/vis_estimator.py, vis_eval.py, vis_dev.py, vis_combo.py   O3 version C, the trained visibility estimator (PREREG_visibility.md)
wad/recoverable.py, stats_tables.py             recoverable recall and statistics per condition
wad/thresh_net.py, thresh_eval.py, thresh_dfl.py, thresh_yolo.py   learned per-image thresholds, versions 1-3 (PREREG_threshold.md)
wad/threshold_headroom.py, threshold_budget.py, threshold_noise.py, graded_response.py,
  detection_rescore.py, label_free_fp.py, summed_map.py, show_boxes.py   exploratory threshold analyses
scripts/run_all.py   one command, cache -> results
scripts/get_dawn.py  verified DAWN download
scripts/get_bdd100k.py  the parts of BDD100K this project uses, from the official host
tests/               pytest suite, 167 tests (matching, loss monotonicity, CRC rule, curves, hazard isolation, the dial, CRC over λ, ablations b and d)
```

`data/`, `cache/` and `weights/` are git-ignored, as the brief specifies. The cache is small
(about 3 MB of parquet); the plan's O6 deliverable ("repo with cached detector outputs") can
be met by committing `cache/*.parquet` and `cache/*.json` once the repo is published.

## Licence note

DAWN is CC BY-NC 3.0: cite Kenk & Hassaballah (2020), non-commercial use only. BDD100K (The
Regents of the University of California) may be used for educational, research and
not-for-profit purposes; cite Yu et al. (CVPR 2020). YOLOv8 weights are Ultralytics' (AGPL-3.0);
the ResNet-18 ImageNet weights are torchvision's.
