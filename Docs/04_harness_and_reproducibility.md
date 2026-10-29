# 04 — The harness and how to reproduce everything

*Sources: `README.md`, `NOTES.md` (environment, caches, GPU), `requirements.txt`, `results/run_timing.json`,
the brief `../Claude Code brief — first results for Weather Adaptive Autonomous Driving (O1, O2).md`, the
four `PREREG_*.md` files, `08_method_lambda_dial.md` §10. The test count was re-collected on 2026-10-22
(`pytest --collect-only`, nothing written).*

> **In short**
> - `wad-harness` is one Python package (`wad/`, 44 modules) plus scripts. It runs the detector **once**
>   at a 0.001 confidence floor and caches every box; every threshold, calibration and dial result is then
>   recomputed from the cache.
> - `scripts/run_all.py --from-cache` reproduces every run 1 number in about 70 s, and two runs give
>   byte-identical CSVs. Runs 2, O3, visibility and the dial each have their own one-line entry points.
> - Four dated pre-registrations fix choices before results are seen. Every departure is written down.
> - 154 tests (the README's "91" is stale). The work ran on one laptop with an RTX 3050 Ti (4 GB). Phase 4
>   can move the GPU-heavy steps to an NRP A100.

---

## 1. The brief's rules (verbatim) and how the repo applies them

The brief (dated Oct 1, 2026) fixed these rules before any code existed:

- *"**No training, no fine-tuning.** The detector is used with released COCO weights. Nothing in this repo
  updates a model."*
- *"**Detect once, cache forever.** Detections are written to disk at a near-zero confidence floor; every
  downstream step reads the cache. Never re-run the detector to evaluate a different threshold."*
- *"**Nothing is chosen on the test half.** Budgets α, the threshold grid, and the default thresholds are
  fixed in a config file before any split is made. Thresholds are fitted on calibration halves only."*
- *"**Splits are by image and stratified by weather.** Calibration and test halves never share an image.
  Check for duplicate or near-duplicate images (file hash and a perceptual hash) and report any found
  before splitting."*
- *"**The loss is image-level and monotone.** ... This is non-decreasing in τ by construction; assert it in
  a test."*
- *"**Matching is IoU at 0.5, class-aware, greedy by score.**"*
- *"**Do not use detector outputs anywhere except as detector outputs.** No hazard or weather signal may be
  derived from scores or boxes (relevant in phase 6)."*
- *"**Report what the data actually is.** If class names, annotation format, image counts, or folder names
  differ from this brief, the real data wins; record the difference in `NOTES.md` and proceed."*
- *"**Every number in `results.md` must be reproducible by one command** from the cache, with fixed seeds."*
- *"**Ask before any download larger than 2 GB or any step that would run longer than 30 minutes.**"*
- *"**Smoke test first.** Every phase runs on 20 images before it runs on all of them."*

**How they are applied.**
- `config.yaml` opens with *"nothing in this file is tuned on a test half"*.
- `wad/hazard.py` imports no detection code, and a test checks this.
- Run 1 used the CPU torch build because the CUDA wheel is larger than 2 GB. The 2.5 GB cu126 wheel was
  installed later for run 2, and no driver change was needed.
- Every data difference from the brief is in `NOTES.md`. For example, DAWN has 1,027/1,005 images, not
  "1,020", and the EXIF mirror tags are real.

**Documented departures (all by the user's decision, all written down).**
1. The brief asked to stop after phase 3 for a go-ahead. Run 1 was asked to finish all phases in one pass
   (`NOTES.md`).
2. "No training" was relaxed for objectives the brief did not cover. Three things are trained: the O3
   classifier (ResNet-18), the visibility estimators (ResNet-18), and the dial's per-detection models
   (gradient boosting). **The detector itself is never trained or fine-tuned.**
3. *"Departure from the brief, by the user's decision (2026-10-13)"* (`PREREG_threshold.md`, version 2):
   the per-image threshold model reads the detector's own outputs. The rule against deriving a hazard signal
   from detector outputs was written for the weather estimator. The dial's per-detection models also read
   detector outputs; that is the point of the method.
4. Several later steps ran longer than 30 minutes: run 2 `analyse` ~55 min, `detect` ~35 min, and the dial's
   per-box feature pass ~85 min. They were run at the user's direction. The files read here do not record a
   per-step approval.

## 2. One command per stage

All commands run from `wad-harness/` with `.venv/Scripts/python.exe` (Windows, uv-managed Python 3.11.15).

| stage | command | what it does |
|---|---|---|
| Run 1, everything | `scripts/run_all.py` (`--force` reruns phases) | DAWN scan → YOLOv8s detection → matching → curves → CRC → hazard → extras → figures → `results/results.md` |
| **Run 1 reproducibility check** | `scripts/run_all.py --from-cache` | recomputes every run 1 number from the cache (byte-identical CSVs) |
| Smoke test | `scripts/run_all.py --smoke` | 20 images, writes only to `cache/smoke/`, `results/smoke/` |
| Single phases | `python -m wad.data / detect / match / curves / crc / hazard / hazard_crc / extras / figures / report` | each takes `--smoke` |
| Resolution sensitivity | `python -m wad.sensitivity` | YOLOv8s at imgsz 1280 (see the bug note in §7) |
| Data download | `scripts/get_dawn.py`; `scripts/get_bdd100k.py` | DAWN 139 MB (SHA-256 checked); BDD100K labels 190 MB + val ~0.6 GB (range requests, CRC-checked) |
| Run 2 + BDD100K control | `python -m wad.data --config config_bdd.yaml`, then `python -m wad.run2 detect`, `analyse`, `report` | `results/run2/run2.md`, fig6–fig8 |
| Per-condition stats | `python -m wad.recoverable`, `wad.visibility`, `wad.stats_tables` | fig9, v1 score, stats CSVs |
| O3 version B | `python -m wad.hazard_cnn select`, fetch, `train`, `predict`; `python -m wad.o3_eval` | `results/o3/o3.md` |
| Visibility v2 | `python -m wad.vis_estimator select / prepare / train / predict`; `python -m wad.vis_eval` | `results/vis2/vis2.md`, fig10, fig11 |
| Visibility v3 | `python -m wad.vis_estimator train / dev_check / predict --version 3`; `python -m wad.vis_eval --version 3` | `results/vis3/vis3.md` |
| Dial training detections | `python -m wad.thresh_net select`, fetch, `scan`, `detect`, `labels` | 21,831 train frames → `cache/thresh/v8x_1280__group/` |
| **The dial (the method)** | `python -m wad.one_threshold`; `wad.yolo_box_features`; `wad.per_detection --features context`; `wad.per_detection --features yolo`; `wad.fbeta_dial`; `wad.label_free_fp`; `wad.dial_demo` | see `08_method_lambda_dial.md` §10 |
| Tests | `python -m pytest tests` | 154 tests |

## 3. Caches

- **Fingerprinted detection cache.** Every shard index records a fingerprint of the detector configuration:
  settings, classes, EXIF policy, weights SHA-256 and device. Resuming under a different fingerprint is
  refused, and `--force` clears the cache first.
- **One image per forward pass.** With batching, Ultralytics letterboxes images to a shared square, so a
  box's score would depend on its batch mates. Shards of 16 images are written atomically: the index is
  written last.
- **Files.**
  - `cache/detections.parquet` (run 1, 57,570 boxes), `matches_gt.parquet`, `matches_pred.parquet`,
    `images.parquet`, `gt.parquet`, `duplicates.csv`, `exif_check.json`, `data_report.json`.
  - `cache/det/{v8x_640, v8x_1280}`, `cache/bdd/det/{v8s_640, v8x_1280}`.
  - `cache/o3`, `cache/vis2` (2.4 GB), `cache/vis3`, `cache/sensitivity`.
  - `cache/thresh/` (dial training detections; per-box features `cache/thresh/yolo_box/*.npy`, 4.3 GB).
  - `data/`, `cache/` and `weights/` are git-ignored. Run 1's cache is about 3 MB of parquet.
- **Why one cache serves every threshold.** Greedy NMS lets only higher-scoring boxes suppress lower ones,
  and greedy matching runs in score order. So filtering the 0.001-floor run at τ is identical to re-running
  the detector at τ. This was checked on 50 BDD100K images at τ = 0.05 / 0.25 / 0.5 and gave identical boxes
  (`tests/test_threshold_equivalence.py`). All 200 thresholds × 8,090 images take 0.2 s instead of about 67
  GPU-hours. The caveat is the 300-box cap.

## 4. Determinism and checks

- A forced rerun reproduced `detections.parquet` bit for bit. Two `--from-cache` runs produce byte-identical
  CSVs, JSON and `results.md`.
- The CUDA build reproduces run 1's CPU detections bit for bit (12 of 12 images,
  `results/run2/gpu_cpu_check.json`). Run 1's config pins `detector.device: cpu`, which is part of the
  fingerprint.
- *"GPU training remains non-bit-reproducible (cuDNN), as for every model in this repository"*
  (`PREREG_visibility.md`). This applies to the O3 and visibility networks. The gradient-boosting dial
  models are seeded (`random_state 0`).
- **Independent reviews.**
  - Run 2: an adversarial review confirmed 25 findings, all fixed.
  - Visibility v2: a review upheld 13 findings, grouped as amendment 3, before any real-image prediction.
  - Visibility v3: an erratum after the evaluation.
  - Per-detection model 2: an independent verification re-extracted 105 images' features on the GPU and
    recomputed every headline number.

## 5. Hardware, software, pins

- **Machine.** Windows 11 laptop, AMD Ryzen 7 4800H (8 cores, 16 threads), NVIDIA RTX 3050 Ti Laptop GPU
  (4 GB, compute 8.6), driver 566.07. *RAM: `NOTES.md` says 31 GB; the Phase 1 reproducibility write-up says
  16 GB.*
- **Python.** 3.11.15 managed by `uv`. torch 2.14.1 (CPU build for run 1; cu126 from run 2 on).
- **Pins** (`requirements.txt`): torch 2.14.1, torchvision 0.29.1, ultralytics 8.4.171, opencv-python
  5.0.0.93, numpy 2.4.6, pandas 3.0.6, pyarrow 25.0.1, matplotlib 3.11.2, pillow 12.3.0, imagehash 4.3.2,
  scipy 1.17.1, scikit-learn 1.9.1, pytest 9.1.1, pyyaml 6.0.3.
- **Detector weights.** `yolov8s.pt` (SHA-256 `1f47a78bf100391c…`) and `yolov8x.pt`, both COCO, AGPL-3.0.
- **Licences.** DAWN CC BY-NC 3.0. BDD100K for educational, research and not-for-profit use.

## 6. Runtimes measured on this laptop

| step | time | device |
|---|---|---|
| `run_all.py --force` (run 1 from raw images) | 5.7 min (341 s) | CPU |
| — detection, 1,005 images, YOLOv8s @640 | 2.5 min (0.128 s/image) | CPU |
| `run_all.py --from-cache` | 69 s (last recorded run: 65.8 s) | CPU |
| `run_all.py --smoke` | 76 s | CPU |
| `wad.sensitivity` (YOLOv8s @1280) | ~7 min (0.38 s/image) | CPU |
| Run 2 `detect` (all detectors, DAWN + BDD100K val) | ~35 min | GPU |
| — YOLOv8x @1280 | 0.152 s/image (BDD100K), 0.165 s/image (DAWN); 0.6 GB VRAM | GPU |
| — YOLOv8x @640 / YOLOv8s @640 | 0.075 / 0.033 s/image | GPU |
| Run 2 `analyse` (every variant, 200 splits) | ~55 min (~1.5 min per DAWN analysis, ~12 min per BDD100K analysis) | CPU |
| O3: select / download / train (8 epochs) / predict / eval | ~3 / ~5 / ~17 / ~3 / ~1 min | GPU for train |
| Visibility: prepare / train (20 epochs) / predict / eval | ~10 / ~30 / ~3 / ~10 min | GPU |
| Dial training detections, `thresh_net detect` (21,831 frames) | ~60 min | GPU |
| `wad.yolo_box_features` (30.9k images: train + val + DAWN) | ~85 min, resumable, 4.3 GB | GPU |
| `wad.one_threshold` | ~1–2 min | CPU |
| `wad.per_detection --features context` / `--features yolo` | ~5 / ~11 min | CPU |
| `wad.fbeta_dial` (incl. the dedicated-model check) | ~30 min | CPU |
| `wad.dial_demo` | ~2–10 min | CPU |
| `pytest tests` | 3 s for the original 91 tests. The full 154-test suite takes longer: collection alone took 53 s on 2026-10-22 under load. | CPU |

## 7. Tests and known report bugs

**Tests.** 154 tests are collected across 13 files: `test_budget`, `test_crc`, `test_curves`, `test_dial`,
`test_fbeta`, `test_hazard`, `test_loss`, `test_match`, `test_one_threshold`, `test_run2`, `test_thresh`,
`test_threshold_equivalence` and `test_vis_estimator`. They cover matching, loss monotonicity, the CRC rule,
curves, hazard isolation, the dial and the Fβ map. The README still says 91 tests, and
`08_method_lambda_dial.md` says 152; both are stale.

**Known bugs in generated reports.** Do not cite these.
1. **`results/results.md` §6.3, the YOLOv8s @1280 sensitivity table, is invalid.** A `--from-cache` recheck
   on 2026-10-08 recomputed the 1280 analysis against the **640** detections. Every 640 → 1280 value is
   identical, it reports "0 boxes gained / 0 lost", and "median side nan px". The likely cause, from the
   code: `wad/config.py` sets `paths.detections = paths.cache` at load time, while `wad/sensitivity.py`
   overrides only `paths.cache` and `paths.results`. **Use the valid 2026-10-01 numbers in `NOTES.md`:**
   - pooled max recall 83.1% → 81.0%, with 504 boxes gained and 662 lost;
   - smallest certifiable α fog 0.229 → 0.309, pooled 0.175 → 0.239;
   - α = 0.2 costs 50–71 FPPI;
   - GO at 1280, with person's worst bin now fog (margin 17.4).
2. **`results/run2/run2.md` "Go/no-go per variant"** repeats the operating-points table instead of the
   verdicts. Its header has 4 named columns but 7 alignment cells. The real verdicts are in
   `results/run2/dawn_variants.csv` and each variant's `go_no_go.json`; they are reproduced in
   `05_results_crc_runs.md` §2.
3. `run2.md` section numbering skips 6 (5 → 7).
4. The README layout list omits run2, hazard_cnn, vis_*, recoverable, stats_tables and `get_bdd100k.py`.
5. Small `NOTES.md` inconsistencies:
   - "RTX 3050 Laptop" vs "RTX 3050 Ti".
   - "per-bin calibration restores night at 42–124 FPPI": these are older figures; `run2.md` has 59 / 117 / 43.
   - "m(b) 0.24–0.30": `run2.md` has 0.232–0.292.
   - "smallest α 0.15–0.18 at night": `run2.md` has 0.143–0.181.
   - "8,091 images ... 97,800 boxes": pre-cleaning; the final count is 8,090 / 97,745.

## 8. Pre-registrations (what was fixed before results)

| file | written | covers | later additions |
|---|---|---|---|
| `PREREG_run2.md` | 2026-10-08, after run 1, before any run 2 or BDD100K number | detectors, L-class/L-group losses, clean labels, budgets, splits, policies, tiers, BDD100K questions Q1–Q4, cross-dataset | erratum (2026-10-08, after results): 23 night images unbinned, 37 zero-area boxes, cross-weather duplicate, per-class AP fix, 200-split rerun added, exploratory items labelled |
| `PREREG_o3.md` | 2026-10-08, before the training subset was downloaded | O3 version B: data, model, hazard score, quartile bins, criteria S1/S2 | argmax bins labelled exploratory |
| `PREREG_visibility.md` | 2026-10-11, before any synthetic sample or model | visibility v2: synthetic physics, targets, model, bins, criteria S1–S4 | amendments 1–2 (2026-10-11), 3 (2026-10-12, after code review); Version 3 (2026-10-12, *not blind*); v3 tuning; v3 erratum |
| `PREREG_threshold.md` | 2026-10-12, before any training-set detection | per-image threshold v1 (pixels); v2 (confidence profile, 2026-10-13); v3 (YOLOv8x features, 2026-10-13/14); **per-detection model 2 (2026-10-15, before training)** | amendments, pilot and outcome notes, each dated |

The pattern is the same throughout. Write down the criteria first. Run once. Report met or not met as
written. Label anything added afterwards *exploratory*. Most pre-registered hypotheses failed: O3's S1,
visibility S1/S2, and per-image threshold versions 1–3. Model 2's adoption criteria were met.

## 9. How to reproduce each figure

| figure (in `Docs/figures/`) | what it shows | command |
|---|---|---|
| fig1 `o1_o2_run1/fig1_missrate_fppi` | miss rate vs FPPI per DAWN bin (YOLOv8s) | `scripts/run_all.py --from-cache` (`wad.figures`) |
| fig2 `o1_o2_run1/fig2_go_no_go` | go/no-go bars, max recall vs recall at 0.25/0.5 | same |
| fig3 `o1_o2_run1/fig3_main` | run 1 CRC: miss and FPPI per bin, three policies, α 0.1/0.2 | same |
| fig4 `o1_o2_run1/fig4_hazard` | O3-A hazard scalar and predicted bins | same (`wad.hazard_crc`) |
| fig5 `o1_o2_run1/fig5_alpha_sweep` | EXPLORATORY α sweep | same (`wad.extras`) |
| fig6 `o2_run2/fig6_clear_vs_hazard_*` | BDD100K: PR per bin, Δ vs clear, clear-calibrated vs per-bin CRC | `python -m wad.run2 report` |
| fig7 `o2_run2/fig7_run2_detectors` | DAWN: smallest α and FPPI per detector × loss | same |
| fig8 `o2_run2/fig8_tiers_fppi1.0` | confirmed tier: m(b) vs held-out miss | same |
| fig9 `o2_run2/fig9_recoverable_*` | found / recoverable / never per condition | `python -m wad.recoverable` |
| fig10 `o3_visibility/fig10_visibility_estimator_v2/_v3` | visibility predictions per condition | `python -m wad.vis_eval` (`--version 3`) |
| fig11 `o3_visibility/v2_/v3_fig11_visibility_examples` | lowest/highest-visibility frames | same |
| `o3_visibility/v2_night_gap`, `v3_night_preview` | v2 night diagnosis; v3 synthetic vs real night | `python -m wad.vis_dev gap`; `python -m wad.vis_dev luma` |
| fig12 `threshold_exploration/fig12_threshold_headroom` | per-image oracle headroom (in-sample) | `python -m wad.threshold_headroom` |
| fig13 `.../fig13_learned_threshold` | learned per-image threshold vs one threshold | `python -m wad.thresh_eval` |
| fig14 `.../fig14_threshold_budget` | what lowering τ recovers and costs | `python -m wad.threshold_budget` |
| fig15 `.../fig15_graded_response` | stop/slow tiers per condition | `python -m wad.graded_response` |
| fig16, fig17, fig19 | all boxes / low thresholds / summed map per condition | `python -m wad.show_boxes` (`--low`, `--summed`) |
| fig18 `.../fig18_summed_map` | summed-confidence map vs thresholding | `python -m wad.summed_map` |
| fig20 `method/fig20_one_threshold` | the dial's one threshold per λ | `python -m wad.one_threshold` |
| fig21 `method/fig21_per_detection` | per-detection model 1 | `python -m wad.per_detection --features context` |
| fig22 `method/fig22_fbeta_dial` | the Fβ dial | `python -m wad.fbeta_dial` |
| fig23 `method/fig23_per_detection_yolo` | per-detection model 2 (adopted) | `python -m wad.per_detection --features yolo` |
| dial page `dial_demo/dial_demo.html` | interactive dial | `python -m wad.dial_demo` |

## 10. Compute plan for Phase 4

**Measured locally (RTX 3050 Ti Laptop, 4 GB, fp32, one image per pass).**
- YOLOv8x @1280: 0.152–0.165 s/image. A full pass over the 21,831 dial training frames took about 60 min.
- The per-box feature pass over 30.9k images took about 85 min and wrote 4.3 GB.
- The per-detection models run on the CPU: model 2 takes about 11 min and the Fβ dial about 30 min.
- CRC analyses also run on the CPU: about 12 min per BDD100K variant with 200 splits.

**NRP A100 (estimate, not measured).**
- For YOLOv8x inference an A100 has roughly 10–20× the GPU throughput of a laptop 3050 Ti. At that rate a
  ~22k-image BDD100K-train detector pass drops from about an hour to minutes.
- With the harness's one-image-per-pass, 0.001-floor pipeline, JPEG decoding and NMS run on the CPU and
  will cap the gain unless data loading is parallelised. Plan about 5–15 min per 22k images.
- Bigger detector-side experiments become cheap:
  - all 70k BDD100K train images (about 3 h locally; tens of minutes on an A100);
  - Foggy Cityscapes (500 val × 3 densities, once the user has registered for Cityscapes);
  - extra detector sizes and input resolutions;
  - DA-Detect weights.
- The CPU-side steps (gradient boosting, bootstraps, 200-split CRC) gain from a node's many cores rather than
  from the GPU.
- The brief's rule still applies: ask before a download larger than 2 GB (full BDD100K train) or a step
  longer than 30 min.
