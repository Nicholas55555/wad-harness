# Pre-registration: run 2 (O2) and the clear-weather control

Written 2026-10-08, after run 1 (DAWN, YOLOv8s@640, CPU) and the proposal of the same day, and
**before** any of the detections, splits or numbers below were produced. Everything here is
encoded in `config_run2.yaml` and `config_bdd.yaml`; nothing in those files is changed after a
run 2 or BDD100K number is seen. Honest limitation: the author has seen run 1's DAWN results, so
run 2's choices (grouped loss, larger detector, cleaned labels) are informed by them, as the
proposal says; the BDD100K control has not been looked at.

## Detectors (fixed COCO weights, no training)

| id | weights | imgsz | device | role |
| :--- | :--- | ---: | :--- | :--- |
| v8s_640 | yolov8s.pt | 640 | CPU (run 1 cache) | run 1 detector; control comparisons |
| v8x_640 | yolov8x.pt | 640 | GPU (RTX 3050 Ti, fp32) | separates capacity from resolution |
| v8x_1280 | yolov8x.pt | 1280 | GPU (fp32) | **run 2 primary** (proposal: "YOLOv8x at 1280") |

All: conf floor 0.001, NMS IoU 0.5, max_det 300, one image per forward pass, EXIF applied.

## Loss and matching

- **L-class** (run 1): class-aware greedy matching at IoU 0.5 over the six classes.
- **L-group** (run 2 primary, proposal: "grouped vehicle loss"): the same greedy matching, but a
  prediction may match any ground truth in its group: vehicle = {car, truck, bus},
  two-wheeler = {bicycle, motorcycle}, person = {person}. IoU 0.5. Per-image loss = missed
  fraction, images without ground truth count 0. A wrong-class box inside a group is a hit; a box
  outside every group's ground truth is a false positive as before.

## Labels

- **all** (run 1): exact duplicates removed, near duplicates kept and grouped in splits.
- **clean** (run 2 primary, proposal: "clean labels"): additionally one image per near-duplicate
  group (lowest image id) and every group whose copies carry different weather labels removed.
  No box is edited by hand.

## Budgets, splits, policies (unchanged from run 1 unless stated)

- α ∈ {0.10, 0.20}; the smallest certifiable α per bin is always reported.
- Splits: 10 repeats (seeds 0–9), half per bin, near-duplicate groups kept together; 200-split
  robustness rerun reported alongside.
- Policies: oracle-bin CRC, global CRC, default τ = 0.25 and 0.5, and **clear-calibrated CRC**
  (the threshold fitted on the clear-weather calibration images only, applied to every bin;
  BDD100K only, where a clear bin exists).
- **Tiers (proposal slide 5).** Confirmed tier: τ_c(b) = smallest grid τ whose calibration FPPI
  in bin b is ≤ c, for c ∈ {0.1, 1.0}. Certified miss rate at it: m(b) = (n·R̂(τ_c) + 1)/(n + 1)
  on the calibration half; reported with the held-out loss at τ_c and the fraction of splits in
  which the held-out loss exceeds m(b). τ_c is chosen from the calibration false positives, so
  m(b) is a calibration-based bound, checked empirically, not a theorem.

## Clear-weather control: BDD100K val

- Data: BDD100K 100k-image val split (10,000 images, original `bdd100k_labels` JSON), official
  host. Never used to train anything; the O3 hazard estimator, if trained, uses train images only.
- Bins by the image's weather and time-of-day tags. Daytime: **clear**, **cloudy** (overcast +
  partly cloudy), **rain**, **snow**. Night: **clear_night**, **rain_night**, **snow_night**.
  Excluded: foggy (13 images), undefined weather or time, dawn/dusk.
- Classes: person (+ rider), bicycle (bike), car, motorcycle (motor), bus, truck; train, traffic
  lights and signs, lanes and areas dropped.
- Detectors: v8s_640 (comparable with DAWN run 1) and v8x_1280 (comparable with run 2).
- Questions, each answered with bootstrap intervals (images resampled within bin):
  1. At the shipped τ = 0.25, how much recall and precision do rain and snow cost relative to
     clear daytime (same camera, same labelling)?
  2. Does the operating point move with weather: per-bin CRC τ̂(α), τ at FPPI 0.1 and 1.0, and
     the F1-optimal τ (descriptive only)?
  3. Does a clear-calibrated threshold hold α in rain and snow, and what does per-bin
     calibration change in miss fraction and FPPI there?
  4. Gains and losses of moving from τ = 0.25 to τ̂(b): recall points gained, precision points
     lost, false positives added per recovered object, per bin.
- Cross-dataset (descriptive, confounded by domain): the BDD100K clear-day calibrated τ̂
  applied to DAWN's four bins with the same detector.

---

## Erratum and deviations (appended 2026-10-08, after the results; the text above is unchanged)

- The BDD100K bin list names only daytime cloudy; overcast/partly-cloudy **night** images (23)
  fell in no bin and were excluded, alongside foggy, undefined and dawn/dusk.
- Data cleaning found after the first analysis: 37 zero-area BDD100K boxes (unmatchable) were
  dropped, and the one exact duplicate whose copies carry different weather tags was dropped
  entirely (before, the alphabetically first copy was kept).
- Under the grouped loss, per-class curves score a matched box for the class of the ground truth
  it matched (before the fix a truck box matching a car counted as a truck hit, which could push
  per-class AP above 1).
- The 200-split robustness rerun (promised above) was missing from the first run 2 report; it is
  now run for every analysis and is the basis of the summary's verdict words.
- Q3/Q4 intervals are percentiles over the 200 random splits rather than an image bootstrap.
- Added after review and labelled **exploratory**: recall standardised to clear daytime's
  object-height mix (BDD100K rain and snow bins hold larger objects than clear), the alpha sweep,
  and O3's argmax-class bins.
- YOLOv8x @640 (pre-registered secondary) turned out best; calling it "best" is a post hoc
  selection and the primary (YOLOv8x @1280) is reported first everywhere.
