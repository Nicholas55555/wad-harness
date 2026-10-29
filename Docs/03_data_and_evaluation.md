# 03 — Data and evaluation protocol

*Sources: `results/results.md` §1–4 (DAWN, run 1), `results/run2/run2.md` §4 (BDD100K val), `PREREG_run2.md`,
`PREREG_o3.md`, `PREREG_visibility.md`, `PREREG_threshold.md`, `config.yaml`, `config_run2.yaml`,
`config_bdd.yaml`, `NOTES.md` (data traps), `cache/data_report.json`, `cache/bdd/data_report.json`. Counts
marked "derived" were computed from `results/run2/dawn_variants.csv` and the clean `splits.csv`; everything
else is copied from the generated reports.*

> **In short**
> - **DAWN** is the real adverse-weather test set: 1,005 images (1,027 found, 22 exact duplicates
>   dropped), 7,749 boxes in four weather bins (fog, rain, sand, snow). It has only daytime images. The
>   "clean" label set (889 images) also drops near-duplicates and cross-weather copies.
> - **BDD100K val** is the clear-weather control and the dial's main grading set: 8,090 images and 97,745
>   boxes in seven weather × time-of-day bins, including night.
> - **BDD100K train** subsets are the only data anything is *fitted* on: the O3 classifier, the
>   visibility estimators, and the dial (21,831 frames).
> - **Protocol.** Greedy IoU-0.5 matching (class-aware in run 1; grouped vehicle / two-wheeler / person
>   from run 2 on). The loss is an image-level miss fraction. CRC uses a 200-point τ grid. Halves are
>   repeated and keep near-duplicate groups together. Nothing is chosen on a test half.

---

## 1. DAWN (the real-weather evaluation set)

**Source.** Kenk & Hassaballah (2020), Mendeley Data `766ygrbt8y` v3, **CC BY-NC 3.0**. It comes as four zips
(139 MB), each checked against its SHA-256. Each weather folder holds Pascal VOC XML next to YOLO-darknet text
labels. The YOLO copy is unused, because its class ids are COCO's shifted by one. The XML holds 7,846 boxes.
Bins are the folders: `{fog: Fog, rain: Rain, sand: Sand, snow: Snow}`. File prefixes vary inside a folder
(`foggy-`, `haze-`, `mist-`; `dusttornado-`, `sand_storm-`, `sand_storm_g2_-`). Classes are the six COCO
road-user classes person, bicycle, car, motorcycle, bus and truck (COCO ids 0, 1, 2, 3, 5, 7).

| bin | images found | images kept | boxes | person | bicycle | car | motorcycle | bus | truck |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fog | 300 | 283 | 1,894 | 109 | 16 | 1,524 | 22 | 71 | 152 |
| Rain | 200 | 199 | 1,584 | 28 | 0 | 1,351 | 4 | 13 | 188 |
| Sand | 323 | 320 | 2,210 | 174 | 9 | 1,724 | 53 | 55 | 195 |
| Snow | 204 | 203 | 2,061 | 158 | 1 | 1,771 | 1 | 22 | 108 |
| **total** | **1,027** | **1,005** | **7,749** | 469 | 26 | 6,370 | 80 | 161 | 643 |

Cars are 82% of the boxes. Several bin × class cells hold fewer than 30 boxes: rain has 28 persons, 0
bicycles and 4 motorcycles; snow has 1 bicycle and 1 motorcycle. Per-class results for those cells rest on a
handful of images.

**Clean labels (run 2 primary, 889 images).** The rule (`PREREG_run2.md`): *"additionally one image per
near-duplicate group (lowest image id) and every group whose copies carry different weather labels removed.
No box is edited by hand."*

| bin | images (clean) | GT boxes (clean) | calibration half n_cal |
|---|---:|---:|---:|
| Fog | 256 | 1,716 | 128 |
| Rain | 187 | 1,486 | 93 |
| Sand | 243 | 1,689 | 121 |
| Snow | 203 | 2,061 | 101 |
| **all** | **889** | **6,952** | 443 |

*(The per-bin image and box counts are derived; n_cal comes from `run2.md`.)* The visibility correlation
tables use 888 DAWN images: the 889 minus `fog/haze-012`, which has no ground truth.

## 2. BDD100K val (the clear-weather control, and the dial's grading set)

**Source.** BDD100K's 100k-image val split: 10,000 images with the original `bdd100k_labels` JSON, from the
official Berkeley host. The ETH mirror no longer resolves. The `bdd100k_det_20_labels.zip` archive is
mispackaged (2,000 images, no labels), so the original labels zip is used instead. Only the val members are
read out of the 5.7 GB image archive, using range requests and CRC checks. **Never used to train anything.**

**Bins** (`config_bdd.yaml`): four daytime bins (clear; cloudy = overcast + partly cloudy; rain; snow) and
three night bins (clear night, rain night, snow night).
- Excluded: foggy (13 images in all, so no fog bin), undefined weather or time, dawn/dusk, and 23
  overcast/partly-cloudy night images. The bin list named only daytime cloudy, so those 23 fell into no bin;
  this is recorded in the erratum.
- 8,092 images fall in the seven bins. That becomes **8,090** after removing exact duplicates; the one exact
  duplicate whose copies carry different weather tags is dropped entirely.
- Classes: rider → person, bike → bicycle, motor → motorcycle. Dropped: 28,504 traffic signs, 21,564 traffic
  lights, 14 trains and 37 zero-area boxes.

| bin | images | boxes | bicycle | bus | car | motorcycle | person | truck | median box height (px) | boxes < 20 px tall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Clear | 1,764 | 23,977 | 207 | 380 | 19,633 | 111 | 2,624 | 1,022 | 34.8 | 27% |
| Cloudy | 1,676 | 24,004 | 193 | 352 | 19,671 | 128 | 2,694 | 966 | 33.8 | 27% |
| Rain | 395 | 4,904 | 41 | 97 | 3,918 | 16 | 565 | 267 | 42.4 | 19% |
| Snow | 422 | 5,550 | 53 | 97 | 4,421 | 7 | 708 | 264 | 38.5 | 22% |
| Clear night | 3,274 | 33,878 | 152 | 251 | 30,489 | 81 | 2,328 | 577 | 37.7 | 21% |
| Rain night | 286 | 2,770 | 15 | 22 | 2,416 | 10 | 235 | 72 | 43.0 | 18% |
| Snow night | 273 | 2,662 | 19 | 19 | 2,251 | 2 | 288 | 83 | 44.0 | 16% |
| **total** | **8,090** | **97,745** | | | | | | | | |

Facts from `cache/bdd/data_report.json`:
- All images are 1280×720 RGB.
- 72 images have no ground truth; the Spearman tables therefore use n = 8,018.
- There are 9 near-duplicate pairs in 5 groups.
- 8,601 boxes carry BDD's "truncated" flag. They are kept, and no document discusses them.

Two cautions:
- **BDD100K's "rainy" and "snowy" frames are mostly wet roads and light precipitation**, not DAWN's storms.
- **The daytime rain and snow bins hold larger objects** than the clear bin. That confounds raw
  comparisons; see `05_results_crc_runs.md` §3 for the size-matched comparison.

> In the threshold/dial work the BDD100K val conditions are named *day* (clear + cloudy + rain + snow
> daytime), *night* (clear + rain + snow night) and *rain at night*. The dial page's "Day" is the 4,257
> daytime images and "Night" the 3,833 night images.

## 3. BDD100K train subsets (the only data anything is fitted on)

| subset | size | used for | selection |
|---|---|---|---|
| **O3 version B** | 5,443 train + 605 hold-out (clear 1,500, cloudy 1,500, fog 48, rain 1,500, snow 1,500) | ResNet-18 weather classifier | daytime only, seed 0, 10% stratified hold-out (`PREREG_o3.md`) |
| **Visibility v2/v3 base frames** | the 1,500 O3 clear frames + 3,000 more clear-daytime frames (seed 1); 49 dark, mis-tagged frames excluded → 3,989 train / 462 synthetic validation | synthetic degradations → visibility estimator | `PREREG_visibility.md` |
| **Visibility v3 development frames** | 400 train night frames (300 clear, 50 rainy, 50 snowy; seed 2) + 400 daytime non-base frames | tuning the night rendering + development check (not independent) | `PREREG_visibility.md`, Version 3 |
| **Threshold / dial frames** | **21,831** train frames (12,383 newly fetched, ~0.85 GB); **3.34 M** detections at the 0.001 floor | per-image threshold models (PREREG v1–v3), the dial's one-threshold fit, per-detection models 1 and 2, the Fβ map | `PREREG_threshold.md`; split 90/10 by near-duplicate group for the per-image models, 5-fold by group for the per-detection stability check |

The train daytime pool has only **48 foggy images**, against 12,477 clear, 11,829 cloudy, 2,532 rainy and
2,873 snowy. The O3 classifier's fog class therefore rests on 43 training images. WEDGE (synthetic weather,
planned in the proposal) was not fetched.

## 4. Data traps and how each was handled

| trap | what was found | handling |
|---|---|---|
| **EXIF orientation** | 40 DAWN JPEGs carry tag 2 (mirror) and 2 carry the invalid value 0; the annotations are in the mirrored frame. Max recall on the 40 is 87.4% with the tag applied vs 9.9% ignored (10 untagged controls: 78.7% both ways). | Every pixel read goes through `wad/imageio.py` with the tag applied (`exif_orientation: apply`). |
| **Exact duplicates** | 20 SHA-256 groups (22 extra copies), all within one folder | first copy by id kept (→ 1,005 images / 7,749 boxes) |
| **Near-duplicates** | 133 pairs (perceptual hash ≤ 4 bits, mirror-aware) in 89 groups covering 197 images; pixel correlation ≥ 0.977 (126 pairs ≥ 0.99). **11 pairs carry different weather labels**, so the weather label itself is noisy. 23% of pairs have different box counts (annotation noise). | Kept in run 1, but every split keeps a group on one side; a sensitivity rerun keeps one image per group; run 2's clean labels drop them. |
| Out-of-set class | one `train` box (snow) | dropped |
| Empty image | `fog/haze-012` has no objects | kept, loss 0 by definition |
| Pixel formats | 1,024 RGB, 2 CMYK, 1 greyscale | converted to RGB on read |
| Image sizes | 403 distinct sizes (widths 254–2,667); most are 1,280 wide, so imgsz 640 halves every object | resolution sensitivity run; run 2 adds the 1280 input |
| `max_det = 300` binds | Run 1: 4 DAWN images (upper bound 24 unmatched boxes, 0.3% of GT). BDD100K YOLOv8x @1280: 84–88 images in clear/cloudy/clear night, 16–59 in the others. | left as configured, and reported |
| Strict confidence filter | Ultralytics keeps only `score > 0.001` | noted |
| Batch letterboxing | a box's score would depend on its batch mates | one image per forward pass |
| BDD mirror / zip | ETH mirror dead; det_20 zip mispackaged | official host; original labels zip |
| BDD zero-area boxes, cross-weather duplicate | 37 boxes; 1 pair | both dropped (erratum in `PREREG_run2.md`) |
| BDD object-size confound | rain/snow bins hold larger objects | size-standardised comparison (exploratory) |
| Mis-tagged night frames in clear/daytime | about 45 visibility base frames | frames with mean luma < 52 excluded (49) |
| DAWN is not 16:9 (median 1.6:1) | visibility estimator trained on 16:9 | centre-crop to 16:9 (stretched kept as a sensitivity check) |
| Depth model saturates near 80 m | used for the synthetic fog veil | ≥ 70 m mapped to far depth, then a smooth extension beyond 40 m (amendments 1–2) |

## 5. Matching, loss and false positives

**Matching** (the same quantity the loss measures):
- **L-class** (run 1): class-aware greedy matching at IoU 0.5 over the six classes. There is one greedy pass
  in score order, so thresholding at τ equals truncating the score-sorted list; this is tested. VOC
  coordinates are used as stored, and IoU is computed on continuous coordinates.
- **L-group** (run 2 primary, the BDD100K control, O3/visibility evaluations and **the dial**): the same
  greedy matching, but *"a prediction may match any ground truth in its group: vehicle = {car, truck, bus},
  two-wheeler = {bicycle, motorcycle}, person = {person}"*. A wrong-class box inside a group is a hit. The
  grouping removes run 1's car/truck/bus confusions, which were 29% of its never-matched boxes.

**Loss (per image).** At threshold τ, the loss is *"the fraction of ground-truth objects with no matched
detection at score ≥ τ. Images with no ground truth have loss 0."* It is non-decreasing in τ, which is
asserted in a test.

**FPPI.** Predictions with `tp = false` and score ≥ τ, divided by the number of images in the bin.

**Summary curves.**
- Max achievable recall: recall at the 0.001 floor.
- "Never matched": 1 − max recall, the share no threshold can recover.
- Miss rate vs FPPI curves.
- LAMR: the arithmetic mean of the interpolated miss rate at nine FPPI points log-spaced in [0.01, 1], as the
  brief specifies, plus Dollár et al.'s geometric-mean "Caltech" version.
- MR@0.1 and MR@1 FPPI.
- mAP50, reported once.

## 6. Calibration (conformal risk control), splits and policies

**Threshold grid.** 150 points log-spaced on [0.001, 0.5] plus 50 linear points on (0.5, 0.99], 200 in all.
Fixed in `config.yaml` before any split.

**CRC rule (per bin b):**

  τ̂(b) = max{ τ on the grid : n/(n+1)·R̂(τ; b) + 1/(n+1) ≤ α }

R̂ is the mean per-image miss fraction on the calibration half of bin b. If no grid point qualifies, τ̂ is set
to the 0.001 floor and the fit is flagged **infeasible**; there the guarantee does not hold. This is SeqCRC's
first step, CRC with B = 1, written as a max over τ (equivalently an inf over λ = 1 − τ).

**Smallest certifiable α.** n/(n+1)·R̂(0.001) + 1/(n+1), averaged over calibration halves. It is the
tightest budget the rule can ever certify with this detector.

**Budgets.** α ∈ {0.10, 0.20}, fixed before any split. Run 1 also previewed per-class budgets (α = 0.05 for
person, 0.10 for the rest).

**Splits.** By image, stratified by weather. Calibration = floor(n/2) per bin (run 1: fog 141, rain 99,
sand 160, snow 101). Near-duplicate groups are assigned whole. Run 1 uses 10 repeats (seeds 0–9), the brief's
number, plus a 200-split robustness rerun. Run 2's verdicts use the pre-registered **200-split** rerun.

**Policies compared:**
- **oracle-bin (per-bin) CRC**: one τ̂ per true weather bin;
- **global CRC**: one τ̂ on all calibration images pooled. This is SeqCRC step 1 applied naively to mixed
  weather, and the fair comparison;
- **defaults** τ = 0.25 (Ultralytics' shipped default) and τ = 0.5;
- **clear-calibrated CRC** (BDD100K only): fitted on clear daytime, applied to every bin;
- **predicted-bin CRC** (O3): bins from a hazard estimator instead of the weather label.

**Tiers (proposal slide 5).**
- Confirmed tier: τ_c(b) is the smallest grid τ whose calibration FPPI in bin b is ≤ c, for c ∈ {0.1, 1.0}.
- At τ_c the miss rate is estimated as m(b) = (n·R̂(τ_c) + 1)/(n + 1) on the calibration half.
- **m(b) is a calibration estimate, not a bound.** The pre-registration called it a "certified miss rate"; the
  run 2 review corrected that. The held-out miss exceeds it in roughly 40–50% of splits. A planner that needs
  a high-probability bound needs a held-out upper confidence bound on top (for example Clopper-Pearson on a
  separate fold).

## 7. Verdict words (computed, not chosen)

From run 2 on, every verdict sentence is generated by one rule (`wad/run2_report.py`):
- *holds*: the mean held-out miss is ≤ α;
- *at the budget*: within 0.01 above α;
- *fails*: more than 0.01 above α.

"Exceeds α" means strictly greater. The share of splits above α is always reported next to the mean,
because a marginal guarantee promises the average, not each split.

## 8. Which data is used for what

Nothing is chosen on a test half or an evaluation set. The table shows each data set's role.

| data | fit / train | calibrate (choose a threshold) | grade (held out) | notes |
|---|---|---|---|---|
| DAWN (1,005; clean 889) | — | CRC calibration halves (runs 1–2; O3 and visibility bins) | CRC test halves; O3/visibility evaluation; **dial grading only** (clean labels, 889 images, YOLOv8x @1280, grouped) | never trains anything; the dial never looks at it to choose |
| BDD100K val (8,090) | — | CRC calibration halves (control) | CRC test halves; O3/visibility evaluation; **dial grading** (95% image bootstrap) | never trains anything |
| BDD100K train, O3 subset (6,048) | O3-B classifier | — | 605-image hold-out for model selection | daytime only |
| BDD100K train, clear-day base frames (4,500 → 3,989/462) | visibility v2/v3 (synthetic labels) | — | synthetic validation (S4) | 49 dark frames excluded |
| BDD100K train, night development frames (400 + 400) | — | v3 rendering tuning | v3 development check (not independent) | v3 is not blind |
| **BDD100K train, threshold frames (21,831)** | **the dial**: one-threshold table, per-detection models 1–2, PCA, Fβ map; per-image threshold models | 5-fold by near-duplicate group (stability, out-of-fold chances for Fβ) | 10% split for the per-image models | the only data the method is fitted on |
| COCO val2017 (5,000) | — | Phase 1 cods reproduction only (2,500/2,500, n_cal 500 used) | Phase 1 only | not part of the harness |

**One caveat on the dial's model choice** (`08_method_lambda_dial.md` §6). Model 2's own choices (feature list,
PCA, one run, adoption) used train only. The 20 context features and the boosting settings it inherits were
picked in exploratory runs scored on val and DAWN. The alternatives differed by about 0.005.

## 9. Metrics used by the method (the λ dial)

Details are in `08_method_lambda_dial.md`. The dial is graded with:
- recall, precision and false alarms per image;
- the **cost ratio** (λ·misses + false alarms) relative to the one threshold at the same λ, with a 95% image
  bootstrap;
- Fβ, with its 95% interval, against the best constant threshold reachable in-sample;
- calibration of the per-detection chances: ECE, and the share actually real near each cut 1/(1+λ).

Its labels come from the same grouped greedy matching at IoU 0.5 against all boxes at the 0.001 floor. If a
dropped box was matched and a kept weaker duplicate was not, the duplicate still counts as false. That makes
per-detection numbers slightly pessimistic; re-matching moves the cost ratio by at most 0.001.
