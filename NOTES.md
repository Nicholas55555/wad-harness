# NOTES

Decisions, surprises, and every place the data differed from the brief, kept as the work went.
Numbers quoted here are from the runs on 2026-10-01; `results/results.md` is the generated source
of truth.

## Environment

- Windows 11, AMD Ryzen 7 4800H (8 cores), 31 GB RAM, NVIDIA RTX 3050 Laptop (4 GB). No system
  Python; Python 3.11.15 managed by `uv`, virtual environment in `.venv/`.
- **CPU build of torch** (2.14.1+cpu). The machine has a CUDA GPU, but the CUDA torch wheel is a
  >2 GB download (the brief asks before anything that size) and detection takes 2 minutes on
  the CPU anyway. `detect.py` uses `cuda:0` automatically if a CUDA build is installed.
- Detection on CPU: 0.12 s of inference per image for 1,005 images, about 2 minutes of wall
  time (`detections.meta.json` records both `inference_s` and `wall_time_s`), well under the
  brief's 0.3–0.6 s/image estimate. Everything after detection runs in about 2 minutes, most of it the
  200-split robustness rerun and the hazard features.

## Data: what DAWN actually is (and where it differs from the brief)

- **Download.** Mendeley Data record `766ygrbt8y` v3 (CC BY-NC 3.0): four zips (`Fog.zip`,
  `Rain.zip`, `Sand.zip`, `Snow.zip`, 139 MB), each verified against the SHA-256 Mendeley
  publishes. `scripts/get_dawn.py` repeats this.
- **Layout.** `data/DAWN/<Folder>/*.jpg` with annotations in `<Folder>/<Folder>_PASCAL_VOC/*.xml`
  and an unused `<Folder>_YOLO_darknet/` copy (its class ids are COCO's, 1-based: person 1,
  bicycle 2, car 3, motorcycle 4, bus 6, train 7, truck 8). Folder names `Fog`, `Rain`, `Sand`,
  `Snow`; file prefixes vary inside a folder (`foggy-`, `haze-`, `mist-`; `dusttornado-`,
  `sand_storm-`, `sand_storm_g2_-`; ...).
- **Counts.** 1,027 images (brief: about 1,020): Fog 300, Rain 200, Sand 323, Snow 204.
  7,846 boxes in the XML.
- **Classes.** The six expected classes plus **one `train` box** (snow); dropped, as the plan
  drops train for Cityscapes (`data.drop_classes` in config.yaml makes this explicit).
- **Flags.** No box is marked `difficult` or `truncated` (all 0); `pose` is always Unspecified.
- **One image has no objects** (`fog/haze-012`): kept, loss 0 by the brief's definition.
- **Pixel formats.** 1,024 RGB, 2 CMYK (`sand/sand_storm-184`, `sand/sand_storm_g2_-067`),
  1 greyscale (`fog/foggy-046`); converted to RGB on read.
- **EXIF orientation (surprise, would have silently broken 4% of the data).** 40 JPEGs carry
  orientation tag 2 (horizontal mirror); 2 carry the invalid value 0 (treated as identity).
  The annotations of the 40 are in the *mirrored* frame: running the detector on them gives
  max recall 87.4% with the tag applied and 9.9% with it ignored (10 untagged controls: 78.7%
  both ways; `cache/exif_check.json`). OpenCV, and so Ultralytics given a path, applies the tag;
  PIL does not. Every pixel read in the repo goes through `wad/imageio.py` with the tag applied
  (`data.exif_orientation: apply`), so detector, hashes and hazard features share one frame.
- **Image sizes.** 403 distinct sizes, widths 254–2,667 px, heights 161–1,950; most are 1280
  wide, so imgsz 640 halves objects.
- **Exact duplicates.** 20 SHA-256 groups (22 extra copies), all inside one weather folder.
  Kept the first by image id, excluded the rest from everything downstream: 1,005 images,
  7,749 boxes remain.
- **Near duplicates (surprise).** 133 pairs within perceptual-hash distance 4 forming 89 groups
  that cover 197 of the 1,005 images. `wad.data` checks every pair with a mirror-aware 64×64
  grey-level correlation (minimum and count in `cache/data_report.json`, per pair in
  `cache/duplicates.csv`): all are the same photograph resized, re-encoded or mirrored. The hash comparison is mirror-aware (min of hash(a)–hash(b) and
  hash(a)–hash(mirror(b))) because some copies differ only by the EXIF mirror tag.
  - **11 pairs carry different weather labels** (e.g. `fog/foggy-075` = `rain/rain_storm-399`,
    `-501`, `-939`; `fog/foggy-037` = `sand/sand_storm-256`). The weather label is noisy.
  - 23% of near-duplicate pairs have a different number of annotated boxes: copies were
    annotated independently, a direct measure of annotation noise.
  - Decision: the brief excludes only exact duplicates, so near duplicates stay, but **every
    split keeps a near-duplicate group on one side** (`wad.crc.split_once`), so calibration and
    test never share a photograph. Sensitivity: phase 4 rerun with one image per group and
    cross-weather groups removed (`results/sensitivity/near_dup_dropped/`) — same picture.
- **Small classes.** Rain has 28 person boxes and no bicycles; bicycle (26 boxes) and
  motorcycle (80) are tiny everywhere. Per-class results for them are anecdotal.

### The duplicate lists (brief: list them here)

Every pair, with Hamming distance and pixel correlation, is in `cache/duplicates.csv`
(written by `python -m wad.data`). The exact-duplicate groups (first id kept, the rest excluded):

```
snow/snow_storm-040 = snow/snow_storm-075        fog/haze-064 = fog/haze-083
fog/haze-002 = fog/haze-139                      fog/foggy-060 = fog/haze-024 = fog/haze-039
fog/haze-031 = fog/haze-061                      fog/haze-080 = fog/haze-128
sand/dusttornado-016 = sand/sand_storm-273       fog/foggy-011 = fog/haze-003
fog/haze-018 = fog/haze-109                      fog/haze-084 = fog/haze-100
fog/haze-036 = fog/haze-059                      sand/dusttornado-023 = sand/sand_storm-057
fog/foggy-056 = fog/foggy-058                    sand/dusttornado-036 = sand/sand_storm-241
fog/haze-072 = fog/haze-090 = fog/haze-142       rain/rain_storm-536 = rain/rain_storm-553
fog/haze-112 = fog/haze-122                      fog/haze-077 = fog/haze-098
fog/haze-145 = fog/haze-149                      fog/haze-026 = fog/haze-049
```

Near-duplicate pairs whose copies carry different weather labels (perceptual-hash distance):

```
fog/foggy-037 = sand/sand_storm-256 (0)          fog/foggy-085 = sand/sand_storm_g2_-061 (0)
fog/foggy-043 = rain/rain_storm-269 (0)          fog/foggy-099 = rain/rain_storm-720 (0)
fog/foggy-075 = rain/rain_storm-399 (0)          fog/haze-038  = sand/sand_storm_g2_-033 (0)
fog/foggy-075 = rain/rain_storm-501 (0)          fog/haze-067  = rain/rain_storm-435 (2)
fog/foggy-075 = rain/rain_storm-939 (0)          fog/haze-082  = sand/sand_storm_g2_-061 (0)
fog/mist-150  = rain/rain_storm-597 (0)
```

The other 122 near-duplicate pairs stay within one weather folder (see the CSV).

## Phase 2: detection cache

- Every shard index records a fingerprint of the detector configuration (detector settings,
  classes, EXIF policy, weights SHA-256, device); resuming or merging shards made under a
  different fingerprint is refused, and `run_all.py --force` clears the detection cache first.
- One image per forward pass. With mixed sizes Ultralytics letterboxes a batch to a shared
  square, so a box's score would depend on which images shared its batch; per-image passes
  make each detection a function of its own image and make resume exact. Shards of 16 images
  are written atomically (rows + an index that also records zero-detection images; the index is
  written last and marks the shard complete).
- Images are passed as BGR arrays from `wad/imageio.py`, not as paths, so the EXIF policy is
  explicit. The detector's `orig_shape` is checked against the XML size for every image.
- Sanity: 3 images with zero detections at the 0.001 floor (`fog/haze-012`, which has no
  objects, `fog/haze-033`, `fog/mist-133`; 0.3%, under the 5% stop rule). Detections per image
  min 0 / median 44 / max 300. Score deciles 0.0010, 0.0013, 0.0016, 0.0022, 0.0030, 0.0044,
  0.0069, 0.0123, 0.0325, 0.2495, 0.9534. Class shares: car 56.3%, person 28.8%, truck 9.7%,
  bus 2.9%, motorcycle 1.5%, bicycle 0.7%.
- **max_det = 300 binds on 4 images** (`rain/rain_storm-580`, `-736`, `-779`,
  `snow/snow_storm-227`; lowest kept score 0.0012–0.0019). Boxes scoring between 0.001 and
  that value were cut there. Upper bound on the effect: the 24 boxes still unmatched in those
  images (0.3% of all ground truth); in practice smaller. Left as configured.
- Ultralytics' `conf` filter is strict (`score > 0.001`), so every cached score exceeds the floor.

## Phase 3: matching and curves — judgment calls

- VOC coordinates are used as stored (no −1 shift); IoU on continuous coordinates without +1.
- Canonical prediction order: image, descending score, then class and box, which makes the
  greedy tie-break deterministic; IoU ties go to the lowest ground-truth index (GT sorted by
  box). The single-pass/truncation equivalence is tested on 40 random images with forced ties.
- **Per-class image loss** is averaged over the images that contain the class (in
  `curves.csv` and in phase 4). The brief's literal "images without ground truth count 0"
  would, for a single class, dilute the loss by the share of images lacking that class and make
  it incomparable with the O4 per-class budget, which the brief defines over images containing
  the class. For the pooled class the brief's definition is used as written (the one empty
  image counts 0).
- **LAMR.** The brief's definition (arithmetic mean of the miss rate at nine FPPI points
  log-spaced in [0.01, 1], linearly interpolated) is the `lamr` column. Interpolation runs along
  the lower envelope: at each distinct FPPI the best miss rate reached there. Dollár et al.'s
  Caltech version (geometric mean, stepwise) is reported beside it as `lamr_caltech`; for the
  person, car and pooled rows they differ by 0.00–0.03. Curves are exact (every distinct score), not the 200-point grid. References
  beyond a curve's largest FPPI are clamped to its end and counted in `lamr_n_clamped`; this
  happens only for bicycle and motorcycle (1–3 of the 9 points), never for person, car or the
  pooled class.
- Miss rate at FPPI 0.1 and 1.0 (the plan's headline operating points) added to the summary.
- mAP@0.5: VOC all-point interpolation, COCO-style greedy matching (best *unmatched* GT, same
  as the loss), mean over the classes present in the bin. Reported only.
- Recall at the default thresholds is evaluated exactly at 0.25 and 0.5, not on the grid.
- τ grid: 150 points log-spaced on [0.001, 0.5] plus 50 linear on (0.5, 0.99] = 200.

## Phase 3: result and how much to trust it

- **GO**: worst person bin sand (max recall 79.9% vs 62.6% at τ = 0.25, margin 17.2 pts), worst
  car bin fog (81.6% vs 58.4%, 23.2 pts).
- **Fragile for person.** Fog's person max recall (80.7%) is 0.8 points above sand's, and fog's
  margin is 14.7 points, under the rule. A cluster bootstrap (near-duplicate groups resampled
  within weather bins, 2,000 draws) gives P(GO) = 0.64 (person passes 64%, car 100%). Worst-bin
  person margin 95% interval 9.4–24.1 points.
- Recoverable vs irrecoverable (pts of recall, pooled classes): fog 23.5 / 20.6, rain 22.9 /
  15.4, sand 23.4 / 18.6, snow 23.8 / 12.9. Thresholding can recover the first number; the
  second is never matched at any threshold.

## Phase 4: CRC — judgment calls

- Splits: per weather bin, calibration target floor(n/2) (fog 141, rain 99, sand 160, snow 101;
  met exactly in every seed), test gets the rest. Near-duplicate groups are visited in random
  order and assigned whole. Seeds 0–9 (`splits.seed` + repeat index). `results/splits.csv`.
- Infeasible fits set τ̂ to the floor 0.001 and are flagged; their test numbers are reported,
  not hidden.
- Default policies don't depend on α; they are evaluated per α only so each table is complete.
- "Exceeds α" means test image loss strictly greater than α.
- The pooled ("all") row of the oracle-bin policy applies each bin's τ̂ to its own test images.
- Per-class preview: n_cal is the number of calibration images in the bin that contain the class.

## Phase 4: result

- **α = 0.10 is not certifiable** in most splits of any weather bin (infeasible in 100% of splits
  for fog, rain and sand, 80% for snow) nor pooled. In the 2 snow splits where it was
  certified, τ̂ sat just above the floor and the held-out loss was still about 0.145, above α:
  the guarantee is marginal over splits, not per split. At the 0.001 floor the per-image miss fraction is still
  0.118 (snow) to 0.218 (fog). Smallest certifiable budget (mean over calibration halves): fog
  0.229, rain 0.154, sand 0.192, snow 0.117, pooled 0.175.
- **α = 0.20**: oracle-bin CRC is feasible in rain and snow (and sand in 8/10 splits), with τ̂
  of 0.015 (rain), 0.04 (snow), 0.002 (sand); infeasible in fog. The global threshold (τ̂ ≈ 0.004)
  over-delivers in rain and snow and misses α in fog (100% of splits) and sand (60%). Default
  τ = 0.25 misses by 0.10–0.22.
- **The price is false positives**: 10–30 FPPI in the feasible oracle bins at α = 0.20 versus
  0.8–1.9 at τ = 0.25. Operationally that is not a usable obstacle list; it is the cost the
  recall-first argument has to justify (the plan's tiered output: confident = obstacle,
  low-confidence = caution zone).
- With 10 splits, the snow oracle-bin mean (0.215) sits above α; the 200-split rerun gives 0.192
  with 37% exceedance, so that was split noise. 10 splits is what the brief fixed; 200 is
  reported as a robustness check.
- Per-class (O4 preview): every class × bin fit is infeasible at least in most splits; person at
  α = 0.05 is structurally infeasible in rain (≈10 calibration images containing a person, so
  1/(n+1) ≈ 0.09 > 0.05).

## Why the loss will not go below ~12–22% (diagnosis, beyond the brief)

- Of the 16.9% of boxes never matched at the floor: 39% have nothing of the right class within
  IoU 0.3 (*no proposal*), 29% are detected as another class at IoU ≥ 0.5 (car→truck 170,
  truck→car 44, car→person 34, bus↔truck 54), 31% have a same-class box at IoU 0.3–0.5 (*loose
  box*), 1% lose their box to a neighbour.
- By size in the detector's 640-px input: boxes under 32 px are 37% of the data but 61% of the
  never-matched boxes; under 16 px, 46% are never matched.
- Implication for the decision layer: a class-aware IoU-0.5 miss counts "truck seen as car" and
  "box drawn 40% too large" as misses. For an obstacle list neither is a safety miss. A loss
  over a vehicle superclass, or at IoU 0.3, would be both more relevant and more certifiable —
  but it is a change of the pre-registered loss and must be fixed *before* it is evaluated
  (ideally on Foggy Cityscapes, so DAWN's test halves are not reused to choose it).

## Phase 6a: hazard scalar

- Features after resizing to a 640-px long side (so edge energy is comparable across DAWN's
  mixed resolutions). All four features are invariant to a horizontal flip, so the EXIF mirror
  question cannot touch h (tested).
- PCA (standardisation + first component) is fitted once on all kept images: label-free and
  detector-free, a symmetric function of the whole pool, so it does not distinguish calibration
  from test images. Quartile cut points come from each split's calibration half, as the brief
  says. Strictly, cut points estimated on the calibration half make the bin boundaries depend on
  the calibration sample; the within-bin guarantee is therefore approximate rather than exact.
- **Surprise: h is a sand axis.** Loadings: dark channel −0.68, saturation +0.71, contrast
  −0.12, edge energy −0.12 (44% of variance). Sign convention satisfied (higher h = lower
  contrast and edges) but weakly; the component mostly separates the orange, saturated sand
  images from everything else. Fog has the highest dark channel, as expected, but the lowest
  saturation too, so it lands at low h. ARI with weather labels 0.074.
- Predicted bins still hold α = 0.20 roughly where feasible (h3, 7/10 infeasible, is the
  exception) and cost +4.4 ± 3.1 FPPI over the oracle bins.
- A fog-specific scalar (dark channel alone) or a learned classifier (O3 version B on BDD100K)
  is the obvious next step.

## Not done, and why

- **Phase 6b (Foggy Cityscapes)** needs a Cityscapes account (registration with the
  data owner) and two multi-GB archives; both are the user's to do. The harness is ready for a
  `config_foggy.yaml` once the files are in place.
- **YOLOv8x** (the SeqCRC paper's model): the brief reserves it for a GPU.
- The brief asks to stop after phase 3 for a go-ahead; this run was asked to complete all
  tasks in one pass, so the phase-3 numbers were reported in the chat and phase 4 ran in the
  form the brief prescribes for either verdict.

## Detector input-resolution sensitivity (YOLOv8s at imgsz 1280)

Same COCO weights, imgsz 1280 instead of 640 (`python -m wad.sensitivity`; 0.38 s/image on
CPU, 77,010 boxes, 34 images now hit max_det). Full table in section 6.3 of `results.md`.

- **Resolution is not the fix.** Pooled max recall *falls* from 83.1% to 81.0%: 504 boxes
  gained, 662 lost (`results/sensitivity/yolov8s_1280/lost_gained.csv`). Gained boxes are small
  (median side 40 px in the original image), lost ones larger (median 98 px). At 1280 the lost
  boxes are mostly still seen: 46% detected with another class, 17% with a box too loose for
  IoU 0.5, 37% with no proposal; 60 of them sit in images where max_det binds. Person improves
  (85.3% → 87.2%), car worsens (84.5% → 82.5%).
- The smallest certifiable budget gets worse in every bin (fog 0.229 → 0.309, pooled
  0.175 → 0.239), and oracle-bin CRC at α = 0.20 costs 50–71 FPPI instead of 10–38.
- So the never-matched loss is not simply a resolution limit of the 640 setting. A multi-scale
  union (640 ∪ 1280 with cross-scale NMS) might recover both ends, but that is test-time
  augmentation, i.e. a different detector, and stays out of scope here. Note also that the go/no-go
  person worst bin flips to fog at 1280 (margin 17.4), another sign of how close fog and sand are.

---

# Run 2 (O2), the clear-weather control and O3 — 2026-10-08

Pre-registered in `PREREG_run2.md` and `PREREG_o3.md` before any of these numbers existed.
Generated results: `results/run2/run2.md`, `results/o3/o3.md`.

## GPU

- NVIDIA RTX 3050 Ti Laptop (4 GB, compute 8.6), driver 566.07 (CUDA 12.7). **No driver change
  was needed or made**: torch's cu126 wheels run on any CUDA-12 driver ≥ 525. The cu130 wheels
  need driver 580+, so cu126 is the right build here. torch 2.14.1+cu126 replaced the CPU build
  in `.venv` (2.5 GB download).
- The CUDA build reproduces run 1's CPU detections bit for bit (checked on 12 images, and run 1's
  config now pins `detector.device: cpu`, which is part of the detection-cache fingerprint).
- GPU speeds (one image per pass, fp32): YOLOv8x @640 0.075 s/image, @1280 0.165 s/image
  (0.6 GB VRAM); YOLOv8s @640 0.033 s/image on BDD100K. ResNet-18 training ~2 min per epoch on
  5,443 images at 224×400.

## BDD100K (clear-weather control)

- The ETH mirror named in BDD100K's docs (`dl.cv.ethz.ch`) no longer resolves; the official
  Berkeley host (`128.32.162.150/bdd100k/`, linked from bdd-data.berkeley.edu) serves the files.
  Its `bdd100k_det_20_labels.zip` is **mispackaged** (2,000 images, no labels); the original
  `bdd100k_labels.zip` (one JSON per image, weather/scene/time-of-day tags, 2D boxes) is used.
- The 5.7 GB image zip supports HTTP range requests, so `scripts/get_bdd100k.py` reads only the
  10,000 val images (and later the 6,048 O3 training images) out of it, CRC-checked. The host
  dropped connections once mid-download; the script now retries per file.
- Val tag counts: daytime clear 1,764, overcast + partly cloudy 1,677, rainy 396, snowy 422,
  foggy 5 (13 in total, so no fog bin); night clear 3,274, rainy 286, snowy 273. 8,091 images
  fall in the seven pre-registered bins; 1 exact duplicate pair, 9 near-duplicate pairs.
- Classes: rider → person (as planned for Cityscapes), bike → bicycle, motor → motorcycle; train,
  traffic lights and signs dropped (28,504 signs, 21,564 lights, 14 trains). 97,800 boxes remain,
  82,843 of them cars: BDD100K labels many tiny distant cars, so its miss rates are dominated by
  small objects in every weather.
- **The control's finding (corrected after review).** Raw, at τ = 0.25, daytime rain and snow
  look no worse than clear daytime (rain −0.3 recall pts, CI −2.5 to +2.0). But the bins do not
  hold the same objects: rain and snow frames contain larger objects (boxes under 20 px tall:
  clear 27%, rain 19%, snow 22%) and fewer highway scenes. At clear daytime's object-height mix
  (exploratory, added after the review) daytime rain costs about 5 recall points (CI roughly −7 to
  −3) and snow about 0; the night gaps widen to 22–29 points. So precipitation *does* cost recall
  per object; the first report's "precipitation does not matter" was confounded by object size.
  What holds: fixed-FPPI thresholds are stable by day and drop at night; a clear-calibrated
  threshold met α = 0.2 in BDD100K's rain and snow bins as sampled but fails at night; per-bin
  calibration restores night at 42–124 FPPI. Exact numbers: `results/run2/run2.md`.
- BDD100K's "rainy"/"snowy" frames are mostly wet roads and light precipitation, not DAWN's
  storms; DAWN fog is the severe case where one threshold breaks the budget.
- With YOLOv8s @640 even clear daytime cannot certify α = 0.2 on BDD100K (smallest certifiable α
  0.25–0.33 in every bin): the pre-registered budgets are infeasible there for that detector, so
  the policy comparison on BDD100K uses YOLOv8x @1280 (smallest α 0.10–0.12 by day, 0.15–0.18 at
  night) and the exploratory α sweep (labelled as such).

## Run 2 on DAWN — what came out

- The grouped loss lifts YOLOv8s max recall from 83% to 87%; YOLOv8x with the grouped loss
  reaches 89–93% per bin, with recall at τ = 0.25 of 67–76%. **A stronger detector leaves less recall below τ = 0.25**, so the go/no-go gate turns
  NO-GO for several YOLOv8x variants (person in rain: nothing left to recover). The gate measures
  how much thresholding *could* help, not how good the system is; the pre-registered primary
  (YOLOv8x @1280, group, clean) is GO by a thin person margin (15.4 pts in fog).
- **1280 px trades the tail for the head.** At 1280 YOLOv8x gains recall at τ = 0.25 in most
  bins but loses low-confidence recall, so the smallest certifiable α and the FPPI at α = 0.2 get
  worse in every bin. YOLOv8x @640 with the grouped loss (pre-registered secondary, singled out
  after the results) is the best variant: smallest certifiable α 0.04–0.07 in every bin, α = 0.1 feasible
  everywhere at 4–14 FPPI, α = 0.2 at 0.6–1.3 FPPI. The primary was fixed in advance as
  YOLOv8x @1280 (the proposal's wording); both are reported, and the proposal should switch the
  primary for the final report only with that history stated.
- Grouped vehicle loss: removes the car/truck/bus confusions (29% of run 1's never-matched
  boxes) and lowers every bin's smallest certifiable α by 0.04–0.09.
- Clean labels (one per near-duplicate group, no cross-weather groups) change CRC results
  little but **flip the go/no-go for YOLOv8x @1280** from NO-GO (person worst bin sand, margin
  9.8) to GO (fog, 15.4); the primary's GO holds in only 28% of bootstrap resamples (person 30%).
- Confirmed tier (FPPI cap 1): the calibration estimate m(b) of the miss rate at τ_c is 0.24–0.30
  per bin for the primary (YOLOv8x @1280 group clean) and 0.17–0.22 for YOLOv8x @640 group (post
  hoc best). m(b) is an expectation-level estimate, not a bound: on BDD100K the held-out miss
  exceeds it in roughly half the splits. A planner that needs a high-probability bound needs a
  held-out upper confidence bound (e.g. Clopper-Pearson on a separate fold).

## O3 version B — decisions

- BDD100K train has only **48 daytime foggy images** (12,477 clear, 11,829 cloudy, 2,532 rainy,
  2,873 snowy). The classifier's fog class rests on 43 training images; WEDGE (planned in the
  proposal) would fill it but was not fetched in this pass.
- Training frames are stored once at 252×448 (16:9; DAWN's non-16:9 frames are squashed the same
  way at prediction time) and augmented on the GPU (crop 60–100% area + flip).

## Review of run 2 (2026-10-08)

An adversarial review (3 reviewers, an independent recomputation, a skeptic per finding)
reproduced every number it recomputed, and confirmed 25 findings, all fixed: the object-size
confound above; zero-area BDD100K boxes and a cross-weather exact duplicate (dropped); per-class AP
above 1 under the grouped loss (TPs now count for the matched ground truth's class); the
pre-registered 200-split rerun and bootstrap intervals for Q2–Q4 (added); "certified" wording for
m(b) (now "calibration estimate"); summary numbers taken from the post hoc best variant instead
of the primary (now primary first, best labelled); inconsistent "holds/fails" wording (now one
computed rule); misses-by-size bins that were not comparable across input sizes (now a fixed
640-px reference); and a label-extraction resume that did not check CRCs. See the erratum at the
end of `PREREG_run2.md`.

# Per-condition analyses and visibility estimation — 2026-10-11/12

## Weather × time of day, recoverable recall, statistics (exploratory, on the run 2 caches)

- `wad/recoverable.py`: recall recoverable by lowering the threshold vs never detected, per
  BDD100K weather × day/night and DAWN weather (fig9). On BDD100K, night gains most from a lower
  threshold (+36–40 pts at night vs +21–24 by day); DAWN (all daytime) gains +16–21 pts. DAWN fog
  has 11.3% of objects never detected at any score.
- `wad/stats_tables.py`: recall, precision, F1, FPPI, AP, per-class numbers per condition and per
  policy. The shipped τ = 0.25 is close to F1-optimal; meeting a recall budget costs precision/F1.
- `wad/visibility.py` (v1, exploratory): h = −(z(log edge energy) + z(RMS contrast))/2. It protects
  the lowest-visibility images (DAWN lowest-visibility quarter 0.196 vs 0.275 per-weather) at higher
  FPPI; blind spot: rain at night (0.238).

## O3 version C: trained visibility estimator (v2) — `PREREG_visibility.md`, `results/vis2/vis2.md`

- Physics-defined targets on synthetic degradations of clear-day BDD100K train frames (veil from
  metric depth, rain/drops/wheel spray at labelled vehicles, darkness), ResNet-18, 20 epochs.
- A first training run was stopped after an independent multi-agent code review upheld 13
  findings (grouped as the 7 items of amendment 3: loss weights, schedule, spray on only 65% of
  vehicles, mis-tagged night base frames, DAWN geometry, determinism, evaluation edge cases). All
  were fixed and the model retrained from scratch; no real-image prediction was made before.
- Pre-registered verdict: S4 met (synthetic MAE log10 V_eff 0.069), S3 met (lower FPPI than v1 on
  both datasets), **S1 not met** (Spearman with misses 0.277 vs v1 0.460 on BDD100K, 0.177 vs 0.245
  on DAWN), **S2 not met** (rain-night 0.248, clear-night 0.236, DAWN fog 0.220).
- What works: scattering. Median predicted MOR on the evaluation images: DAWN fog 208 m, sand
  216 m, rain 2,078 m, snow 2,901 m, BDD100K clear day 9,461 m. The lowest-V_eff DAWN rain images are
  downpours with spray (MOR 33–74 m), the lowest fog/snow images near white-outs (fig11). On DAWN
  the overall FPPI of the v2 policy is 4.05 vs 6.37 for v1 at the same overall miss (0.191 vs
  0.192), but each score protects the images it ranks least visible: v2's thresholds hold v2's
  lowest quarter at 0.195 (9.9 FPPI) and leave v1's lowest quarter at 0.256; v1's hold its own at
  0.196 (19.8 FPPI) and leave v2's at 0.206.
- What fails: night. The estimator reads BDD100K night as near-daylight (median k 0.94; ROC AUC
  night vs day 0.852), so BDD100K V_eff sits near its 2,000 m daylight ceiling and 8,073 of 8,090
  BDD100K images fall in the top fixed band. Diagnosis (`python -m wad.vis_dev gap`,
  `results/vis2/night_gap.csv`; it looked at 24 BDD100K **val** images with their v2 light
  predictions, no detection results): v2's synthetic darkness is dimmed daylight (median sky
  luma 62 above road 40, daylight shadows, floating lights), while real clear night has a dark
  sky (21 vs road 32), light pooled by headlights and street lamps, and lower overall luma
  (8–47). A sim-to-real gap in the darkness model, not in the scattering model.

## Version 3: night relighting — `PREREG_visibility.md` (Version 3 + erratum), `results/vis3/vis3.md`

- **Not blind**: designed after v2's results, from the diagnosis above (which viewed 24 val
  images). Darkness is a relighting of the daytime frame: ambient light, ego headlights with
  inverse-square fall-off on metric depth, street-light pools, dark sky, head/tail lights on the
  labelled vehicles, dashcam auto-exposure; k = illumination of driving-zone points 10–60 m away.
  Rendering was tuned against the first 200 of 400 BDD100K *train* night frames (final median luma
  synthetic vs real: mean 22.7 vs 28.1, sky 15.2 vs 16.6, road 28.7 vs 22.2;
  `python -m wad.vis_dev luma`, `results/vis3/dev_luma.csv`).
- Development check on the 400 train night frames (the same frames used for tuning, so not an
  independent check): night vs day ROC AUC 0.991 (v2: 0.868); median k on real night 0.32 (v2:
  0.94). On BDD100K val: AUC 0.987 (v2 0.852); median k clear-night 0.33, rain-night 0.57,
  snow-night 0.41. v3 separates night from day but underestimates how dark it is (synthetic night
  k is 0.002–0.018), so median V_eff at night is 1,180–1,510 m vs 1,930–1,970 m by day, and 59.5% of
  night images are still in the top band (≥ 1000 m).
- Criteria pre-registered for v2, applied unchanged to v3: S3, S4 met; **S1 not met** (BDD100K
  0.373, up from 0.277; v1 0.460; DAWN 0.179 vs v1 0.245); **S2 not met** (rain-night 0.260,
  clear-night 0.220, DAWN fog 0.216). Against v2, v3 improves clear-night (0.220 vs 0.236) but is
  worse for rain-night (0.260 vs 0.248, at 45.4 vs 52.8 FPPI).
- Why rain at night still fails: v3 reads rain-night as *brighter* than clear night (median k 0.57
  vs 0.33). The target itself ignores glare and reflections (k is the scene illumination; light
  cores and wet-road streaks are added after k is fixed), but in the synthesis lamp glare appears
  only together with street-light pools, which raise k. The model has plausibly learned that
  visible lights and bright road reflections mean more light, so real wet-road reflections read
  as brighter. For the detector, glare and reflections are a hazard. A next version would need
  glare/contrast loss in the target (e.g. veiling luminance from light sources).
- Exploratory (`python -m wad.vis_combo`, `results/vis3/exploratory_combo_*.csv`): v3's light level
  alone reaches Spearman 0.437 on BDD100K (v1 0.460), but that is mostly "is it night": within day
  images v1 0.091, v3 0.028, light 0.017; within night images v1 0.175, v3 0.132, light 0.141. An
  equal-weight rank mean of v1 and v3 does not beat v1 (0.438 vs 0.460 on BDD100K, 0.226 vs 0.245
  on DAWN).

### What the visibility work says, in one paragraph

A trained, physics-labelled estimator measures the scattering it is built to measure: fog, haze
and downpour veil (median MOR on the evaluation images, v2/v3: DAWN fog 208/180 m, sand 216/196 m,
rain 2.1/2.5 km, snow 2.9/4.0 km, BDD100K clear day 9.5/9.7 km; the lowest DAWN rain scores are
downpours with spray). Night needed a second, not-blind version (v3) and is now separated from
day, but its darkness is underestimated. On DAWN the v2/v3 thresholds reach the same overall miss
as v1's at lower FPPI (4.05 vs 6.37), though each score protects a different set of images. As a
predictor of *detector* difficulty, the simple contrast + detail score v1 remains better on both
datasets and within day and within night, and no visibility score, simple or trained, holds the
budget for rain at night, where per-weather (oracle) thresholds do it at ~117 FPPI. Physical
visibility is one cause of misses; v1 measures image contrast and fine detail directly, which is
closer to what the detector needs, though it too misses rain at night (0.238).

# Threshold work at a glance (2026-10-12/14)

**The method: a dial λ (false alarms accepted per avoided miss), applied as one confidence threshold
per λ or per detection (keep iff the model's chance ≥ 1/(1+λ)), both fitted on BDD100K train** (the
two DECISION sections at the end of this file; `wad/dial.py`, `wad/one_threshold.py`,
`wad/per_detection.py`; README "Threshold").
The sections below follow the order the work was done; everything except the decision is
exploratory and kept so its numbers reproduce:

1. Per-image threshold from pixels (PREREG v1): captured ~none of the oracle's headroom.
2. What lowering the threshold buys per condition, and a stop / slow-down graded response.
3. Per-image threshold from the confidence profile (PREREG v2): no better than one threshold.
4. Per-image threshold from YOLOv8x's own features (PREREG v3): pilot 1.01–1.06× one threshold's cost;
   full run on all 21,831 train images 1.00–1.03× on BDD100K val, 0.99–1.05× on DAWN (E1–E3 not met).
5. Luck vs structure: ~90% of the per-image oracle's edge on BDD100K is luck.
6. Per-detection rescoring is 6–8% cheaper than one threshold on BDD100K, 4–7% on DAWN (became the dial's
   per-detection option; with YOLOv8x's own per-box features, model 2, 4–13% cheaper on BDD100K);
   softmax / noisy-OR box combinations add nothing; the summed-confidence map is worse than thresholding.
7. The dial page, label-free false-alarm estimates, and the decision.

# Learned per-image threshold with a cost dial — 2026-10-12 (`PREREG_threshold.md`, `results/threshold/threshold.md`)

- Idea (user's): no single ideal threshold exists; the right recall/precision trade-off depends on
  the use case (camera set-up, redundancy, weather). One ResNet-18 sees the image (pixels only)
  and predicts, at 40 thresholds, the detector's missed objects and false positives; a dial λ
  ("false positives accepted to avoid one miss") gives each image its cheapest threshold. CRC can
  turn a target miss rate α into a calibrated λ (overall, or per condition).
- Headroom first (`wad/threshold_headroom.py`, in-sample, ground truth known): at mean image miss
  0.2 a perfect per-image threshold needs 9.2 FPPI on BDD100K val vs 28.0 for one threshold (night:
  19.9 vs 66.1) and 0.33 vs 2.24 on DAWN; per-weather thresholds chosen for the same aggregate goal
  gain almost nothing (26.8, 2.09).
- Training: 21,831 BDD100K *train* frames (12,383 fetched; 9,000 night, 2,000 dusk), labelled by
  YOLOv8x @1280 with grouped matching (~60 min GPU detection); best validation epoch 6 of 15
  (validation loss then rose: the net memorises).
- Pre-registered verdict: **P1 not met** (BDD100K α = 0.2: 27.82 vs 28.00 FPPI, 1% fewer, miss
  0.198), **P2 not met** (DAWN: 2.34 vs 2.38), **P3 met** (per-condition calibrated dial,
  rain-night: miss 0.195 at 108 vs 117 FPPI for per-weather thresholds).
- Why: the network predicts *how many* objects an image will lose (Spearman 0.57 with true misses
  on BDD100K), mostly by counting, but not *where* in score space its true and false detections
  sit, which is what sets the right threshold. At the calibrated λ it uses only 10 distinct
  thresholds over 8,090 BDD100K images (6 on DAWN) while the per-image oracle spans 0.002–0.58;
  Spearman of learned vs oracle threshold 0.19 (DAWN 0.09) (`results/threshold/diagnosis.csv`).
  Its trade-off curve lies on the single-threshold curve (fig13).
- What does work: the dial as a knob. Direct presets on BDD100K val give a clean menu, from
  λ = 1 (recall 0.53, precision 0.86, 1.0 FPPI) to λ = 30 (recall 0.77, precision 0.32, 19.9 FPPI);
  on DAWN from 0.66/0.87 to 0.85/0.49. Calibrated per condition it holds α everywhere (as
  per-weather thresholds do) and costs a little less in rain-night.
- Next options (not done): decision-focused training (learn the chosen threshold's actual cost
  across λ rather than the counts); adding the detector's own score distribution as input (likely
  much stronger, but breaks the brief's rule that no hazard signal comes from detector outputs);
  speed adaptation by condition (user's later idea: drive more conservatively when the miss
  budget is expensive to hold, e.g. rain at night).

# Threshold budget and graded response — 2026-10-12 (exploratory; `results/threshold/`)

## What lowering the threshold buys, per condition (`python -m wad.threshold_budget`)

User's hypothesis: night / low-visibility images have more room to lower the threshold. In-sample
on BDD100K val and DAWN (YOLOv8x @1280), bootstrap intervals in `budget_bands.csv`:
- **Night has about twice as much to recover, at the same or lower price per object.** Lowering
  0.25 → 0.1 recovers 12.3 recall pts at night vs 8.2 by day for 2.7 vs 2.9 false positives per
  recovered object; 0.1 → 0.05: +6.9 vs +3.7 pts (6.4 vs 7.8 FP each); 0.01 → 0.001: +8.2 vs
  +3.8 pts (114 vs 207 FP each).
- The detector is much less confident at night: median matched score 0.28 vs 0.62 by day; 39% of
  night objects are found only below 0.25 (day 22%); 17% never (day 12%) (`budget_scores.csv`).
- **Rain at night is the expensive exception** (3.3 FP per recovered object at 0.25 → 0.1; 42 at
  0.02 → 0.01), consistent with glare and wet-road reflections. **DAWN fog is the cheapest** (2.1).
- For the same dial λ the best threshold differs little between conditions (λ = 10: night 0.045,
  day 0.053, DAWN fog 0.032; `budget_optimum.csv`): conditions differ much more in what lowering
  recovers than in where the best cut lies.

## Graded response from confidence (`python -m wad.graded_response`)

User's idea: use confidence to choose between stopping and slowing down. Stop tier: score ≥ τ_stop
(calibration false stops per image ≤ cap); slow tier: τ_slow ≤ score < τ_stop, τ_slow from CRC
(mean image miss ≤ α). 200 held-out halves; cap 1, α = 0.2 on BDD100K val:
- One threshold that stops for everything: 28.0 false stops per image to hold the miss budget.
- Two tiers, global (τ_stop 0.33, τ_slow 0.012): 0.97 false stops and 27 false slow-downs per
  image; overall miss 0.199, but night 0.249 (over budget), day 0.154.
- Two tiers, per condition: every condition holds α (0.192–0.198) with ~0.95 false stops per image
  everywhere; night needs τ_slow 0.004 (day 0.037), so the extra caution goes to cheap slow-downs:
  39% of night objects are caught only by the slow tier (day 18%), at 61 false slow-downs per image
  at night vs 9.3 by day (rain-night 116). Slow-tier precision: day 0.21, night 0.06, rain-night 0.03.
- DAWN, per condition: fog needs the most caution (τ_slow 0.025, 11.5% of objects slow-tier only,
  5.2 false slow-downs per image vs 1.6 rain, 0.6 snow).
- Caveat: detections are not filtered to the driving path; a planner would react to far fewer.
  The comparison across conditions is the result: holding the same miss budget in hazardous
  conditions means slowing down far more often, mostly for nothing, but rarely stopping falsely.

# Decision-focused threshold from the detector's confidence — 2026-10-13 (`PREREG_threshold.md` v2)

- User's direction: learn per image which threshold the detector does best at, for a tunable
  trade-off, from the detector's own behaviour (no pixels, no hand-picked scene measurements).
  An MLP reads the image's confidence profile (detections per class group above each of 40
  thresholds, top scores, box sizes by score band) and the dial λ, and is trained on the cost of
  its pick, λ·misses + false positives (plus a soft-oracle cross-entropy after the first run
  saturated onto one threshold; amendment 2).
- Result: **no better than one threshold**, even on its training images (1.001–1.009× the best
  single threshold's cost; the per-image oracle 0.80–0.85×). On BDD100K val 1.005–1.017×, DAWN
  0.97–1.09×; its picks barely track the oracle (Spearman 0.06–0.26). D3 not met.
- Why (`results/threshold/dfl/calibration_by_condition.csv`): YOLOv8x's confidence means nearly
  the same thing in every condition. Share of detections that are real at score 0.1–0.25: day
  0.264, night 0.277, rain-night 0.239, DAWN fog 0.327, DAWN rain 0.223; at 0.25–0.5: 0.583,
  0.568, 0.525, 0.585, 0.464. For a fixed trade-off the expected-cost-best threshold is where the
  share real of what a lower threshold admits falls below 1/(1+λ), which is then about the same
  for every image. The per-image oracle wins only by knowing which particular detections turned
  out real, which nothing observed beforehand predicts. Conditions differ in how much is
  recoverable (night has ~2× more), not in what a given score is worth; equalising the miss rate
  across conditions (per-condition CRC, graded response) therefore means accepting more false
  positives per recovered object in hazardous conditions, by choice.
- Exceptions worth following: DAWN rain (and, weaker, BDD100K rain-night), where a given score is
  less often real (glare, reflections, spray), are where richer per-detection context could help.

# Oracle threshold from YOLOv8x's own features: pilot — 2026-10-13 (`PREREG_threshold.md` v3)

- User's direction: predict, from the input YOLOv8 itself is trained on (the image, read the way
  YOLOv8 reads it), the per-image threshold at which the detector does best for a tunable
  trade-off. Each image goes through YOLOv8x's frozen backbone and neck (same preparation as
  detection, verified to reproduce the cached detections); its three feature maps are pooled to
  10,880 numbers; a head maps them (and the dial λ) to a threshold, trained on the cost of its pick
  plus a soft-oracle target (`wad/thresh_yolo.py`).
- Pilot first (user's request): 1,000 random BDD100K train images (random, not the literal first
  1,000, which would all be clear daytime), checked on 1,000 random BDD100K val images and DAWN.
- The pre-registered head ignored the dial (its median pick stayed at grid step 26 for λ 1–100,
  the best single threshold moves 27 → 8): λ was 1 of 10,881 inputs. A structured head fixes this:
  it predicts each image's miss and false-positive curves and applies the dial exactly.
- Result (after an independent verification; `results/threshold/yolo/pilot_steps_matched_curves/`):
  cost 1.01–1.04× one threshold on BDD100K val, 1.02–1.06× on DAWN; the per-image oracle is
  0.80–0.86 / 0.71–0.75. No better than the pixel CNN (v1) or the confidence profile (v2) on the
  same images. With 10,880 features and 900 images the head overfits (predicted vs true misses:
  Spearman 0.94 on its training images, 0.43 held out); given the true curves as input, the same
  pipeline reaches 0.88–0.96×, so the code can learn when the signal is there.
- Open: the full run (21,831 training images, ~80 min of GPU extraction; 2,500 BDD100K val images
  already extracted and resumable) would show whether more data lets YOLOv8x's features beat one
  threshold. Versions 1–2 at full size did not.

# Oracle threshold from YOLOv8x's own features: full run — 2026-10-14 (`PREREG_threshold.md` v3 full run)

The user asked for the YOLOv8x-feature threshold model trained on all the data (plan registered in
PREREG before the run). Features of all 21,831 train / 8,090 BDD100K val / 1,005 DAWN images
(extraction resumed at val row 2,500; ~67 min GPU; `cache/thresh/yolo_feats_*.npy`, 0.67 GB); both
heads trained in seconds (`results/threshold/yolo_curves/`, `results/threshold/yolo/`,
`yolo_full_run.log`).

- Structured curves head (primary), cost vs the best single threshold on the same images, λ = 1 / 3
  / 10 / 30 / 100: BDD100K 1.005 / 1.001 / 1.005 / 1.022 / 1.028; DAWN 1.054 / 0.994 / 0.990 / 1.025 /
  1.030. **E1 1.005, E2 0.990 (95% interval 0.976–1.009), E3 0.217: none met.** The head as first
  registered (dial as an input): E1 1.068, E2 1.161, E3 0.149.
- Verified independently (three agents): numbers recomputed outside the module agree to CSV
  rounding; no train/val/DAWN overlap (names, sha256, perceptual hash); split by near-duplicate
  group; standardisation and checkpointing on the fit / train-validation split only; feature rows
  around the resume point re-extracted on the GPU bit-exact; PREREG note written before the run.
- Why it fails: more data cut the memorising (predicted-vs-true misses Spearman on fit images 0.93 →
  0.85) but held-out ranking of misses barely moved (0.45 → 0.48; an image's object count alone ranks
  them at 0.65). False positives generalise well (0.78 held out); misses are the bottleneck. On
  BDD100K val the head acts like one constant threshold set 1–3 grid steps too high (its predicted
  curves are not calibrated counts); its image-to-image variation is worth −0.001 at λ = 10. It does
  lower the threshold at night (as the per-condition best does), but that is worth < 0.5%: even one
  in-sample best threshold per weather is only 0.997× one threshold on BDD100K. Per condition nothing
  gains reliably (best: DAWN snow 0.960, interval 0.935–1.006; DAWN fog significantly worse, 1.032).
- Conclusion: confirms the decision for one threshold per dial setting.
- Erratum (pilot): the pilot's "0.94 on its training images vs 0.43 held out" came from the checkpoint
  before the verification fixes; the saved pilot checkpoint (`weights/thresh_yolo_pilot_steps_curves.pt`)
  gives 0.933 vs 0.445. Same conclusion.

# Is the right threshold learnable? Luck vs structure — 2026-10-13 (exploratory)

- `python -m wad.threshold_noise` (`results/threshold/noise_floor.csv`): keep each image's
  detections and scores, redraw which are real from the dataset-wide chance at that score and class
  (no per-image structure), and see how much the per-image oracle threshold still gains. On BDD100K
  val 87–91% of the oracle's 15–19% advantage over one threshold is luck; at most ~1–2% of cost is
  learnable structure for any one-threshold-per-image policy (DAWN: 70–89% luck, 3–9% learnable).
  Adding the condition label to the null explains almost none of the rest. This is why every
  per-image threshold model (pixel CNN, confidence profile, YOLOv8x features) matched one threshold.
- The luck is about which detections are real, so the learnable pattern is per detection.
  `python -m wad.detection_rescore` (`rescore_{detections,decision,importance}.csv`): a
  gradient-boosted model on the detector's own outputs per detection (score, class, box size,
  shape and position, rank, overlap with stronger boxes, confidence-weighted support from all
  overlapping boxes, image crowding), trained on BDD100K train detections, keeps a detection iff its
  estimated chance of being real ≥ 1/(1+λ). Cost vs one threshold: BDD100K val 0.92–0.94, DAWN
  0.93–0.96 (score + class only: 0.99–1.00). In the 0.05–0.5 score band, AUC real vs false 0.75 →
  0.82 (BDD100K), 0.73 → 0.80 (DAWN).
- Not just duplicate cleanup: without the overlap / support features BDD100K still reaches
  0.93–0.96 (DAWN 0.96–0.99, where duplicates matter more). Strongest cues after the score: overlap
  with a stronger box of the same group, horizontal position, aspect ratio, width, top edge, class,
  height (box plausibility). The summed-area "support" feature adds little beyond the overlap cue.
- Next (not done): the stochastic version of the user's Monte-Carlo idea needs a source of
  variation — test-time augmentation (flips, scales, small shifts: does a box reappear?) or video
  frames (does it persist?) — added as per-detection consistency features; and YOLOv8's own
  features around each box instead of hand-made box geometry.

## Performance at every threshold from one detector run

The detector runs once at the confidence floor (0.001) and every threshold is computed from the
cache by sorting scores and cumulative sums (`Cache._count_ge`): all 200 thresholds × 8,090
BDD100K val images in 0.2 s, versus ~67 GPU hours of re-runs. This is exact, not approximate:
greedy NMS lets only higher-scoring boxes suppress lower ones, and greedy matching is score-ordered,
so filtering the floor run at τ equals running at τ. Checked: re-running YOLOv8x at τ = 0.05 / 0.25
/ 0.5 on 50 BDD100K val images reproduces the cached boxes filtered at τ exactly (817 / 427 / 285
boxes, IoU 1.0, same class, score difference 0; `tests/test_threshold_equivalence.py`, GPU). The
only caveat is the 300-box cap, which can drop the lowest-scoring boxes at the floor. Extra runs
are needed only when the input or model changes (test-time augmentation, input size, NMS IoU).
- The threshold never changes a box's size or position: it only keeps or drops boxes (hard NMS, no
  box merging; re-runs at τ give identical boxes). It changes the mix of boxes kept. On BDD100K
  val, real detections at score ≥ 0.5 have median height 60 px and median IoU 0.86 with their
  object; at 0.1–0.25, 25 px and 0.66; at 0.001–0.01, 26 px and 0.62. Lowering the threshold adds
  smaller, less precisely placed boxes, near the IoU 0.5 matching limit. (Methods that merge
  overlapping boxes, such as weighted box fusion, would make box shape depend on the threshold.)

## Combining overlapping boxes: softmax (competitive) vs sigmoid-like (cooperative) — exploratory

- `python -m wad.show_boxes`: at essentially zero (cut-off 1e-6, cap 10,000) YOLOv8x puts
  2,300–2,900 boxes on an image (BDD100K ~100 at 0.001, 5–12 at 0.25); every object in the night
  example has a box, and confidence summed over all boxes concentrates on the objects (fig16).
- YOLOv8 scores each class with an independent sigmoid (non-competitive). Across boxes, two
  overlap-weighted combinations were added (`wad/detection_rescore.py`): a cooperative noisy-OR
  1 − (1 − sᵢ)·Π(1 − IoUᵢⱼ·sⱼ) ("is something there?") and a competitive softmax share among
  overlapping boxes ("is this THE box?"), plus a learned sigmoid on top (logistic regression on own
  score, noisy-OR, share, class; weights 0.69 / 0.49 / 0.26).
- Per-box decision cost vs one raw-score threshold: competitive 1.02–1.12 (BDD100K), 0.99–1.09
  (DAWN); cooperative 1.03–1.32 / 1.06–1.41; both 1.01–1.02 / 0.99–1.02; learned sigmoid 0.99–1.00 /
  0.99–1.02. As extra inputs to the context model they change nothing (0.92–0.94 / 0.93–0.96).
- Why the cooperative version loses here: many low-score "false positives" are partial or extra
  boxes on real objects (fig16 rain night). Per-box matching allows one box per object, so they
  count as false although an object is there. Noisy-OR answers "is something there?", so it needs a
  region-level evaluation (false alarm = kept region with no object; miss = object covered by no
  kept region), which is closer to what a planner needs. Not done yet.

## The summed-confidence map vs ordinary thresholding, region level (`python -m wad.summed_map`)

- From every detection ≥ 0.001, three maps per image (4-px cells): max (cutting it at τ is exactly
  ordinary thresholding), sum of confidences (the user's method) and noisy-OR. Scored as a planner
  would: an object is found if ≥ half its box is covered; a false alarm is a connected covered
  region touching no object; methods compared at equal object recall (`summed_map_*.csv`, fig18).
- At 80% of objects found, false-alarm regions per image (max / sum / noisy-OR): BDD100K day 0.24 /
  0.38 / 0.34, night 1.18 / 1.43 / 1.42, rain night 1.87 / 2.26 / 2.22; DAWN fog 0.41 / 0.43 / 0.42.
  Ordinary thresholding is best or tied at every recall level and condition; the curves meet at
  very high recall. Summing does cover 10–17% less wrong area on BDD100K at the same recall (night
  1.9% vs 2.3% of the image): where boxes agree, its regions hug the object more tightly. No
  difference on DAWN.
- Why: weak boxes also pile up on clutter (lamps, reflections, the hood; fig17), and summed they
  reach the level of a single moderate real detection, while real objects mostly already have one
  strong box after non-maximum suppression, so agreement adds little there.

## The trade-off dial in action (`python -m wad.dial_demo`, `results/threshold/dial_demo/index.html`)

Interactive page for the dial λ (false alarms accepted per avoided miss), both options: one
threshold fitted on BDD100K train and used in every condition, or per detection (keep iff the
model's chance ≥ 1/(1+λ)); four conditions (BDD100K day, night, rain at night; DAWN fog), two example
frames each. Since 2026-10-14 every number on it comes from `wad.per_detection.grade`, the function
behind `per_detection_eval.csv` (checked: all preset rows equal the CSVs to rounding), plus exact
in-sample best constants, kept counts and precision-recall curves; the presets sit exactly on the
slider (λ = 1, 10, 30, 100; before, they snapped to 0.95 / 9.5 / 30 / 95). The readout shows the
calibration at the cut for the condition (boxes within ×1.5 of the threshold or cut: share real vs
what the dial assumes, flagged when they differ by more than 25%), cost ratios with 95% intervals,
and no tuning (0.25). At λ = 10, recall / precision / false alarms per image:

- BDD100K day: one threshold (0.0541) 77.5% / 55.7% / 8.45; per detection 78.5% / 59.8% / 7.26
  (cost ×0.935, 0.929–0.940); no tuning 65.7% / 84.1% / 1.70 (×1.240).
- BDD100K night: 62.1% / 42.1% / 8.76; per detection 65.7% / 41.4% / 9.53 (×0.939, 0.933–0.944);
  no tuning 43.4% / 76.6% / 1.36 (×1.249).
- BDD100K rain at night: 58.4% / 38.7% / 8.96; per detection 61.6% / 38.1% / 9.69 (×0.951,
  0.935–0.969); no tuning 41.3% / 74.6% / 1.36 (×1.183).
- DAWN fog: 78.6% / 65.3% / 2.80; per detection 78.4% / 72.5% / 1.99 (×0.957, 0.922–0.993); no
  tuning 68.1% / 85.9% / 0.75 (×1.290). The page flags fog: boxes near the 9.1% cut are real 15.4%,
  so there the per-detection option leans to precision.
(Before 2026-10-13 the page's one-threshold view used each condition's own in-sample best.)
The per-detection numbers in this section are model 1's; since 2026-10-15 the page uses model 2
(DECISION 2026-10-15 section), and it is rebuilt from `per_detection_yolo_*` (all rows checked against the CSVs).

Frame scorecard (2026-10-15, user's request): for the selected example frame the page shows the dial's
precision, recall, F1 and Fβ against the best each could reach on that frame with hindsight (the best
'keep value ≥ t' on the frame's own labels), the gap, which boxes the hindsight choice keeps (drawable),
why the frame is off (objects dropped by the setting vs never detected at any score; false alarms on an
object, loose, or on background), the frame's F curve across thresholds, and how the gap compares with
every frame of the condition (`frame_gaps` in `wad/dial_demo.py`; checked against brute force and, on all
8 example frames, against an independent recomputation). Per-frame F1 at the F1 dial (β = 1), median gap
to each frame's own best and share within 0.05, one threshold / per detection: day −0.046 / −0.042,
53% / 57% (mean per-frame F1 0.736 / 0.763 vs 0.798 / 0.818 with hindsight); night −0.075 / −0.069, 35% /
38%; rain at night −0.073 / −0.083, 34% / 36%; DAWN fog −0.023 / −0.025, 57% / 58%. At λ = 10 the same F1
gaps look about twice as large (median −0.08 to −0.15), but most of that is a different goal: λ = 10 aims
for more recall than F1 rewards (F1 ≈ λ 1.9). Night frames are furthest from their own best, as the
earlier analyses found, and that per-frame gap is mostly luck rather than something a per-image rule can learn.

## Knowing false alarms without the image's labels (`python -m wad.label_free_fp`) — exploratory

- A threshold never knows which box is false. From labelled BDD100K **train** detections it can
  learn the chance a box at each score is real (isotonic fit): score 0.05 → 8%, 0.1 → 17%,
  0.25 → 42%, 0.5 → 74%, 0.75 → 97%. The best single threshold for dial λ (chosen on train) is
  exactly the score where that chance is 1/(1+λ): 0.306 / 0.054 / 0.019 for λ = 1 / 10 / 30 vs
  0.307 / 0.054 / 0.019 from the calibration (`label_free_calibration.csv`).
- Expected false alarms on an unlabelled frame = sum of (1 − chance real) over kept boxes. Checked
  on val (labels used only for the check), λ = 10, one threshold, estimated vs actual per image:
  day 8.46 vs 8.45, night 8.90 vs 8.76, rain at night 8.76 vs 8.97; DAWN fog 2.94 vs 2.80, sand
  3.16 vs 3.16, snow 5.25 vs 5.36, rain 3.75 vs 4.33 (DAWN rain under by 13%, 38% at λ = 1: the
  known calibration exception). Per frame it ranks images by false alarms better than counting
  kept boxes (Spearman 0.94 vs 0.84 on day, 0.96 vs 0.93 at night).
- Real objects among dropped boxes = sum of chance real over them: close on day (1.55 est vs 1.38),
  too optimistic at night (1.62 vs 2.11) and in fog (0.40 vs 0.68). It cannot see objects that got
  no correct box even at 0.001: 12% of objects on day, 17% at night, 18% in rain at night, 11%
  in DAWN fog (YOLOv8x @1280, grouped matching).
- Per box it stays odds: among the boxes kept at λ = 10 half are false on BDD100K and the score
  separates real from false with AUC 0.90 on day, 0.84 at night (per-detection model 0.91 / 0.85).
  The per-detection model's probabilities pass the same check (`label_free_fp.csv`).

## DECISION (2026-10-13): one threshold with a dial (`python -m wad.one_threshold`)

The threshold method going forward is the simplest one: **one constant confidence threshold per dial
setting λ, the same in every frame and condition**, chosen on BDD100K train. Reasons: per-image
threshold models (v1–v3) tie with it at best; the per-detection rescorer is 6–8% cheaper (BDD100K; 4–7% on DAWN) but needs a
second trained model; one threshold is a single number per setting, trivially computed and audited.
The dial still adapts it to the use case (and λ can be set per condition, e.g. more caution in fog).
Per-image, per-detection and summed-map work above stays as exploratory comparison.

- Fit: for each λ the threshold minimising λ·misses + false alarms over the 3.34M train detections
  (21,831 images): λ = 1 / 3 / 10 / 30 / 100 → 0.306 / 0.151 / 0.054 / 0.019 / 0.0057
  (`one_threshold_dial.csv`). It is the score where a box is real 1 in (1+λ) times on train.
- Stable: 200-resample image bootstrap 90% band at λ = 10: 0.051–0.056; 5-fold inside train, the
  threshold fitted on four folds costs at most 0.2% more on the fifth than that fold's own best
  (`one_threshold_folds.csv`).
- A win over no tuning (Ultralytics' default 0.25, which is the dial's choice at λ ≈ 1.4): at
  λ = 10 no tuning costs 1.24× on BDD100K val (day 1.24, night 1.25, rain at night 1.18) and 1.31× on
  DAWN; at λ = 30 1.55× / 1.65×; at λ = 1 about 1.01×. Concretely at λ = 10, all BDD100K val: recall
  71.3% vs 56.7% at the cost of 8.6 vs 1.5 false alarms per image; DAWN 82.3% vs 72.1% (3.8 vs 1.2).
- One threshold per weather (each fitted on its own train weather) gains nothing: 0.98–1.03× the
  one threshold across conditions and settings, i.e. noise. Even each condition's own in-sample best
  constant is at most 1–3% cheaper on BDD100K and 5% on DAWN fog at λ = 30 (`one_threshold_eval.csv`,
  fig20). Consistent with YOLOv8x's confidence being about equally calibrated across conditions.

## DECISION (2026-10-14): the dial has two options, one threshold and per detection (`python -m wad.per_detection`, `wad/dial.py`)

User's framing: the goal is not to beat a universal threshold such as 0.25 on F1, but to replace an
arbitrary, set-once cut-off (changed, if at all, by guessing) with a dial in meaningful units, λ =
false alarms accepted per avoided miss, where every setting is statistically backed. So both ways of
applying the dial are supported: one threshold per λ (`wad/one_threshold.py`) and per detection,
keep a box iff the model's chance it is real ≥ 1/(1+λ) (`wad/per_detection.py`, model saved as
`weights/per_detection.joblib`; same model and features as `wad/detection_rescore.py`'s context model,
predictions identical). `wad/dial.py` applies either to new detections (one threshold from the exact
switch points of the fit, `weights/one_threshold_table.csv`, 258 rows since the 2026-10-15 tie fix, so any λ > 0 gives the fitted
threshold; inputs validated: class names, image sizes, rows per image, λ > 0).

- Calibration on images never used in training (`per_detection_calibration.csv`): expected
  calibration error 0.12 percentage points on all of BDD100K val, 0.15 by day, 0.21 at night,
  0.19–0.28 per weather; DAWN 0.9–1.8 (fog worst). These averages are diluted by the half of the
  boxes with a chance below 0.5%; among boxes with a chance of at least 1% they are 0.29 points on
  BDD100K and 4.0 in DAWN fog. So the check that matters is right at the cuts
  (`per_detection_cuts.csv`, boxes within ×1.5 of the cut), at the graded settings λ = 1 / 3 / 10 /
  30 / 100 (the four named presets plus 3): cut 50 / 25 / 9.1 / 3.2 / 1.0%, BDD100K real 53.0 / 25.1 /
  9.7 / 3.5 / 1.1%. Day and night are close (λ = 30: day 3.3%, night 3.7% real; night slightly
  under-stated). So on BDD100K a dial setting means what it says. DAWN fog under-states the chance
  (boxes near the 9.1% cut are real 15%): in fog the option drops some boxes it should keep, i.e. it
  errs toward the precision side there.
- Stability, 5-fold by near-duplicate group inside train (`per_detection_folds.csv`): held-out
  calibration error 0.06–0.09 points; held-out cost vs the one threshold fitted on the other folds
  0.915–0.949; the five fold models on BDD100K val 0.918–0.943 and DAWN 0.920–0.949.
- Grading vs one threshold (both from train; `per_detection_eval.csv`, fig21; 95% image bootstrap):
  per detection costs 0.90–0.97× in every BDD100K condition and graded setting (every interval
  below 1) and 0.90–0.97× on DAWN (intervals below 1 except fog and sand at λ = 100). E.g. λ = 10, all BDD100K val:
  recall 73.4% vs 71.3% with 8.3 vs 8.6 false alarms per image (0.937, 0.933–0.940); day 78.5% vs
  77.5%, 7.3 vs 8.4 false alarms; DAWN 82.9% vs 82.3%, 3.2 vs 3.8. No tuning (0.25) costs 1.01–2.08×
  on BDD100K and 0.99–2.35× on DAWN (DAWN all 1.01–2.14×); it ties only near λ = 1.5, where 0.25 is
  the dial's own choice.
- Caveat: each detection keeps its real / false label from matching all boxes ≥ 0.001; if the model
  drops a matched box but keeps a weaker duplicate, the duplicate still counts as false, so the
  per-detection numbers are slightly pessimistic (re-matching the kept boxes moves the ratio by at
  most 0.001). The model's early stopping uses a random 10% of train detections (train only), and
  its features and settings were picked in the exploratory `detection_rescore` runs that were scored
  on val / DAWN (alternatives within ~0.005): the fit uses train only, that earlier choice did not.
- Verified independently (three agents): every number above recomputed outside the module (to CSV
  rounding, own bootstrap and own 5-fold retrain); no leakage; features bit-identical between
  training and `dial.keep` on raw detections (all 1.18M BDD100K and 49,814 DAWN detections); the
  rule p ≥ 1/(1+λ) is the expected-cost-optimal one.

## The dial in F-score units, and should a model be trained for Fβ? (`python -m wad.fbeta_dial`) — 2026-10-14

User's request: a variant of the dial set by β (how many times more recall counts than precision,
Fβ = (1+β²)PR/(β²P+R)) instead of λ, and whether a separate model should be trained for Fβ.

- Fit (train only): for each β the one threshold with the highest Fβ over all train detections, and
  the per-detection cut with the highest Fβ on 5-fold out-of-fold chances
  (`cache/thresh/per_detection_oof_train.parquet`, now saved by `per_detection`); table
  `weights/fbeta_table.csv` (401 settings, β 0.25–4), `dial.keep_f`. β = 0.5 / 1 / 2 / 4 → threshold
  0.412 / 0.196 / 0.082 / 0.024, per-detection cut 62.9 / 36.3 / 13.8 / 4.3%. Every β is a λ setting,
  λ = (1+β²)/F* − 1: F0.5 ≈ λ 0.6, F1 ≈ λ 1.9, F2 ≈ λ 6.4, F4 ≈ λ 22 (so "F1" means accepting about two
  false alarms per avoided miss); exact for one threshold, approximate for per detection (cut 0.363 vs
  1/(1+λ) = 0.353 at F1). β ≤ 0.33 maps below the λ dial's range (0.3). Image bootstrap of the train choice (`fbeta_stability.csv`), 90%:
  F1 threshold 0.194–0.215, cut 35.4–38.2%.
- Grading (`fbeta_eval.csv`, fig22; 95% image-bootstrap intervals): the one threshold reaches 97.7–100%
  of the best Fβ any constant could reach with the graded condition's own labels (BDD100K 0.983–1.000,
  DAWN 0.977–1.000; at β 0.5–2 at least 0.980). Per detection goes beyond that ceiling: BDD100K
  1.004–1.051×, DAWN 0.98–1.03×. F1 on all BDD100K val: one threshold 0.672 (0.668–0.676), per detection
  0.695 (0.691–0.699), no tuning 0.670. No tuning (0.25) happens to sit near the F1 optimum (it is the
  dial's choice at β ≈ 0.85, λ ≈ 1.4–1.6; F1 itself picks 0.196) but not elsewhere: F2 0.604 vs 0.661 (dial), F0.5 0.751 vs 0.771
  on all BDD100K val; down to 0.69 of the best at β = 4 (clear night).
- Should a model be trained for Fβ? Theory says no: for any β (and any λ) the best rule is a cut on
  the calibrated chance (at F*/(1+β²)), so one calibrated model serves every β and only the cut moves.
  Checked (`fbeta_dedicated.csv`, exploratory): the per-detection model retrained with real boxes
  weighted by β's equivalent λ (cost-sensitive learning), its cut chosen the same way on its own 5-fold
  out-of-fold chances, for β = 0.5 / 1 / 2: Fβ 0.995–1.001× the single calibrated model plus its cut on
  BDD100K val and DAWN, a tie. Its own best cut lands at ≈ 0.51: weighting by λ only moves the
  calibrated chance so that 0.5 becomes the cut, it does not rank boxes better. Better ranking (new
  information: YOLOv8 per-box features, video consistency) is what would help every β and λ at once.
- The dial page has both dials ("Dial in: λ | β"), F1 and Fβ on every setting, the Fβ dial's choice
  shown on the λ dial and vice versa (all 164 β rows checked against `fbeta_eval.csv`).

## DECISION (2026-10-15): the λ dial, Fβ through equivalent λ, and per-detection model 2 with YOLOv8x's own per-box features

User's decision: the method is the λ dial; the Fβ dial is the λ dial at β's equivalent λ; and the
per-detection model should also learn from YOLOv8's own per-box features.

- Fβ through λ (`wad/fbeta_dial.py`, `dial.keep_f` = `dial.keep` at λ_eq): β → λ_eq = (1+β²)/F* − 1 with
  F* measured on train per option; the one threshold there is exactly the Fβ-best threshold, the
  per-detection cut is 1/(1+λ_eq) = F*/(1+β²) (the theory cut; on model 2's out-of-fold chances it
  reaches within 0.0005 of the directly searched cut's train F, median 0.0001).
- Per-box features (`wad/yolo_box_features.py`): YOLOv8x re-run on all 30.9k images exactly as for
  detection; Ultralytics' NMS reports each kept box's anchor, and the head's outputs there are kept:
  the 80 class probabilities, the four box sides' DFL distributions (16 bins each), the neck feature
  vector (channels mean-reduced to 320, as Ultralytics' own `get_obj_feats`) and the pyramid level.
  The re-run reproduced every cached detection (38 of 30.9k images needed box-by-box matching, none
  unmatched). `Detector` returns detections and per-box features for new images.
- Model 2 (pre-registered in `PREREG_threshold.md` before training): model 1's 20 context features
  plus 48 per-box features (six class probabilities, strongest other class, top-1 − top-2 margin, sum
  and entropy of the 80 probabilities, DFL entropy per side and mean, level, 32 neck components from a
  PCA on 200,000 train detections). Same model settings, train only. **All three criteria met**:
  all BDD100K val, λ = 10, cost vs model 1 0.972 (0.969–0.975); DAWN 0.995 (0.981–1.010); share real near
  the preset cuts 1.01–1.13 × the cut. Adopted (`per_detection.ADOPTED = "yolo"`).
- Model 2 against the one threshold (`per_detection_yolo_eval.csv`, fig23): 0.87–0.96× on BDD100K in
  every condition and preset (every interval below 1; model 1: 0.90–0.97×), 0.90–0.97× on DAWN
  (intervals below 1 except rain at λ 1 (upper end 1.000), fog at λ 30 (1.003, borderline) and λ 100,
  sand at λ 100). λ = 10: all BDD100K val
  recall 74.4%, precision 51.4%, 8.49 false alarms per image (×0.910 of one threshold's cost); day
  79.1% / 59.6% / 7.35 (×0.918); night 67.6% / 41.5% / 9.76 (×0.903); rain at night 63.4% / 38.6% /
  9.78 (×0.918); DAWN fog 78.6% / 71.6% / 2.09 (×0.956). Against model 1 the gains are largest at night
  (snow at night ×0.950 at λ = 10, clear night ×0.956 at λ = 30, all night ×0.959 at λ = 30; all
  BDD100K 0.966–0.982); DAWN ties (0.97–1.04).
- Calibration: error 0.11 points on all BDD100K val, 0.11–0.29 per weather; DAWN 0.9–1.5. Near the cuts
  on BDD100K real 52.1 / 25.1 / 9.6 / 3.4 / 1.1% at 50 / 25 / 9.1 / 3.2 / 1.0%. DAWN fog still
  under-states (17% real at the 9.1% cut): there the option leans to precision.
- Stable: 5-fold held-out calibration error 0.05–0.11 points, held-out cost vs one threshold
  0.886–0.933; fold models on BDD100K val 0.896–0.928, DAWN 0.916–0.958.
- What the new features add (log-loss increase when a group is shuffled, BDD100K val boxes scored
  0.05–0.5): context 0.182, class probabilities 0.091, neck components 0.022, box-side uncertainty
  0.004, level 0.001. YOLOv8's own class-probability vector (e.g. car vs truck confusion) carries most
  of the gain.
- The Fβ dial rebuilt on model 2 and through λ (`fbeta_*.csv`, `weights/fbeta_table.csv`): β = 0.5 / 1 / 2
  / 4 → per-detection λ_eq 0.56 / 1.81 / 6.18 / 21.8, cut 64.3 / 35.5 / 13.9 / 4.4%; one threshold
  unchanged (λ_eq 0.60 / 1.92 / 6.45 / 22.6). The λ route gives exactly the Fβ-best one threshold at all
  401 β (after the tie fix below). Image bootstrap of the settings in use (`fbeta_stability.csv`, 90%):
  per-detection cut F0.5 64.2–64.4%, F1 35.5–35.6%, F2 13.90–13.97%; one threshold F1 0.194–0.215. Per detection now reaches 1.006–1.069× the best Fβ any one threshold could
  reach on BDD100K (in-sample), 0.98–1.03× on DAWN; F1 on all BDD100K val 0.701 (0.697–0.705) vs one
  threshold 0.6725 and no tuning 0.670. The dedicated β-weighted check, now with model 2's features,
  ties again (0.997–1.001×).
- Tie fix (found by the verification): many train detections share exactly the same score, and the
  λ → threshold search could pick a point inside such a block, which no threshold can make ('score ≥ t'
  keeps the whole block). `train_curve` now offers only block ends. Effects: the one threshold changed
  at a single dial setting (λ = 169: 0.00303246 → 0.0030329); the exact table has 258 switch points;
  the 5-fold tables moved by at most 0.00004; everything else (all graded CSVs, the page's preset rows)
  reproduced bit for bit.

Housekeeping (2026-10-13): the input-frame caches of finished models (`cache/thresh/frames_*.npy`,
`cache/o3/*frames*.npy`, 17.5 GB) were moved out of the project to
`C:\Users\nicho\Downloads\WAD_removed_2026-10-13\` (manifest there). Their models and outputs are
kept and every number above still reproduces; the commands rebuild the frames if ever rerun.
