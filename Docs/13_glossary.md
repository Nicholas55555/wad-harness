# 13 — Glossary

*Every term, symbol and label used across docs 00–12, alphabetical, with the doc that defines or uses it most. Symbols
come first. Where the same letter means different things in different sources (α, β, λ), each meaning is listed.*

> **In short.**
> - **λ** (the dial): false alarms accepted per avoided miss. **α**: the CRC miss budget. **β**: how many times more recall counts than precision in Fβ.
> - Option A is **one threshold** τ(λ). Option B is **per detection**: keep a box iff its chance p ≥ 1/(1+λ).
> - Run 2's **primary** is YOLOv8x @1280, grouped loss, clean DAWN labels.
> - Verdict words **holds / at the budget / fails** are computed, not chosen.

---

## Symbols

| symbol | meaning | doc |
|---|---|---|
| **α** (CRC budget) | The miss budget: the target expected image-level miss fraction, fixed before any split (0.10 and 0.20 in runs 1–2). | 03, 05 |
| α (AdvGRL) | In DA-Detect, the loss threshold (0.63) below which AdvGRL raises the reversal strength. Unrelated to the CRC budget. | 02 |
| α^cnf, α^loc, α^cls, α^tot | SeqCRC's per-step budgets. Global risk is bounded by α^tot = α^loc + α^cls; α^cnf sits outside it. | 02 |
| **β** (Fβ) | Weight on recall in Fβ: recall counts β times as much as precision. β = 1 is F1; β = 2 is F2 (recall counts double). | 08 |
| β (fog) | Extinction coefficient of a fog veil: transmission t = exp(−β·d). Foggy Cityscapes uses β = 0.02; the visibility renderer uses β = 3.912/MOR. | 02, 06 |
| β (AdvGRL) | Cap (30) on AdvGRL's reversal coefficient. | 02 |
| B | Upper bound of a CRC loss (1 for a miss fraction). It enters the rule as the B/(n+1) term. | 02, 03 |
| C_λ | Cost of a keep rule at dial setting λ: C_λ = λ·M + F, summed over images. | 08 |
| F (false alarms) | Number of kept boxes that match no object. | 08 |
| **F\*** | The best Fβ an option reaches on BDD100K train: the maximum over thresholds (one threshold), or over cuts on out-of-fold chances (per detection). | 08 |
| **Fβ** | (1+β²)·P·R / (β²·P + R). From counts: (1+β²)·TP / ((1+β²)·TP + β²·FN + FP). | 08, 09 |
| F1 | Fβ with β = 1, the harmonic mean of precision and recall. On this detector it corresponds to λ ≈ 1.8–1.9. | 08 |
| **λ** (the dial) | False alarms you accept to avoid one missed object. The user-facing parameter of the method. Presets: 1 "precision first", 10 "balanced", 30 "recall first", 100 "catch everything". | 08, 09 |
| λ (CRC / SeqCRC) | The calibrated parameter in the papers. For the confidence step, λ^cnf = 1 − τ, so the CRC rule's *inf over λ* is a *max over τ*. Not the same quantity as the dial λ. | 02 |
| λ^loc, λ^cls | SeqCRC's localisation margin and class-set parameters (step 2). | 02 |
| **λ_eq(β)** | The dial setting equivalent to an Fβ target: λ_eq = (1+β²)/F\* − 1. F2 ≈ λ 6.2–6.4; F0.5 ≈ λ 0.6. | 08 |
| M (misses) | Number of ground-truth objects with no kept matching box. | 08 |
| m(b) | Miss rate at the confirmed tier's threshold in bin b: (n·R̂(τ_c)+1)/(n+1). A **calibration estimate, not a certified bound**: the held-out miss exceeds it in 38–42% of splits. | 05 |
| n, n_b | Number of calibration images (in bin b). It sets the 1/(n+1) term that makes small bins conservative. | 03 |
| p(d) | Calibrated chance that detection d is real, from the per-detection model. | 08 |
| R̂(τ; b) | Mean per-image miss fraction at threshold τ over the calibration images of bin b. | 03 |
| **τ** | Confidence threshold: keep boxes with score ≥ τ. | 03 |
| τ grid | 200 candidate thresholds: 150 log-spaced on [0.001, 0.5] plus 50 linear on (0.5, 0.99]. | 03 |
| **τ̂(b)** | CRC threshold of bin b: max{τ on the grid : n/(n+1)·R̂(τ;b) + 1/(n+1) ≤ α}. It falls to 0.001 and is flagged *infeasible* when no τ qualifies. | 02, 03, 05 |
| τ(λ) | Option A's threshold for dial setting λ: argmin over BDD100K train of λ·M + F. τ(10) = 0.054. | 08 |
| τ_c(b) | Confirmed-tier threshold: the smallest grid τ with calibration FPPI ≤ c (c = 0.1 or 1.0). | 05 |

## A–Z

**Adjusted Rand index (ARI).** Agreement between hazard bins and the true weather labels (0 = chance). O3 version A
scored 0.074 and version B 0.102 on run 1. *(06)*

**AdvGRL.** DA-Detect's adversarial gradient-reversal layer. Its reversal coefficient rises for "hard" examples:
λ_adv = min(λ₀/L_c, β) if L_c < α, otherwise λ₀. *(02)*

**Anchor (YOLOv8).** The grid cell and pyramid level a box was predicted from. The per-box features are read at the
anchor that produced each kept box. *(08)*

**At the budget.** Verdict word: the mean held-out miss is within 0.01 above α. *(05)*

**BDD100K.** Berkeley DeepDrive dashcam dataset, 1280×720.
- **val** (8,090 images in 7 pre-registered weather × time bins: clear, cloudy, rain and snow by day; clear, rain and snow at night) is used only for grading.
- **train** frames are used for fitting: 21,831 for the dial, 5,443 + 605 daytime for O3 version B, and clear-day frames for the visibility estimators. *(03)*

**Bayes-optimal cut.** With a calibrated chance p, keeping a box costs (1 − p) expected false alarms and dropping it
costs λ·p expected misses. So keep iff p ≥ 1/(1+λ). *(08)*

**Bootstrap (image bootstrap).** Resampling images with replacement to get 95% intervals for cost ratios, Fβ and
differences. Images, not boxes, are the unit, so within-image dependence is kept. *(05, 08)*

**Box-count-recall / box-count-threshold.** SeqCRC's confidence-step losses: the fraction of objects outnumbering
the kept boxes, and a 0/1 indicator of too few boxes. They count boxes and do not locate them. *(02)*

**Calibration (of a model).** Whether predicted chances match observed frequencies: of boxes given 9%, about 9% are
real. Checked with ECE and near-cut shares. *(08)*

**Calibration half / calibration set.** The images CRC fits on. The other half is the held-out test half. *(03)*

**Class-aware loss (L-class).** Matching requires the exact class. Used in run 1. *(03, 05)*

**Class probabilities (feature group).** 10 per-box features: the six target-class probabilities, the strongest
other COCO class, the top-1 minus top-2 margin, and the sum and entropy of all 80 class probabilities. It is the
most useful new group in model 2. *(08)*

**Clean labels.** DAWN with one image kept per near-duplicate group, and every group whose copies carry different
weather labels removed. 889 images and 6,952 boxes (fog 256, rain 187, sand 243, snow 203). Run 2 primary. *(03)*

**Clear-calibrated policy.** BDD100K control: a threshold fitted on clear daytime images and applied everywhere. It
fails every night bin. *(05)*

**cods.** The official SeqCRC toolkit (github.com/leoandeol/cods @ 269ff11), reproduced in Phase 1. *(02)*

**Conformal risk control (CRC).** Choose the parameter so that the calibration risk, inflated by B/(n+1), is ≤ α.
Then the expected loss on an exchangeable test image is ≤ α (Angelopoulos et al., ICLR 2024). *(02, 03)*

**Confirmed / possible tiers.** The tiered output.
- Boxes above an FPPI-capped threshold τ_c(b) are *confirmed* and act.
- Boxes between τ_c(b) and the CRC threshold τ̂(b) are *possible*: tracked, never braking alone.
- m(b) sets speed and gap open loop. *(02, 05)*

**Context features.** The 20 inputs of per-detection model 1, all from the detector's output list:
- logit of the score, class, log height and width, aspect, x-centre, top and bottom edge;
- rank and rank within the class group;
- overlap with stronger boxes, and confidence-weighted support from overlapping boxes;
- image crowding (boxes above 0.25 / 0.05 / 0.01, max score, group count). *(08)*

**Cost ratio.** A policy's cost C_λ divided by the one-threshold cost at the same λ. Below 1 is better. *(08)*

**CRC over λ.** The planned extension: calibrate the dial setting λ itself by CRC per condition, to certify a miss
budget on top of the per-detection operating curve. This works because the keep rules are nested in λ. *(08, 10, 12)*

**DA-Detect.** Li et al. (WACV 2023): a domain-adaptive Faster R-CNN for fog, using AdvGRL plus a triplet
regulariser with a rainy auxiliary domain. Paper A of the literature review and the optional second detector (O5). *(02)*

**DAWN.** Kenk & Hassaballah 2020: real adverse-weather images in four folders (fog, rain, sand, snow).
- 1,027 images found, 1,005 kept after exact duplicates, 7,749 boxes in six classes.
- 889 images with clean labels. *(03)*

**Decision layer.** The planner-facing output, to be specified in O6: what to do with confirmed and possible boxes
and m(b), or with the dial's stop/slow tiers. *(01, 12)*

**Dedicated β model.** A per-detection model retrained with real boxes weighted by λ_eq(β). It ties with the single
calibrated model (0.997–1.001). *(08)*

**Detect once, cache forever.** Run the detector once per image at the 0.001 floor and cache every box. Because
NMS and matching are score-ordered, filtering the cache at τ equals rerunning at τ. *(04, 07)*

**DFL entropy (box-side uncertainty).** YOLOv8 predicts each box side as a distribution over bins (distribution
focal loss). The entropy of each side's distribution, plus their mean, measures how sure the box edges are: 5 features. *(08)*

**Exchangeability / i.i.d.** CRC's assumption that calibration and test images are interchangeable draws. Persistent
weather and consecutive video frames break it. *(02)*

**Expected calibration error (ECE).** The count-weighted mean |mean chance − share real| over probability bins, in
percentage points. Model 2: BDD100K 0.10–0.29, DAWN 0.98–1.49. *(08)*

**Exploratory.** An analysis that was not pre-registered, labelled EXPLORATORY in the outputs. Its numbers inform
but do not count as tests. *(04, 05)*

**Feasible / infeasible.** A CRC fit is infeasible when no threshold certifies α, even keeping every box ≥ 0.001.
τ̂ then falls to the floor and the fit is flagged. *(03, 05)*

**Floor (confidence floor).** The detector runs at score 0.001 (NMS IoU 0.5, at most 300 boxes), so every threshold
above it can be evaluated from cache. *(03, 04)*

**Foggy Cityscapes.** Cityscapes re-rendered with synthetic fog at three densities. DA-Detect's test set. Planned
for O1 and O4 but not run: it needs a Cityscapes account. *(01, 02, 12)*

**Found / recoverable / never matched.** Per object:
- *found*: matched at τ = 0.25;
- *recoverable*: matched only below 0.25;
- *never*: matched at no threshold. The reasons are no proposal, wrong class, loose box, or taken by another object.

12–18% of objects are never matched, which is the detector's ceiling. *(05, 07)*

**FPPI (false positives per image).** Kept boxes that match no object, divided by the number of images. "The price"
of a miss budget. *(03, 05)*

**Frame scorecard.** A dial page panel for the selected frame. It compares the dial's kept boxes, precision,
recall, F1 and Fβ with the frame's **hindsight best**, and gives the gap distribution across the whole condition. *(09)*

**Global CRC.** One CRC threshold calibrated on all calibration images pooled. This is SeqCRC step 1 as published,
and the baseline that fails in fog. *(05)*

**Go/no-go.** O1's gate: GO if, in the worst bin for person and for car, max achievable recall beats recall at the
default threshold by ≥ 15 points. P(GO) is the share of 2,000 bootstrap resamples that give GO. *(05)*

**Graded response (stop / slow tiers).** Exploration #9. A *stop* tier is capped at ≤ 1 false stop per image, and a
*slow* tier is set by CRC at α = 0.2, per condition. Every condition holds α (0.192–0.198) at about 0.95 false stops
per image. *(07)*

**Grouped loss (L-group).** A prediction may match any ground truth in its class group: vehicle {car, truck, bus},
two-wheeler {bicycle, motorcycle}, person. A wrong-class box inside a group counts as a hit. Run 2 primary. *(03, 05)*

**Group-aware split.** A calibration/test split that keeps every near-duplicate group on one side. *(03)*

**Group-conditional validity.** A CRC guarantee within each bin, as in Mondrian conformal prediction. It holds per
*predicted* bin, approximately, not per true weather. *(02, 06)*

**Hazard estimator h.** A per-image score from pixels only (never from detector outputs), cut into bins.
- **Version A:** first principal component of the dark channel, RMS contrast, edge energy and saturation; quartile bins.
- **Version B:** ResNet-18 on BDD100K train daytime weather tags, with h_B = p(rain) + p(snow) + p(fog). *(06)*

**Headroom (oracle per-image threshold).** Exploration #0: the in-sample gain from choosing each image's best
threshold with its labels. 87–91% of it is luck. *(07)*

**Hindsight best.** The best Fβ a single frame could reach with a rule 'keep value ≥ t', choosing t with that
frame's own labels. Not achievable by any deployed setting; it is a ceiling for comparison. *(09)*

**Holds / at the budget / fails.** Computed verdict words: *holds* = mean held-out miss ≤ α; *at the budget* =
within 0.01 above α; *fails* = more than 0.01 above. *(05)*

**Image-level miss fraction.** Per image, the fraction of ground-truth objects with no matched detection at score ≥
τ; 0 for an image with no objects. The loss CRC controls. *(03)*

**k (light level).** Illumination relative to daylight.
- In v2's renderer it is sampled log-uniform on [0.003, 0.3].
- In v3 it is the mean illumination of driving-zone pixels 10–60 m away.
- Predicted median k: BDD100K clear night 0.33, rain night 0.57 (v3). *(06)*

**LAMR (log-average miss rate).** The mean miss rate at 9 FPPI values log-spaced on [0.01, 1] (Caltech
convention). *(03, 05)*

**Luck vs structure (null simulation).** Exploration #5. It redraws which detections are real from the dataset-wide
chance at their score and class, and finds that most per-image headroom is luck. The learnable pattern is per
detection. *(07)*

**mAP@0.5 / AP@50.** Average precision at IoU 0.5, integrated over thresholds. Reported once; it cannot show an
operating point. *(02, 03)*

**Max achievable recall.** Recall when every box ≥ 0.001 is kept, i.e. at the floor. *(05)*

**max_det.** The cap of 300 boxes per image. It binds on a few DAWN images, and on 16–88 images per BDD100K bin with
YOLOv8x @1280. *(03, 04)*

**Model 1 / model 2.** The per-detection models.
- Model 1 uses the 20 context features.
- Model 2 adds 48 YOLOv8x per-box features. It was adopted on 2026-10-15 under pre-registered criteria, at a cost
  0.972 (0.969–0.975) of model 1 on BDD100K val at λ 10.
- Both use `HistGradientBoostingClassifier`. *(08)*

**MOR (meteorological optical range).** The visibility distance in fog or haze. The renderer samples MOR log-uniform
on [30, 5000] m and uses β = 3.912/MOR. *(06)*

**Near-cut share.** Among detections whose chance lies within a factor 1.5 of a cut 1/(1+λ), the share actually real.
If calibrated, it equals the cut. Pre-registered criterion: a ratio in [0.8, 1.25]. *(08)*

**Near-duplicate group.** DAWN images within pHash distance 4 (mirror-aware): 133 pairs in 89 groups, covering 197
images; 11 pairs carry different weather labels. *(03)*

**Neck PCA-32.** YOLOv8x's 320-d neck feature at the box's anchor, reduced to 32 principal components (PCA fitted on
200k train detections). One group of model 2's per-box features. *(08)*

**No-label false-alarm estimate.** Σ(1 − p) over the kept boxes of an unlabelled frame: the expected false alarms.
Close on BDD100K; under-estimated in DAWN rain. A run-time health check. *(07, 08)*

**No tuning (0.25).** Ultralytics' default confidence threshold, the shipped setting. It is near F1-optimal
(λ ≈ 1.4–1.5), and costs 1.2–1.3× at λ 10 and 1.5–1.7× at λ 30. *(08)*

**One threshold (option A).** Keep d iff s(d) ≥ τ(λ). Exact, stored as switch points, and the same in every
condition: per-weather thresholds add nothing (0.98–1.03×). *(08)*

**OOF (out-of-fold).** The chance for each train detection comes from a model fitted on the other 4 of 5 folds,
with folds by near-duplicate group. Used for F\*, λ_eq and stability. *(08)*

**Oracle bin / predicted bin.** An *oracle* bin uses the true weather label; it is the ceiling for O3. A *predicted*
bin uses the hazard or visibility estimator. *(05, 06)*

**Per-bin CRC (oracle-bin policy).** One CRC threshold per weather bin, SeqCRC step 1 run in each bin. **The Phase 3
baseline.** *(02, 05)*

**Per detection (option B).** Keep d iff p(d) ≥ 1/(1+λ), with p from the calibrated per-detection model. 0.87–0.96×
the one-threshold cost on BDD100K. *(08)*

**Per-box features.** The 48 YOLOv8x outputs read at the box's anchor:
- class probabilities (10);
- DFL entropy (5);
- pyramid level (1);
- neck PCA-32 (32). *(08)*

**Post hoc.** Chosen after seeing results. Example: the YOLOv8x @640 grouped variant called "best" in run 2. Always
labelled. *(05)*

**Pre-registration.** Criteria and choices written down before results, in `PREREG_run2.md`, `PREREG_o3.md` (both
2026-10-08), `PREREG_visibility.md` (10-11) and `PREREG_threshold.md` (10-12 onward). *(04, 11)*

**Primary / secondary.** Run 2's pre-registered **primary** is YOLOv8x @1280, grouped loss, clean labels. The
secondaries are the other registered detector × loss × label variants. *(05)*

**Pyramid level.** The FPN level (stride 8, 16 or 32) a box came from. One per-box feature. *(08)*

**Run 1 / run 2.**
- **Run 1** (Oct 1): YOLOv8s @640 on CPU, class-aware loss, all DAWN labels, 10 splits plus a 200-split rerun.
- **Run 2** (Oct 8, pre-registered): YOLOv8s @640, YOLOv8x @640 and YOLOv8x @1280, each with class or grouped loss and all or clean labels, plus the BDD100K control. *(05)*

**S1–S4 (success criteria).**
- O3 (`PREREG_o3`): **S1** = the predicted-bin policy meets α in every weather bin where the global threshold
  fails; **S2** = FPPI gap to oracle bins ≤ 5.
- Visibility (`PREREG_visibility`): **S1** = Spearman ≥ v1's on both datasets; **S2** = holds α = 0.2 in BDD100K
  rain-night and clear-night and DAWN fog; **S3** = overall FPPI ≤ v1's; **S4** = synthetic MAE of log10 V_eff ≤ 0.15. *(06)*

**SeqCRC.** Andéol et al. 2025, *Conformal Object Detection by Sequential Risk Control*. CRC for detection in two
steps: the confidence threshold, then localisation and classification sets. Paper B and the project's **base paper**;
its step 1 is the baseline. *(02)*

**Smallest certifiable α.** The lowest budget CRC can certify for a detector and bin: n/(n+1)·R̂(0.001) + 1/(n+1),
averaged over calibration halves. *(05)*

**Spearman (visibility check).** Rank correlation between a visibility score and the per-image miss at τ = 0.25.
v1: 0.46 on BDD100K and 0.245 on DAWN, the best of v1–v3. *(06)*

**Summed-confidence map.** Exploration #8: confidence summed over all boxes on a 4-px grid, compared with ordinary
thresholding at region level. Not better: weak boxes pile up on clutter. *(07)*

**Switch points (convex hull).** Option A's exact table: the lower convex hull of (real kept, false kept) over
candidate thresholds. Every λ between two consecutive hull slopes picks the same threshold. 258 rows; tie-aware
(candidates only at the ends of equal-score blocks). *(08)*

**V_eff (effective visibility).** min(MOR on the road, V_light), with V_light = 2000 m·k^0.5. The visibility
estimators' target; the bins use −log10 V_eff. *(06)*

**Visibility v1 / v2 / v3.**
- **v1** is a pixel score: −(z(log edge energy) + z(RMS contrast))/2.
- **v2** is a ResNet-18 trained on physics-rendered synthetic fog, rain and darkness over clear-day BDD100K frames,
  using metric depth. It predicts MOR, k and V_eff.
- **v3** is v2 with night relighting (headlights, street lights, dark sky, auto-exposure).
- v1 remains best. *(06)*

**WEDGE.** A synthetic adverse-weather dataset (CVPRW 2023) proposed for the O3 estimator. Not fetched. *(01, 06)*

**Within 0.05 (frame gaps).** The share of a condition's frames whose dial F is within 0.05 of that frame's hindsight
best; reported with gap quantiles. *(09)*
