# Pre-registration: a trained visibility estimator (O3 version C)

Written 2026-10-11, before any synthetic sample was generated, any model trained, or any of its
predictions seen. Honest limitation: the author has already seen how a simple pixel score
(contrast + fine detail, `wad/visibility.py`, "v1") relates to detection difficulty on BDD100K
val and DAWN, which motivated this estimator; the estimator itself, its targets and its
evaluation are fixed below.

## Goal

An **objective** hazard score: the visibility of the scene measured from the camera frame, not a
weather label. It must cover what reduces visibility for a driver and a detector: fog and haze,
the veil of a heavy downpour, spray thrown up by other vehicles' wheels, and darkness (including
glare and wet-road reflections at night, the blind spot of v1). It never reads detector output.

## Training data (synthetic, physics-defined labels)

- Base frames: BDD100K **train**, daytime, weather tag "clear" only: the 1,500 already fetched for
  O3 plus 3,000 more drawn at random (seed 1) from the remaining clear-daytime pool. 90% of base
  frames train, 10% (by frame, seed 0) form a synthetic validation set. No BDD100K val or DAWN
  image is used for training or model selection.
- Metric depth d(x) per base frame from Depth Anything V2 Metric Outdoor Small (fixed weights),
  clipped to [1, 1000] m; depth ≥ 300 m is set to 1000 m (sky).
- Each training sample draws, independently:
  - **Scattering veil** (fog, haze, downpour veil), probability 0.75: meteorological optical range
    MOR ~ log-uniform[30, 5000] m, extinction β = 3.912 / MOR, transmission t = exp(−β d),
    I = J·t + A·(1 − t) in linear light; airlight A grey-white (0.65–0.95), with probability 0.15
    tinted yellow-orange (dust/sand). Otherwise no veil (MOR = 10,000 m).
  - **Rain**, probability 0.35: streak layer (density, length and opacity scale with a drawn rain
    intensity), windshield drops (blurred refracting blobs) with probability 0.5, and **wheel
    spray** with probability 0.6: a local extinction plume (local MOR ~ log-uniform[15, 150] m) in
    an ellipse over the lower part of each labelled car/truck/bus box (ground-truth labels, not
    detections), scaled by box size.
  - **Darkness**, probability 0.35: light level k ~ log-uniform[0.003, 0.3] multiplies linear
    radiance; Poisson-Gaussian sensor noise; partial auto-gain (gain = k^−0.5); light sources
    (street lights / headlights) as Gaussian blooms, with probability 0.5 vertical wet-road
    reflection streaks below them. Otherwise k = 1.
  - Camera nuisances on every sample: JPEG quality 40–95, slight blur, horizontal flip.
- Targets (all log10):
  - MOR of the global veil; light level k;
  - a 7×13 **visibility map**: per cell, MOR_cell = 3.912 / (β_global + mean spray β in the cell);
  - **effective visibility** V_eff = min(MOR_road, V_light), where MOR_road is the 10th percentile
    of the map over the lower two thirds of rows (the driving zone) and V_light = 2000 m · k^0.5
    (clipped to [30, 10000] m). The V_light mapping is a convention that puts darkness on the same
    scale as scattering (k = 0.01 → 200 m); it is not a measured quantity.

## Model and training

ResNet-18 (ImageNet weights), input 224×400; heads on the pooled feature: log MOR, log k,
log V_eff; a 1×1-conv head on the last feature map for the 7×13 map. L1 loss on all log10 targets
(equal weights). AdamW lr 3e-4, weight decay 1e-4, cosine schedule, batch 32, 20 epochs (each
epoch visits every training base frame once with fresh random degradations), seed 0, fp32. The
checkpoint with the lowest synthetic-validation loss is kept.

## Hazard score and bins

- h_v2 = −log10(predicted V_eff). Higher = less visible.
- Primary bins (comparable with v1): quartiles of h_v2 on each calibration half.
- Secondary bins (fixed physical bands, no estimated cut points): predicted V_eff ≥ 1000 m,
  300–1000 m, 100–300 m, < 100 m.

## Evaluation (real images; the estimator never saw them)

Detector: run 2 primary (YOLOv8x @1280, grouped vehicles; DAWN clean labels). α = 0.2,
200 calibration/test splits. Policies: one global threshold, per-weather (oracle) thresholds,
per-visibility v1 quartiles, per-visibility v2 quartiles (and v2 fixed bands). Reported per
condition (BDD100K weather × time of day; DAWN fog/rain/sand/snow and the fog subtypes
foggy/haze/mist), per each score's lowest-visibility quarter, and overall: recall, precision, F1,
FPPI, held-out image miss fraction, share of splits above α.

Success criteria:
- **S1** Spearman correlation of h_v2 with the per-image miss fraction at τ = 0.25 is at least
  v1's on both BDD100K and DAWN.
- **S2** Per-visibility v2 thresholds hold α = 0.2 on average in BDD100K rain-night and
  clear-night and in DAWN fog (v1 failed rain-night at 0.238 and fog at 0.213).
- **S3** Overall FPPI of the v2 policy is no higher than v1's on each dataset.
- **S4** Synthetic validation: mean absolute error of log10 V_eff ≤ 0.15 (about ±40%).
- Sanity (reported, not pass/fail): predicted light level separates BDD100K day from night
  (ROC AUC); predicted MOR orders DAWN fog below DAWN rain/snow below BDD100K clear day.

Anything added after the results is labelled exploratory.

## Amendment (2026-10-11, before any synthetic sample was generated or model trained)

The metric depth model is fine-tuned on Virtual KITTI and saturates near 80 m (on 20 sampled
training frames the per-image maximum is 73–80 m, the 99th percentile 74 m; sky reads ≈ 77 m).
The rule "depth ≥ 300 m is set to 1000 m" would never fire, so it becomes **depth ≥ 70 m is set
to 1000 m** (sky and far background take the full airlight). Consequence: between ~70 m and the
true distance the depth of far road and buildings is underestimated, so long-MOR veils (≥ 500 m)
are rendered somewhat weaker than reality; recorded as a limitation.

## Amendment 2 (2026-10-11, after a visual preview of synthetic samples, before training)

A preview of 24 synthetic samples (no model, no evaluation data) showed two rendering artifacts:
(1) the hard "depth ≥ 70 m → 1000 m" rule produced blotchy full-airlight patches where the
depth map jumps across 70 m on trees and buildings; (2) light blooms were far too large and
were amplified by the auto-gain, saturating most of the frame. Fixes, fixed now:
- Depth beyond 40 m is extended smoothly, d' = 40·exp((d − 40)/12) (70 m → ≈ 490 m,
  80 m → ≈ 1,130 m), then lightly smoothed in log space.
- Darkness: radiance × k × gain (gain = k^−0.5) and noise first; light sources are then added
  to the exposed image as small saturated cores (σ 1.5–4 px) with faint halos (σ 8–20 px,
  5–20% amplitude), half near road level (head/tail lights), half higher (street lights), with
  thin reflection streaks (σ 1.5–4 px) on the road surface (starting no higher than 55% of
  the frame height) decaying over 30% of the frame height.

## Amendment 3 (2026-10-12, after an independent code review, before any real-image prediction)

A first training run (stopped at epoch 16 of 20; only its synthetic-validation numbers were seen,
never a BDD100K val or DAWN prediction) was reviewed against this document by independent
reviewers with adversarial verification. Upheld deviations, all fixed before retraining from
scratch:
- **Loss weights.** The loss averaged the three scalar heads into one term and added the map,
  so the map weighed 3x each scalar target. Now each of the four targets (log MOR, log k,
  log V_eff, map) is one mean-L1 term with weight 1, the same sum used for checkpoint selection.
- **Schedule.** One-cycle (1-epoch warm-up, cycled AdamW beta1) replaced by plain per-step cosine
  annealing from 3e-4, as written above.
- **Spray on every vehicle.** Plumes were drawn only for boxes >= 8 px tall (working size) and at
  most 15 per frame (65% of labelled vehicles). Now every labelled car/truck/bus box with
  positive area gets a plume (54,017 boxes over 4,500 frames; at most 46 per frame).
- **Mis-tagged night frames.** About 45 base frames tagged clear/daytime are night scenes (seen
  on a contact sheet sorted by brightness). Base frames with mean Rec. 601 luma < 52 (0-255) are
  excluded from training and synthetic validation (49 frames; a pixel rule, no detector).
- **Real-image geometry.** All training frames are 16:9; DAWN images are not (median 1.6:1, some
  portrait or panoramic). Primary: centre-crop every real image to 16:9 before resizing
  (BDD100K val is 16:9, unaffected). The stretched version is reported as a sensitivity check.
- **Determinism.** Rain-dot placement used index assignment with repeated indices
  (non-deterministic on CUDA); now a max-reduction. GPU training remains non-bit-reproducible
  (cuDNN), as for every model in this repository.
- **Evaluation.** Fixed physical bands now put V_eff exactly at 1000/300/100 m in the
  higher-visibility band (>= 1000, [300, 1000), [100, 300), < 100 m); per-band calibration sizes
  and infeasibility rates are reported; the MOR-ordering sanity check is reported, not pass/fail.

Recorded, not changed: V_eff <= 2000 m in daylight by construction (V_light = 2000 m at k = 1), so
clear daytime always falls in the top band; light sources and wet-road reflections are drawn only
in darkness samples.

# Version 3: night relighting (2026-10-12, after the v2 results)

Written after v2 was evaluated (`results/vis2/vis2.md`), so this version is **not blind** to the
evaluation data: v2 showed that its synthetic darkness (dimmed daylight) does not transfer to
real night (BDD100K night read as k ≈ 0.92). The change is motivated by that sanity check and by a
visual comparison with real night frames, not by detection results. Everything not listed here
is as in v2 (amendments included): base frames, veil, rain, spray, camera nuisances, targets
other than k, model, training, bins, evaluation, criteria S1–S4 (v1 remains the comparator; v2 is
reported alongside).

- **Development frames (never trained on, never evaluated on):** 400 BDD100K *train* night frames
  (300 clear, 50 rainy, 50 snowy; seed 2; `cache/vis3/dev_night.csv`). Used only to compare the
  synthetic night with real night (luma of sky and road, contact sheets) while setting the
  rendering below, and for a development check after training: predicted k on these frames vs
  on 400 BDD100K train daytime frames that are not base frames.
- **Darkness, probability 0.40:** night relighting with probability 0.75, otherwise dusk (v2's
  uniform dimming, k ~ log-uniform[0.03, 0.5]).
- **Night relighting** of the (veiled, rained) daytime radiance J, treated as albedo × daylight:
  an illumination field E(x) relative to daylight multiplies J.
  - ambient (moon, sky glow, diffuse urban light): a ~ log-uniform[0.0005, 0.01], bluish or sodium tint;
  - ego headlights, probability 0.9: E = P · exp(−θ²/2σ²) / max(d, 3 m)² below a horizon row,
    θ the normalised horizontal image angle, σ ~ U[0.25, 0.45], P such that E at 10 m on axis is
    U[0.02, 0.2]; d the metric depth (not the far extension);
  - street-light pools, probability 0.6: 2–8 elliptical pools on the lower image (amplitude
    log-uniform[0.003, 0.05]), each with a lamp core above it;
  - sky (extended depth > ~300 m) replaced by a dark glow (a × U[0.3, 1.5]);
  - camera: auto-exposure towards a mean sRGB level U[0.06, 0.25] (gain ≤ 200), Poisson-Gaussian
    noise growing with gain, optional 3×3 denoise; then light cores: lamp cores, head/tail lights
    at the labelled vehicle boxes, a few small random lights; wet-road reflection streaks with
    probability 0.5 (as v2).
- **Light-level target k (night):** mean of E over driving-zone pixels (lower 5 of 7 map rows)
  whose metric depth is 10–60 m (median of E over the driving zone if none), clipped to
  [0.0003, 1]. Dusk and daylight as v2. V_light = 2000 m · k^0.5 as before.
- Outputs: `weights/visibility_v3_resnet18.pt`, `cache/vis3/`, `results/vis3/`.

**Version 3 development tuning (2026-10-12, before training, using only the development night
frames and synthetic samples).** Three preview rounds of 200 synthetic night samples against 200
development night frames (mean, sky and road luma). Round 1 was twice as bright as real night
(median luma 55 vs 28), with oversized halos and reflection bars from every light. Final values:
auto-exposure target U[0.03, 0.15] (sRGB mean); headlight illumination at 10 m U[0.01, 0.1]; night
light halos σ 3–8 px at 2–8% amplitude; wet road with probability 0.25, streak amplitude
0.05–0.2, and only 30% of vehicle lights mirrored. Final luma, synthetic vs real (median): mean
23 vs 28, sky 15 vs 17, road 29 vs 22. Night k then spans about 0.002–0.018 (V_light 90–270 m).

**Version 3 erratum (2026-10-12, after the v3 evaluation; documentation only, nothing retrained).**
An independent verification of the write-up against the code and result files found that the
Version 3 text above does not describe four details of what was trained and evaluated, and one
number in its preamble is imprecise. Recorded here as they are in the code:
- *Night diagnosis frames.* The diagnosis that motivated Version 3 viewed 24 BDD100K **val**
  images (8 clear day, 8 clear night, 8 rain night; seed 0) with v2's light predictions; no
  detection result. Reproduce with `python -m wad.vis_dev gap` (`results/vis2/night_gap.csv`).
  The preamble's "k ≈ 0.92" is the mean; the median over BDD100K val night is 0.94.
- *Tuning frames and numbers.* Tuning compared 200 synthetic renders with the first 200 of the
  400 development frames; the 400 are used for the development check, so that check is not
  independent of tuning. The final round is reproduced by `python -m wad.vis_dev luma`
  (`results/vis3/dev_luma.csv`: median mean/sky/road luma 22.7/15.2/28.7 synthetic vs 28.1/16.6/22.2
  real). Rounds 1–2 (e.g. "55 vs 28") were scratch previews of earlier code and cannot be rerun.
- *Dusk lights.* Dusk uses v2's dimming with 0–5 light sources (v2: 3–19).
- *k vs the rendered light.* k is computed from the untinted illumination field; the image uses
  the same field with colour tints (sodium, bluish, white) normalised to a maximum channel of 1,
  not to unit luminance, so the light reaching the image can be up to ~22% (−0.11 log10) below k
  (sodium ambient, no headlights); median gap about 7%.
- *Exposure.* Auto-exposure sets the mean *linear* luminance to t^2.2 with t ~ U[0.03, 0.15]
  (not the mean of sRGB values), gain clamped to [1, 200]; the floor of 1 applies in about a
  third to a half of night renders (strong near-field headlights), the cap of 200 was never reached.
