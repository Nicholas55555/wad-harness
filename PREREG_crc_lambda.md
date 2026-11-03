# Pre-registration: CRC over the dial λ vs per-weather CRC over the threshold (Phase 4)

Written 2026-11-03, **before any CRC-over-λ fit or evaluation was run**. Not blind: the per-detection model
(model 2, adopted 2026-10-15) and its graded results at fixed λ are known (`NOTES.md`, DECISION 2026-10-15),
and so are the baseline's run 2 numbers (`results/run2/run2.md`).

## Question

At the same miss budget α per condition, certified the same way, does calibrating the dial's per-detection
cut cost fewer false alarms than calibrating the score threshold, while still holding the budget?

- **Ours:** CRC over λ on the per-detection chance p(d) (model 2).
- **Baseline:** per-bin CRC over the score threshold τ (SeqCRC step 1 per weather bin, the run 2 policy).

## Data and splits (fixed)

- **Evaluation sets** (YOLOv8x @1280, grouped matching, the run 2 caches, `wad.one_threshold.eval_configs()`):
  - BDD100K val: 8,090 images in the bins clear, cloudy, rain, snow, clear_night, rain_night and snow_night.
  - DAWN, clean labels: 889 images in the bins fog, rain, sand and snow.
- **Splits:** the 200 calibration/test splits of run 2's pre-registered 200-split rerun (`wad.crc.make_splits`
  with `n_repeats = 200`, seeds 0–199). The calibration fraction is 0.5, the split is stratified by weather,
  and near-duplicate groups stay on one side.
- **p(d):** model 2 as adopted (`weights/per_detection_yolo.joblib`), fitted on BDD100K train only. It is not
  refitted for this test.

## Methods (each fitted per weather bin on the calibration half)

- **Baseline:** τ̂(b) = max{τ ∈ grid : n/(n+1)·R̂(τ; b) + 1/(n+1) ≤ α} on the detector score, exactly as in
  run 2.
- **Ours:** ĉ(b) = max{c ∈ grid : n/(n+1)·R̂_p(c; b) + 1/(n+1) ≤ α} on the chance p(d). The cut is
  c = 1/(1+λ), so λ̂(b) = 1/ĉ(b) − 1 is the smallest dial setting whose bound holds.
- **Grid:** both use the same 200-point grid (150 log-spaced points on [0.001, 0.5], plus 50 linear points on
  (0.5, 0.99]). If no grid point qualifies, the fit takes the floor 0.001 and is flagged infeasible.
- **Loss:** the image-level miss fraction, with the fixed matching at the floor (an object is found iff its
  matched box is kept), as in run 2. Both losses are non-increasing in λ and non-decreasing in τ.

## Metrics (on the test half)

- The image-level miss (the controlled quantity) and the share of splits above α.
- Object recall, precision and false alarms per image (FPPI).
- The paired per-split FPPI ratio, ours / baseline.

## Criteria (α = 0.2, the pre-registered run 2 budget)

- **C1 validity:** in every condition where the baseline holds α (mean held-out miss ≤ α + 0.01, i.e. "holds"
  or "at the budget"), ours also has mean held-out miss ≤ α + 0.01.
- **C2 price, per condition:** the median per-split FPPI ratio (ours / baseline) is below 1 in at least 9 of
  the 11 conditions.
- **C3 price, pooled:** pooled over the 11 conditions (each split's false alarms summed over conditions and
  divided by its test images), the 97.5th percentile of the per-split ratio is below 1.

**Reported regardless (exploratory):**
- α = 0.1: feasibility, and the smallest certifiable α for both methods;
- the price curves over α ∈ {0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30};
- a global version (one fit pooled over all bins of a dataset) for both methods.

## Sanity check before reading the result

The baseline arm must reproduce run 2's 200-split per-bin summary
(`results/run2/{bdd/v8x_1280__group, dawn/v8x_1280__group__clean}/seeds200/crc_summary.csv`, policy
oracle_bin, α = 0.2) to rounding. If it does not, stop and fix the code before looking at ours.

## Not pre-registered (Phase 4 ablations, exploratory)

1. **Training-set size:** model 2 refitted on 10 / 25 / 50% of the BDD100K train images, chosen by
   near-duplicate group with seeds 0–2.
2. **Feature groups:** model 2 refitted without each per-box group (class probabilities, DFL entropies, neck
   components, pyramid level).

Both are graded like doc 08: cost vs one threshold at λ = 10 with 95% image-bootstrap intervals, and the
calibration error.
