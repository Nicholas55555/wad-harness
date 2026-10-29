# Pre-registration: a learned per-image threshold with a cost dial

Written 2026-10-12, before any training-set detection was run, any label computed or any model
trained. Proposed by the user (2026-10-12): one network predicts, per image, how many objects the
detector would miss and how many false positives it would produce at every threshold; a single
dial ("how many false positives am I willing to pay to avoid one miss") then picks each image's
threshold, and presets such as recall-first or precision-first are just dial values. The right
trade-off depends on the use case (camera set-up, redundancy, weather), so the dial is left to the
user, with an optional calibration that turns a target miss rate into a guaranteed dial setting.

Honest limitations, known now: the author has seen an in-sample headroom analysis on the
evaluation data (`wad/threshold_headroom.py`, `results/threshold/headroom.csv`: at mean image miss
0.2 a perfect per-image threshold needs 9.2 FPPI on BDD100K val vs 28.0 for one global threshold,
and 0.33 vs 2.24 on DAWN; per-weather thresholds gain almost nothing). The brief forbids
deriving any hazard signal from detector outputs; here the detector's outputs on *training*
images are used only as labels, and the model sees pixels only at test time (user's decision).

## Detector and matching

Run 2 primary: YOLOv8x, imgsz 1280, conf floor 0.001, NMS 0.5, max 300; grouped matching
(vehicle = car/truck/bus, two-wheeler, person), IoU 0.5, greedy by score. The model is specific to
this detector; another detector needs its own labels.

## Training data (BDD100K train only)

- Frames: every BDD100K train frame already on disk (O3 subset, the 3,000 extra clear-day frames,
  the 400 version-3 development night frames) plus a stratified draw (seed 3) from the remaining
  train pool: clear night 4,000; rainy night 1,950; snowy night 1,950; dawn/dusk 2,000; all
  foggy; the rest of rainy and snowy daytime. List: `cache/thresh/train_names.txt`.
- Same scan rules as BDD100K val (degenerate boxes dropped, exact duplicates removed, classes
  mapped), in bins clear, cloudy, rain, snow, clear_night, rain_night, snow_night, plus dusk and
  fog for training only.
- 90% train / 10% validation (by near-duplicate group, seed 0). Validation is used only for early
  stopping. BDD100K val and DAWN are never used for training or model selection. No k-fold: the
  training set is separate from the evaluation sets.

## Labels

At K = 40 thresholds (every 5th point of the project's 200-point grid, 0.001 to ~0.97), per image:
m_k = ground-truth objects not matched at τ_k (non-decreasing in k) and f_k = false positives
with score ≥ τ_k (non-increasing in k).

## Model and training

ResNet-18 (ImageNet weights), input 288×512 (real images centre-cropped to 16:9, then resized),
pooled features → MLP (512→256→2K). Monotone outputs: m̂_k = Σ_{j≤k} softplus(a_j),
f̂_k = Σ_{j≥k} softplus(b_j). Loss: Poisson negative log-likelihood per threshold, each threshold
column divided by (1 + its mean training count) so that every column weighs alike; equal weight
for misses and false positives. AdamW lr 3e-4, weight decay 1e-4, cosine schedule, batch 32,
15 epochs, horizontal flip only (colour changes would change the labels), seed 0; the checkpoint
with the lowest validation loss is kept.

## The dial

For λ > 0 (false positives accepted to avoid one miss), image i gets τ_i(λ) = τ_k* with k* the
largest k minimising λ·m̂_ik + f̂_ik. Because m̂ rises and f̂ falls with τ, τ_i(λ) can only fall
as λ grows, so every image's true miss fraction is non-increasing in λ (the monotonicity CRC needs).

- **Direct presets (no guarantee):** λ = 1 (precision-first), 3, 10 (balanced), 30 (recall-first)
  false positives per avoided miss.
- **Calibrated presets:** for α ∈ {0.1, 0.2, 0.3}, CRC on a calibration half picks the smallest λ
  on a log grid (200 values, 0.01–10⁴) with n/(n+1)·R̂(λ) + 1/(n+1) ≤ α, R̂ the mean image miss
  fraction (the project's loss); none feasible → the largest λ, recorded as infeasible. (b1) one
  λ for all images (primary); (b2) one λ per weather bin (condition labels known; per-condition
  guarantee).

## Evaluation (BDD100K val and DAWN, never seen in training)

Same 200 random calibration/test halves as the rest of the project (`wad.crc.make_splits`).
Comparators on the same halves: one global CRC threshold, per-weather CRC thresholds, v1
visibility-quartile CRC thresholds. Reported on test halves, overall and per condition (BDD100K
weather × time of day; DAWN weather and fog subtypes): mean image miss and share of splits above
α, recall, precision, F1, FPPI. Also: prediction quality (Spearman of predicted vs true miss and
false-positive counts at τ = 0.25) and the share of the in-sample headroom captured,
(FPPI_global − FPPI_learned) / (FPPI_global − FPPI_oracle) at α = 0.2.

Success criteria:
- **P1** BDD100K val, α = 0.2: the calibrated learned policy (b1) has mean test FPPI at least 10%
  lower than the global CRC threshold, with mean test image miss ≤ α + 0.01.
- **P2** The same on DAWN (out of distribution: different cameras and weather).
- **P3** Per-condition variant (b2) on BDD100K val, α = 0.2: rain-night mean test image miss
  ≤ α + 0.01 at lower FPPI than per-weather CRC thresholds in rain-night.

Anything added after the results is labelled exploratory.

## Implementation note (2026-10-12, during training; no evaluation result seen)

The first training run (fp32) took 810 s per epoch on the 4 GB GPU (memory at 3.94 GB; likely
spilling to shared system memory) and was stopped after epoch 1. Training restarted from scratch
with mixed precision (fp16 autocast for the network, Poisson loss in fp32, gradient scaling);
everything else as above. A channels-last memory layout was tried in a speed benchmark and
dropped (7× slower on this GPU and driver). The real cause of the slow first run was the same:
the input tensor was a permuted (channels-last) view; it is now made contiguous, giving ~0.17 s
per batch. A second start with mixed precision was stopped during epoch 1 for this fix.

# Version 2: decision-focused threshold from confidence and scene features (2026-10-13)

Written before any feature was computed or model trained. Not blind: designed after version 1
(pixels-only curve prediction) failed P1/P2 and after the exploratory threshold-budget and
graded-response analyses. **Departure from the brief, by the user's decision (2026-10-13):** the
model reads the detector's own outputs on the image (confidence and boxes) as inputs, because how
much room a threshold has depends on many interacting factors (street lighting, taillight glare,
fog, splashes from puddles, night with rain) that the detector's confidence profile reflects.

## Inputs (per image; one feature table per set: BDD100K train, BDD100K val, DAWN)

- **Confidence profile** (from cached YOLOv8x @1280 detections, conf ≥ 0.001): per class group
  (vehicle = car/truck/bus, two-wheeler, person) and for all groups together, log(1 + number of
  detections with score ≥ τ_k) at the 40 thresholds; the 10 highest scores; per score band
  ([0.25, 1], [0.05, 0.25), [0.01, 0.05)) the median log relative box height, the share of small
  boxes (height < 3% of the image) and the mean relative box bottom; log(1 + all detections).
- **Scene measurements** (288×512 frames, as in version 1): mean luma, RMS contrast, log Laplacian
  energy, dark share (luma < 20), saturated share (luma > 245), bright-red share (R > 180, G < 90,
  B < 90: taillights, brake lights), top- and bottom-third mean luma, mean HSV saturation; the
  version-3 visibility estimator's predicted log MOR, log k, log V_eff and road-map MOR.
- Variants: **full** (both; primary), **confidence only**, **scene only** (ablations).
  Features standardised with the training-set mean and s.d.

## Model and decision-focused training

MLP: [features, z(log λ)] → 256 → 256 (ReLU, dropout 0.1) → 40 logits, one per threshold τ_k.
Loss: for each training image a dial λ ~ log-uniform[0.1, 1000] is drawn; with p = softmax(logits),
loss = Σ_k p_k (λ·m_k + f_k) / N(λ), where m_k, f_k are the image's true missed objects and false
positives at τ_k (version 1's labels) and N(λ) = min_k (λ·M̄_k + F̄_k) is the mean cost of the best
single threshold on the training set (so 1 = as good as one threshold on average). AdamW lr 1e-3,
weight decay 1e-4, cosine, batch 512, 40 epochs, seed 0; same 90/10 train/validation split as
version 1; checkpoint with the lowest validation cost (hard argmax policy, mean over
λ ∈ {0.3, 1, 3, 10, 30, 100, 300}).

Policy: τ_i(λ) = τ at argmax logits (λ clamped to [0.1, 1000]); then a running minimum over an
ascending λ grid, so each image's threshold can only fall as λ grows (the monotonicity CRC needs).

## Evaluation (BDD100K val, DAWN; never used in training)

As version 1: 200 calibration/test halves; calibrated presets (CRC picks λ on the calibration half
for α ∈ {0.1, 0.2, 0.3}; one λ for all images, and one per weather bin) and direct presets
λ = 1, 3, 10, 30. Comparators on the same halves: global, per-weather and v1-quartile CRC
thresholds, and the version-1 pixels-only dial. New headline metric, **decision cost at a fixed
dial**: for λ ∈ {1, 3, 10, 30, 100}, the mean test-half cost λ·misses + FPs per image of each policy,
with the global comparator = the best single threshold chosen on the calibration half for that λ,
and the share of the in-sample per-image oracle's improvement captured (test half).

Success criteria (version 2, full variant):
- **D1** BDD100K val, α = 0.2, calibrated dial (one λ): mean test FPPI ≥ 10% lower than the global
  CRC threshold, mean test image miss ≤ α + 0.01.
- **D2** The same on DAWN.
- **D3** BDD100K val: at λ = 10 the dial captures ≥ 20% of the oracle's improvement over the best
  single threshold (test halves, mean over splits).
- **D4** BDD100K rain-night, per-weather calibrated dial, α = 0.2: test miss ≤ α + 0.01 at lower
  FPPI than per-weather CRC thresholds.

**Version 2 amendment (2026-10-13, user's decision, before any feature was computed or model
trained).** No scene measurements and no visibility estimator: the model learns only from the
detector's own behaviour on the image, i.e. the confidence profile above (the "confidence only"
variant becomes the one model; the full and scene-only variants are dropped). Everything else in
version 2 is unchanged; D1–D4 apply to this model. Also reported, since the aim is to predict the
per-image oracle threshold at the chosen trade-off: at λ ∈ {1, 3, 10, 30, 100} on BDD100K val and
DAWN, Spearman correlation between the model's threshold and the oracle's (argmin of
λ·misses + FPs on the image), the share of images within one step of the 40-point grid of an
oracle-optimal threshold, and the mean regret (cost of the model's pick minus the oracle's).

**Version 2 amendment 2 (2026-10-13, after a failed first training run; only training and
validation data seen).** Trained on expected cost alone, the softmax saturated within the first
epoch onto one threshold for every image and every λ (logit spread ~40; validation cost 1.48× the
best single threshold, constant over all 40 epochs). Changes: (1) loss = expected normalised cost
+ cross-entropy to a soft oracle target q_k ∝ exp(−r_k / 0.05), r_k = (λ·m_k + f_k − min_j(λ·m_j +
f_j)) / N(λ), so thresholds within a few percent of the image's best cost share the target and the
gradient cannot vanish; (2) standardised features clipped to [−5, 5] (BDD100K val reached 85 s.d.);
(3) learning rate 3e-4. Everything else unchanged.

**Version 2 outcome note (2026-10-13).** The trained model (confidence profile only, amendment 2)
does not beat one threshold even on its own training images (decision cost 1.001–1.009× the best
single threshold; oracle 0.80–0.85×). A quick check on all BDD100K val / DAWN images
(`python -m wad.thresh_dfl check`) gives 1.005–1.017× on BDD100K and 0.97–1.09× on DAWN. D3 is
therefore not met; the full held-out CRC evaluation for D1, D2 and D4 was not run, since a policy
no better than one threshold at every λ cannot give the ≥ 10% FPPI reduction D1/D2 require.

# Version 3: predict the oracle threshold from pixels, with the trade-off as input (2026-10-13)

Written before training. User's direction: a network that sees only the image (pixels) and the
desired trade-off, trained to output the threshold at which the detector (here YOLOv8x @1280,
grouped matching; the pipeline only needs another detector's cached detections to retarget it)
does best on that image. Not blind: designed after versions 1–2; prior evidence (version 2 check)
is that the detector's confidence is about equally reliable across conditions, so gains, if any,
should come from what the image shows and the scores do not (glare, spray, lens rain).

- Data, labels, splits, input frames: as version 1 (21,831 BDD100K train frames, 90/10 split by
  near-duplicate group; per-image misses m_k and false positives f_k at 40 thresholds; 288×512).
- Model: ResNet-18 (ImageNet weights) → 512 pooled features; head [512 + z(log λ)] → 256 (ReLU)
  → 40 logits. Each image is seen with 4 dial values per step, λ ~ log-uniform[0.1, 1000].
- Loss (version 2, amendment 2): expected normalised cost Σ_k p_k(λ·m_k + f_k)/N(λ) + cross-entropy
  to the soft oracle q_k ∝ exp(−(cost_k − min cost)/N(λ)/0.05).
- Training: AdamW lr 3e-4, weight decay 1e-4, cosine, batch 32 images, 15 epochs, mixed precision,
  horizontal flip, seed 0; checkpoint with the lowest validation decision cost (hard argmax policy,
  mean over λ ∈ {0.3, 1, 3, 10, 30, 100, 300}, relative to N(λ)).
- Policy: argmax, then a running minimum over ascending λ (monotone, as version 2).

Evaluation (BDD100K val, DAWN; never used in training): decision cost per image at
λ ∈ {1, 3, 10, 30, 100} relative to the best single threshold chosen on the same images (which
favours the single threshold), the per-image oracle, Spearman with the oracle threshold, and the
share within one grid step of it; versions 1 (pixels, curve prediction) and 2 (confidence
profile) on the same footing.
- **E1** BDD100K val: cost at λ = 10 ≤ 0.97× the best single threshold.
- **E2** The same on DAWN.
- **E3** BDD100K val: Spearman with the oracle threshold at λ = 10 ≥ 0.3.
If E1 is met, the held-out CRC evaluation of version 1 (calibrated dial, α = 0.2, 200 halves) is
run for this model as well.

**Version 3 amendment (2026-10-13, user's clarification, before any training).** "Pixels" means
the input YOLOv8 itself is trained on, read the way YOLOv8 reads it, not a separate network of
ours. Input: each image prepared exactly as for detection (EXIF orientation applied, YOLOv8's own
letterbox to 1280, scaling to [0, 1]) and passed through YOLOv8x's frozen backbone and neck; the
three feature maps it hands to its detection head (strides 8, 16, 32) are pooled into one vector
per image: global mean and global max per channel at each scale, plus the stride-32 map averaged
over a 3×4 grid (rough spatial layout: sky / road, left / right). Model: MLP [features + z(log λ)]
→ 512 → 256 (ReLU, dropout 0.2) → 40 logits; features standardised on the training set and
clipped to [−5, 5]; loss as above (expected cost + soft-oracle cross-entropy); AdamW lr 3e-4,
weight decay 1e-4, cosine, batch 512, 40 epochs, seed 0, one λ per image per step; same split,
checkpointing, policy and criteria E1–E3. The ResNet-18 design above is not used.

**Version 3 pilot (2026-10-13, user's request, before the full extraction; exploratory).** Same
design trained on 1,000 random BDD100K train images (seed 0; ~100 of them held out for early
stopping) and checked on 1,000 random BDD100K val images and all DAWN images (`--pilot`;
`results/threshold/yolo/pilot/`). The full run (21,831 / 8,090 / 1,005) and E1-E3 follow only if the
user decides so after the pilot. The interrupted full extraction (2,500 of 8,090 BDD100K val images) is
kept and resumable.

**Version 3 pilot finding (2026-10-13; pilot data only, nothing from the full run).** The
pre-registered head (dial as one extra input next to 10,880 image features) ignores the dial: on
its own training images its median pick stays at grid step 26 for lambda 1 to 100 while the best
single threshold moves from step 27 to 8 (pilot BDD100K val cost 1.18x at lambda 10 with the
pre-registered 80 updates, 1.24x with ~1,700 updates). Pilot follow-up: a structured head that
predicts each image's miss and false-positive curves from the features and applies the dial
analytically (argmin lambda*m + f; a larger lambda can only lower the threshold), trained with the
same decision cost + soft-oracle loss plus a weak (0.1) Poisson anchor on the curves (`--curves`).

**Version 3 pilot result (2026-10-13; exploratory).** Structured head, 900 training images
(~1,700 updates), after an independent verification (alignment, metrics and code confirmed; minor
fixes applied: oracle ties broken like the dial, fit-split-only standardisation, validation cost
against the validation images' own best single threshold, pilot-id check): cost 1.01-1.04x the best
single threshold on 1,000 BDD100K val images and 1.02-1.06x on DAWN (oracle 0.80-0.86 / 0.71-0.75);
E1 1.012, E2 1.018, E3 0.194: none met. The verification also showed that the same loss and head
reach 0.88-0.96x when given the true curves as input (the pipeline can learn when the signal is
there) and that with 10,880 features the pilot head overfits (Spearman of predicted vs true misses
0.94 on its training images, 0.43 on held-out ones).

**Version 3 full run (2026-10-13, user's decision after the pilot; written before the full extraction
finished and before any full-run training).** The user asked to train the threshold model on
YOLOv8x's features with all the data. By then the project had chosen one threshold per dial setting
(`wad/one_threshold.py`, NOTES "DECISION"); this run asks whether the full training set changes the
pilot's answer. Data: features of all 21,831 BDD100K train, 8,090 BDD100K val and 1,005 DAWN images
(the interrupted extraction resumed; same extraction code). Primary model: the structured curves
head (`train --curves`), because the pilot showed the version 3 head as first registered ignores the
dial; that head (`train`) is also trained and reported. Everything else as registered: 90/10 split by
near-duplicate group, batch 512, 40 epochs, seed 0, checkpoint by validation decision cost, `check`
criteria E1-E3 unchanged (cost vs the best single threshold on the same images, λ = 10). Also
reported: cost vs the single threshold chosen on train (`cost_vs_train_chosen_single`), which is
the chosen method on the 40-threshold grid. Not changed after seeing any full-run result.

**Version 3 full run result (2026-10-14; criteria as registered).** Structured curves head (primary),
19,650 fit / 2,181 validation train images, best epoch 22: cost vs the best single threshold on the
same images, λ = 1 / 3 / 10 / 30 / 100, BDD100K val (8,090) 1.005 / 1.001 / 1.005 / 1.022 / 1.028,
DAWN (889) 1.054 / 0.994 / 0.990 / 1.025 / 1.030 (oracle 0.81-0.85 / 0.71-0.75). **E1 1.005, E2
0.990, E3 0.217: none met.** The head as first registered: E1 1.068, E2 1.161, E3 0.149, none met.
Against the single threshold chosen on train (the chosen method) the curves head is 1.000-1.028 on
BDD100K and 0.987-1.054 on DAWN. An independent verification (recomputation outside the module,
leakage / alignment audit incl. bit-exact GPU re-extraction of rows around the resume point, and
diagnostics) confirmed every number. With 22x the pilot's data the head still overfits the misses
(Spearman of predicted vs true misses 0.85 on its fit images vs 0.48 held out), and on BDD100K val
it behaves like one constant threshold set one to three grid steps too high.

## Per-detection option, model 2: with YOLOv8x's own per-box features (written 2026-10-15, before training)

User's request: train the dial's per-detection model with YOLOv8's own per-box features. The dial
stays the λ dial (false alarms accepted per avoided miss; the Fβ dial maps β to its equivalent λ).

- Features: the current model's 20 context features (model 1, `wad/per_detection.py`) plus, from
  YOLOv8x's detection head at the anchor each kept box came from (`wad/yolo_box_features.py`; the
  re-run reproduces the cached detections exactly): the probabilities of the six classes, the
  largest probability among the other 74 COCO classes, the top-1 minus top-2 margin, the sum and the
  entropy of the 80 probabilities, the entropy of each box side's DFL distribution and their mean,
  the pyramid level, and the neck feature vector at the anchor reduced to 32 principal components
  (PCA fitted on a random 200,000 train detections, seed 0).
- Same data, labels, model and settings as model 1 (HistGradientBoostingClassifier, max_iter 400,
  learning rate 0.1, 63 leaves, early stopping on 10%, random_state 0), BDD100K train only; same rule
  (keep iff chance ≥ 1/(1+λ)); same 5-fold check by near-duplicate group. No setting is tuned on
  BDD100K val or DAWN.
- Primary criterion (adoption): on all BDD100K val at λ = 10, cost (λ·misses + false alarms) of model 2
  relative to model 1 with a 95% image-bootstrap interval entirely below 1. Also required: DAWN (all)
  at λ = 10 not worse than model 1 (point estimate ≤ 1), and calibration at the preset cuts on all
  BDD100K val within ×1.25 of the cut (as model 1). If all three hold, model 2 becomes the dial's
  per-detection option (and the Fβ table and the dial page are rebuilt from it); otherwise model 1
  stays and model 2 is reported.
- Reported either way: every condition and preset, F1 / Fβ, the 5-fold out-of-fold check, and the
  importance of the new feature groups.

**Per-detection model 2 result (2026-10-15; criteria as registered).** All three met, so model 2
became the dial's per-detection option (`wad/per_detection.py` ADOPTED = "yolo"). All BDD100K val,
λ = 10: cost relative to model 1 0.972 (95% 0.969–0.975); DAWN 0.995 (0.981–1.010), not worse;
share real near the preset cuts on all BDD100K val 1.01–1.13 × the cut. Against the one threshold at
λ = 10 the per-detection option now costs 0.910× on BDD100K val (model 1: 0.937×) and 0.933× on DAWN
(0.938×). Feature groups by importance: context 0.182, class probabilities 0.091, neck components 0.022,
DFL 0.004, level 0.001 (log-loss increase when shuffled).
Caveat: model 2's own choices (feature list, PCA, one run, adoption by the criteria above) used train
only; the 20 context features and the model settings it inherits from model 1 were picked earlier in
exploratory `detection_rescore` runs that were scored on BDD100K val and DAWN (alternatives within
about 0.005). An independent verification (re-extraction of 105 images' per-box features on the GPU,
recomputation of the criteria and of every headline number, routing of the adopted model) confirmed
the result.
