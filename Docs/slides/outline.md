# Phase 3 slides: outline, visuals and speaker notes

*7 slides, about 6 minutes, plus Q&A. One visual per slide and at most 3 short bullets. The diagram
(slide 4) and the pseudocode (slide 5) are the same files as the Overleaf Methodology section
(`../latex/methodology.tex`). Assets are in `../figures/phase3/` and `../dial_demo/`. Numbers are from doc 10
§4–§8, which cites the source CSVs.*

Design notes for the deck:
- **Background:** white. The figures are drawn on white with Segoe UI (diagram) and Computer Modern
  (pseudocode).
- **Accent colour:** orange (#D97706) marks "new", the same as the diagram. Grey marks the baseline; blue
  (#2F6FDB) marks "planned".
- **Images:** insert the SVGs in PowerPoint (Insert → Pictures accepts `.svg`). Google Slides needs the `.png`
  copies.

---

## Slide 1 — Title (0:15)

**On the slide**
- Title: **Weather Adaptive Autonomous Driving**
- Subtitle: *A calibrated trade-off dial for camera object detection in bad weather*
- Nicholas Taylor · Zander Picon · CAI 6930 Autonomous Vehicles · Florida Gulf Coast University ·
  Phase 3: Methodology · *date of the talk*
- Visual: a full-bleed rain-at-night frame, `../figures/threshold_exploration/fig16_all_boxes_rain_night.png`
  (cropped to the image half), or a crop of `../dial_demo/dial_F2_per_detection_rain_night.png`.

**Say:** "We're Nicholas and Zander. Our project asks how a car's camera detector should decide which detections
to trust when the weather turns bad."

---

## Slide 2 — Baseline recap: what we reproduced and where it stops (1:00)

**On the slide**
- Left (small): the rule. **SeqCRC step 1, per weather bin:**
  τ̂(b) = max{τ : n/(n+1)·R̂(τ;b) + 1/(n+1) ≤ α}, the strictest threshold whose miss rate is certified ≤ α.
- Right (large): `../figures/phase3/baseline_price.svg`.
- Three bullets:
  - Reproduced (Phase 1): the guarantee held 5/5 on COCO. On DAWN with YOLOv8x, **α = 0.2 holds in every
    weather bin** (miss 0.188–0.194); one global threshold fails in fog (0.251).
  - **Gap:** the miss budget is fixed, and false alarms are whatever results: 1.6 per image in DAWN snow,
    **117 at night in rain**.
  - One cut for every box, and the weather label must be known at run time.

**Say:**
- "Phase 1 reproduced SeqCRC, conformal risk control on the detector's threshold. Our harness runs its first
  step separately per weather bin.
- It works: at a 20% miss budget, every DAWN weather bin holds, where one global threshold fails in fog.
- But look at the price. The budget is fixed, and the false alarms are whatever it takes: under two per image
  in DAWN snow, 117 per image in rain at night.
- The literature review called this 'false positives unpriced'. Nothing lets the user say how many false
  alarms one avoided miss is worth."

---

## Slide 3 — Proposed method: make the trade-off the input (1:00)

**On the slide**
- One sentence, large: *Each detection gets a calibrated chance p(d) that it is real, computed from the
  frozen detector's own outputs; keep it iff **p(d) ≥ 1/(1+λ)**, where **λ = false alarms accepted per avoided
  miss**.*
- What changes (three-row mini-table; orange = new):

  | | baseline | ours |
  |---|---|---|
  | you choose | miss budget α | **trade-off λ** (or Fβ; α later as a floor) |
  | rule | one threshold per weather bin | **per-detection cut on a calibrated chance** |
  | needs | weather label | **nothing beyond the detector's outputs** |

- Evidence strip:
  - **0.87–0.96×** the cost of the best single threshold on BDD100K (0.90–0.97× on DAWN).
  - At λ = 10: recall **74%** vs 57% for the default 0.25.

**Say:**
- "Our change: stop fixing the miss rate and start pricing the trade-off. λ is a number with units: how many
  false alarms you'll accept to avoid one miss.
- For that cost, the best rule is to keep a box when its calibrated chance of being real exceeds 1/(1+λ).
- The detector stays frozen. What's new is a calibrated chance per box, built from YOLOv8x's own outputs, and
  the dial on top.
- Graded on data it never saw, it beats the best single threshold in every condition. The default 0.25 is
  only right near λ ≈ 1.4."

---

## Slide 4 — Architecture (1:00)

**On the slide:** `../figures/phase3/architecture.svg`, full width (identical to Fig. 1 in Overleaf). No
bullets.

**Say (left to right):**
- "The camera frame goes through the frozen YOLOv8x once. That part is unchanged.
- New: we also keep its per-box outputs, meaning the class probabilities, how sure each box edge is, and the
  neck feature.
- A feature builder adds context such as size, rank, overlaps and crowding. A boosted-tree model turns that
  into the chance the box is real.
- The dial keeps boxes above 1/(1+λ). λ is set by the use case, or from an F-score weight β.
- Out come the kept boxes, plus an estimate of their false alarms that needs no labels.
- Everything in green is fitted once on BDD100K train and graded on val and DAWN.
- The grey lane is the baseline, kept for comparison. The blue box is Phase 4: CRC over λ, to certify a
  miss budget per condition."

---

## Slide 5 — Algorithm and key equations (1:00)

**On the slide**
- Left: `../figures/phase3/pseudocode_alg2.svg`, Algorithm 2 (online, per frame).
- Right: `../figures/phase3/pseudocode_equations.svg`: cost, one threshold, per-detection rule, log-loss,
  Fβ map, CRC over λ.
- Footer: "Algorithm 1 (offline fitting): in the report; backup slide B1."

**Say:**
- "The cost is λ times misses plus false alarms. Two options.
- Option A: the single threshold that minimises that cost on train, solved exactly for every λ at once.
- Option B: per detection. Keeping a box costs 1 − p in expected false alarms, and dropping it costs λp in
  expected misses. So keep it when p ≥ 1/(1+λ). That's the Bayes rule, and it is only right if p is
  calibrated, which is why the model is trained with log-loss and checked at every cut.
- An F-score weight maps onto λ: F1 is about λ = 1.8, F2 about 6.2.
- Phase 4 adds the baseline's own guarantee on top. Because the kept set grows with λ, CRC can pick the
  smallest λ that certifies a miss budget, and we never go below it."

---

## Slide 6 — Design choices and alternatives (0:50)

**On the slide:** a 6-row table. Each row reads "choice — instead of — because":

| choice | instead of | because |
|---|---|---|
| frozen detector, decide after it | domain adaptation / fine-tuning | DA-Detect gave no gain on real fog (20.4 vs 21.4 AP50); retraining voids calibration |
| λ: false alarms per avoided miss | fixed α, max F1, FPPI cap | the use case should set the balance; the optimal rule is explicit |
| per detection | per-image or per-weather thresholds | per-weather adds nothing (0.98–1.03×); ~90% of the per-image edge is luck |
| one threshold kept as option A | none | exact, auditable fallback |
| boosted trees on YOLO's own per-box outputs | context features only; a crop CNN | pre-registered gain 0.972×; calibrated (ECE 0.1 pt); 11 min on CPU |
| fit on train, grade on val + DAWN | calibrate on halves of the test pool | nothing is chosen on graded data |

**Say:** "Most alternatives here weren't hypothetical; we ran them, several under pre-registered criteria. The
biggest lesson: the learnable signal is in individual boxes, not in images or weather labels."

---

## Slide 7 — Experiment plan for Phase 4, due Nov 5 (0:55)

**On the slide:** four blocks.
- **Data:**
  - fit on BDD100K train (21,831 frames);
  - grade on BDD100K val (8,090, 7 weather × time bins) and DAWN (889: fog, rain, sand, snow);
  - CRC: 200 calibration/test halves per condition.
- **Metrics and baselines:**
  - cost vs one threshold (95% bootstrap); recall, precision and FA/image; Fβ; ECE; held-out miss vs α;
  - baselines: default 0.25, global CRC, **per-weather CRC (Phase 1)**, one-threshold dial, model 1.
- **Ablations:** (a) feature groups, (b) per-condition recalibration (fog), (c) training size, (d) detector
  and resolution, (e) **CRC over λ vs CRC over τ**.
- **Compute:**
  - laptop RTX 3050 Ti (measured: 0.15–0.17 s/frame; per-box pass ~85 min; model ~11 min);
  - NRP A100 for (d) plus parallel fits, an estimated few GPU-hours;
  - timeline: Oct 23–26 CRC over λ; Oct 27–30 NRP runs; Oct 31–Nov 3 tables; Nov 4–5 write-up.

**Say:**
- "Most of the method is already measured. Phase 4 adds the certified version, CRC over λ, and four ablations.
- The key comparison is (e): at the same certified 20% miss budget, how many false alarms does the dial save
  over the per-weather baseline?
- The detector variants run on NRP. Everything else fits on the laptop."

---

## Backup slides (for Q&A)

| # | content | asset |
|---|---|---|
| B1 | Algorithm 1, offline fitting | `../figures/phase3/pseudocode_alg1.svg` |
| B2 | Calibration: share real near each cut, BDD100K vs DAWN fog | doc 08 §6 table T5; `../figures/method/fig23_per_detection_yolo.png` (left panel) |
| B3 | Results at λ = 1 / 10 / 30, both options vs no tuning | doc 08 §5 tables |
| B4 | The Fβ dial | `../figures/method/fig22_fbeta_dial.png` |
| B5 | Why not per-image thresholds (headroom, then luck vs structure) | `../figures/threshold_exploration/fig12_threshold_headroom.png`, doc 07 #5 |
| B6 | Live dial | `../dial_demo/dial_demo.html` (opens offline in any browser) or the 4 screenshots |
| B7 | Run 2 per-bin CRC on DAWN | `../figures/o1_o2_run1/fig3_main.png`, `../figures/o2_run2/fig7_run2_detectors.png` |
| B8 | Night is where one threshold breaks (BDD100K control) | `../figures/o2_run2/fig6_clear_vs_hazard_v8x_1280_group.png` |
