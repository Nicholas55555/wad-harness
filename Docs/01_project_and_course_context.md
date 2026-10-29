# 01 — Project and course context

*Sources: the project plan PDF (Sep 30, 2026), the Claude Code brief (Oct 1), the proposal deck (Oct 8), the
SeqCRC presentation deck, the Assignment 1 PDF, the literature review and reproducibility write-up (Oct 10), all in
`Downloads\Weather Adaptive Autonomous Driving\` and `Downloads\weather adaptive lit review\`; the harness `NOTES.md`
and pre-registrations for the status. Facts not found in any source are marked **TODO**.*

> **In short.**
> - Graduate course project for CAI 6930 Autonomous Vehicles at Florida Gulf Coast University, by Nicholas Taylor and Zander Picon.
> - Question: can a confidence threshold chosen per weather condition recover the recall a frozen detector loses in bad weather, at a stated false-alarm cost, without retraining?
> - The baseline is SeqCRC's first step (a conformal risk control threshold), calibrated per weather bin. It was reproduced in Phase 1 and extended in runs 1 and 2.
> - Since Oct 12–15 the method has been the **λ dial**: a stated trade-off (false alarms accepted per avoided miss), applied as one threshold or per detection, fitted on BDD100K train. See doc 08.
> - Next: Phase 3 is the methodology (Overleaf section plus a 5–7 slide talk). Phase 4, full experiments, is due Nov 5.

---

## 1. Course

| item | value | source |
|---|---|---|
| course | **CAI 6930 Autonomous Vehicles** | LR and RW title blocks (not in the assignment PDF) |
| university | Florida Gulf Coast University (FGCU) | LR, RW, proposal slide 1 |
| level | graduate course project | brief |
| platform | Canvas: the harness link is posted there | proposal slide 10 |
| instructor | **TODO**: not in any source | — |
| syllabus dates | **TODO**: the plan assumes Dec 4 for the final presentation; "confirm against the syllabus" | plan p.7, p.10 |

**Course phases vs harness phases.** The course numbers its deliverables as phases. "Phase 1" is Assignment 1 (the
literature review and reproduction), "Phase 3" is the methodology, and "Phase 4" is the full experiments. The
sources never name "Phase 2"; it is probably the proposal (**TODO**, confirm). The harness brief separately numbers
its own build steps "phase 1–6" (data, detection, matching, CRC, report, hazard). In these docs, "Phase N" with a
capital P means the **course** phase.

## 2. Team

| source | what it says |
|---|---|
| proposal slide 1 | "Nicholas Taylor · Zander", "Florida Gulf Coast University · October 8, 2026" |
| SeqCRC deck slide 1 | "Authors: Nicholas Taylor" |
| literature review, write-up | single author (Nicholas Taylor), written as "we" |
| plan p.9 roles | threshold policy and hazard estimator (O2, O3, O4): Nicholas; harness, data, figures (O1, O6 repo): TBD; DA-Detect and stretch items (O5): TBD; report and presentation (O6): all |

- **Team members:** Nicholas Taylor and **Zander Picon** (surname supplied on 2026-10-22; the proposal deck
  shows only "Zander").
- **Group name:** not found. The plan lists "group name and members" as an open question.

## 3. Deliverables so far and ahead

| deliverable | date | status | where |
|---|---|---|---|
| Research Paper Presentation 1: DA-Detect (the "perceive" layer) | not in sources | done | deck not among the sources |
| Research Paper Presentation 2: SeqCRC (the "decide" layer), 21 slides with timed notes | deck created 2026-10-01 | done | `SeqCRC_Paper_Presentation_2_Nicholas_Taylor (2).pptx` |
| Research Project Proposal, 10 slides, about 10 min | **Oct 8, 2026** | done | `Research_Project_Proposal_..._Nicholas_Taylor (3).pptx` |
| Assignment 1, Part 1: literature review (Overleaf link plus PDF; 2 papers × 7 headings) | due **Oct 10** | done | `deliverables\Literature_Review_Nicholas_Taylor.pdf`, Overleaf source in `deliverables\Literature_Review_Overleaf\` |
| Assignment 1, Part 2: reproducibility write-up (2 repos: setup, test, train, compare) | due **Oct 10** | done | `deliverables\Reproducibility_Writeup_Nicholas_Taylor.pdf` |
| **Phase 3: Methodology.** Update the Overleaf Methodology section; 5–7 slides for a 5–7 min talk plus Q&A | now (this package is dated 2026-10-22) | in progress | doc 10, `latex/`, `slides/` |
| **Phase 4: Full experiments** | due **Nov 5** | planned | doc 10 §experiment plan, doc 12 |
| Final report and presentation | Nov 23–Dec 4 per the plan (Dec 4 assumed) | planned | plan Gantt |

**Assignment 1 rubric** (100 points):

| criterion | points |
|---|---|
| depth and relevance of the 2 reviewed papers | 30 |
| clarity of writing and synthesis | 15 |
| repo 1 | 20 |
| repo 2 | 20 |
| comparison write-up | 15 |

**Phase 3 slide list** (as given in the assignment, 5–7 slides):
1. Title.
2. Baseline recap.
3. Proposed method.
4. Architecture diagram (**.svg**).
5. Algorithm and pseudocode.
6. Design choices.
7. Experiment plan.

The diagram and pseudocode must be the same ones as in Overleaf. Doc 10 maps each slide to content.

## 4. Research question and hypotheses

**Research question, plan version** (Sep 30; also the SeqCRC deck and the brief):
> "Can a weather-conditional confidence threshold recover the recall a detector loses in adverse weather, at a bounded false-positive cost, without retraining the detector?"

**Research question, proposal version** (Oct 8, slide 4):
> "can a threshold calibrated per weather bin by conformal risk control keep the expected miss fraction under a budget α in every bin — and what does it cost in false positives per image?"

**System under test** (plan): "a fixed object detector, a per-frame hazard estimate computed from the camera image,
and a threshold chosen per hazard bin so that missed detections stay under a budget."

**Why recall first** (plan): "in bad weather a false positive costs a slowdown, a false negative costs a collision."

| hypothesis | statement (plan p.1) | what the evidence says now |
|---|---|---|
| **H1** | "A per-bin threshold meets a target miss rate in every weather bin where a single global threshold cannot, and the extra false positives it costs are bounded and reported." | **Supported at α = 0.2.** In the run 2 primary, per-bin CRC holds in all four DAWN bins (0.188–0.194 at 1.6–6.2 FPPI). The global threshold fails in fog (0.251). On BDD100K, a clear-day threshold fails every night bin; per-bin calibration restores them at 43–117 FPPI. Docs 05 and 06. |
| **H2** | "A camera-only hazard estimate is good enough that calibrating on predicted bins costs little more in false positives than calibrating on true weather labels." | **Not supported.** The FPPI gap is small (S2 met in most settings), but predicted bins do not protect real fog or rain at night (S1 never met; visibility v1–v3 included). Doc 06. |
| **falsifier** | If the worst bin's max achievable recall is close to its default recall, "the problem is architectural (YOLOv8-SCS territory), not a policy problem." | **Not triggered.** Run 1 worst-bin margins are 17.2 (person, sand) and 23.2 (car, fog) recall points, against a 15-point target. Doc 05. |

## 5. Scope

**Overall framing.** The wider project has three layers (SeqCRC deck slide 2):
- **ROUTE**, "pick the road": deferred.
- **PERCEIVE**, "see through it": DA-Detect, Presentation 1.
- **DECIDE**, "drive to the conditions": SeqCRC, Presentation 2.

The plan: "This plan covers the decision layer's first knob (the detector's operating point) and the part of the
perception layer it needs (the hazard estimate)."

| in scope (plan) | out of scope (plan) |
|---|---|
| camera 2-D detection on real and synthetic fog, rain, snow and sand | retraining or modifying any detector |
| per-frame hazard estimation from the image | lidar ("deferred; the hazard score is designed so a lidar signal can be added later") |
| a per-bin threshold policy with a miss-rate guarantee, including per-class budgets | satellite data and route planning |
| two fixed detectors: COCO YOLOv8, and DA-Detect weights (adapted and source-only) | CARLA and tracking (stretch items "a teammate may own") |
| evaluation on DAWN and Foggy Cityscapes (3 fog densities) | new datasets or annotation |

**How the scope moved in practice:**
- Foggy Cityscapes was **not run**. It needs a Cityscapes account, which is the user's to create. BDD100K val
  (7 weather × time bins, 8,090 images) became the second evaluation set instead.
- "Nothing is trained except the small hazard estimator" still holds for the **detector**, which is always frozen
  YOLOv8. The λ dial's per-detection option does train a gradient-boosting model on BDD100K train detections. It is
  a post-hoc calibration model on the detector's own outputs, not a detector change.

## 6. Objectives O1–O6

The plan: "O1 to O3 are the spine; O4 and O5 extend it; O6 closes the project. Each is done only when its criterion
is met on held-out data, not when the code runs."

| # | objective | done when (plan) | plan dates | status 2026-10-22 | docs |
|---|---|---|---|---|---|
| O1 | Evaluation harness and go/no-go: COCO YOLOv8 on DAWN and Foggy Cityscapes; cache every box; PR curves and max achievable recall per bin and class | PR curves for every bin and class; GO if the worst bin's max recall beats default-threshold recall by ≥ 15 points | Oct 5–16, go/no-go Oct 16 | **Done on DAWN and BDD100K val** (Foggy Cityscapes not run). GO in run 1 (person 17.2, car 23.2 points), but fragile: P(GO) = 0.64 over bootstrap resamples; run 2 primary 0.28. | 03, 04, 05 |
| O2 | Oracle-bin policy: CRC threshold per true weather bin vs one global threshold at the same α | ≥ 5 repeated splits; miss ≤ α in every bin, FPPI reported, main figure | Oct 19–30 | **Done.** Run 1 (YOLOv8s @640), pre-registered run 2 (YOLOv8x @1280, grouped loss, clean labels), BDD100K clear-weather control. α = 0.2 holds per bin; α = 0.1 is not certifiable in fog for the primary (smallest 0.148). | 05 |
| O3 | Camera hazard estimator: hand-crafted h and a small CNN on BDD100K weather tags; rerun O2 on predicted bins | predicted-bin policy still meets α; FPPI gap quantified; agreement reported; no detector confidences used | Nov 2–13 | **Done; criterion not met.** Versions A (pixel statistics) and B (ResNet-18), then visibility estimators v1 (pixels), v2 (physics-trained ResNet-18) and v3 (night relighting). S1 not met anywhere: predicted bins do not protect true fog or rain at night. | 06 |
| (new) | Threshold exploration and the **λ dial** (not in the plan) | — | Oct 12–15 | **Done; it is now the method.** Per-image thresholds do not beat one threshold. Per-detection rescoring does: 0.87–0.96× the cost on BDD100K. | 07, 08, 09 |
| O4 | Per-class budgets: a tighter α for people than for vehicles | per-class miss meets its budget in every bin | Nov 16–20 | **Preview only.** Run 1: 24/24 bin × class fits infeasible in at least half the splits (about 10 person images per rain calibration half). | 05, 12 |
| O5 | Second detector: DA-Detect (adapted and source-only) through the same harness | both weight sets under O2's policy; stacking figure; dropped if it won't build by week 5 | build Oct 19–Nov 6, figure Nov 9–20 | **Not started in the harness.** Phase 1 showed it builds and runs, but on DAWN fog the full model is not better than its baseline (20.41 vs 21.44 AP@50). Drop call per proposal: Nov 13. | 02, 12 |
| O6 | Decision-layer spec and write-up: tiered output ("confident detection = obstacle; low-confidence = caution zone that caps speed"), report, presentation, repo | report submitted; repo reproduces the main figure from cache with one command | Nov 23–Dec 4 | **Pending.** One-command reproducibility already exists (doc 04). The stop/slow graded response (doc 07 #9) is the starting point for the spec. | 04, 07, 12 |

**Proposal timeline** (slide 8):

| dates | milestone |
|---|---|
| Sep 22–Oct 1 | O1 ("Done — GO: 23 pts (car), 17 pts (person)") |
| Oct 8 | proposal: "direction fixed, α and policies pre-registered" |
| Oct 12–23 | O2 |
| Oct 26–Nov 6 | O3 |
| Nov 9–20 | O4 / O5 |
| Nov 23–Dec 4 | O6 |

## 7. Working rules (from the brief)

Short list; doc 04 shows how the harness enforces each rule.
- No training and no fine-tuning of the detector.
- Detect once, cache forever.
- Nothing is chosen on the test half: α, the τ grid and the default thresholds are fixed in config before any split.
- Splits are by image, stratified by weather, with near-duplicate groups kept on one side.
- The loss is image-level and monotone; matching is IoU 0.5, greedy by score.
- No hazard signal may come from detector outputs.
- Report what the data actually is; the real data wins over the brief.
- Every number in a report is reproducible by one command.
- Ask before any download over 2 GB or any step over 30 minutes.
- Smoke test on 20 images first.
- Pre-register before running (the project's own practice from run 2 on: `PREREG_*.md` written before results).

## 8. Risks

| risk | plan response | run 1 signal (proposal slide 9) | proposal backup | outcome |
|---|---|---|---|---|
| recall not recoverable (go/no-go fails) | reframe; make O5 the main result | GO, 23 / 17 points | — | GO, though fragile |
| budget not certifiable | — | α = 0.10 infeasible in every bin; floor miss 0.12–0.22 | grouped vehicle loss, YOLOv8x @1280; always report the smallest certifiable α | α = 0.2 holds per bin in run 2; α = 0.1 only with the post-hoc YOLOv8x @640 grouped variant |
| FPPI too high | — | 9.5–30 FPPI at α = 0.20 | per-box actions only above an FPPI-capped τ | 1.6–6.2 FPPI with the run 2 primary; led to the tiers and then the λ dial (a chosen price) |
| per-class budgets uncertifiable | — | 24/24 fits infeasible | pool bins, group classes, add Foggy Cityscapes | still open (O4) |
| calibration ≠ deployment | exchangeability failure is a reported limitation | 133 near-duplicate pairs; P(GO) = 0.64 | group-aware splits; bootstrap every verdict; non-exchangeable CRC | group-aware splits done; non-exchangeable CRC open |
| DA-Detect will not build | drop O5; not on the critical path | — | drop O5 by Nov 13 | builds (Phase 1); not yet in the harness |
| DAWN small per bin (200–320) | repeated splits, pool bins, fine-grained claims on Foggy Cityscapes | — | — | 10 + 200 splits; BDD100K val added |
| leakage in the hazard estimator | train on BDD100K only; pixels only | — | — | respected (doc 06) |
| scope creep (route, simulation) | belongs to teammates | — | — | — |
| BDD100K is large | weather-balanced subset of about 10,000 images | — | — | val 8,090 for grading; train subsets for fitting |

## 9. Success criteria and contributions as proposed

**Success criteria:**
- Plan go/no-go: "Go if, in the worst weather bin for person and for car, maximum achievable recall exceeds recall at the detector's default threshold by at least 15 points."
- Proposal slide 7: "Success = predicted-bin thresholds meet α in every weather bin where the global threshold fails, at a reported FPPI; the predicted-bin policy costs ≤ 5 FPPI more than oracle bins; per-class or grouped budgets certifiable on at least one domain."
- The O3 pre-registration encodes these as S1 (meet α where the global threshold fails) and S2 (FPPI gap ≤ 5).

**Contributions as proposed** (slide 9), with what the evidence now says:

| proposed contribution | status |
|---|---|
| "Weather-conditional conformal thresholds calibrated per hazard bin on real adverse-weather images (DAWN), not synthetic fog" | delivered with **oracle** (true-label) bins; predicted bins fall short (S1) |
| "The price tag: FPPI for every miss budget α, per weather bin" | delivered (runs 1 and 2, BDD100K control); generalised by the λ dial, which makes the price a chosen setting |
| "A guarantee that survives a wrong bin ... plus a one-page decision-layer spec" | **do not claim as stated.** Validity holds per *predicted* bin and only approximately; it does not carry to true fog. The spec is pending (O6). |

## 10. How the direction evolved

| date | event | doc |
|---|---|---|
| Sep 30 | Project plan: RQ, H1/H2, O1–O6, nine-week Gantt | this doc |
| Oct 1 | Brief for the harness; **run 1** (YOLOv8s @640, CPU, DAWN): GO; α = 0.1 infeasible; α = 0.2 at 9.5–30 FPPI | 05 |
| Oct 1 | SeqCRC presentation deck created (Presentation 2; the talk date is not in the sources) | 02 |
| Oct 8 | **Proposal** presented; **run 2 pre-registered** (YOLOv8x, grouped loss, clean labels, BDD100K control) and run; O3 version B pre-registered and run; review of run 2 (25 findings fixed) | 05, 06 |
| Oct 10 | **Assignment 1** submitted: literature review and reproducibility write-up | 02 |
| Oct 11–12 | Per-condition analyses (recoverable recall); visibility estimators v2 and v3 pre-registered and run; v1 stays best | 06 |
| Oct 12 | Threshold question reframed by the user: there is no single ideal threshold, and the trade-off depends on the use case → **cost dial λ**; learned per-image threshold (PREREG_threshold v1) | 07 |
| Oct 12–14 | Per-image threshold from confidence profile (v2) and YOLOv8x features (v3) fail; luck-vs-structure shows the learnable part is **per detection** | 07 |
| Oct 13 | **Decision:** one threshold with a dial (option A) | 08, 11 |
| Oct 14 | **Decision:** the dial has two options, one threshold and per detection (model 1); Fβ dial | 08, 11 |
| Oct 15 | **Decision:** λ is the method; Fβ goes through λ_eq; per-detection **model 2** (YOLOv8x per-box features) adopted under pre-registered criteria; tie-aware switch-point table | 08, 11 |
| Oct 15 | Dial page: frame scorecard (dial vs the frame's hindsight best), artifact v10 | 09 |
| Oct 22 | Phase 3 documentation package (this folder) | 00, 10 |

## 11. Which paper is the baseline?

The sources disagree:
- **Plan p.1** names Li et al. (DA-Detect) as the base paper.
- **Proposal** slides 3 and 10 say "SeqCRC (base paper)"; slide 10's notes say "the base paper is Andéol et al. 2025".
- **SeqCRC deck** slide 3 says the same.
- **Reproducibility write-up** calls cods (SeqCRC's code) "our project's base paper".

**Recommendation for Phase 3: SeqCRC.** The reproduced model that the project builds on is SeqCRC's first step
(conformal risk control of the confidence threshold), via `cods`, run per weather bin in the harness. DA-Detect is
the optional detector swap (O5). Doc 02 §"The Phase 3 baseline, stated precisely" gives the exact baseline and its
numbers.

## 12. Cautions when quoting earlier documents

Earlier decks and drafts contain claims that later evidence corrected. Carry these into Phase 3:
1. Call m(b) a **calibration estimate**, not a "certified miss rate" as the proposal and plan do. The held-out miss exceeds it in 38–42% of splits.
2. "Validity survives a wrong bin" holds only within **predicted** bins, and only approximately. It does not hold in true fog (O3, S1 not met).
3. The CRC rule is an **inf over λ**, or equivalently a **max over τ = 1 − λ**. The Overleaf skeleton's "max over λ" was wrong.
4. DAWN has **1,027 images found and 1,005 kept** (889 with clean labels), not the "1,020" of the plan and brief. The DAWN paper itself says 1,000.
5. The SeqCRC deck quotes **v1** numbers (DETR-50). The literature review quotes **v2** (DETR-101).
6. The proposal's description of DA-Detect ("consistency", "RTTS") is wrong. There is no consistency term and no RTTS result.
7. α = 0.1 on DAWN is certifiable only with the **post-hoc** YOLOv8x @640 grouped-loss variant. Say so whenever it is quoted.
8. "Nothing is trained" (README) is stale: O3 trains a ResNet-18, and the dial trains a gradient-boosting rescorer. The detector is never trained.
9. Do not cite `results.md` §6.3, which is broken (the 1280-px sensitivity reran against 640 detections). Do not cite `run2.md`'s go/no-go table, which is mis-rendered. Use doc 05's tables instead.
10. The proposal and README say 91 tests; the harness now collects 154.
