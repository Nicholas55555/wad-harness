# 11 — Decision log: what was decided, when, and on what evidence

*Chronological. Sources:*
- *the project plan (Sep 30), the brief (Oct 1), the proposal (Oct 8) and the Assignment 1 deliverables (Oct 10);*
- *`NOTES.md` (dated sections and the three DECISION sections);*
- *`PREREG_run2.md`, `PREREG_o3.md`, `PREREG_visibility.md` and `PREREG_threshold.md`.*

*"User's decision" marks choices the project owner made. Everything else followed from a pre-registered
criterion or a documented check.*

> **In short.** The project began as "calibrate a confidence threshold per weather bin" (SeqCRC step 1 per bin,
> chosen in Phase 1). That works, but it fixes the miss rate and leaves the false alarms unpriced. Three
> attempts followed, and none beat per-weather labels or a simple pixel score:
> - learn the weather (O3);
> - learn visibility (v1–v3);
> - learn a per-image threshold (PREREG v1–v3).
>
> A null simulation showed why: the learnable signal is per *detection*. The method became the **λ dial**
> (Oct 13–15): one threshold or a calibrated per-detection cut, set in units of false alarms per avoided miss
> or in Fβ, with CRC over λ planned to restore a certified miss budget.

---

| date | decision | why / evidence | where |
|---|---|---|---|
| Sep 30 | Research question: a weather-conditional threshold that recovers recall at a bounded false-positive cost, **without retraining**. Inference-only scope. Go/no-go margin 15 recall points | plan; falsifier: if max recall ≈ default recall, the problem is architectural | 01 §4–5 |
| Oct 1 | Harness rules: detect once and cache; nothing chosen on the test half; image-level monotone loss; IoU 0.5 greedy matching; one command reproduces everything; ask before >2 GB downloads or >30 min steps | the brief | 04 §1 |
| Oct 1 | **Run 1: GO** (person worst bin sand +17.2 pts, car fog +23.2) | max recall vs recall at 0.25 on DAWN, YOLOv8s @640; fragile for person (bootstrap P = 0.64) | 05 §1 |
| Oct 1 | Keep near-duplicates but split by group; apply EXIF orientation | 133 near-duplicate pairs (11 with conflicting weather labels); 40 mirrored JPEGs whose labels follow the mirrored frame | 03 §4 |
| Oct 8 | **Run 2 pre-registered:** primary YOLOv8x @1280, grouped loss, clean labels. BDD100K clear-weather control added; 200-split rerun | run 1: α = 0.1 infeasible in every bin, so more capacity and resolution, and a loss that does not count car↔truck as misses | `PREREG_run2.md`, 05 §2 |
| Oct 8 | O3 version B pre-registered: ResNet-18 weather classifier on BDD100K train; criteria S1/S2 | O3 version A (pixel statistics) had ARI 0.07, "mostly sand vs rest" | `PREREG_o3.md`, 06 §2 |
| Oct 8 | Adversarial review of run 2: 25 findings confirmed, all fixed before the report | independent review | 05 §6 |
| Oct 8 | Proposal: contributions are weather-conditional CRC on real weather, the FPPI price of every budget, and a tiered decision-layer spec | proposal deck | 01 §9 |
| Oct 10 | **Baseline fixed (Phase 1 conclusion):** "keep SeqCRC's first step as the calibration rule, calibrate per weather bin, and treat DA-Detect as an optional detector swap" | cods reproduced the guarantee 5/5; DA-Detect 20.41 vs 21.44 AP@50 on real fog (no gain) | 02 §4–6 |
| Oct 8–10 | Run 2 results: per-bin α = 0.2 holds in every DAWN bin at 1.6–6.2 FPPI; global fails in fog; a clear-calibrated threshold fails every BDD100K night bin | 200 splits | 05 §2–3 |
| Oct 11 | Visibility v2 pre-registered (physics-labelled synthetic fog, rain and darkness). The first training run was stopped after a review upheld 13 findings (amendment 3) | O3-B failed S1: fog is confused with sand | `PREREG_visibility.md`, 06 §5 |
| Oct 11–12 | v2: S3/S4 met, S1/S2 not. Night diagnosed as a sim-to-real gap in the darkness model. v3 night relighting, declared **not blind**: S1/S2 still not met. **Verdict: v1, the simple contrast + detail score, stays better; no visibility score holds rain at night** | Spearman and policy results | 06 §5–8 |
| Oct 12 | **User's idea:** there is no single ideal threshold; the trade-off depends on the use case. Introduce the dial λ, false alarms accepted per avoided miss | user | 07 |
| Oct 12 | PREREG threshold v1: per-image threshold from pixels. P1, P2 not met; P3 met | 27.82 vs 28.00 FPPI, i.e. no gain | 07 #1 |
| Oct 12 | Exploratory threshold budget and graded response (stop/slow tiers) | night recovers about 2× more recall per threshold step; per condition, α holds at about 0.95 false stops per image | 07 #2, #9 |
| Oct 13 | **User's decision, a departure from the brief:** the threshold model may read the detector's own outputs (confidence profile) | street lighting, glare and spray are reflected in the confidence profile | `PREREG_threshold.md` v2 |
| Oct 13 | v2 (confidence profile) no better than one threshold; v3 pilot (YOLOv8x image features) 1.01–1.06×; luck-vs-structure null: **87–91% of the per-image oracle's edge is luck** | null simulation | 07 #3–5 |
| Oct 13 | **DECISION: one threshold with a dial**, one constant threshold per λ fitted on BDD100K train | per-image models tie at best; per-weather thresholds 0.98–1.03×; no tuning costs 1.24× at λ = 10 | 08 §3 |
| Oct 13 | Housekeeping: 17.5 GB of input-frame caches moved (not deleted) to `Downloads\WAD_removed_2026-10-13\` | disk; every number still reproduces | `NOTES.md` |
| Oct 14 | v3 full run (21,831 frames): E1–E3 not met; it behaves like one threshold set 1–3 steps too high | pre-registered | 07 #4 |
| Oct 14 | **DECISION: the dial has two options**, one threshold and per detection (model 1: 20 context features, keep iff p ≥ 1/(1+λ)) | per-detection rescoring 0.90–0.97× one threshold; user: the goal is a tunable, statistically backed trade-off, not beating 0.25 on F1 | 08 §4 |
| Oct 14 | Fβ dial added; dedicated β-weighted models tested | they tie with one calibrated model | 08 §7 |
| Oct 15 | **DECISION (user's):** λ is the method; Fβ goes through λ_eq(β) = (1+β²)/F* − 1; the per-detection model also learns from YOLOv8x's own per-box outputs (model 2) | model 2 pre-registered before training: all three criteria met (BDD100K 0.972×, DAWN 0.995, near-cut calibration ×1.01–1.13) | 08, `PREREG_threshold.md` |
| Oct 15 | Tie fix: only ends of equal-score blocks are candidate thresholds; the exact table has 258 switch points | verification found a threshold inside a tie block | 08 §3 |
| Oct 15 | Frame scorecard added to the dial page (dial vs per-frame hindsight best) | user's request | 09 |
| Oct 22 | Phase 3 package: Methodology LaTeX, architecture SVG, pseudocode, slide outline. Baseline for Phase 3 = SeqCRC step 1 per weather bin; **planned Phase 4 extension: CRC over λ** | this documentation set | 10 |

## Decisions still open (see doc 12)

- The CRC-over-λ criteria: pre-register them before running (as for every earlier test).
- Whether to switch the run-2 primary to YOLOv8x @640 for the final report. It is better on DAWN, but was
  singled out post hoc, so any switch must state that history (`NOTES.md`).
- Whether the decision layer (stop/slow tiers) goes into the final report as a spec (O6).
