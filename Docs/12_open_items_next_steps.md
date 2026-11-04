# 12 — Open items and next steps

*As of 2026-10-22. Phase 4 (full experiments) is due Nov 5. The Phase 4 plan itself is in doc 10 §8. This
list adds the loose ends found while building the documentation: known report bugs, missing facts, and
objectives not yet done.*

> **In short.**
> - **For Phase 4:** implement CRC over λ (pre-register it first) and run ablations (a)–(e).
> - **Fix two report bugs before citing those sections:** the invalid resolution-sensitivity section in
>   `results.md` §6.3, and the mis-rendered go/no-go table in `run2.md`.
> - **Fill in:** the group name and the syllabus dates.
> - **Decide** the scope of O4 (per-class budgets), O5 (DA-Detect in the harness) and O6 (the decision-layer
>   spec) for the final report.

---

## 1. Phase 4 work (method)

**Status 2026-11-03:**
- **Done (doc 14):**
  - 1, CRC over λ (pre-registered, all criteria met);
  - 2, per-condition recalibration (ablation b, `wad/dial_recalibration.py`): fixes DAWN fog's under-stated
    chances only with labelled frames from the condition; no effect on BDD100K;
  - 3, feature-group ablation;
  - 4, training-size curve (10 / 25 / 50 / 100%);
  - 5, detector transfer (ablation d, `wad/detector_ablation.py`): YOLOv8x @640, YOLOv8s @640 and @1280, run on
    the laptop;
  - 6, per-frame timing;
  - 7, tables and figures.
- **Still open:** training on all 70,000 BDD100K train frames (needs about 3 GB more of BDD100K images).

| # | item | detail | effort |
|---|---|---|---|
| 1 | **CRC over λ** | per condition, λ̂_c = inf{λ : n/(n+1)·R̂_c(λ) + 1/(n+1) ≤ α}; deploy max(λ_user, λ̂_c); 200 calibration/test halves, near-duplicate-aware; compare FA/image with per-bin CRC over τ at equal α; report the smallest certifiable α. **Pre-register the criteria first** (e.g. "held-out miss ≤ α in every condition where per-bin CRC holds; FA/image below per-bin CRC's in ≥ x of y conditions") | laptop, about 1 day |
| 2 | per-condition recalibration | isotonic per condition on BDD100K train conditions; does DAWN fog's near-cut share move toward the cut? | hours |
| 3 | feature-group ablation (retrain) | drop class probabilities / neck PCA / DFL / level / context, one at a time | 5 fits × ~11 min CPU (5-fold: NRP CPU) |
| 4 | training-size curve | 25 / 50 / 100% × 3 seeds | 9 fits |
| 5 | detector transfer | YOLOv8x @640, YOLOv8s @640 / @1280: detections + per-box outputs for 30.9k frames each, then refit | NRP A100 (est. under ~15 min per pass); laptop ~1–1.5 h per pass |
| 6 | per-frame timing | time the per-detection model per frame (features + trees) for the real-time claim | minutes |
| 7 | report tables and figures | same harness commands; regenerate the dial page; `--from-cache` check | half a day |

## 2. Known bugs and stale text (fix before quoting)

| where | problem | fix |
|---|---|---|
| `results/results.md` §6.3 (resolution sensitivity) | **Invalid.** The 2026-10-08 recheck recomputed the 1280-px analysis against the **640-px** detections: every 640 → 1280 value is identical, "0 boxes gained", "median side nan", and `lost_gained.csv` has only a header. Likely cause: `wad/config.py:45-46` sets `paths.detections = paths.cache` at load time, but `wad/sensitivity.py:25-33` overrides only `paths.cache`/`paths.results` | set `paths.detections` in the sensitivity override, then rerun `python -m wad.sensitivity` (~7 min). Until then, quote the 2026-10-01 numbers in `NOTES.md` (max recall 83.1% → 81.0%; smallest pooled α 0.175 → 0.239) |
| `results/run2/run2.md` "Go/no-go per variant" | prints the operating-point table again (4 header names, 7 alignment cells) | fix `wad/run2_report.py`; doc 05 §2.3 has the correct table rebuilt from `dawn_variants.csv` and `go_no_go*.json` |
| `run2.md` | section numbers skip 6 | cosmetic |
| `README.md` | says 91 tests (154 are collected now); layout list omits run2, hazard_cnn, vis_*, recoverable, stats_tables, get_bdd100k, the threshold modules | update the counts and the layout |
| `NOTES.md` | small inconsistencies with `run2.md`: night FPPI "42–124" vs 59/117/43 (200 splits); m(b) "0.24–0.30" vs 0.232–0.292; night smallest α "0.15–0.18" vs 0.143–0.181; pre-cleaning BDD100K counts; "RTX 3050 Laptop" vs 3050 Ti; RAM 31 GB vs 16 GB in the Phase 1 write-up | the docs use the `run2.md` / CSV values; correct NOTES when convenient |
| old decks and plan | "certified miss rate" for m(b) (it is a calibration estimate); "validity survives a wrong bin" (only within *predicted* bins, approximately); DAWN "1,020 images" (1,027 found, 1,005 kept); the base paper named as DA-Detect in the plan | use the corrected wording (doc 01 §12) |

## 3. Facts missing from every source

- The group name and the instructor's name. (Zander's surname, Picon, was supplied on 2026-10-22.)
- The syllabus dates: the Phase 3 talk date, the final report and presentation (the plan assumes Dec 4).
- The Presentation 1 deck (DA-Detect) is not among the files.

## 4. Objectives not finished (O1–O6)

| objective | status | suggestion for the final report |
|---|---|---|
| O1 harness + go/no-go | done (DAWN, BDD100K) | done |
| O2 oracle-bin CRC | done (run 1, run 2, BDD100K control) | it is the baseline |
| O3 hazard estimator | done: S1 not met (A, B, v1–v3) | report as a negative result; it motivated the per-detection design |
| O4 per-class budgets | preview only (run 1: 24/24 fits infeasible at α_person = 0.05) | cheap version: run CRC over λ with person-only loss on pooled bins; or drop with a reason |
| O5 second detector (DA-Detect) | builds (Phase 1); no gain on real fog; not run in the harness | either drop, or reuse its released weights as one more "detector transfer" row (needs maskrcnn-benchmark in Docker, GPU) |
| O6 decision layer + write-up | stop/slow tiers analysed (doc 07 #9); spec not written | one-page spec: confirmed tier (low λ, or an FPPI cap) → obstacle; possible tier (high λ) → slow-down; the certified α floor per condition |

## 5. Data that would strengthen the evaluation (the user's steps)

- **Foggy Cityscapes** (controlled fog density; DA-Detect's benchmark): needs a Cityscapes account and two
  multi-GB archives. The harness is ready for a `config_foggy.yaml`.
- **ACDC** (real fog, night, rain, snow with labels): needs registration.
- **WEDGE** (synthetic weather, for estimators): not fetched.

## 6. Housekeeping

- 17.5 GB of frame caches were moved (not deleted) to `C:\Users\nicho\Downloads\WAD_removed_2026-10-13\` on
  Oct 13. They can be rebuilt by the commands if ever needed. Deleting that folder is the user's call.
- Old intermediate caches (`cache/vis2` 2.4 GB, `cache/thresh/yolo_box` 4.3 GB) are still needed for
  reproduction. Keep them through Phase 4.
