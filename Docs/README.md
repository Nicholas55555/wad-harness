# Docs — Weather Adaptive Autonomous Driving (reference set, 2026-10-22)

A complete, self-contained reference to the project so far. It covers:
- the course and the plan;
- the literature and the Phase 1 baseline;
- data and protocol;
- the harness;
- every result (weather-conditional CRC, hazard and visibility estimators, threshold exploration);
- the method (the λ dial);
- the interactive dial demo;
- a ready-to-use **Phase 3 kit**: Methodology LaTeX, architecture SVG, pseudocode, slide outline.

Numbers are copied or generated from the harness result files. Each doc names its sources, and nothing was
re-run to make them.

## Start here

| if you are… | read |
|---|---|
| preparing the **Phase 3 slides** | [`10_phase3_methodology_kit.md`](10_phase3_methodology_kit.md) → [`slides/outline.md`](slides/outline.md); figures in [`figures/phase3/`](figures/phase3/) |
| updating the **Overleaf Methodology** section | [`latex/methodology.tex`](latex/methodology.tex) + [`latex/preamble_additions.tex`](latex/preamble_additions.tex) + [`latex/refs_methodology.bib`](latex/refs_methodology.bib); compiled preview [`latex/methodology_standalone.pdf`](latex/methodology_standalone.pdf) |
| new to the project | [`00_summary.md`](00_summary.md), then [`01`](01_project_and_course_context.md) and [`08`](08_method_lambda_dial.md) |
| planning **Phase 4** (due Nov 5) | doc 10 §8, [`12_open_items_next_steps.md`](12_open_items_next_steps.md), [`04`](04_harness_and_reproducibility.md) §10 |
| answering "where does this number come from?" | docs 05 / 08, the CSVs in [`tables/`](tables/), doc 04 §9 (command per figure) |

## The documents

| doc | contents |
|---|---|
| [`00_summary.md`](00_summary.md) | one page: baseline and its gap, what failed, the method, headline results, what's next |
| [`01_project_and_course_context.md`](01_project_and_course_context.md) | course, team, deliverables and dates, research question and hypotheses, scope, objectives O1–O6 with status, risks, how the direction evolved, which paper is the baseline |
| [`02_literature_and_phase1_baseline.md`](02_literature_and_phase1_baseline.md) | DA-Detect and SeqCRC in depth (equations, data, results, limits); the Phase 1 reproduction (cods, DA-Detect); the gap in the lit review's own words; the Phase 3 baseline stated precisely |
| [`03_data_and_evaluation.md`](03_data_and_evaluation.md) | DAWN, BDD100K val/train subsets, data traps, matching, loss, FPPI, CRC protocol, verdict words, which data is used for what |
| [`04_harness_and_reproducibility.md`](04_harness_and_reproducibility.md) | the brief's rules, commands, caches, determinism, hardware, runtimes, tests, pre-registrations, how to reproduce each figure, compute plan |
| [`05_results_crc_runs.md`](05_results_crc_runs.md) | run 1, run 2 (pre-registered), the BDD100K clear-weather control, cross-dataset, found / recoverable / never, deviations; what it means for the method |
| [`06_results_hazard_and_visibility.md`](06_results_hazard_and_visibility.md) | O3 hazard estimator A and B; visibility v1, v2, v3; why none protects true fog or rain at night |
| [`07_threshold_exploration.md`](07_threshold_exploration.md) | ideas 0–10: per-image thresholds, luck vs structure, per-detection rescoring, box combinations, summed maps, graded response, label-free false alarms |
| [`08_method_lambda_dial.md`](08_method_lambda_dial.md) | **the method:** notation, option A (one threshold), option B (per detection, model 2), results, statistical backing, Fβ through λ, pseudocode, relation to CRC, files and commands |
| [`09_dial_demo_guide.md`](09_dial_demo_guide.md) | the interactive page: controls, panels, frame scorecard, four screenshots, a 30-second live-demo script |
| [`10_phase3_methodology_kit.md`](10_phase3_methodology_kit.md) | **Phase 3:** checklist → assets, core idea wording, changes vs baseline, diagram walk-through, equations, design choices with evidence, experiment plan, Q&A, timing |
| [`11_decision_log.md`](11_decision_log.md) | every decision with date and evidence |
| [`12_open_items_next_steps.md`](12_open_items_next_steps.md) | Phase 4 work, known report bugs, missing facts, unfinished objectives, data to request |
| [`13_glossary.md`](13_glossary.md) | symbols and about 90 terms, each pointing to where it is defined |
| [`14_phase4_experiments.md`](14_phase4_experiments.md) | **Phase 4:** dataset and splits, setup, the pre-registered comparison of CRC over λ with the per-weather baseline, the dial results, the ablations, figures, and the paper and submission package |

## Folders

| folder | contents |
|---|---|
| [`figures/phase3/`](figures/phase3/) | `architecture.svg/.pdf/.png` (v2, drawn with real frames and data; its inputs are in `architecture_assets.json`; the text-only v1 is kept as `architecture_v1.*`), `pseudocode_alg1/alg2/equations.svg/.png`, `baseline_price.svg/.png`, and their generators: `make_architecture.py` (`--prepare` extracts the assets from the harness caches), `render_architecture.ps1`, `make_pseudocode.py`, `make_baseline_price.py` |
| [`figures/o1_o2_run1/`](figures/o1_o2_run1/) | fig1–fig5: run 1 curves, go/no-go, main CRC figure, hazard (O3-A), α sweep |
| [`figures/o2_run2/`](figures/o2_run2/) | fig6–fig9: BDD100K clear vs hazard (two detectors), run-2 detectors, tiers, recoverable recall |
| [`figures/o3_visibility/`](figures/o3_visibility/) | fig10 (v2, v3) and the example / night-gap contact sheets |
| [`figures/threshold_exploration/`](figures/threshold_exploration/) | fig12–fig19: headroom, learned threshold, budget, graded response, all-boxes / low-threshold / summed-map examples |
| [`figures/method/`](figures/method/) | fig20–fig23: one-threshold dial, per-detection models 1 and 2, Fβ dial |
| [`figures/phase1_reproduction/`](figures/phase1_reproduction/) | the 13 figures of the Phase 1 reproducibility write-up |
| [`tables/`](tables/) | the result CSVs behind doc 08 (one threshold, per detection, Fβ, label-free, noise floor) |
| [`dial_demo/`](dial_demo/) | `dial_demo.html` (opens offline) and four screenshots |
| [`latex/`](latex/) | the Methodology section, preamble additions, new BibTeX, a compile test (`methodology_standalone.tex/.pdf`), the slide-algorithm builder, the lit review's `refs.bib` and its two TikZ figures |
| [`slides/`](slides/) | `outline.md`: 7 slides with visuals, timing, speaker notes, backup slides |

## Conventions

- **Data use:** "train" is BDD100K train (fitting only). "val" is BDD100K val, and "DAWN" is the 889 clean-label
  images unless stated otherwise. Graded sets are never used to choose anything.
- **Words that are defined, not chosen:**
  - *holds* / *at the budget* / *fails* (doc 03 §7);
  - *exploratory* means not pre-registered; *post hoc* means chosen after seeing results.
- **m(b)** is a *calibration estimate*, never a "certified miss rate". Validity of predicted bins is only
  within predicted bins.
- **Known-bad report sections** (doc 12 §2): do not cite `results.md` §6.3 or the go/no-go table in `run2.md`.
- **Regenerating Phase 3 assets:**
  - `python make_architecture.py` (run it with `--prepare` first to re-extract the photos and numbers from the
    harness caches), then `.\render_architecture.ps1`, in `figures/phase3/`;
  - `python make_pseudocode.py` and `python make_baseline_price.py`;
  - `tectonic methodology_standalone.tex` in `latex/` (tectonic is in `Downloads\weather adaptive lit review\tools\`).
- **Team:** Nicholas Taylor and Zander Picon. **Missing facts** (group name, syllabus dates) are marked **TODO**
  where they appear.
