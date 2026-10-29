"""Write results/results.md from the cache and result CSVs. Generated, never hand-edited.

Run: python -m wad.report [--smoke]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from wad.config import CANONICAL, Cfg, bins, cache_path, load_config, results_path

BL = {"fog": "Fog", "rain": "Rain", "sand": "Sand", "snow": "Snow", "all": "All"}
PL = {"oracle_bin": "Oracle-bin CRC", "global": "Global CRC", "default_0.25": "Default τ=0.25",
      "default_0.5": "Default τ=0.5", "predicted_bin": "Predicted-bin CRC"}


def md_table(df: pd.DataFrame, align: str | None = None) -> str:
    cols = list(df.columns)
    align = align or ("l" + "r" * (len(cols) - 1))
    sep = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join(sep) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def pct(v, d=1):
    return "–" if pd.isna(v) else f"{100 * v:.{d}f}%"


def num(v, d=3):
    return "–" if pd.isna(v) else f"{v:.{d}f}"


def pm(m, s, d=3):
    return "–" if pd.isna(m) else f"{m:.{d}f} ± {0 if pd.isna(s) else s:.{d}f}"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def exif_line(cfg: Cfg, r: dict) -> str:
    e = load_json(cache_path(cfg, "exif_check.json"))
    line = f"- **EXIF orientation:** counts {r['exif_orientation_counts']}."
    if e:
        line += (f" The {e['tagged/apply']['n_images']} images tagged 2 (mirrored) are annotated in the mirrored "
                 f"frame: on them, max recall is {pct(e['tagged/apply']['max_recall'])} with the tag applied and "
                 f"{pct(e['tagged/ignore']['max_recall'])} with it ignored ({e['control/apply']['n_images']} untagged "
                 f"controls: {pct(e['control/apply']['max_recall'])} both ways; cache/exif_check.json).")
    return line + " Every pixel read in this repo applies the tag."


def section_data(cfg: Cfg) -> list[str]:
    r = load_json(cache_path(cfg, "data_report.json"))
    im = pd.read_parquet(cache_path(cfg, "images.parquet"))
    gt = pd.read_parquet(cache_path(cfg, "gt.parquet"))
    kept = im[im.keep]
    groups = kept.groupby("group").size()
    in_groups = int(groups[groups > 1].sum())
    cross = [p for p in r["near_duplicate_pairs"] if p["a"].split("/")[0] != p["b"].split("/")[0]]
    n_dup_extra = sum(len(g) - 1 for g in r["exact_duplicate_groups"])
    t = gt.groupby(["weather", "cls"]).size().unstack(fill_value=0).reindex(columns=CANONICAL, fill_value=0)
    t.insert(0, "boxes", t.sum(axis=1))
    t.insert(0, "images kept", kept.groupby("weather").size())
    t.insert(0, "images found", im.groupby("weather").size())
    t.loc["total"] = t.sum()
    t = t.reset_index().rename(columns={"weather": "bin"})
    t["bin"] = t["bin"].map(lambda b: BL.get(b, b))
    out = ["## 1. Data as found", "",
           f"DAWN (Kenk & Hassaballah 2020, Mendeley Data v3, CC BY-NC 3.0): four weather folders, Pascal VOC XML "
           f"beside YOLO-darknet text labels. {r['n_images_found']} images found, {r['n_images_kept']} kept after "
           f"removing exact duplicates; {int(t.iloc[-1]['boxes'])} ground-truth boxes in the six canonical classes.", "",
           md_table(t.astype({c: int for c in t.columns if c != 'bin'}).astype(str)), "",
           f"- **Exact duplicates (SHA-256):** {len(r['exact_duplicate_groups'])} groups; the {n_dup_extra} extra "
           f"copies are excluded from everything downstream (all groups lie within one weather folder).",
           f"- **Near duplicates (perceptual hash ≤ {cfg.data.phash_max_hamming} bits, mirror-aware):** "
           f"{len(r['near_duplicate_pairs'])} pairs forming {r['n_near_duplicate_groups']} groups that cover "
           f"{in_groups} of the {r['n_images_kept']} kept images. A mirror-aware 64×64 grey-level correlation over "
           f"all {r['near_duplicate_pixel_check']['n_pairs']} pairs has minimum "
           f"{r['near_duplicate_pixel_check']['min_rho']:.3f} ({r['near_duplicate_pixel_check']['n_rho_ge_0_99']} pairs "
           f"≥ 0.99): the same photograph resized, re-encoded or mirrored (full list: `cache/duplicates.csv`). They "
           f"are kept, as the brief asks, but every "
           f"calibration/test split keeps a group on one side. {len(cross)} pairs carry *different* weather labels "
           f"(e.g. {cross[0]['a']} = {cross[0]['b']}), so the weather label itself is noisy.",
           f"- **Classes:** {r['dropped_boxes']} box(es) of a class outside the six were dropped; no box is flagged "
           f"difficult ({r['n_flagged_difficult']}) or truncated ({r['n_flagged_truncated']}).",
           f"- **Images without ground truth:** {', '.join(r['images_without_gt'])} (loss 0 by definition).",
           f"- **Pixel formats:** {r['image_modes']}; non-RGB files converted on read ({', '.join(r['non_rgb_images'])}).",
           exif_line(cfg, r),
           f"- **Image sizes:** {r['n_distinct_sizes']} distinct sizes, widths {r['width_range']}, heights "
           f"{r['height_range']}.", ""]
    return out


def section_detector(cfg: Cfg) -> list[str]:
    m = load_json(cache_path(cfg, "detections.meta.json"))
    s = load_json(cache_path(cfg, "detection_sanity.json"))
    return ["## 2. Detector and cache", "",
            f"`{m['model_file']}` (COCO weights, SHA-256 `{m['model_sha256'][:16]}…`), Ultralytics "
            f"{m['ultralytics_version']}, torch {m['torch_version']}, Python {m['python_version']}, device "
            f"`{m['device']}`; imgsz {m['imgsz']}, conf floor {m['conf_floor']}, NMS IoU {m['nms_iou']}, max_det "
            f"{m['max_det']}, classes {list(m['classes'])}. One forward pass per image; {m['n_images']} images in "
            f"{m['wall_time_s']:.0f} s ({m['mean_s_per_image']:.3f} s/image); {m['n_detections']:,} cached boxes.", "",
            f"Sanity: {s['n_zero_detection_images']} images ({100 * s['frac_zero_detection']:.1f}%) have no detection "
            f"at the floor; detections per image min {s['dets_per_image']['min']}, median "
            f"{s['dets_per_image']['median']:.0f}, max {s['dets_per_image']['max']} "
            f"({s['dets_per_image']['n_at_max_det']} images hit max_det); score deciles "
            f"{', '.join(f'{v:.4f}' for v in s['score_deciles'].values())}; class shares "
            f"{', '.join(f'{k} {100 * v:.1f}%' for k, v in s['class_fraction'].items())}.", ""]


def section_phase3(cfg: Cfg) -> list[str]:
    s = pd.read_csv(results_path(cfg, "summary_phase3.csv"))
    v = load_json(results_path(cfg, "go_no_go.json"))
    bs = load_json(results_path(cfg, "extra/go_no_go_bootstrap.json"))
    bt = pd.read_csv(results_path(cfg, "extra/go_no_go_bootstrap.csv")) if bs else None
    t25, t50 = cfg.default_thresholds
    rows = []
    for _, d in s[s.cls.isin(["all", "person", "car"])].iterrows():
        rows.append({"bin": BL[d["bin"]], "class": d["cls"], "images": d["n_images"], "GT": d["n_gt"],
                     "max recall": pct(d["max_recall"]), f"recall@{t25}": pct(d[f"recall_at_{t25}"]),
                     f"recall@{t50}": pct(d[f"recall_at_{t50}"]),
                     "margin (pts)": num(d["margin_points"], 1), "margin vs 0.5": num(d["margin_points_vs_0_5"], 1),
                     "never matched (pts)": num(d["irrecoverable_points"], 1),
                     "LAMR": num(d["lamr"]), "LAMR Caltech": num(d["lamr_caltech"]),
                     "MR@0.1 FPPI": num(d["mr_at_fppi_0.1"]), "MR@1 FPPI": num(d["mr_at_fppi_1.0"]),
                     "mAP50": num(d["map50"])})
    out = ["## 3. Phase 3: curves, maximum achievable recall, go/no-go", "",
           f"**{v['line']}**", ""]
    if bs:
        pb = bt[(bt.cls == "person") & (bt.bin != "worst")].sort_values("p_worst_bin", ascending=False)
        w1, w2 = pb.iloc[0], pb.iloc[1]
        wc = bt[(bt.cls == "person") & (bt.bin == "worst")].iloc[0]
        cc = bt[(bt.cls == "car") & (bt.bin == "worst")].iloc[0]
        out += [f"Bootstrap check (beyond the brief; {bs['n_boot']} resamples of near-duplicate groups within each "
                f"weather bin, seed {bs['seed']}): the verdict is GO in **{100 * bs['p_go']:.0f}%** of resamples "
                f"(person passes {100 * bs['p_pass']['person']:.0f}%, car {100 * bs['p_pass']['car']:.0f}%). "
                f"For person the worst bin is a near-tie (P(worst) {BL[w1.bin].lower()} {w1.p_worst_bin:.2f}, "
                f"{BL[w2.bin].lower()} {w2.p_worst_bin:.2f}; margins {w1.margin_points:.1f} and {w2.margin_points:.1f} "
                f"points on {int(w1.n_gt)} and {int(w2.n_gt)} boxes), so the person verdict is fragile. Worst-bin "
                f"margin 95% interval: person {wc.margin_ci_lo:.1f} to {wc.margin_ci_hi:.1f} points, car "
                f"{cc.margin_ci_lo:.1f} to {cc.margin_ci_hi:.1f}.", ""]
    out += ["Per weather bin (object-level recall at IoU 0.5, class-aware). *Margin* = max recall − recall at "
            f"τ={t25} (recoverable by lowering the threshold); *never matched* = 100 − max recall (not recoverable by "
            "any threshold). LAMR follows the brief (arithmetic mean of the interpolated miss rate at nine FPPI points "
            "in [0.01, 1]); the Caltech column is Dollár et al.'s geometric-mean, stepwise version. MR@x FPPI is the "
            "plan's headline operating point. Every class, including the four minor ones, is in "
            "`results/summary_phase3.csv`; full curves in `results/curves.csv`.", "",
            md_table(pd.DataFrame(rows), "llrr" + "r" * 11), ""]
    m = s[s.bin != "all"].pivot_table(index="cls", columns="bin", values="margin_points").reindex(CANONICAL)
    m2 = s[s.bin != "all"].pivot_table(index="cls", columns="bin", values="irrecoverable_points").reindex(CANONICAL)
    g = s[s.bin != "all"].pivot_table(index="cls", columns="bin", values="n_gt").reindex(CANONICAL)
    tab = pd.DataFrame({"class": CANONICAL})
    for b in bins(cfg):
        tab[BL[b]] = [f"{m.loc[c, b]:.1f} / {m2.loc[c, b]:.1f} (n={int(g.loc[c, b])})" if g.loc[c, b] > 0 else "no boxes"
                      for c in CANONICAL]
    out += [f"Recoverable / never-matched points per bin and class (margin against τ={t25} / 100 − max recall; "
            "n = ground-truth boxes):", "", md_table(tab, "lrrrr"), ""]
    return out


def section_phase4(cfg: Cfg) -> list[str]:
    s = pd.read_csv(results_path(cfg, "crc_summary.csv"))
    s = s[s.cls == "all"]
    n_seeds = int(s.n_seeds.max())
    out = ["## 4. Phase 4: conformal risk control per weather bin", "",
           f"Rule: τ̂ = max{{τ on the 200-point grid : n/(n+1)·R̂(τ) + 1/(n+1) ≤ α}} with R̂ the mean per-image miss "
           f"fraction on the calibration half; τ̂ = {cfg.detector.conf_floor} and *infeasible* when no grid point "
           f"qualifies. {n_seeds} splits (seeds {cfg.splits.seed}–{cfg.splits.seed + n_seeds - 1}), each half of every "
           "weather bin, near-duplicate groups kept together. Test metrics are on the held-out half: image loss = mean "
           "per-image miss fraction (the controlled quantity), FPPI = false positives per test image, exceed = "
           "fraction of splits whose test image loss is above α. Values are mean ± sd over splits; n_cal / n_test are "
           "images per split.", ""]
    for alpha in cfg.alphas:
        d = s[s.alpha == alpha]
        rows = []
        for p in ["oracle_bin", "global", "default_0.25", "default_0.5"]:
            for b in bins(cfg) + ["all"]:
                r = d[(d.policy == p) & (d.bin == b)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append({"policy": PL[p], "bin": BL[b],
                             "test image loss": pm(r.image_loss_mean, r.image_loss_std),
                             "object miss rate": pm(r.miss_rate_mean, r.miss_rate_std),
                             "FPPI": pm(r.fppi_mean, r.fppi_std, 2),
                             "τ̂": "per bin" if (p == "oracle_bin" and b == "all") else pm(r.tau_hat_mean, r.tau_hat_std, 4),
                             "exceed α": f"{r.exceed_frac:.1f}",
                             "infeasible": ("–" if pd.isna(r.infeasible_frac) else
                                            f"{r.infeasible_frac:.2f}" + ("†" if p == "oracle_bin" and b == "all" else "")),
                             "n_cal": "–" if pd.isna(r.n_cal_mean) else f"{r.n_cal_mean:.0f}",
                             "n_test": f"{r.n_test_mean:.0f}"})
        out += [f"### α = {alpha}", "", md_table(pd.DataFrame(rows), "llrrrrrrrr"), "",
                "† for the oracle-bin *All* row: share of all (bin, split) fits that were infeasible; elsewhere the "
                "share of splits.", ""]
    out += [phase4_reading(cfg, s), ""]
    return out


def phase4_reading(cfg: Cfg, s: pd.DataFrame) -> str:
    """The reading paragraph, with every number pulled from the result files."""
    a1, a2 = cfg.alphas[0], cfg.alphas[1]
    cur = pd.read_csv(results_path(cfg, "curves.csv"))
    fl = cur[(cur.cls == "all") & (cur.bin.isin(bins(cfg))) & (cur.tau == cur.tau.min())].set_index("bin")
    o1 = s[(s.alpha == a1) & (s.policy == "oracle_bin") & (s.bin != "all")].set_index("bin")
    o2 = s[(s.alpha == a2) & (s.policy == "oracle_bin") & (s.bin != "all")].set_index("bin")
    g2 = s[(s.alpha == a2) & (s.policy == "global") & (s.bin != "all")].set_index("bin")
    d2 = s[(s.alpha == a2) & (s.policy == "default_0.25") & (s.bin != "all")].set_index("bin")
    feas2 = o2[o2.infeasible_frac < 0.5]
    mp = results_path(cfg, "extra/min_feasible_alpha.csv")
    ma = pd.read_csv(mp) if mp.exists() else None
    fits = pd.read_csv(results_path(cfg, "crc_fits.csv"))
    ev = pd.read_csv(results_path(cfg, "crc_eval.csv"))
    ok1 = fits[(fits.alpha == a1) & (fits.policy == "oracle_bin") & (~fits.infeasible)]
    txt = (f"**Reading.** At α = {a1}, oracle-bin fits are infeasible in "
           + ", ".join(f"{BL[b]} {o1.loc[b, 'infeasible_frac']:.0%}" for b in bins(cfg))
           + f" of splits: even at the {cfg.detector.conf_floor} floor the mean per-image miss fraction is "
           f"{fl.image_loss.min():.3f}–{fl.image_loss.max():.3f} across bins, so α = {a1} cannot be certified in most "
           f"splits of any bin, and both CRC policies collapse to (or next to) the floor at "
           f"{o1.fppi_mean.min():.0f}–{o1.fppi_mean.max():.0f} FPPI. ")
    if len(ok1):
        t = ok1.merge(ev[(ev.alpha == a1) & (ev.policy == "oracle_bin") & (ev.cls == "all")], on=["seed", "alpha", "policy", "bin"])
        txt += (f"In the {len(ok1)} splits where it was certified ("
                + ", ".join(f"{BL[r.bin]} seed {r.seed}" for r in t.itertuples())
                + f", τ̂ just above the floor), the held-out loss was "
                + ", ".join(f"{r.image_loss:.3f}" for r in t.itertuples())
                + f", above α: a marginal guarantee does not promise any single split. ")
    if ma is not None:
        txt += ("The smallest budget the rule can certify (calibration halves, mean over splits) is "
                + ", ".join(f"{BL[r.bin]} {r.min_alpha_mean:.3f}" for r in ma.itertuples() if r.bin != "all")
                + f" (pooled {ma[ma.bin == 'all'].min_alpha_mean.iloc[0]:.3f}). ")
    txt += (f"At α = {a2}, oracle-bin CRC is feasible in "
            + (", ".join(BL[b] for b in feas2.index) or "no bin")
            + " (test loss " + ", ".join(f"{BL[b]} {feas2.loc[b, 'image_loss_mean']:.3f}" for b in feas2.index)
            + f"; τ̂ " + ", ".join(f"{BL[b]} {feas2.loc[b, 'tau_hat_mean']:.4f}" for b in feas2.index)
            + ") and infeasible in " + (", ".join(BL[b] for b in o2.index if b not in feas2.index) or "none")
            + ". The global threshold's test loss per bin is "
            + ", ".join(f"{BL[b]} {g2.loc[b, 'image_loss_mean']:.3f} (exceeds α in {g2.loc[b, 'exceed_frac']:.0%})"
                        for b in bins(cfg))
            + f": one threshold over-delivers in the easy bins and under-delivers in the hard ones. The default "
            f"τ = 0.25 misses α by {(d2.image_loss_mean - a2).min():.2f}–{(d2.image_loss_mean - a2).max():.2f} in every "
            f"bin. The price of the budget is FPPI: {feas2.fppi_mean.min():.1f}–{feas2.fppi_mean.max():.1f} false "
            f"positives per image in the feasible oracle bins at α = {a2}, against "
            f"{d2.fppi_mean.min():.2f}–{d2.fppi_mean.max():.2f} at τ = 0.25. The 10-split means carry split noise; "
            "section 6.1 repeats this with 200 splits.")
    return txt


def section_perclass(cfg: Cfg) -> list[str]:
    p = pd.read_csv(results_path(cfg, "crc_perclass.csv"))
    rows = []
    for r in p.itertuples():
        rows.append({"class": r.cls, "bin": BL[r.bin], "α": r.alpha,
                     "test image loss": pm(r.image_loss_mean, r.image_loss_std),
                     "object miss rate": pm(r.miss_rate_mean, r.miss_rate_std),
                     "FPPI (class)": pm(r.fppi_mean, r.fppi_std, 2),
                     "τ̂": "–" if r.bin == "all" else pm(r.tau_hat_mean, r.tau_hat_std, 4),
                     "infeasible": "–" if r.bin == "all" else f"{r.infeasible_frac:.1f}",
                     "exceed α": f"{r.exceed_frac:.1f}",
                     "n_cal (imgs w/ class)": "–" if r.bin == "all" else f"{r.n_cal_mean:.1f}",
                     "n_test (imgs w/ class)": f"{r.n_loss_images_mean:.1f}"})
    pa = cfg.perclass_alphas
    pr = p[(p.cls == "person") & (p.bin != "all")].sort_values("n_cal_mean").iloc[0]
    n_inf = int((p[p.bin != "all"].infeasible_frac >= 0.5).sum())
    n_all = int((p.bin != "all").sum())
    return ["## 5. Per-class budgets (O4 preview)", "",
            f"Oracle-bin CRC with a per-class loss (miss fraction among class-c boxes, over images containing class c); "
            f"α = {pa['person']} for person, {pa['default']} for the rest. FPPI counts that class's false positives per "
            "test image. Preview only, one table.", "",
            md_table(pd.DataFrame(rows), "llrrrrrrrrr"), "",
            f"{n_inf} of {n_all} bin × class fits are infeasible in at least half the splits: the per-class losses do "
            "not fall below the budgets even at the floor, and for the rare classes the 1/(n+1) reserve alone exceeds "
            f"α (e.g. {BL[pr.bin].lower()} has {pr.n_cal_mean:.1f} calibration images containing a person on average, "
            f"so 1/(n+1) ≈ {1 / (pr.n_cal_mean + 1):.2f} against α = {pa['person']}). A tighter budget for people is "
            "not certifiable on DAWN with this detector.", ""]


def section_extras(cfg: Cfg) -> list[str]:
    out = ["## 6. Beyond the brief: robustness and diagnosis", "",
           "None of these changes a pre-registered choice; each is labelled with what it is.", ""]
    p200 = results_path(cfg, "sensitivity/seeds200/crc_summary.csv")
    pnd = results_path(cfg, "sensitivity/near_dup_dropped/crc_summary.csv")
    base = pd.read_csv(results_path(cfg, "crc_summary.csv"))
    if p200.exists() and pnd.exists():
        a = cfg.alphas[1]
        rows = []
        for name, path in [("primary (10 splits)", results_path(cfg, "crc_summary.csv")), ("200 splits", p200),
                           ("near-duplicates dropped (10 splits)", pnd)]:
            d = pd.read_csv(path)
            d = d[(d.cls == "all") & (d.alpha == a)]
            for p in ["oracle_bin", "global"]:
                row = {"analysis": name, "policy": PL[p]}
                for b in bins(cfg) + ["all"]:
                    r = d[(d.policy == p) & (d.bin == b)].iloc[0]
                    row[BL[b]] = f"{r.image_loss_mean:.3f} / {r.fppi_mean:.1f} / {r.exceed_frac:.2f}"
                rows.append(row)
        d200 = pd.read_csv(p200)
        d200 = d200[(d200.cls == "all") & (d200.alpha == a)]
        o = d200[(d200.policy == "oracle_bin") & (d200.bin.isin(bins(cfg)))].set_index("bin")
        g = d200[(d200.policy == "global") & (d200.bin.isin(bins(cfg)))].set_index("bin")
        feas = o[o.infeasible_frac < 0.5]
        worst_g = g.image_loss_mean.idxmax()
        out += [f"### 6.1 Robustness of the phase 4 result (α = {a}; cells: test image loss / FPPI / exceed fraction)", "",
                md_table(pd.DataFrame(rows), "llrrrrr"), "",
                f"With 200 splits the oracle-bin test loss in the bins where it is feasible averages "
                + ", ".join(f"{BL[b]} {feas.loc[b, 'image_loss_mean']:.3f} (exceeds α in {feas.loc[b, 'exceed_frac']:.0%})"
                            for b in feas.index)
                + f", and the global policy's worst bin, {BL[worst_g]}, averages {g.loc[worst_g, 'image_loss_mean']:.3f} "
                f"and exceeds α in {g.loc[worst_g, 'exceed_frac']:.0%} of splits. Differences from the 10-split table are "
                "split noise. Dropping near-duplicates (one image per group, cross-weather groups removed) leaves the "
                "pattern unchanged.", ""]
    mr = results_path(cfg, "extra/miss_reasons.csv")
    if mr.exists():
        m = pd.read_csv(mr)
        cf = pd.read_csv(results_path(cfg, "extra/miss_class_confusions.csv"))
        sz = pd.read_csv(results_path(cfg, "extra/misses_by_size.csv"))
        tab = m.pivot_table(index="reason", columns="bin", values="share_of_all_gt").reindex(
            ["no_proposal", "wrong_class", "loose_box", "taken"])[bins(cfg) + ["all"]]
        tab = tab.map(lambda v: pct(v)).reset_index()
        tab.columns = ["reason (share of all GT boxes)"] + [BL[b] for b in bins(cfg) + ["all"]]
        s_all = sz[(sz.bin == "all") & (sz.cls == "all") & (sz["size"] != "all")]
        size_tab = pd.DataFrame({"box size in the 640-px input (√area, px)": s_all["size"],
                                 "share of boxes": s_all.share_of_boxes.map(pct),
                                 "never matched": s_all.never_proposed.map(pct),
                                 "missed at τ=0.25": s_all.missed_at_0_25.map(pct),
                                 "share of all never-matched": s_all.share_of_never_proposed.map(pct)})
        out += ["### 6.2 What the irrecoverable miss is made of", "",
                "Each ground-truth box that no detection matches even at the 0.001 floor, classified in order: a "
                "same-class box overlapped it at IoU ≥ 0.5 but was taken by another object (*taken*); a box of another "
                "class overlapped it at IoU ≥ 0.5 (*wrong_class*); the best same-class box reached only IoU 0.3–0.5 "
                "(*loose_box*); nothing of the right class came within IoU 0.3 (*no_proposal*).", "",
                md_table(tab, "lrrrrr"), "",
                "Most frequent confusions (ground truth → predicted): "
                + ", ".join(f"{r.cls}→{r.other} {r.n}" for r in cf.head(5).itertuples()) + ".", "",
                md_table(size_tab, "lrrrr"), ""]
        small = s_all[s_all["size"].isin(["<16", "16-32"])]
        mall = m[m.bin == "all"].set_index("reason")
        out += [f"Small objects dominate: boxes under 32 px in the detector's input are {pct(small.share_of_boxes.sum(), 0)} "
                f"of the data but {pct(small.share_of_never_proposed.sum(), 0)} of the never-matched boxes. Only "
                f"{pct(mall.loc['no_proposal', 'share_of_never_matched'], 0)} of the never-matched boxes are a true "
                f"non-detection; {pct(mall.loc['wrong_class', 'share_of_never_matched'], 0)} are found with the wrong "
                f"class (mostly car/truck/bus) and {pct(mall.loc['loose_box', 'share_of_never_matched'], 0)} with a box "
                "too loose for IoU 0.5. A decision layer that treats any vehicle as an obstacle would not count a truck "
                "detected as a car as a miss (NOTES.md).", ""]
        sens = results_path(cfg, "sensitivity/yolov8s_1280/summary_phase3.csv")
        if sens.exists():
            out += section_resolution(cfg)
    sw = results_path(cfg, "extra/alpha_sweep_EXPLORATORY.csv")
    if sw.exists():
        w = pd.read_csv(sw)
        w = w[(w.policy == "oracle_bin") & (w.bin != "all") & (w.alpha.isin([0.15, 0.2, 0.25, 0.3]))]
        piv = w.assign(cell=w.apply(lambda r: f"{r.image_loss_mean:.3f} / {r.fppi_mean:.1f}"
                                    + (" (inf.)" if r.infeasible_frac >= 0.5 else ""), axis=1)).pivot(
            index="alpha", columns="bin", values="cell")[bins(cfg)].reset_index()
        piv.columns = ["α"] + [BL[b] for b in bins(cfg)]
        out += ["### 6.4 EXPLORATORY, not pre-registered: the price of each budget", "",
                "The phase 4 procedure (same splits and rule) over a grid of budgets, oracle-bin policy; cells are test "
                "image loss / FPPI, *(inf.)* = infeasible in at least half the splits. This answers 'what would it "
                "cost' and must not be used to pick α after the fact (`results/figures/extra/fig5_alpha_sweep.png`).",
                "", md_table(piv, "lrrrr"), ""]
    return out


def section_resolution(cfg: Cfg) -> list[str]:
    """Sensitivity: the same YOLOv8s weights at imgsz 1280 (wad/sensitivity.py)."""
    base = pd.read_csv(results_path(cfg, "summary_phase3.csv"))
    hi = pd.read_csv(results_path(cfg, "sensitivity/yolov8s_1280/summary_phase3.csv"))
    vb = load_json(results_path(cfg, "go_no_go.json"))
    vh = load_json(results_path(cfg, "sensitivity/yolov8s_1280/go_no_go.json"))
    mb = pd.read_csv(results_path(cfg, "extra/min_feasible_alpha.csv")).set_index("bin")
    mh = pd.read_csv(results_path(cfg, "sensitivity/yolov8s_1280/min_feasible_alpha.csv")).set_index("bin")
    meta = load_json(cfg.paths.cache / "sensitivity" / "yolov8s_1280" / "detections.meta.json")
    ch = pd.read_csv(results_path(cfg, "sensitivity/yolov8s_1280/crc_summary.csv"))
    cb = pd.read_csv(results_path(cfg, "crc_summary.csv"))
    rows = []
    for b in bins(cfg) + ["all"]:
        for cls in ["all", "person", "car"]:
            x = base[(base.bin == b) & (base.cls == cls)].iloc[0]
            y = hi[(hi.bin == b) & (hi.cls == cls)].iloc[0]
            rows.append({"bin": BL[b], "class": cls,
                         "max recall 640 → 1280": f"{pct(x.max_recall)} → {pct(y.max_recall)}",
                         "recall@0.25 640 → 1280": f"{pct(x['recall_at_0.25'])} → {pct(y['recall_at_0.25'])}",
                         "LAMR 640 → 1280": f"{x.lamr:.3f} → {y.lamr:.3f}",
                         "smallest certifiable α 640 → 1280": (f"{mb.loc[b, 'min_alpha_mean']:.3f} → "
                                                              f"{mh.loc[b, 'min_alpha_mean']:.3f}") if cls == "all" else ""})
    a = cfg.alphas[1]
    crow = []
    for name, d in [("640", cb), ("1280", ch)]:
        d = d[(d.cls == "all") & (d.alpha == a) & (d.policy == "oracle_bin")].set_index("bin")
        crow.append({"imgsz": name, **{BL[b]: f"{d.loc[b, 'image_loss_mean']:.3f} / {d.loc[b, 'fppi_mean']:.1f}"
                                         + (" (inf.)" if d.loc[b, "infeasible_frac"] >= 0.5 else "")
                                         for b in bins(cfg)}})
    return ["### 6.3 Sensitivity: the same YOLOv8s weights at input size 1280", "",
            f"DAWN images are mostly 1280 px wide, so imgsz 640 halves every object. Re-running the unchanged COCO "
            f"weights at imgsz 1280 ({meta['mean_s_per_image']:.2f} s/image on {meta['device']}, "
            f"{meta['n_detections']:,} boxes) through the identical pipeline tests whether the irrecoverable loss is a "
            f"resolution limit. Phase 3 verdict at 1280: {vh['verdict']} (640: {vb['verdict']}).", "",
            md_table(pd.DataFrame(rows), "llrrrr"), "",
            f"Oracle-bin CRC at α = {a} (cells: test image loss / FPPI over {int(ch.n_seeds.max())} splits):", "",
            md_table(pd.DataFrame(crow), "lrrrr"), "", resolution_reading(cfg, base, hi, meta), ""]


def resolution_reading(cfg: Cfg, base: pd.DataFrame, hi: pd.DataFrame, meta: dict) -> str:
    d = pd.read_csv(results_path(cfg, "sensitivity/yolov8s_1280/lost_gained.csv"))
    lost, gained = d[d.status == "lost"], d[d.status == "gained"]
    why = lost.variant_reason.value_counts(normalize=True)
    x = base[(base.bin == "all") & (base.cls == "all")].iloc[0]
    y = hi[(hi.bin == "all") & (hi.cls == "all")].iloc[0]
    return (f"Pooled max recall goes from {pct(x.max_recall)} to {pct(y.max_recall)}: {len(gained)} boxes are matched "
            f"only at 1280 and {len(lost)} only at 640 (`lost_gained.csv`). The gained boxes are small (median side "
            f"{gained.side_px.median():.0f} px in the original image), the lost ones larger (median "
            f"{lost.side_px.median():.0f} px). At 1280 the lost boxes are mostly still seen: "
            f"{pct(why.get('wrong_class', 0), 0)} are detected with another class, {pct(why.get('loose_box', 0), 0)} with "
            f"a box too loose for IoU 0.5, and {pct(why.get('no_proposal', 0), 0)} have no proposal; "
            f"{int(lost.variant_max_det_binds.sum())} sit in images where max_det = {cfg.detector.max_det} binds. "
            "Doubling the input trades large-object accuracy for small-object recall, so the never-matched loss is "
            "not simply a resolution limit of the 640 setting.")


def section_hazard(cfg: Cfg) -> list[str]:
    hp = results_path(cfg, "hazard_summary.csv")
    if not hp.exists():
        return []
    s = pd.read_csv(hp)
    pca = load_json(cache_path(cfg, "hazard_pca.json"))
    head = load_json(results_path(cfg, "hazard_headline.json"))
    cont = pd.read_csv(results_path(cfg, "hazard_contingency.csv"))
    cont["weather"] = cont["weather"].map(BL)
    hz = pd.read_parquet(cache_path(cfg, "hazard.parquet"))
    means = hz.groupby("weather")[["dark_channel", "rms_contrast", "edge_energy", "saturation", "h"]].mean()
    means = means.reset_index()
    means["weather"] = means["weather"].map(BL)
    means = means.round(4).astype(str)
    rows = []
    for alpha in cfg.alphas:
        for b in ["h1", "h2", "h3", "h4", "all"]:
            r = s[(s.alpha == alpha) & (s.policy == "predicted_bin") & (s.bin == b)].iloc[0]
            o = s[(s.alpha == alpha) & (s.policy == "oracle_bin") & (s.bin == b)].iloc[0]
            rows.append({"α": alpha, "predicted bin": b, "test image loss": pm(r.image_loss_mean, r.image_loss_std),
                         "FPPI": pm(r.fppi_mean, r.fppi_std, 2), "τ̂": "–" if b == "all" else pm(r.tau_hat_mean, r.tau_hat_std, 4),
                         "infeasible": "–" if b == "all" else f"{r.infeasible_frac:.1f}", "exceed α": f"{r.exceed_frac:.1f}",
                         "oracle-bin policy on the same images: loss / FPPI": f"{o.image_loss_mean:.3f} / {o.fppi_mean:.1f}",
                         "n_test": f"{r.n_test_mean:.0f}"})
    gap = head["fppi_gap"]
    return ["## 7. Phase 6a: hazard scalar and predicted bins (O3, version A)", "",
            "h is computed from pixels only (`wad/hazard.py` imports no detection code; tested). Loadings of the first "
            "principal component of the standardised features: "
            + ", ".join(f"{k} {v:+.3f}" for k, v in pca["loadings"].items())
            + f" ({100 * pca['explained_variance_ratio']:.0f}% of variance), signed so higher h means lower contrast "
            "and edge energy. PCA is fitted on all kept images without labels; quartile cut points on each calibration "
            "half.", "", md_table(means, "lrrrrr"), "",
            f"Agreement with weather labels is low: adjusted Rand index {head['ari_mean']:.3f} ± {head['ari_std']:.3f}. "
            "The first component is mostly a sand-versus-rest axis (saturation against dark channel), not a monotone "
            "degradation score: sand fills h4, fog and snow sit low. Contingency (mean image counts over splits):", "",
            md_table(cont.round(1).astype(str), "lrrrr"), "",
            md_table(pd.DataFrame(rows), "lrrrrrrrr"), "",
            "FPPI gap of the predicted-bin policy to the oracle-bin policy over all test images (mean ± sd over "
            "splits): " + ", ".join(f"α = {a}: {v['mean']:+.2f} ± {v['std']:.2f}" for a, v in gap.items())
            + ". Despite the low agreement the predicted bins hold the budget about as well where feasible: validity "
            "does not depend on how well h tracks the weather (only approximately here, because the quartile cut "
            "points are estimated on the calibration half), but they spend more false positives than the oracle bins "
            "at α = 0.2.", ""]


def section_caveats(cfg: Cfg) -> list[str]:
    s = pd.read_csv(results_path(cfg, "crc_summary.csv"))
    inf = s[(s.policy == "oracle_bin") & (s.cls == "all") & (s.bin != "all") & (s.infeasible_frac > 0)]
    san = load_json(cache_path(cfg, "detection_sanity.json"))
    r = load_json(cache_path(cfg, "data_report.json"))
    p3 = pd.read_csv(results_path(cfg, "summary_phase3.csv"))
    v = load_json(results_path(cfg, "go_no_go.json"))
    bs = load_json(results_path(cfg, "extra/go_no_go_bootstrap.json"))
    pers = p3[(p3.cls == "person") & (p3.bin.isin(bins(cfg)))].sort_values("max_recall")
    runner = pers.iloc[1]
    cross = [p for p in r["near_duplicate_pairs"] if p["a"].split("/")[0] != p["b"].split("/")[0]]
    gt = pd.read_parquet(cache_path(cfg, "gt.parquet"))
    cnt = gt.groupby(["weather", "cls"]).size().unstack(fill_value=0).reindex(columns=CANONICAL, fill_value=0)
    rare = [f"{BL[b].lower()} {cnt.loc[b, k]} {k}" for b in cnt.index for k in CANONICAL if cnt.loc[b, k] < 30]
    return ["## 8. Caveats", "",
            "- **Infeasible bins.** Oracle-bin fits with no feasible threshold (fraction of splits): "
            + ", ".join(f"α={x.alpha} {BL[x.bin]} {x.infeasible_frac:.1f}" for x in inf.itertuples())
            + f". There τ̂ is the {cfg.detector.conf_floor} floor and the conformal guarantee does not hold: the loss "
            "does not vanish at the floor.",
            f"- **The {v['verdict']} verdict is fragile for person**"
            + (f" (bootstrap P(GO) = {bs['p_go']:.2f})" if bs else "")
            + f": the worst person bin is a near tie between {BL[pers.iloc[0].bin].lower()} and "
            f"{BL[runner.bin].lower()} (max recall {pct(pers.iloc[0].max_recall)} vs {pct(runner.max_recall)}), and "
            f"{BL[runner.bin].lower()}'s margin is {runner.margin_points:.1f} points against the "
            f"{cfg.go_no_go.margin_points}-point rule.",
            f"- **Duplicates.** {sum(len(g) - 1 for g in r['exact_duplicate_groups'])} exact duplicates excluded; "
            f"{len(r['near_duplicate_pairs'])} near-duplicate pairs kept but never split across halves; {len(cross)} "
            "pairs carry different weather labels.",
            f"- **Zero-detection images:** {san['n_zero_detection_images']} ({', '.join(san['zero_detection_images'])}); "
            f"{san['dets_per_image']['n_at_max_det']} images reach max_det = {cfg.detector.max_det}, which may cut "
            "low-score true positives there (NOTES.md).",
            "- **Small samples** (fewer than 30 boxes): " + ", ".join(rare)
            + "; per-class results for these rest on a handful of images per bin.",
            "- **The guarantee is marginal and per bin**, an expectation over calibration draws; a split-half design on "
            "a fixed pool makes calibration and test halves negatively correlated, so single splits swing around α.",
            "- **Detector input resolution.** DAWN images are mostly 1280 px wide and are downscaled to 640; most "
            "never-detected objects are small at that scale (section 6.2).", ""]


def write(cfg: Cfg) -> str:
    v = load_json(results_path(cfg, "go_no_go.json"))
    lines = ["# Weather Adaptive Autonomous Driving — first results (O1, O2, O3-A)", "",
             "Generated by `python -m wad.report` from `cache/` and `results/*.csv`; do not edit by hand. "
             "`python scripts/run_all.py --from-cache` recomputes every number below from the cached detections (fixed "
             "seeds; identical CSVs on every run). Section 6.3 additionally needs the imgsz-1280 detections, produced "
             "once by `python -m wad.sensitivity` (about 7 minutes); run_all recomputes 6.3 from them when present.", "",
             "**Figures:** `results/figures/fig1_missrate_fppi.png`, `fig2_go_no_go.png`, `fig3_main.png` (main), "
             "`fig4_hazard.png`; exploratory `figures/extra/fig5_alpha_sweep.png`.", "",
             f"**{v['line']}**", ""]
    for sec in (section_data, section_detector, section_phase3, section_phase4, section_perclass, section_extras,
                section_hazard, section_caveats):
        lines += sec(cfg)
    text = "\n".join(lines).rstrip() + "\n"
    results_path(cfg, "results.md").write_text(text, encoding="utf-8")
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke)
    write(cfg)
    print(results_path(cfg, "results.md"))


if __name__ == "__main__":
    main()
