"""Phase 4 paper assets: the LaTeX tables and figures of the Experiments section, generated from result files.

Reads results/crc_lambda (wad.crc_lambda), results/ablation (wad.dial_ablation, wad.dial_recalibration,
wad.detector_ablation), results/threshold (per_detection_yolo_eval.csv, the training log) and the adopted model, and
writes Docs/latex/phase4/:
  tables/tab_data.tex, tab_crc.tex, tab_crc_floor.tex, tab_dial.tex, tab_ablation.tex, tab_recal.tex, tab_detectors.tex
  figures/fig_p4_price.pdf/.png        CRC over lam vs the per-weather baseline: false alarms at alpha 0.2, ratio vs alpha
  figures/fig_p4_training.pdf/.png     training evidence: log-loss per boosting iteration, learning curve
  figures/fig_p4_qualitative.pdf/.png  baseline vs ours on held-out frames (split 0, alpha 0.2)
  figures/fig_p4_logs.png              excerpts of the saved run logs
Run: python -m wad.phase4_paper
"""
from __future__ import annotations

import copy
import textwrap

import numpy as np
import pandas as pd

from wad.config import REPO, THRESH_RESULTS

OUT = REPO / "Docs" / "latex" / "phase4"
CRC = REPO / "results" / "crc_lambda"
ABL = REPO / "results" / "ablation"
ALPHA = 0.2
COND = [("BDD100K", "clear", "BDD clear day"), ("BDD100K", "cloudy", "BDD cloudy day"), ("BDD100K", "rain", "BDD rain day"),
        ("BDD100K", "snow", "BDD snow day"), ("BDD100K", "clear_night", "BDD clear night"),
        ("BDD100K", "rain_night", "BDD rain night"), ("BDD100K", "snow_night", "BDD snow night"),
        ("DAWN", "fog", "DAWN fog"), ("DAWN", "rain", "DAWN rain"), ("DAWN", "sand", "DAWN sand"), ("DAWN", "snow", "DAWN snow")]
GREY, ORANGE, INK = "#7B8794", "#C26A00", "#1F2933"
REAL, FALSE, MISS = "#22C55E", "#EF4444", "#F59E0B"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


# ------------------------------------------------------------------------------------------------ tables
def tab_crc() -> str:
    s = pd.read_csv(CRC / "crc_lambda_summary.csv")
    r = pd.read_csv(CRC / "crc_lambda_ratio.csv")
    s = s[(s.scope == "per_bin") & np.isclose(s.alpha, ALPHA)]
    r = r[(r.scope == "per_bin") & np.isclose(r.alpha, ALPHA)].set_index(["dataset", "bin"])
    lines = []
    for ds, b, lab in COND:
        g = s[(s.dataset == ds) & (s.bin == b)].set_index("method")
        bl, ou, rr = g.loc["baseline"], g.loc["ours"], r.loc[(ds, b)]
        lines.append(f"{lab} & {bl.n_test_mean:.0f} & {bl.image_loss_mean:.3f} & {pct(bl.exceed_frac, 0)} & {pct(bl.recall_mean)} & "
                     f"{pct(bl.precision_mean)} & {bl.fppi_mean:.2f} & {ou.image_loss_mean:.3f} & {pct(ou.exceed_frac, 0)} & "
                     f"{pct(ou.recall_mean)} & {pct(ou.precision_mean)} & {ou.fppi_mean:.2f} & {ou.lam_median:.1f} & "
                     f"{rr.ratio_median:.2f} ({rr.ratio_q025:.2f}--{rr.ratio_q975:.2f}) \\\\")
        if (ds, b) == ("BDD100K", "snow_night"):
            lines.append("\\addlinespace[2pt]")
    lines.append("\\midrule")
    for ds, lab in (("BDD100K", "pooled, BDD100K val"), ("DAWN", "pooled, DAWN"), ("both", "pooled, all 11")):
        rr = r.loc[(ds, "all")]
        lines.append(f"\\multicolumn{{13}}{{@{{}}l}}{{{lab}}} & {rr.ratio_median:.2f} ({rr.ratio_q025:.2f}--{rr.ratio_q975:.2f}) \\\\")
    return "\n".join(lines)


def tab_crc_floor() -> str:
    m = pd.read_csv(CRC / "crc_lambda_min_alpha.csv").pivot_table(index=["dataset", "bin"], columns="method", values="min_alpha")
    s = pd.read_csv(CRC / "crc_lambda_summary.csv")
    s = s[(s.scope == "per_bin") & np.isclose(s.alpha, 0.15)].pivot_table(index=["dataset", "bin"], columns="method",
                                                                          values="infeasible_frac")
    out = []
    for ds, b, lab in COND:
        out.append(f"{lab} & {m.loc[(ds, b), 'baseline']:.3f} & {m.loc[(ds, b), 'ours']:.3f} & "
                   f"{pct(s.loc[(ds, b), 'baseline'], 0)}\\% & {pct(s.loc[(ds, b), 'ours'], 0)}\\% \\\\")
    return "\n".join(out)


def tab_dial() -> str:
    e = pd.read_csv(THRESH_RESULTS / "per_detection_yolo_eval.csv")
    e = e[e.lam == 10]
    rows = [("BDD100K", "all", "BDD100K val, all"), ("BDD100K", "day", "day"), ("BDD100K", "night", "night"),
            ("BDD100K", "rain_night", "rain at night"), ("DAWN", "all", "DAWN, all"), ("DAWN", "fog", "DAWN fog")]
    out = []
    for ds, g, lab in rows:
        x = e[(e.dataset == ds) & (e.group == g)].set_index("policy")
        o, p, n = x.loc["one threshold (train)"], x.loc["per detection"], x.loc["no tuning (0.25)"]
        out.append(f"{lab} & {pct(n.recall)} / {pct(n.precision)} / {n.fppi:.2f} & {n.cost_vs_one:.2f} & "
                   f"{pct(o.recall)} / {pct(o.precision)} / {o.fppi:.2f} & 1 & "
                   f"{pct(p.recall)} / {pct(p.precision)} / {p.fppi:.2f} & {p.cost_vs_one:.3f} ({p.ci_lo:.3f}--{p.ci_hi:.3f}) \\\\")
    return "\n".join(out)


def tab_ablation() -> str:
    a = pd.read_csv(ABL / "dial_ablation.csv")
    out = []

    def cell(x, key):
        return f"{x[key + '_cost']:.3f} ({x[key + '_lo']:.3f}--{x[key + '_hi']:.3f})"
    full = a[a.fraction == 1]
    for x in full.to_dict("records"):
        name = x["variant"].replace("model 2 (adopted, 68 features)", "model 2, all 68 features (adopted)")
        out.append(f"{name} & {x['n_features']} & {x['train_images']:,} & {x['iterations']} & {cell(x, 'BDD100K_all_lam10')} & "
                   f"{cell(x, 'BDD100K_night_lam10')} & {cell(x, 'DAWN_all_lam10')} & {x['BDD100K_ece_points']:.2f} / {x['DAWN_ece_points']:.2f} \\\\")
    out.append("\\addlinespace[2pt]")
    sub = a[a.fraction < 1].groupby("fraction")
    for frac, g in sub:
        def ms(col):
            return f"{g[col].mean():.3f} $\\pm$ {g[col].std(ddof=1):.3f}"
        out.append(f"model 2, {frac:.0%} of train images (3 seeds) & 68 & {g.train_images.mean():,.0f} & {g.iterations.mean():.0f} & "
                   f"{ms('BDD100K_all_lam10_cost')} & {ms('BDD100K_night_lam10_cost')} & {ms('DAWN_all_lam10_cost')} & "
                   f"{g.BDD100K_ece_points.mean():.2f} / {g.DAWN_ece_points.mean():.2f} \\\\")
    return "\n".join(out).replace("%", "\\%")


def tab_recal() -> str:
    """Ablation (b): cost at lam = 10 of the recalibrated rule / raw chance (median over the 200 splits, 95% interval),
    and the share of real boxes near the 9.1% cut before and after the per-condition map."""
    s = pd.read_csv(ABL / "dial_recalibration.csv")
    s = s[s.lam == 10].set_index(["dataset", "condition", "variant"])
    c = pd.read_csv(ABL / "dial_recalibration_calibration.csv").groupby(["dataset", "condition", "variant"]).mean(numeric_only=True)
    rows = ([("BDD100K", "all", "BDD100K val, all")] + [x for x in COND if x[0] == "BDD100K"]
            + [("DAWN", "all", "DAWN, all")] + [x for x in COND if x[0] == "DAWN"])
    out = []
    for ds, cond, lab in rows:
        def cell(v):
            r = s.loc[(ds, cond, v)]
            return f"{r.vs_raw_median:.3f} ({r.vs_raw_q025:.3f}--{r.vs_raw_q975:.3f})"
        raw = s.loc[(ds, cond, "raw")]
        if cond == "all":
            near = "-- & --"
        else:
            a, b = c.loc[(ds, cond, "raw")], c.loc[(ds, cond, "condition")]
            near = f"{pct(a.mean_chance_near_10)} / {pct(a.share_real_near_10)} & {pct(b.mean_chance_near_10)} / {pct(b.share_real_near_10)}"
        out.append(f"{lab} & {raw.vs_one_median:.3f} & {cell('condition')} & {cell('pooled')} & {cell('train')} & {near} \\\\")
        if (ds, cond) in (("BDD100K", "all"), ("DAWN", "all")):
            out.append("\\addlinespace[2pt]")
        if (ds, cond) == ("BDD100K", "snow_night"):
            out.append("\\midrule")
    return "\n".join(out)


def tab_detectors() -> str:
    """Ablation (d): per detector, the share of objects it ever proposes, its one threshold's cost against the adopted
    detector's, the per-detection rule against its own one threshold (lam = 10), and CRC over lam against
    per-weather CRC over its score (pooled false-alarm ratio at the first alpha every condition can certify)."""
    from wad.detector_ablation import ADOPTED, DETS

    e = pd.read_csv(ABL / "detector_ablation_eval.csv")
    e = e[e.lam == 10].set_index(["detector", "dataset", "group", "policy"])
    s = pd.read_csv(ABL / "detector_ablation_crc.csv")
    r = pd.read_csv(ABL / "detector_ablation_ratio.csv")
    d = pd.read_csv(ABL / "detector_ablation_data.csv").set_index(["detector", "set"])
    m = pd.read_csv(ABL / "detector_ablation_min_alpha.csv")
    labels = {ADOPTED["det_id"]: "YOLOv8x @1280 (adopted)",
              **{k: v["label"] + ("" if v.get("per_box", True) else r"$^\dagger$") for k, v in DETS.items()}}
    out = []
    for det in [ADOPTED["det_id"]] + [k for k in DETS if k in set(e.index.get_level_values(0))]:
        per_box = DETS.get(det, {}).get("per_box", True)  # without per-box outputs the per-detection rule is model 1

        def cost(ds, grp, pol=None, ci=True):
            x = e.loc[(det, ds, grp, pol or ("per detection (model 2)" if per_box else "per detection (model 1)"))]
            return f"{x.cost_vs_one:.3f}" + (f" ({x.ci_lo:.3f}--{x.ci_hi:.3f})" if ci else "")
        one = e.loc[(det, "BDD100K", "all", "one threshold (train)")].cost_vs_adopted_one
        ss = s[(s.detector == det) & (s.scope == "per_bin")]
        floor = m[(m.detector == det) & (m.method == "baseline")].min_alpha.max()  # every condition certifiable above it
        feasible = [a for a in sorted(ss.alpha.unique()) if a >= floor]
        if feasible:
            a = feasible[0]
            rr = r[(r.detector == det) & (r.scope == "per_bin") & (r.alpha == a) & (r.dataset == "both") & (r.bin == "all")].iloc[0]
            crc = f"{a:.2f} & {rr.ratio_median:.2f} ({rr.ratio_q025:.2f}--{rr.ratio_q975:.2f})"
        else:
            crc = "-- & --"
        mb = m[(m.detector == det) & (m.method == "baseline") & (m.dataset == "BDD100K")].min_alpha
        out.append(f"{labels[det]} & {pct(d.loc[(det, 'BDD100K')].objects_ever_detected)} & {one:.3f} & {cost('BDD100K', 'all')} & "
                   f"{cost('BDD100K', 'night', ci=False)} & {cost('DAWN', 'all')} & {cost('BDD100K', 'all', 'per detection (model 1)', ci=False)} & "
                   f"{crc} & {mb.min():.3f}--{mb.max():.3f} \\\\")
    return "\n".join(out)


def tab_data() -> str:
    return "\n".join([
        "BDD100K train (subset) & fit & 21{,}831 & 261{,}699 & 3{,}338{,}916 & clear 4{,}500, cloudy 1{,}500, rain 2{,}532, snow 2{,}873, "
        "fog 130, dawn/dusk 1{,}996; night: clear 4{,}300, rain 2{,}000, snow 2{,}000 \\\\",
        "BDD100K val & test & 8{,}090 & 97{,}745 & 1{,}178{,}828 & clear 1{,}764, cloudy 1{,}676, rain 395, snow 422; night: clear 3{,}274, "
        "rain 286, snow 273 \\\\",
        "DAWN (clean labels) & test & 889 & 6{,}952 & 49{,}814 & fog 256, rain 187, sand 243, snow 203 \\\\"]
        + _full_train_row())


def _full_train_row() -> list:
    """All BDD100K train frames (wad.full_train), once that run has finished: counts from its results and scan."""
    import json

    if not (ABL / "full_train_data.csv").exists():
        return []
    d = pd.read_csv(ABL / "full_train_data.csv", header=None, index_col=0)[1]
    b = json.loads((REPO / "cache" / "thresh_full" / "data_report.json").read_text(encoding="utf-8"))["images_per_bin_kept"]

    def n(x):
        return f"{int(x):,}".replace(",", "{,}")
    bins = (f"clear {n(b['clear'])}, cloudy {n(b['cloudy'])}, rain {n(b['rain'])}, snow {n(b['snow'])}, fog {n(b['fog'])}, "
            f"dawn/dusk {n(b['dusk'])}, weather undefined {n(b['undefined_weather'])}; night: clear {n(b['clear_night'])}, "
            f"rain {n(b['rain_night'])}, snow {n(b['snow_night'])}, cloudy {n(b['cloudy_night'])}; time undefined "
            f"{n(b['undefined_time'])}")
    return [f"BDD100K train (all frames) & check & {n(d['frames'])} & {n(d['objects'])} & {n(d['detections'])} & {bins} \\\\"]


# ------------------------------------------------------------------------------------------------ figures
def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
                         "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    return plt


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / "figures" / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)


def fig_price():
    plt = _style()
    s = pd.read_csv(CRC / "crc_lambda_summary.csv")
    r = pd.read_csv(CRC / "crc_lambda_ratio.csv")
    s = s[s.scope == "per_bin"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.45, 4.1), gridspec_kw=dict(height_ratios=[1.45, 1]))
    y = np.arange(len(COND))[::-1]
    for yy, (ds, b, lab) in zip(y, COND):
        g = s[(s.dataset == ds) & (s.bin == b) & np.isclose(s.alpha, ALPHA)].set_index("method")
        bl, ou = g.loc["baseline"].fppi_mean, g.loc["ours"].fppi_mean
        a1.plot([ou, bl], [yy, yy], color="#D9DEE5", lw=2.5, zorder=1)
        a1.plot(bl, yy, "o", color=GREY, ms=5, zorder=2)
        a1.plot(ou, yy, "o", color=ORANGE, ms=5, zorder=3)
    a1.set_yticks(y, [c[2] for c in COND])
    a1.set_xscale("log")
    a1.set_xlabel(f"false alarms per image at $\\alpha$ = {ALPHA} (mean over 200 splits)")
    a1.plot([], [], "o", color=GREY, label="baseline: per-weather CRC over $\\tau$")
    a1.plot([], [], "o", color=ORANGE, label="ours: per-weather CRC over $\\lambda$")
    a1.legend(loc="lower right", frameon=False)
    a1.grid(axis="x", alpha=0.3, which="both")
    rr = r[(r.scope == "per_bin") & (r.bin == "all")]
    for ds, col, lab in (("BDD100K", "#2F6FDB", "BDD100K val (7 bins)"), ("DAWN", "#2BA36B", "DAWN (4 bins)"), ("both", INK, "all 11 conditions")):
        g = rr[rr.dataset == ds].sort_values("alpha")
        a2.plot(g.alpha, g.ratio_median, "-o", color=col, ms=3, lw=1.3, label=lab)
        a2.fill_between(g.alpha, g.ratio_q025, g.ratio_q975, color=col, alpha=0.15, lw=0)
    a2.axhline(1, color=GREY, lw=0.8, ls="--")
    a2.set_xlabel("miss budget $\\alpha$")
    a2.set_ylabel("false alarms, ours / baseline")
    a2.legend(frameon=False, loc="best")
    a2.grid(alpha=0.3)
    fig.tight_layout(h_pad=1.0)
    _save(fig, "fig_p4_price")
    plt.close(fig)


def fig_training():
    import joblib

    plt = _style()
    m = joblib.load(REPO / "weights" / "per_detection_yolo.joblib")["model"]
    a = pd.read_csv(ABL / "dial_ablation.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.45, 1.85))
    it = np.arange(len(m.train_score_))
    a1.plot(it, -m.train_score_, color=ORANGE, lw=1.3, label="train (90%)")
    a1.plot(it, -m.validation_score_, color=INK, lw=1.1, ls="--", label="validation (10%)")
    a1.axvline(m.n_iter_, color=GREY, lw=0.8, ls=":")
    a1.set_xlabel("boosting iteration")
    a1.set_ylabel("log-loss")
    a1.set_title(f"(a) training, early stop at {m.n_iter_}", pad=3)
    a1.legend(frameon=False)
    a1.grid(alpha=0.3)
    full = a[(a.fraction == 1) & a.variant.str.startswith("model 2 (adopted")]
    rows = pd.concat([a[a.fraction < 1], full])
    xs = rows.groupby("fraction").train_images.mean()
    full_train = None
    if (ABL / "full_train_eval.csv").exists():  # all BDD100K train frames (wad.full_train), same reference threshold
        ft = pd.read_csv(ABL / "full_train_eval.csv")
        ft = ft[(ft.lam == 10) & (ft.group == "all") & (ft.one_threshold_from == "adopted (subset)")
                & (ft.policy == "per detection (model 2, all train)")].set_index("dataset").cost_vs_one
        frames = pd.read_csv(ABL / "full_train_data.csv", header=None, index_col=0)[1]["frames"]
        full_train = (float(frames), ft)
    for key, col, lab in (("BDD100K_all_lam10_cost", "#2F6FDB", "BDD100K val"), ("DAWN_all_lam10_cost", "#2BA36B", "DAWN")):
        g = rows.groupby("fraction")[key].mean()
        a2.plot(xs.loc[g.index], g.values, "-o", color=col, ms=3, lw=1.2, label=lab)
        a2.scatter(rows.train_images, rows[key], s=6, color=col, alpha=0.45, lw=0)
        if full_train is not None:
            y = full_train[1][key.split("_")[0]]
            a2.plot([xs.loc[1.0], full_train[0]], [g.loc[1.0], y], ":", color=col, lw=1.0)
            a2.plot([full_train[0]], [y], "D", mfc="white", mec=col, ms=3.5, mew=1.0)
    a2.set_xscale("log")
    ticks = list(xs.values) + ([full_train[0]] if full_train is not None else [])
    a2.set_xticks(ticks, [f"{v / 1000:.1f}k" if v < 10000 else f"{v / 1000:.0f}k" for v in ticks])
    a2.minorticks_off()
    a2.axhline(1, color=GREY, lw=0.8, ls="--")
    a2.set_xlabel("train images (10, 25, 50, 100%" + (r"; all $\diamond$)" if full_train is not None else ")"))
    a2.set_ylabel("cost vs one threshold")
    a2.set_title("(b) learning curve, $\\lambda$ = 10", pad=3)
    a2.legend(frameon=False, loc="upper right")
    a2.grid(alpha=0.3)
    fig.tight_layout(w_pad=1.0)
    _save(fig, "fig_p4_training")
    plt.close(fig)


def fig_qualitative():
    """Held-out frames of split 0: the first frame (by image id) with 6-15 objects in the test half of each
    condition; boxes kept by the baseline (score >= its tau-hat) and by ours (chance >= its cut), alpha 0.2."""
    plt = _style()
    from matplotlib.patches import Rectangle

    from wad.config import tau_grid
    from wad.crc import fit_threshold, make_splits
    from wad.crc_lambda import chance_cache
    from wad.curves import ALL, Cache
    from wad.imageio import load_rgb
    from wad.one_threshold import eval_configs
    from wad.per_detection import pred_path

    picks = [("BDD100K", "clear", "BDD100K clear day"), ("BDD100K", "clear_night", "BDD100K clear night"),
             ("BDD100K", "rain_night", "BDD100K rain at night"), ("DAWN", "fog", "DAWN fog")]
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 2.55))
    det_path = {"BDD100K": "cache/bdd/det/v8x_1280/detections.parquet", "DAWN": "cache/det/v8x_1280/detections.parquet"}
    gt_path = {"BDD100K": "cache/bdd/gt.parquet", "DAWN": "cache/gt.parquet"}
    cfgs = eval_configs()
    for j, (ds, b, title) in enumerate(picks):
        cfg = cfgs[ds]
        c = Cache(cfg)
        p = pd.read_parquet(pred_path("bdd" if ds == "BDD100K" else "dawn")).p.to_numpy()
        cp = chance_cache(c, p)
        v = copy.deepcopy(cfg)
        v.splits["n_repeats"] = 1
        sp = make_splits(v, c.images).set_index("image_id").loc[c.image_ids]
        cal = (sp.half == "cal").to_numpy()
        grid, floor = tau_grid(cfg), cfg.detector.conf_floor
        rows = cal & (c.weather == b)
        t_b = fit_threshold(c.loss_grid(grid, ALL)[rows], grid, ALPHA, floor)["tau_hat"]
        t_o = fit_threshold(cp.loss_grid(grid, ALL)[rows], grid, ALPHA, floor)["tau_hat"]
        test_ids = c.images[(~cal) & (c.weather == b) & c.images.n_gt.between(6, 15).to_numpy()].image_id.sort_values()
        iid = test_ids.iloc[0]
        sel = (c.pred.image_id == iid).to_numpy()
        pr = c.pred[sel].reset_index(drop=True)
        d = pd.read_parquet(REPO / det_path[ds])
        d = d[d.image_id == iid].reset_index(drop=True)
        assert np.allclose(d.score.to_numpy(), pr.score.to_numpy())
        pi, tp = p[sel], pr.tp.to_numpy().astype(bool)
        gtb = pd.read_parquet(REPO / gt_path[ds])
        gt = gtb[(gtb.image_id == iid) & gtb.gt_id.isin(c.gt[c.gt.image_id == iid].gt_id)]
        im = load_rgb(REPO / c.images.set_index("image_id").loc[iid, "path"], "apply")
        for i, (keep, lab) in enumerate(((d.score.to_numpy() >= t_b, f"baseline: score $\\geq$ {t_b:.3f}"),
                                         (pi >= t_o, f"ours: $p \\geq$ {t_o:.3f} ($\\lambda$ = {1 / t_o - 1:.0f})"))):
            ax = axes[i, j]
            ax.imshow(im)
            found = set(pr.gt_id[keep & tp])
            for g in gt.itertuples():
                if g.gt_id not in found:
                    ax.add_patch(Rectangle((g.x1, g.y1), g.x2 - g.x1, g.y2 - g.y1, fill=False, ec=MISS, lw=0.9, ls=(0, (2, 1.5))))
            for bb, k, t in zip(d.itertuples(), keep, tp):
                if k:
                    ax.add_patch(Rectangle((bb.x1, bb.y1), bb.x2 - bb.x1, bb.y2 - bb.y1, fill=False, ec=REAL if t else FALSE, lw=0.9))
            n_f, n_fa = len(found), int((keep & ~tp).sum())
            ax.set_xticks([])
            ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
            ax.set_title(f"{title}\n{lab}" if i == 0 else lab, fontsize=6.5, pad=2)
            ax.text(0.01, 0.02, f"found {n_f}/{len(gt)}, false {n_fa}", transform=ax.transAxes, fontsize=6, color="white",
                    va="bottom", bbox=dict(facecolor="black", alpha=0.6, pad=1.2, lw=0))
    fig.tight_layout(pad=0.2, w_pad=0.3, h_pad=0.4)
    _save(fig, "fig_p4_qualitative")
    plt.close(fig)


def fig_logs():
    plt = _style()
    def read(path):
        return [l.rstrip() for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]

    train = [l for l in read(THRESH_RESULTS / "per_detection_yolo.log") if not l.startswith("findfont")][:9]
    abl = [l for l in read(ABL / "dial_ablation.log")
           if l.startswith(("train:", "model 2 (adopted", "without class", "context only", "score and class",
                            "model 2, training subset [10%, seed 0]", "done"))]
    crc = []
    for l in read(CRC / "crc_lambda.log"):
        if l.startswith(("BDD100K: evaluated", "DAWN: evaluated", "sanity:", "done")):
            crc.append(l)
        elif l.startswith("| C"):
            cells = [c.strip() for c in l.strip("|").split("|")]
            crc.append(f"{cells[0]} -> {cells[1]} -> {cells[2]}")
    det = [l for l in read(ABL / "detector_ablation.log")
           if (" detections, " in l and " real, " in l) or " model 2: " in l or "check passed" in l or ": done, " in l
           or (l.startswith("rtdetr") and " model 1: " in l)]
    blocks = [("python -m wad.per_detection --features yolo   (training the per-detection model)", train),
              ("python -m wad.dial_ablation   (ablations: refits on BDD100K train, graded on BDD100K val / DAWN)", abl),
              ("python -m wad.crc_lambda   (pre-registered: CRC over lambda vs the per-weather baseline, 200 splits)", crc),
              ("python -m wad.detector_ablation fit   (ablation d: the pipeline refitted for four other detectors)", det)]
    if (ABL / "full_train.log").exists():  # training on all BDD100K train frames: the passes, then the (last) fit
        ft = read(ABL / "full_train.log")
        last = max(i for i, l in enumerate(ft) if l.startswith("features:"))
        keep = [l.split("; per bin")[0] for l in ft[:last] if l.startswith(("scan:", "detection done", "per-box outputs done"))]
        keep += [l for l in ft[last:] if l.startswith(("features:", "model 1:", "model 2:", "BDD100K: graded", "DAWN: graded",
                                                      "- pooled false-alarm", "done,"))]
        blocks.append(("python -m wad.full_train scan / gpu / fit   (the model refitted on all 69,998 BDD100K train frames)", keep))
    text = []
    for title, lines in blocks:
        text.append("$ " + title)
        for l in lines:
            text += textwrap.wrap(l, 118, subsequent_indent="    ") or [""]
        text.append("")
    fig = plt.figure(figsize=(7.16, 0.118 * len(text) + 0.1))
    fig.text(0.005, 0.995, "\n".join(text), family="monospace", fontsize=5.6, va="top", ha="left",
             bbox=dict(facecolor="#F4F6F8", edgecolor="#C9CED6", pad=4))
    _save(fig, "fig_p4_logs")
    plt.close(fig)


def main():
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    tables = [("tab_crc", tab_crc), ("tab_crc_floor", tab_crc_floor), ("tab_dial", tab_dial), ("tab_ablation", tab_ablation),
              ("tab_data", tab_data), ("tab_recal", tab_recal)]
    if (ABL / "detector_ablation_eval.csv").exists():
        tables.append(("tab_detectors", tab_detectors))
    for name, fn in tables:
        (OUT / "tables" / f"{name}.tex").write_text(fn() + "\n", encoding="utf-8")
    fig_price()
    fig_training()
    fig_logs()
    fig_qualitative()
    print(f"wrote tables and figures to {OUT}")


if __name__ == "__main__":
    main()
