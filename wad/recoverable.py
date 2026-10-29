"""Recoverable recall by weather condition and time of day.

For every condition (BDD100K weather x time-of-day bins; DAWN's severe-weather bins), each
ground-truth object falls in one of three parts:
  found        matched at the shipped threshold tau = 0.25
  recoverable  matched only below 0.25 (some threshold down to the 0.001 floor recovers it)
  never        not matched at any threshold (thresholding cannot help)
plus the price of recovery: recall reached at fixed false-positive budgets (FPPI 1, 2, 5, 10, 20),
the share of the recoverable gap recovered there, and the extra false positives per image needed to
recover half of it. Intervals are 95% bootstraps over images (near-duplicate groups) within the
condition. BDD100K rows also give found / recoverable at clear daytime's object-height mix
(exploratory), since its conditions hold different object sizes.

Run: python -m wad.recoverable   -> results/run2/recoverable_by_condition.csv, figures/fig9_recoverable
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wad.config import REPO, Cfg
from wad.curves import ALL, Cache, mr_at_fppi, operating_curve

OUT = REPO / "results" / "run2"
FPPI_BUDGETS = [1, 2, 5, 10, 20]
TAU = 0.25
HEIGHT_EDGES = [0, 10, 20, 40, 80, np.inf]
LABEL = {"clear": "Clear · day", "cloudy": "Cloudy · day", "rain": "Rain · day", "snow": "Snow · day",
         "clear_night": "Clear · night", "rain_night": "Rain · night", "snow_night": "Snow · night",
         "fog": "Fog (DAWN)", "sand": "Sand (DAWN)"}
TOD = {"clear": "day", "cloudy": "day", "rain": "day", "snow": "day", "clear_night": "night", "rain_night": "night",
       "snow_night": "night"}


def configs():
    from wad.run2 import bdd_cfg, dawn_cfg, load_run2

    r2 = load_run2()
    return [("BDD100K", "v8x_1280", bdd_cfg(r2, "v8x_1280", "group")),
            ("DAWN", "v8x_1280", dawn_cfg(r2, "v8x_1280", "group", "clean")),
            ("BDD100K", "v8s_640", bdd_cfg(r2, "v8s_640", "class")),
            ("DAWN", "v8s_640", dawn_cfg(r2, "v8s_640", "class", "all"))]


def condition_rows(ds: str, det: str, v: Cfg, n_boot: int = 2000, seed: int = 0) -> list[dict]:
    from wad.data import load_gt

    c = Cache(v)
    rng = np.random.default_rng(seed)
    floor = v.detector.conf_floor
    groups = c.images.group.to_numpy()
    pm = c.pred.image_id.map(c.idx).to_numpy()
    # per-object status and height
    gt = load_gt(v).set_index("gt_id")
    height = (gt.y2 - gt.y1).reindex(c.gt.gt_id).to_numpy()
    band = np.digitize(height, HEIGHT_EDGES[1:-1])
    found_obj = c.gt_score >= TAU
    ever_obj = c.gt_score >= floor
    ref_rows = c.weather == "clear"
    w_ref = None
    if ref_rows.any():
        m = ref_rows[c.gt_img]
        w_ref = np.bincount(band[m], minlength=len(HEIGHT_EDGES) - 1) / m.sum()
    out = []
    for b in sorted(set(c.weather), key=lambda x: list(LABEL).index(x) if x in LABEL else 99):
        rows = np.nonzero(c.weather == b)[0]
        sel = np.zeros(c.n_img, bool)
        sel[rows] = True
        gm = sel[c.gt_img]
        n_gt = int(gm.sum())
        found, ever = found_obj[gm].mean(), ever_obj[gm].mean()
        # exact recall-vs-FPPI curve for the condition
        msk = sel[pm]
        oc = operating_curve(c.pred.score.to_numpy()[msk], c.pred.tp.to_numpy()[msk], n_gt, len(rows))
        fppi_025 = float(((~c.pred.tp.to_numpy()[msk]) & (c.pred.score.to_numpy()[msk] >= TAU)).sum() / len(rows))
        rec_at = {f: 1 - float(mr_at_fppi(oc["fppi"], oc["miss_rate"], np.array([float(f)]))[0][0]) for f in FPPI_BUDGETS}
        gap = ever - found
        # extra FP per image to recover half the gap (first operating point reaching it)
        target = found + 0.5 * gap
        rec_curve = 1 - oc["miss_rate"]
        k = np.nonzero(rec_curve >= target)[0]
        half_cost = float(oc["fppi"][k[0]] - fppi_025) if len(k) and gap > 0 else np.nan
        # bootstrap over images (groups) for the three shares
        g_ids, inv = np.unique(groups[rows], return_inverse=True)
        pos = {r: i for i, r in enumerate(rows)}
        gi = np.array([inv[pos[r]] for r in c.gt_img[gm]])
        per = np.zeros((len(g_ids), 3))
        np.add.at(per, gi, np.c_[np.ones(n_gt), found_obj[gm], ever_obj[gm]])
        bs = per[rng.integers(0, len(g_ids), size=(n_boot, len(g_ids)))].sum(1)
        f_b, e_b = bs[:, 1] / bs[:, 0], bs[:, 2] / bs[:, 0]
        row = dict(dataset=ds, detector=det, condition=b, time_of_day=TOD.get(b, "day"), n_images=len(rows), n_objects=n_gt,
                   found=found, found_lo=np.percentile(f_b, 2.5), found_hi=np.percentile(f_b, 97.5),
                   recoverable=gap, recoverable_lo=np.percentile(e_b - f_b, 2.5), recoverable_hi=np.percentile(e_b - f_b, 97.5),
                   never=1 - ever, never_lo=np.percentile(1 - e_b, 2.5), never_hi=np.percentile(1 - e_b, 97.5),
                   fppi_at_025=fppi_025, fppi_at_floor=float(oc["fppi"][-1]), extra_fppi_for_half_gap=half_cost)
        for f in FPPI_BUDGETS:
            row[f"recall_at_fppi{f}"] = rec_at[f]
            row[f"share_of_gap_recovered_at_fppi{f}"] = (max(rec_at[f] - found, 0.0) / gap) if gap > 0 else np.nan
        for cls in ("person", "car"):
            cm = gm & (c.gt_cls == cls)
            row[f"{cls}_n"] = int(cm.sum())
            row[f"{cls}_found"] = found_obj[cm].mean() if cm.any() else np.nan
            row[f"{cls}_recoverable"] = (ever_obj[cm].mean() - found_obj[cm].mean()) if cm.any() else np.nan
            row[f"{cls}_never"] = (1 - ever_obj[cm].mean()) if cm.any() else np.nan
        if w_ref is not None:  # exploratory: at clear daytime's object-height mix
            masks = [gm & (band == k) for k in range(len(w_ref))]
            fb = np.array([found_obj[m].mean() if m.any() else 0.0 for m in masks])
            eb = np.array([ever_obj[m].mean() if m.any() else 0.0 for m in masks])
            row["found_size_matched"] = float((w_ref * fb).sum())
            row["recoverable_size_matched"] = float((w_ref * (eb - fb)).sum())
            row["never_size_matched"] = float(1 - (w_ref * eb).sum())
            row["share_under_20px"] = float((height[gm] < 20).mean())
        out.append(row)
    return out


def table() -> pd.DataFrame:
    rows = []
    for ds, det, v in configs():
        rows += condition_rows(ds, det, v)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "recoverable_by_condition.csv", index=False, float_format="%.6g")
    return d


def figure(d: pd.DataFrame, det: str = "v8x_1280"):
    import matplotlib.pyplot as plt

    from wad.figures import INK, INK2, MUTED, ORDINAL_BLUE, SERIES, SURFACE, save, style

    style()
    d = d[d.detector == det].copy()
    order = ["clear", "cloudy", "rain", "snow", "clear_night", "rain_night", "snow_night", "fog", "rain", "sand", "snow"]
    keys = [("BDD100K", b) for b in order[:7]] + [("DAWN", b) for b in order[7:]]
    d = d.set_index(["dataset", "condition"]).loc[keys].reset_index()
    labels = [LABEL.get(b, b) if ds == "BDD100K" else f"{b.capitalize()} (DAWN)" for ds, b in keys]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    y = np.r_[np.arange(7)[::-1] + 5.6, np.arange(4)[::-1]]  # BDD100K block above, DAWN block below
    h = 0.62
    left = np.zeros(len(d))
    parts = [("found", "found at τ = 0.25", ORDINAL_BLUE[0]), ("recoverable", "recoverable by lowering τ", ORDINAL_BLUE[2]),
             ("never", "never detected at any τ", "#d6d4cc")]
    for col, lab, colour in parts:
        ax.barh(y, 100 * d[col], left=100 * left, height=h, color=colour, edgecolor=SURFACE, linewidth=1.5, label=lab)
        left = left + d[col].to_numpy()
    for yi, r in zip(y, d.itertuples()):
        ax.text(100 * r.found / 2, yi, f"{100 * r.found:.0f}%", ha="center", va="center", fontsize=7.5, color="white")
        ax.text(100 * (r.found + r.recoverable / 2), yi, f"+{100 * r.recoverable:.0f}", ha="center", va="center", fontsize=7.5,
                color=INK)
        ax.text(100 * (r.found + r.recoverable) + 1, yi, f"{100 * r.never:.0f}%", ha="left", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of ground-truth objects (%)")
    ax.grid(False, axis="y")
    ax.axhline(4.8, color=MUTED, lw=0.8)
    ax.text(99, 4.95, "BDD100K (same cameras): weather × time of day", ha="right", va="bottom", fontsize=7.5, color=INK2)
    ax.text(99, 3.75, "DAWN: severe weather, daytime", ha="right", va="bottom", fontsize=7.5, color=INK2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, fontsize=7.5)
    ax.set_title("Where each condition's objects end up", fontsize=10)
    # (b) share of the recoverable gap recovered vs false-positive budget
    ax = axes[1]
    styles = {"clear": (SERIES[0], "-"), "rain": (SERIES[1], "-"), "snow": (SERIES[2], "-"), "cloudy": (SERIES[3], "-"),
              "clear_night": (SERIES[0], "--"), "rain_night": (SERIES[1], "--"), "snow_night": (SERIES[2], "--")}
    for r in d[d.dataset == "BDD100K"].itertuples():
        col, ls = styles[r.condition]
        xs = [r.fppi_at_025] + FPPI_BUDGETS
        ys = [0] + [100 * getattr(r, f"share_of_gap_recovered_at_fppi{f}") for f in FPPI_BUDGETS]
        keep = [i for i, x in enumerate(xs) if i == 0 or x > r.fppi_at_025]
        ax.plot([xs[i] for i in keep], [ys[i] for i in keep], color=col, ls=ls, lw=1.6, marker="o", ms=4, label=LABEL[r.condition])
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(["1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlabel("False positives per image (log scale; the line starts at τ = 0.25)")
    ax.set_ylabel("Share of the recoverable gap recovered (%)")
    ax.set_ylim(0, 100)
    ax.set_title("What recovery costs (BDD100K; solid = day, dashed = night)", fontsize=10)
    ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle(f"Recoverable recall by weather and time of day ({'YOLOv8x @1280, grouped vehicles' if det == 'v8x_1280' else 'YOLOv8s @640'})",
                 x=0.02, ha="left", fontsize=10.5, color=INK, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, Cfg(dict(paths=Cfg(dict(results=OUT)))), f"fig9_recoverable_{det}")


if __name__ == "__main__":
    d = table()
    for det in ("v8x_1280", "v8s_640"):
        for p in figure(d, det):
            print(p)
