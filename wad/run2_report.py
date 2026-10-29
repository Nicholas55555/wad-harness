"""results/run2/run2.md and the run 2 figures, generated from results/run2/**.

Verdict words are computed, never typed: for a policy's mean held-out miss fraction m over the
200 split repeats, "holds" if m <= alpha, "at the budget" if alpha < m <= alpha + 0.01, "fails"
otherwise, always shown with the share of splits above alpha.

fig6_clear_vs_hazard   BDD100K: precision-recall per daytime bin; recall/precision at tau = 0.25 vs
                       clear daytime, raw and size-standardised; clear-calibrated vs per-bin CRC
fig7_run2_detectors    DAWN: smallest certifiable alpha and FPPI at alpha = 0.2 per detector and loss
fig8_tiers             confirmed tier: calibration estimate m(b) vs held-out miss per bin
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from wad.config import REPO, Cfg

OUT = REPO / "results" / "run2"
BL = {"fog": "Fog", "rain": "Rain", "sand": "Sand", "snow": "Snow", "all": "All", "clear": "Clear", "cloudy": "Cloudy",
      "clear_night": "Clear night", "rain_night": "Rain night", "snow_night": "Snow night"}
DAY = ["clear", "cloudy", "rain", "snow"]
NIGHT = ["clear_night", "rain_night", "snow_night"]
B4 = ["fog", "rain", "sand", "snow"]
PL = {"oracle_bin": "Per-bin CRC", "global": "Global CRC", "ref_calibrated": "Clear-calibrated CRC",
      "default_0.25": "Default τ=0.25", "default_0.5": "Default τ=0.5"}
DET = {"v8s_640": "YOLOv8s @640", "v8x_640": "YOLOv8x @640", "v8x_1280": "YOLOv8x @1280"}
HEIGHT_BANDS = ["<10", "10–20", "20–40", "40–80", "≥80"]


def md(df: pd.DataFrame, align: str | None = None) -> str:
    cols = list(df.columns)
    align = align or ("l" + "r" * (len(cols) - 1))
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "| " + " | ".join("---:" if a == "r" else ":---" for a in align) + " |"]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def pct(v, d=1):
    return "–" if pd.isna(v) else f"{100 * v:.{d}f}%"


def num(v, d=3):
    return "–" if pd.isna(v) else f"{v:.{d}f}"


def pts(v, lo=None, hi=None):
    s = f"{100 * v:+.1f} pts"
    return s if lo is None else f"{s} [{100 * lo:+.1f}, {100 * hi:+.1f}]"


def word(mean, alpha, exceed=None, infeasible=None):
    w = "holds" if mean <= alpha + 1e-12 else ("at the budget" if mean <= alpha + 0.01 else "fails")
    s = f"{w} ({mean:.3f}"
    if exceed is not None and not pd.isna(exceed):
        s += f", above α in {exceed:.0%} of splits"
    if infeasible is not None and not pd.isna(infeasible) and infeasible > 0:
        s += f", infeasible in {infeasible:.0%}"
    return s + ")"


def bdd_dir(det, loss):
    return OUT / "bdd" / f"{det}__{loss}"


def dawn_dir(det, loss, labels):
    return OUT / "dawn" / f"{det}__{loss}__{labels}"


def pi(path: Path, alpha: float, policy: str):
    s = pd.read_csv(path / "seeds200" / "policy_intervals.csv")
    return s[(s.alpha == alpha) & (s.policy == policy)].set_index("bin")


from pathlib import Path  # noqa: E402


# ---------------------------------------------------------------- figures
def _save(fig, name):
    from wad.figures import save

    return save(fig, Cfg(dict(paths=Cfg(dict(results=OUT)))), name)


def fig6(det: str, loss: str):
    import matplotlib.pyplot as plt

    from wad.curves import ALL, Cache, operating_curve
    from wad.figures import CRITICAL, INK, INK2, SERIES, SURFACE, style
    from wad.run2 import bdd_cfg, load_run2

    style()
    v = bdd_cfg(load_run2(), det, loss)
    c = Cache(v)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    ax = axes[0]
    pm_ = c.pred.image_id.map(c.idx).to_numpy()
    for i, b in enumerate(DAY):
        sel = np.zeros(c.n_img, bool)
        sel[c.weather == b] = True
        m = sel[pm_]
        oc = operating_curve(c.pred.score.to_numpy()[m], c.pred.tp.to_numpy()[m], int(c.n_gt(ALL)[sel].sum()), int(sel.sum()))
        ax.plot(oc["recall"], oc["precision"], color=SERIES[i], lw=1.6, label=f"{BL[b]} (n={int(sel.sum())})")
        j = np.nonzero(oc["threshold"][1:] >= 0.25)[0].max()
        ax.plot(oc["recall"][j], oc["precision"][j], marker="o", ms=6.5, mfc=SERIES[i], mec=SURFACE, mew=1.2, ls="none")
    ax.set_xlabel("Recall (all six classes, IoU 0.5)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Precision–recall by daytime weather (dot: τ = 0.25)")
    ax.legend(loc="lower left", fontsize=7.5)
    # (b) recall difference from clear daytime: raw and size-standardised; precision difference
    ax = axes[1]
    d = pd.read_csv(bdd_dir(det, loss) / "pr_vs_ref.csv")
    d = d[(d.cls == "all") & (d.bin != "clear")].set_index("bin")
    sz = pd.read_csv(bdd_dir(det, loss) / "pr_vs_ref_sizeadj_EXPLORATORY.csv").set_index("bin")
    order = [b for b in DAY + NIGHT if b in d.index]
    y = np.arange(len(order))
    series = [("raw recall", 100 * d.loc[order, "d_recall"], 100 * d.loc[order, "d_recall_lo"], 100 * d.loc[order, "d_recall_hi"], "o", 0),
              ("recall at clear's object-size mix (exploratory)", 100 * sz.loc[order, "d_recall_std"],
               100 * sz.loc[order, "d_recall_std_lo"], 100 * sz.loc[order, "d_recall_std_hi"], "D", 2),
              ("raw precision", 100 * d.loc[order, "d_precision"], 100 * d.loc[order, "d_precision_lo"],
               100 * d.loc[order, "d_precision_hi"], "s", 1)]
    for k, (lab, val, lo, hi, mk, col) in enumerate(series):
        yy = y + (k - 1) * 0.22
        ax.errorbar(val, yy, xerr=[val - lo, hi - val], fmt="none", ecolor=SERIES[col], elinewidth=1.2)
        ax.plot(val, yy, ls="none", marker=mk, ms=6, mfc=SERIES[col], mec=SURFACE, mew=1.2, label=lab)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([BL[b] for b in order])
    ax.invert_yaxis()
    ax.set_xlabel("Difference from clear daytime at τ = 0.25 (points, 95% bootstrap CI)")
    ax.set_title("What the condition costs at the shipped threshold")
    ax.legend(loc="center left", fontsize=7)
    # (c) held-out miss under clear-calibrated vs per-bin CRC, alpha = 0.2, 200 splits
    ax = axes[2]
    a = 0.2
    bins_ = DAY + NIGHT
    all_inf = True
    for k, pol in enumerate(["ref_calibrated", "oracle_bin", "default_0.25"]):
        dd = pi(bdd_dir(det, loss), a, pol).reindex(bins_)
        x = np.arange(len(bins_)) + (k - 1) * 0.22
        ax.errorbar(x, dd.loss_mean, yerr=[dd.loss_mean - dd.loss_lo, dd.loss_hi - dd.loss_mean], fmt="none",
                    ecolor=SERIES[k], elinewidth=1.2)
        inf = dd.infeasible_frac.fillna(0).to_numpy() >= 0.5
        if pol != "default_0.25":
            all_inf &= bool(inf.all())
        for xi, yi, hollow in zip(x, dd.loss_mean, inf):
            ax.plot(xi, yi, ls="none", marker="osD"[k], ms=6.5, mfc=SURFACE if hollow else SERIES[k],
                    mec=SERIES[k] if hollow else SURFACE, mew=1.3)
        ax.plot([], [], ls="none", marker="osD"[k], ms=6.5, mfc=SERIES[k], mec=SURFACE, label=PL[pol])
    ax.plot([], [], ls="none", marker="o", ms=6.5, mfc=SURFACE, mec=INK2, label="hollow: infeasible (τ at floor)")
    ax.axhline(a, color=CRITICAL, lw=1.0)
    ax.text(len(bins_) - 0.6, a, f"α = {a}", fontsize=8, color=INK, va="bottom", ha="right")
    ax.set_xticks(range(len(bins_)))
    ax.set_xticklabels([BL[b].replace(" night", "\nnight") for b in bins_], fontsize=7.5)
    ax.tick_params(axis="x", length=0)
    ax.grid(False, axis="x")
    ax.set_ylabel("Held-out image-level miss fraction (200 splits, 95% interval)")
    ax.set_title("Calibrate in clear daylight, deploy everywhere (α = 0.2)" if not all_inf else
                 "α = 0.2 is infeasible in every bin for this detector")
    ax.legend(loc="upper left", fontsize=7)
    fig.suptitle(f"Clear-weather control: BDD100K val, {DET[det]}, {loss} loss", x=0.02, ha="left", fontsize=10.5,
                 color=INK, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, f"fig6_clear_vs_hazard_{det}_{loss}")


def fig7(dv: pd.DataFrame):
    import matplotlib.pyplot as plt

    from wad.figures import CRITICAL, INK, SERIES, SURFACE, style

    style()
    d = dv[dv.labels == "clean"]
    combos = [(det, loss) for det in ["v8s_640", "v8x_640", "v8x_1280"] for loss in ["class", "group"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for k, (det, loss) in enumerate(combos):
        dd = d[(d.detector == det) & (d.loss == loss)].set_index("bin").reindex(B4)
        x = np.arange(len(B4)) + (k - 2.5) * 0.13
        col = SERIES[["v8s_640", "v8x_640", "v8x_1280"].index(det)]
        mk = "o" if loss == "group" else "s"
        axes[0].plot(x, dd["min_alpha"], ls="none", marker=mk, ms=6.5, mfc=col, mec=SURFACE, mew=1.2,
                     label=f"{DET[det]}, {loss} loss")
        inf = dd["oracle_infeasible_a0.2"].fillna(0).to_numpy() >= 0.5
        for xi, yi, hollow in zip(x, dd["oracle_fppi_a0.2"], inf):
            axes[1].plot(xi, yi, ls="none", marker=mk, ms=6.5, mfc=SURFACE if hollow else col,
                         mec=col if hollow else SURFACE, mew=1.3)
    axes[1].plot([], [], ls="none", marker="o", ms=6.5, mfc=SURFACE, mec="#52514e", label="hollow: infeasible (τ at floor)")
    axes[0].axhline(0.1, color=CRITICAL, lw=1.0)
    axes[0].axhline(0.2, color=CRITICAL, lw=1.0, ls="--")
    axes[0].set_ylabel("Smallest certifiable α (mean over splits)")
    axes[0].set_title("Which budgets can be certified at all (lines: α = 0.1, 0.2)")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Test FPPI, per-bin CRC at α = 0.2")
    axes[1].set_title("The false-positive price of α = 0.2")
    for ax in axes:
        ax.set_xticks(range(len(B4)))
        ax.set_xticklabels([BL[b] for b in B4])
        ax.tick_params(axis="x", length=0)
        ax.grid(False, axis="x")
    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    axes[1].legend(loc="upper right", fontsize=7)
    fig.suptitle("Run 2 on DAWN (clean labels): detector capacity, input size and grouped vehicle loss "
                 "(squares: class loss, dots: group loss)", x=0.02, ha="left", fontsize=10.5, color=INK, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "fig7_run2_detectors")


def fig8(tiers: list[tuple[str, pd.DataFrame]], cap: float):
    import matplotlib.pyplot as plt

    from wad.figures import INK, INK2, SERIES, SURFACE, style

    style()
    fig, axes = plt.subplots(1, len(tiers), figsize=(6 * len(tiers), 4.2), squeeze=False)
    for ax, (title, t) in zip(axes[0], tiers):
        t = t[t.fppi_cap == cap]
        x = np.arange(len(t))
        for k, (m, s, lab, mk, dx) in enumerate([("m_cert_mean", "m_cert_std", "m(b): calibration estimate", "o", -0.12),
                                                  ("test_image_loss_mean", "test_image_loss_std", "held-out miss at τ_c", "s", 0.12)]):
            ax.errorbar(x + dx, t[m], yerr=t[s], fmt="none", ecolor=SERIES[k], elinewidth=1.2)
            ax.plot(x + dx, t[m], ls="none", marker=mk, ms=6.5, mfc=SERIES[k], mec=SURFACE, mew=1.2, label=lab)
        for xi, (tau, ex) in enumerate(zip(t.tau_c_mean, t.exceed_m_frac)):
            ax.text(xi, 0.01, f"τ_c={tau:.2f}\n>m: {ex:.0%}", ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([BL.get(b, b).replace(" night", "\nnight") for b in t.bin], fontsize=7.5)
        ax.tick_params(axis="x", length=0)
        ax.grid(False, axis="x")
        ax.set_ylim(0, max(0.6, float(np.nanmax(t.m_cert_mean)) + 0.1))
        ax.set_ylabel("Image-level miss fraction (mean ± sd, 200 splits)")
        ax.set_title(title, fontsize=9.5)
    axes[0][0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle(f"Confirmed tier: τ_c = smallest threshold with calibration FPPI ≤ {cap}; m(b) estimates the miss rate "
                 "there (expectation-level, not a per-deployment bound)", x=0.02, ha="left", fontsize=10, color=INK,
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, f"fig8_tiers_fppi{cap}")


# ---------------------------------------------------------------- summary
def sec_summary(r2: Cfg, dv: pd.DataFrame) -> list[str]:
    p = r2.primary
    out = ["## Summary", "", "Verdicts use the pre-registered 200-split rerun; *holds* = mean held-out miss ≤ α, "
           "*at the budget* = within 0.01 above α, *fails* = more than 0.01 above.", ""]
    P = dawn_dir(p.detector, p.loss, p.labels)
    prim = dv[(dv.detector == p.detector) & (dv.loss == p.loss) & (dv.labels == p.labels)].set_index("bin")
    o2, g2, o1 = pi(P, 0.2, "oracle_bin"), pi(P, 0.2, "global"), pi(P, 0.1, "oracle_bin")
    r1 = pi(dawn_dir("v8s_640", "class", "all"), 0.2, "oracle_bin")
    r1f = [b for b in B4 if r1.loc[b, "infeasible_frac"] < 0.5]
    bad01 = [b for b in B4 if o1.loc[b, "infeasible_frac"] >= 0.5 or prim.loc[b, "min_alpha"] > 0.1]
    part01 = [b for b in B4 if b not in bad01 and o1.loc[b, "infeasible_frac"] > 0]
    out.append(f"1. **Run 2 primary on DAWN ({DET[p.detector]}, {p.loss} loss, {p.labels} labels; pre-registered).** "
               "At α = 0.2 per-bin CRC " + "; ".join(f"{BL[b]} {word(o2.loc[b, 'loss_mean'], 0.2, o2.loc[b, 'exceed_frac'], o2.loc[b, 'infeasible_frac'])}" for b in B4)
               + f", at {o2.loc[B4, 'fppi_mean'].min():.1f}–{o2.loc[B4, 'fppi_mean'].max():.1f} FPPI (run 1, YOLOv8s: "
               f"{r1.loc[r1f, 'fppi_mean'].min():.1f}–{r1.loc[r1f, 'fppi_mean'].max():.1f} in its feasible bins, fog infeasible at "
               f"{r1.loc['fog', 'fppi_mean']:.0f}). The global threshold: " + "; ".join(
                   f"{BL[b]} {word(g2.loc[b, 'loss_mean'], 0.2, g2.loc[b, 'exceed_frac'])}" for b in B4)
               + ". α = 0.1 is out of reach for the primary in " + ", ".join(
                   f"{BL[b]} (smallest certifiable α {prim.loc[b, 'min_alpha']:.3f}, infeasible in {o1.loc[b, 'infeasible_frac']:.0%} of splits)"
                   for b in bad01) + "; it is feasible in most splits of "
               + ", ".join(f"{BL[b]} (infeasible in {o1.loc[b, 'infeasible_frac']:.0%}, smallest certifiable α {prim.loc[b, 'min_alpha']:.3f})"
                           for b in part01 + [b for b in B4 if b not in bad01 and b not in part01]) + ".")
    best = dv[(dv.labels == "clean") & (dv.detector == "v8x_640") & (dv.loss == "group")].set_index("bin")
    if not best.empty:
        Bd = dawn_dir("v8x_640", "group", "clean")
        b1, b2 = pi(Bd, 0.1, "oracle_bin"), pi(Bd, 0.2, "oracle_bin")
        out.append(f"2. **{DET['v8x_640']}, group, clean (a pre-registered secondary, singled out after the results were "
                   "seen).** Smallest certifiable α " + ", ".join(f"{BL[b]} {best.loc[b, 'min_alpha']:.3f}" for b in B4)
                   + ". At α = 0.1 per-bin CRC " + "; ".join(f"{BL[b]} {word(b1.loc[b, 'loss_mean'], 0.1, b1.loc[b, 'exceed_frac'], b1.loc[b, 'infeasible_frac'])}" for b in B4)
                   + f", at {b1.loc[B4, 'fppi_mean'].min():.1f}–{b1.loc[B4, 'fppi_mean'].max():.1f} FPPI; α = 0.2 costs "
                   f"{b2.loc[B4, 'fppi_mean'].min():.1f}–{b2.loc[B4, 'fppi_mean'].max():.1f} FPPI.")
        x12 = dv[(dv.labels == "clean") & (dv.detector == "v8x_1280") & (dv.loss == "group")].set_index("bin")
        d25 = 100 * (x12.loc[B4, "recall_at_025"] - best.loc[B4, "recall_at_025"])
        dmx = 100 * (x12.loc[B4, "max_recall"] - best.loc[B4, "max_recall"])
        dma = x12.loc[B4, "min_alpha"] - best.loc[B4, "min_alpha"]
        out.append(f"3. **640 vs 1280 px for YOLOv8x (group, clean).** At 1280 recall at τ = 0.25 changes by "
                   + ", ".join(f"{BL[b]} {d25[b]:+.1f}" for b in B4) + " pts and max recall by "
                   + ", ".join(f"{BL[b]} {dmx[b]:+.1f}" for b in B4) + " pts; the smallest certifiable α rises by "
                   + ", ".join(f"{BL[b]} {dma[b]:+.3f}" for b in B4)
                   + ". Higher resolution helps at the shipped threshold but loses objects in the low-confidence tail, "
                   "which is what a miss budget needs.")
    d = bdd_dir("v8x_1280", "group")
    if (d / "pr_vs_ref.csv").exists():
        vs = pd.read_csv(d / "pr_vs_ref.csv")
        vs = vs[vs.cls == "all"].set_index("bin")
        sz = pd.read_csv(d / "pr_vs_ref_sizeadj_EXPLORATORY.csv").set_index("bin")
        op = pd.read_csv(d / "operating_point_boot.csv").set_index("bin")
        ref, orc = pi(d, 0.2, "ref_calibrated"), pi(d, 0.2, "oracle_bin")
        c200 = pd.read_csv(d / "seeds200" / "crc_summary.csv")
        th = c200[(c200.alpha == 0.2) & (c200.policy == "oracle_bin") & (c200.cls == "all")].set_index("bin")
        small = {b: sz.loc[b, "share_band0"] + sz.loc[b, "share_band1"] for b in DAY + NIGHT}
        out.append(f"4. **Clear-weather control (BDD100K val, {DET['v8x_1280']}, group loss).** Raw, at τ = 0.25 and "
                   "relative to clear daytime: rain recall " + pts(vs.loc["rain", "d_recall"], vs.loc["rain", "d_recall_lo"], vs.loc["rain", "d_recall_hi"])
                   + ", precision " + pts(vs.loc["rain", "d_precision"], vs.loc["rain", "d_precision_lo"], vs.loc["rain", "d_precision_hi"])
                   + "; snow recall " + pts(vs.loc["snow", "d_recall"], vs.loc["snow", "d_recall_lo"], vs.loc["snow", "d_recall_hi"])
                   + ". But the bins do not hold the same objects: boxes under 20 px tall are "
                   + ", ".join(f"{BL[b]} {small[b]:.0%}" for b in DAY) + " of each daytime bin. At clear daytime's object-height "
                   "mix (exploratory, added after review) the differences are rain " + pts(sz.loc["rain", "d_recall_std"], sz.loc["rain", "d_recall_std_lo"], sz.loc["rain", "d_recall_std_hi"])
                   + ", snow " + pts(sz.loc["snow", "d_recall_std"], sz.loc["snow", "d_recall_std_lo"], sz.loc["snow", "d_recall_std_hi"])
                   + ", cloudy " + pts(sz.loc["cloudy", "d_recall_std"], sz.loc["cloudy", "d_recall_std_lo"], sz.loc["cloudy", "d_recall_std_hi"])
                   + ", and at night " + ", ".join(f"{BL[b]} " + pts(sz.loc[b, "d_recall_std"], sz.loc[b, "d_recall_std_lo"], sz.loc[b, "d_recall_std_hi"]) for b in NIGHT)
                   + ": daytime rain does cost recall per object; the bin as sampled hides it behind larger objects.")
        out.append("5. **Does the operating point move?** The τ that gives 1 FPPI is "
                   + ", ".join(f"{BL[b]} {op.loc[b, 'tau_fppi1.0']:.3f} [{op.loc[b, 'tau_fppi1.0_lo']:.3f}, {op.loc[b, 'tau_fppi1.0_hi']:.3f}]" for b in DAY + NIGHT)
                   + " (95% bootstrap): fixed-FPPI thresholds are stable by day and drop at night. The per-bin CRC threshold "
                   "at α = 0.2 does move: " + ", ".join(f"{BL[b]} {th.loc[b, 'tau_hat_mean']:.4f}" for b in DAY + NIGHT)
                   + ". The clear-calibrated threshold " + "; ".join(
                       f"{BL[b]} {word(ref.loc[b, 'loss_mean'], 0.2, ref.loc[b, 'exceed_frac'])}" for b in DAY[1:] + NIGHT)
                   + ". Per-bin calibration (" + "; ".join(f"{BL[b]} {word(orc.loc[b, 'loss_mean'], 0.2, orc.loc[b, 'exceed_frac'])}" for b in NIGHT)
                   + ") restores the night budget at " + ", ".join(f"{orc.loc[b, 'fppi_mean']:.0f}" for b in NIGHT)
                   + " FPPI; by day it changes FPPI by " + ", ".join(
                       f"{BL[b]} {ref.loc[b, 'd_fppi_mean'] * -1:+.1f} [{-ref.loc[b, 'd_fppi_hi']:+.1f}, {-ref.loc[b, 'd_fppi_lo']:+.1f}]" for b in DAY[1:])
                   + " relative to the clear-calibrated threshold.")
        dop = dawn_dir(p.detector, p.loss, p.labels) / "operating_point_boot.csv"
        if dop.exists():
            o = pd.read_csv(dop).set_index("bin")
            out[-1] += (" On DAWN's severe weather, by contrast, the τ that gives 1 FPPI differs between bins (primary: "
                        + ", ".join(f"{BL[b]} {o.loc[b, 'tau_fppi1.0']:.3f} [{o.loc[b, 'tau_fppi1.0_lo']:.3f}, "
                                    f"{o.loc[b, 'tau_fppi1.0_hi']:.3f}]" for b in B4) + ").")
        gl = pd.read_csv(d / "seeds200" / "gains_losses.csv")
        g = gl[(gl.alpha == 0.2) & (gl.policy == "oracle_bin")].set_index("bin")
        gr = gl[(gl.alpha == 0.2) & (gl.policy == "ref_calibrated")].set_index("bin")
        out.append("6. **Gains and losses of tuning (α = 0.2, 200 splits).** From τ = 0.25 to the per-bin threshold: "
                   + "; ".join(f"{BL[b]} {100 * g.loc[b, 'd_recall_mean']:+.0f} recall / {100 * g.loc[b, 'd_precision_mean']:+.0f} "
                               f"precision pts, {g.loc[b, 'fp_per_recovered']:.1f} FP per recovered object" for b in DAY + NIGHT)
                   + ". At the one shared clear-calibrated threshold a recovered object costs "
                   + ", ".join(f"{BL[b]} {gr.loc[b, 'fp_per_recovered']:.1f}" for b in DAY + NIGHT)
                   + " FP: the exchange rate is similar day and night at the same τ; what night changes is how far τ must "
                   "drop to meet the budget, and the rate rises steeply as it does.")
    cd = OUT / "cross_dataset.csv"
    if cd.exists():
        c = pd.read_csv(cd)
        c = c[(c.detector == "v8x_1280") & (c.loss == "group") & (c.alpha == 0.2)].set_index("dawn_bin")
        if not c.empty:
            out.append(f"7. **A BDD100K clear-daytime threshold on DAWN** (same detector and loss, clean labels, α = 0.2, median "
                       f"τ̂ {c.tau_bdd_clear_median.iloc[0]:.3f}): " + "; ".join(
                           f"{BL[b]} {word(c.loc[b, 'image_loss_mean'], 0.2)} [95% image bootstrap "
                           f"{c.loc[b, 'image_loss_at_median_lo']:.3f}–{c.loc[b, 'image_loss_at_median_hi']:.3f}]" for b in B4)
                       + ". Descriptive only: DAWN and BDD100K differ in more than weather.")
    tP = P / "seeds200" / "tiers_summary.csv"
    if tP.exists():
        tt = pd.read_csv(tP)
        tt = tt[tt.fppi_cap == 1.0].set_index("bin")
        tb = pd.read_csv(dawn_dir("v8x_640", "group", "clean") / "seeds200" / "tiers_summary.csv")
        tb = tb[tb.fppi_cap == 1.0].set_index("bin")
        out.append("8. **Speed-prior input (confirmed tier, FPPI ≤ 1).** The calibration estimate m(b) of the miss rate at "
                   f"τ_c is, for the primary, " + ", ".join(f"{BL[b]} {tt.loc[b, 'm_cert_mean']:.3f}" for b in B4)
                   + " (held-out " + ", ".join(f"{tt.loc[b, 'test_image_loss_mean']:.3f}" for b in B4)
                   + f"; above m(b) in {tt.loc[B4, 'exceed_m_frac'].min():.0%}–{tt.loc[B4, 'exceed_m_frac'].max():.0%} of splits), and "
                   f"for {DET['v8x_640']} group (post hoc best) " + ", ".join(f"{BL[b]} {tb.loc[b, 'm_cert_mean']:.3f}" for b in B4)
                   + ". m(b) is an expectation-level estimate, not a per-deployment bound; a planner that needs a "
                   "high-probability bound needs a held-out upper confidence bound on top of it.")
    out += ["", "**What this changes for the project.** Night (and, on DAWN, fog) is where one threshold breaks the miss "
            "budget and per-bin calibration earns its false positives; daytime rain also costs recall per object, but "
            "on BDD100K a clear-calibrated threshold still met α = 0.2 in the rain bin as sampled. α = 0.1 was certifiable "
            "in every DAWN bin only with YOLOv8x @640 and the grouped loss, a variant singled out after the results; the "
            "pre-registered primary reaches α = 0.2 everywhere. The hazard estimator (O3) should therefore separate "
            "visibility (night, fog) first.", ""]
    return out


# ---------------------------------------------------------------- sections
def sec_setup(r2: Cfg) -> list[str]:
    rows, caps = [], []
    for det, d in r2.detectors.items():
        for base, ds in [(REPO / d["detections"], "DAWN"), (REPO / "cache" / "bdd" / "det" / det, "BDD100K")]:
            p = base / "detections.meta.json"
            if not p.exists():
                continue
            m = json.loads(p.read_text())
            rows.append(dict(dataset=ds, detector=DET[det], device=m["device"], images=m["n_images"],
                             boxes=f"{m['n_detections']:,}", **{"s/image": f"{m['mean_s_per_image']:.3f}"}, torch=m["torch_version"]))
            det_df = pd.read_parquet(base / "detections.parquet", columns=["image_id"])
            n = det_df.groupby("image_id").size()
            capped = n[n >= 300]
            if len(capped):
                byb = pd.Series(capped.index.str.split("/").str[0]).value_counts()
                caps.append(f"{ds} {DET[det]}: " + ", ".join(f"{BL.get(b, b)} {v}" for b, v in byb.items()))
    gc = json.loads((OUT / "gpu_cpu_check.json").read_text()) if (OUT / "gpu_cpu_check.json").exists() else None
    out = ["## 1. Setup", "",
           "Pre-registered in `PREREG_run2.md` before any run 2 or BDD100K number existed (deviations are listed in "
           "section 7). GPU: NVIDIA RTX 3050 Ti Laptop (4 GB), driver 566.07, torch 2.14.1+cu126.", ""]
    if gc:
        out += [f"Run 1 (CPU) reproducibility with this build: CPU detections bit-identical to the run 1 cache on "
                f"{gc['bit_identical']} of {gc['n_images']} DAWN images checked (`results/run2/gpu_cpu_check.json`).", ""]
    out += [md(pd.DataFrame(rows)), ""]
    if caps:
        out += ["Images where max_det = 300 binds (low-score boxes cut there, which matters most for per-bin thresholds "
                "near the 0.001 floor): " + "; ".join(caps) + ".", ""]
    return out


def sec_dawn(r2: Cfg, dv: pd.DataFrame) -> list[str]:
    out = ["## 2. Run 2 on DAWN", "",
           "One row per detector × loss × label set and weather bin, 10 splits (the 200-split numbers for the "
           "primary and the best variant are in the summary and in each folder's `seeds200/`). *min α* = smallest "
           "budget the CRC rule can certify; cells are held-out miss fraction / FPPI (*inf* = share of splits with no "
           "feasible threshold, *exc* = share of splits above α).", ""]
    for labels in ["clean", "all"]:
        rows = []
        for _, r in dv[dv.labels == labels].iterrows():
            rows.append({"detector": DET[r.detector], "loss": r.loss, "bin": BL[r.bin], "max recall": pct(r.max_recall),
                         "recall@0.25": pct(r.recall_at_025), "min α": num(r.min_alpha),
                         "per-bin α=0.1": f"{r['oracle_loss_a0.1']:.3f} / {r['oracle_fppi_a0.1']:.1f} (inf {r['oracle_infeasible_a0.1']:.1f})",
                         "per-bin α=0.2": f"{r['oracle_loss_a0.2']:.3f} / {r['oracle_fppi_a0.2']:.1f} (inf {r['oracle_infeasible_a0.2']:.1f})",
                         "global α=0.2": f"{r['global_loss_a0.2']:.3f} / {r['global_fppi_a0.2']:.1f} (exc {r['global_exceed_a0.2']:.1f})"})
        out += [f"### Labels: {labels}", "", md(pd.DataFrame(rows), "lllrrrrrr"), ""]
    rows = []
    for det in r2.detectors:
        for loss in r2.losses:
            for labels in r2.labels:
                d = dawn_dir(det, loss, labels)
                if not (d / "go_no_go.json").exists():
                    continue
                v = json.loads((d / "go_no_go.json").read_text())
                b = json.loads((d / "go_no_go_bootstrap.json").read_text())
                rows.append({"detector": DET[det], "loss": loss, "labels": labels, "verdict": v["verdict"],
                             "person: worst bin, margin": f"{BL[v['classes']['person']['worst_bin']]}, {v['classes']['person']['margin_points']:.1f}",
                             "car: worst bin, margin": f"{BL[v['classes']['car']['worst_bin']]}, {v['classes']['car']['margin_points']:.1f}",
                             "bootstrap P(GO)": f"{b['p_go']:.2f}"})
    rows = []
    for det, loss in [("v8s_640", "class"), (r2.primary.detector, r2.primary.loss), ("v8x_640", "group")]:
        labels = "all" if det == "v8s_640" else "clean"
        f = dawn_dir(det, loss, labels) / "operating_point_boot.csv"
        if not f.exists():
            continue
        for _, r in pd.read_csv(f).iterrows():
            rows.append({"detector": f"{DET[det]}, {loss}, {labels}", "bin": BL[r.bin],
                         "τ at FPPI 0.1": f"{r['tau_fppi0.1']:.3f} [{r['tau_fppi0.1_lo']:.3f}, {r['tau_fppi0.1_hi']:.3f}]",
                         "τ at FPPI 1": f"{r['tau_fppi1.0']:.3f} [{r['tau_fppi1.0_lo']:.3f}, {r['tau_fppi1.0_hi']:.3f}]",
                         "F1-optimal τ": f"{r.tau_f1opt:.3f} [{r.tau_f1opt_lo:.3f}, {r.tau_f1opt_hi:.3f}]"})
    out += ["### Operating points per DAWN weather bin (95% image bootstrap)", "",
            "Unlike BDD100K's daytime bins (section 4), DAWN's weather bins need different thresholds for the same "
            "false-positive rate.", "", md(pd.DataFrame(rows), "llrrr"), ""]
    out += ["### Go/no-go per variant", "",
            "A stronger detector leaves less recall below τ = 0.25, so the gate can turn NO-GO because the detector is "
            "better; clean labels flip YOLOv8x @1280 from NO-GO to GO. Bootstrap: 2,000 resamples of near-duplicate "
            "groups within bin.", "", md(pd.DataFrame(rows), "llllrrr"), ""]
    return out


def sec_tiers(r2: Cfg) -> list[str]:
    p = r2.primary
    out = ["## 3. Confirmed tier (speed-prior input)", "",
           "Per bin, τ_c is the smallest threshold whose calibration FPPI is at most the cap; m(b) = (n·R̂(τ_c)+1)/(n+1) "
           "on the calibration half estimates the miss rate the planner should assume at that operating point. It is an "
           "expectation-level estimate (the held-out miss exceeds it in a large share of splits), not a per-deployment "
           "bound. 200 splits, mean ± sd.", ""]
    for name, path in [(f"DAWN, {DET[p.detector]}, {p.loss}, {p.labels} (primary)", dawn_dir(p.detector, p.loss, p.labels)),
                       (f"DAWN, {DET['v8x_640']}, group, clean (post hoc best)", dawn_dir("v8x_640", "group", "clean")),
                       (f"BDD100K, {DET['v8x_1280']}, group", bdd_dir("v8x_1280", "group"))]:
        f = path / "seeds200" / "tiers_summary.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        rows = [{"FPPI cap": r.fppi_cap, "bin": BL.get(r.bin, r.bin), "τ_c": f"{r.tau_c_mean:.3f} ± {r.tau_c_std:.3f}",
                 "m(b) estimate": f"{r.m_cert_mean:.3f} ± {r.m_cert_std:.3f}",
                 "held-out miss": f"{r.test_image_loss_mean:.3f} ± {r.test_image_loss_std:.3f}",
                 "held-out FPPI": f"{r.test_fppi_mean:.2f} ± {r.test_fppi_std:.2f}", "above m(b)": f"{r.exceed_m_frac:.2f}",
                 "n_cal": f"{r.n_cal_mean:.0f}"} for r in t.itertuples()]
        out += [f"### {name}", "", md(pd.DataFrame(rows), "rlrrrrrr"), ""]
    return out


def sec_bdd(r2: Cfg) -> list[str]:
    rep = json.loads((REPO / "cache" / "bdd" / "data_report.json").read_text())
    im = pd.read_parquet(REPO / "cache" / "bdd" / "images.parquet")
    gt = pd.read_parquet(REPO / "cache" / "bdd" / "gt.parquet")
    t = gt.groupby(["weather", "cls"]).size().unstack(fill_value=0)
    t.insert(0, "boxes", t.sum(axis=1))
    t.insert(0, "images", im[im.keep].groupby("weather").size())
    hgt = gt.assign(h=gt.y2 - gt.y1).groupby("weather").h
    t["median box height (px)"] = hgt.median().round(1)
    t["boxes < 20 px tall"] = hgt.apply(lambda s: f"{(s < 20).mean():.0%}")
    t = t.reindex(DAY + NIGHT).reset_index().rename(columns={"weather": "bin"})
    t["bin"] = t["bin"].map(BL)
    out = ["## 4. Clear-weather control: BDD100K val", "",
           f"{rep['n_images_found']:,} val images fall in the seven pre-registered bins ({rep['n_images_kept']:,} after "
           f"removing exact duplicates; the one exact duplicate whose copies carry different weather tags is dropped "
           f"entirely). Excluded by tag: " + ", ".join(f"{k.replace('excluded ', '')} {v}" for k, v in
                                                     sorted(rep["excluded_images"].items(), key=lambda x: -x[1]))
           + f". Dropped boxes: {rep['dropped_boxes']}. Same cameras and labelling protocol, but **not the same scenes**: "
           "rain and snow bins hold larger objects and fewer highway frames than clear daytime, so raw bin differences "
           "mix weather with object size; the size-standardised comparison below separates them.", "",
           md(t.astype(str)), ""]
    for det, loss in [("v8s_640", "class"), ("v8x_1280", "group")]:
        d = bdd_dir(det, loss)
        if not (d / "pr_table.csv").exists():
            continue
        pr = pd.read_csv(d / "pr_table.csv")
        rows = []
        for _, r in pr[pr.cls == "all"].iterrows():
            rows.append({"bin": BL[r.bin], "images": r.n_images, "boxes": r.n_gt, "max recall": pct(r.max_recall),
                         "R@0.25": pct(r["t0.25_recall"]), "P@0.25": pct(r["t0.25_precision"]),
                         "FPPI@0.25": num(r["t0.25_fppi"], 2), "R@0.5": pct(r["t0.5_recall"]), "P@0.5": pct(r["t0.5_precision"]),
                         "F1-opt τ": num(r.f1opt_tau, 3), "R / P at F1-opt": f"{pct(r.f1opt_recall)} / {pct(r.f1opt_precision)}"})
        out += [f"### {DET[det]}, {loss} loss", "", "Precision and recall per bin (all classes; per-class precision under the "
                "group loss counts a matched box for the class of the ground truth it matched):", "", md(pd.DataFrame(rows)), ""]
        vs = pd.read_csv(d / "pr_vs_ref.csv")
        sz = pd.read_csv(d / "pr_vs_ref_sizeadj_EXPLORATORY.csv").set_index("bin")
        rows = []
        for _, r in vs[vs.cls.isin(["all", "person", "car"]) & (vs.bin != "clear")].iterrows():
            row = {"bin": BL[r.bin], "class": r.cls, "Δ recall": pts(r.d_recall, r.d_recall_lo, r.d_recall_hi),
                   "Δ precision": pts(r.d_precision, r.d_precision_lo, r.d_precision_hi),
                   "Δ FPPI": f"{r.d_fppi:+.2f} [{r.d_fppi_lo:+.2f}, {r.d_fppi_hi:+.2f}]"}
            if r.cls == "all":
                row["Δ recall at clear's size mix (exploratory)"] = pts(sz.loc[r.bin, "d_recall_std"], sz.loc[r.bin, "d_recall_std_lo"],
                                                                         sz.loc[r.bin, "d_recall_std_hi"])
            else:
                row["Δ recall at clear's size mix (exploratory)"] = ""
            rows.append(row)
        out += ["Difference from clear daytime at τ = 0.25 (95% bootstrap CI, images resampled within bin; Q1):", "",
                md(pd.DataFrame(rows), "llrrrr"), ""]
        op = pd.read_csv(d / "operating_point_boot.csv")
        rows = [{"bin": BL[r.bin], "τ at FPPI 0.1": f"{r['tau_fppi0.1']:.3f} [{r['tau_fppi0.1_lo']:.3f}, {r['tau_fppi0.1_hi']:.3f}]",
                 "τ at FPPI 1": f"{r['tau_fppi1.0']:.3f} [{r['tau_fppi1.0_lo']:.3f}, {r['tau_fppi1.0_hi']:.3f}]",
                 "F1-optimal τ": f"{r.tau_f1opt:.3f} [{r.tau_f1opt_lo:.3f}, {r.tau_f1opt_hi:.3f}]",
                 "Δ τ at FPPI 1 vs clear": f"{r['d_tau_fppi1.0']:+.3f} [{r['d_tau_fppi1.0_lo']:+.3f}, {r['d_tau_fppi1.0_hi']:+.3f}]"}
                for _, r in op.iterrows()]
        out += ["Operating points per bin (95% bootstrap CI; Q2):", "", md(pd.DataFrame(rows), "lrrrr"), ""]
        ma = pd.read_csv(d / "min_feasible_alpha.csv").set_index("bin")
        rows = []
        for a in [0.1, 0.2]:
            for b in DAY + NIGHT:
                row = {"α": a, "bin": BL[b]}
                for pol in ["ref_calibrated", "oracle_bin", "global", "default_0.25"]:
                    x = pi(d, a, pol)
                    if b not in x.index:
                        continue
                    x = x.loc[b]
                    inf = "" if pd.isna(x.infeasible_frac) or x.infeasible_frac == 0 else f", inf {x.infeasible_frac:.1f}"
                    row[PL[pol]] = f"{x.loss_mean:.3f} [{x.loss_lo:.3f}, {x.loss_hi:.3f}] / {x.fppi_mean:.1f} (exc {x.exceed_frac:.2f}{inf})"
                rows.append(row)
        out += ["Policies on the held-out halves, 200 splits (Q3): miss fraction [2.5–97.5% over splits] / FPPI (share of "
                "splits above α, share of fits infeasible). Smallest certifiable α: "
                + ", ".join(f"{BL[b]} {ma.loc[b, 'min_alpha_mean']:.3f}" for b in DAY + NIGHT) + ".", "",
                md(pd.DataFrame(rows), "rlrrrr"), ""]
        g = pd.read_csv(d / "seeds200" / "gains_losses.csv")
        rows = []
        for _, r in g[(g.alpha == 0.2) & g.policy.isin(["oracle_bin", "ref_calibrated"])].iterrows():
            rows.append({"policy": PL[r.policy], "bin": BL[r.bin],
                         "Δ recall": pts(r.d_recall_mean, r.d_recall_lo, r.d_recall_hi),
                         "Δ precision": pts(r.d_precision_mean, r.d_precision_lo, r.d_precision_hi),
                         "Δ FPPI": f"{r.d_fppi_mean:+.2f} [{r.d_fppi_lo:+.2f}, {r.d_fppi_hi:+.2f}]",
                         "FP per recovered object": num(r.fp_per_recovered, 2)})
        out += ["Gains and losses of moving from τ = 0.25 to the calibrated threshold (α = 0.2, 200 splits, 2.5–97.5% over "
                "splits; FP per recovered object = ΣΔFP / ΣΔTP; Q4):", "", md(pd.DataFrame(rows), "llrrrr"), ""]
    return out


def sec_cross() -> list[str]:
    p = OUT / "cross_dataset.csv"
    if not p.exists():
        return []
    d = pd.read_csv(p)
    rows = [{"detector": DET[r.detector], "loss": r.loss, "α": r.alpha, "DAWN bin": BL[r.dawn_bin],
             "median τ̂ (BDD clear)": num(r.tau_bdd_clear_median, 4) + (" (floor: infeasible)" if r.bdd_fit_infeasible_frac >= 0.5 else ""),
             "DAWN miss fraction": num(r.image_loss_mean),
             "95% CI at median τ̂": f"{r.image_loss_at_median_lo:.3f}–{r.image_loss_at_median_hi:.3f}",
             "P(miss > α)": f"{r.p_loss_above_alpha:.2f}", "recall": pct(r.recall_mean), "precision": pct(r.precision_mean),
             "FPPI": num(r.fppi_mean, 2)} for r in d.itertuples()]
    return ["## 5. Cross-dataset: a BDD100K clear-daytime threshold applied to DAWN", "",
            "Descriptive only (camera, framing and annotation differ). Each clear-calibrated BDD100K threshold is applied "
            "to the DAWN images (clean labels, as the primary); the CI is an image bootstrap within the DAWN bin at the "
            "median threshold.", "", md(pd.DataFrame(rows), "llrlrrrrrrr"), ""]


def sec_deviations() -> list[str]:
    return ["## 7. Deviations from the pre-registration", "",
            "- Added after an adversarial review, labelled exploratory: the size-standardised recall comparison, the "
            "α sweep (`alpha_sweep_EXPLORATORY.csv`), O3's argmax-class bins.",
            "- Q3 and Q4 intervals are percentiles over the 200 random calibration/test splits, not an image bootstrap; "
            "Q1 and Q2 use the pre-registered image bootstrap.",
            "- Data: 37 zero-area BDD100K boxes and one exact duplicate filed under two weathers were dropped (both "
            "unmatchable or ambiguous); overcast/partly-cloudy *night* images were not in any bin (the bin list named "
            "only daytime cloudy).",
            "- YOLOv8x @640 group clean was pre-registered as a secondary detector; calling it the best variant is a "
            "post hoc selection and is labelled so everywhere.", ""]


def write_report(r2: Cfg) -> str:
    dv = pd.read_csv(OUT / "dawn_variants.csv")
    figs = []
    for det, loss in [("v8s_640", "class"), ("v8x_1280", "group")]:
        if (bdd_dir(det, loss) / "seeds200" / "policy_intervals.csv").exists():
            figs += fig6(det, loss)
    figs += fig7(dv)
    p = r2.primary
    tiers = []
    for name, path in [(f"DAWN, {DET[p.detector]}, {p.loss}, {p.labels} (primary)", dawn_dir(p.detector, p.loss, p.labels)),
                       ("BDD100K, YOLOv8x @1280, group", bdd_dir("v8x_1280", "group"))]:
        if (path / "seeds200" / "tiers_summary.csv").exists():
            tiers.append((name, pd.read_csv(path / "seeds200" / "tiers_summary.csv")))
    if tiers:
        figs += fig8(tiers, 1.0)
    lines = ["# Run 2 (O2) and the clear-weather control", "",
             "Generated by `python -m wad.run2 report` from `results/run2/**`; do not edit by hand. Reproduce with "
             "`python -m wad.run2 analyse` (from cached detections) after `python -m wad.run2 detect`. O3 results: "
             "`results/o3/o3.md`.", "",
             "Figures: " + ", ".join(f"`{q.relative_to(REPO).as_posix()}`" for q in figs if q.suffix == ".png"), ""]
    for sec in (sec_summary(r2, dv), sec_setup(r2), sec_dawn(r2, dv), sec_tiers(r2), sec_bdd(r2), sec_cross(),
                sec_deviations()):
        lines += sec
    text = "\n".join(lines).rstrip() + "\n"
    (OUT / "run2.md").write_text(text, encoding="utf-8")
    print(OUT / "run2.md")
    return text
