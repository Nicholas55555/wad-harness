"""Figures (PNG and PDF, 300 dpi, one style), all drawn from the cache and the result CSVs.

fig1_missrate_fppi  miss rate vs FPPI per weather bin (person, car), defaults marked
fig2_go_no_go       max achievable recall vs recall at 0.25 / 0.5, margin and verdict
fig3_main           achieved test miss and FPPI per bin, oracle-bin vs global vs default
fig4_hazard         (phase 6a) hazard scalar per weather label; predicted vs oracle bins

Colour: categorical slots in fixed order (blue, orange, aqua, yellow) from a palette validated
for colour-vision deficiency on the adjacent pairlist (lines/bars); the three policies use the
first three slots, which also pass on all pairs. Aqua and yellow sit below 3:1 contrast on
white, so every series is also identified by a legend and a direct label or marker shape.

Run: python -m wad.figures [--smoke]
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

from wad.config import Cfg, bins, load_config, results_path  # noqa: E402

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
ORDINAL_BLUE = ["#104281", "#2a78d6", "#86b6ef"]  # dark -> light, validated as an ordinal ramp
CRITICAL = "#d03b3b"
BIN_LABEL = {"fog": "Fog", "rain": "Rain", "sand": "Sand", "snow": "Snow", "all": "All"}
POLICY_LABEL = {"oracle_bin": "Oracle-bin CRC", "global": "Global CRC", "default_0.25": "Default τ = 0.25"}
MARKERS = {"oracle_bin": "o", "global": "s", "default_0.25": "D"}


def style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def save(fig, cfg: Cfg, name: str) -> list:
    out = []
    for ext in ("png", "pdf"):
        p = results_path(cfg, f"figures/{name}.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight", metadata={"CreationDate": None} if ext == "pdf" else None)
        out.append(p)
    plt.close(fig)
    return out


def rounded_bar(ax, x0, width, height, color, radius_px=3.0, baseline=0.0):
    """A bar with a rounded data-end and a square baseline. Call after the axis limits are final."""
    if height <= 0:
        return
    fig = ax.figure
    fig.canvas.draw_idle()
    t = ax.transData
    px_x = abs(t.transform((1, 0))[0] - t.transform((0, 0))[0])
    px_y = abs(t.transform((0, 1))[1] - t.transform((0, 0))[1])
    rx = min(radius_px / px_x, width / 2)
    ry = min(radius_px / px_y, height / 2)
    x1, top = x0 + width, baseline + height
    k = 0.5523  # cubic bezier circle constant
    verts = [(x0, baseline), (x0, top - ry), (x0, top - ry + k * ry), (x0 + rx - k * rx, top), (x0 + rx, top),
             (x1 - rx, top), (x1 - rx + k * rx, top), (x1, top - ry + k * ry), (x1, top - ry), (x1, baseline),
             (x0, baseline)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4,
             MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


# ---------------------------------------------------------------- figure 1
def fig1(cfg: Cfg):
    from wad.curves import Cache, operating_curve

    c = Cache(cfg)
    t25, t50 = cfg.default_thresholds
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, cls in zip(axes, ["person", "car"]):
        ends = []
        for i, b in enumerate(bins(cfg)):
            rows = np.nonzero(c.weather == b)[0]
            sel = np.zeros(c.n_img, bool)
            sel[rows] = True
            pm = c.pred.image_id.map(c.idx).to_numpy()
            m = sel[pm] & (c.pred.eval_cls.to_numpy() == cls)
            n_gt = int(c.n_gt(cls)[rows].sum())
            oc = operating_curve(c.pred.score.to_numpy()[m], c.pred.tp.to_numpy()[m], n_gt, len(rows))
            f, mr, thr = oc["fppi"], oc["miss_rate"], oc["threshold"]
            keep = f > 0
            ax.plot(f[keep], mr[keep], color=SERIES[i], lw=1.6, zorder=3)
            for t, mk in [(t25, "o"), (t50, "s")]:
                j = np.nonzero(thr >= t)[0].max()  # last operating point still at score >= t
                ax.plot(f[j], mr[j], marker=mk, ms=6.5, mfc=SERIES[i], mec=SURFACE, mew=1.2, ls="none", zorder=4)
            ends.append(f"{BIN_LABEL[b]} {n_gt}")
        # The curves converge at high FPPI, so identity comes from the shared legend rather than
        # nudged end labels; the per-bin box counts sit under the panel title.
        ax.text(0.5, 1.01, "ground-truth boxes: " + " · ".join(ends), transform=ax.transAxes, ha="center",
                va="bottom", fontsize=7.5, color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(0.01, 10)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False positives per image (FPPI, log scale)")
        ax.set_title(f"{cls.capitalize()}", pad=16)
        ax.grid(True, which="major")
        ax.grid(False, which="minor")
    axes[0].set_ylabel("Miss rate (1 − recall), object level")
    from matplotlib.lines import Line2D

    h = [Line2D([], [], color=SERIES[i], lw=1.6, label=BIN_LABEL[b]) for i, b in enumerate(bins(cfg))]
    h += [Line2D([], [], marker="o", ls="none", mfc=MUTED, mec=SURFACE, ms=6.5, label=f"τ = {t25} (Ultralytics default)"),
          Line2D([], [], marker="s", ls="none", mfc=MUTED, mec=SURFACE, ms=6.5, label=f"τ = {t50} (hand-picked)")]
    fig.legend(handles=h, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.01), fontsize=8)
    fig.suptitle("Miss rate vs FPPI per weather bin, YOLOv8s on DAWN (axis cut at FPPI 10; curves continue to the "
                 "0.001 floor at FPPI 11–74, see summary_phase3.csv)",
                 x=0.02, ha="left", fontsize=10.5, color=INK, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95), w_pad=3)
    return save(fig, cfg, "fig1_missrate_fppi")


# ---------------------------------------------------------------- figure 2
def fig2(cfg: Cfg):
    s = pd.read_csv(results_path(cfg, "summary_phase3.csv"))
    v = json.loads(results_path(cfg, "go_no_go.json").read_text())
    t25, t50 = cfg.default_thresholds
    margin = cfg.go_no_go.margin_points
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    series = [("max_recall", "Max achievable recall (τ = 0.001)"), (f"recall_at_{t25}", f"Recall at τ = {t25}"),
              (f"recall_at_{t50}", f"Recall at τ = {t50}")]
    bw, gap = 0.24, 0.03
    for ax, cls in zip(axes, cfg.go_no_go.classes):
        d = s[(s.cls == cls) & (s.bin.isin(bins(cfg)))].set_index("bin").loc[bins(cfg)]
        ax.set_xlim(-0.6, len(d) - 0.4)
        ax.set_ylim(0, 1.12)
        worst = v["classes"][cls]["worst_bin"]
        for i, b in enumerate(d.index):
            for k, (col, _) in enumerate(series):
                x0 = i + (k - 1) * (bw + gap) - bw / 2
                rounded_bar(ax, x0, bw, d.loc[b, col], ORDINAL_BLUE[k])
            # bracket from the max-recall bar to the tau=0.25 bar, labelled with the margin
            hi, lo = d.loc[b, "max_recall"], d.loc[b, f"recall_at_{t25}"]
            pts = 100 * (hi - lo)
            c0, c1 = i - (bw + gap), i
            ax.plot([c0, c0, c1, c1], [hi + 0.015, hi + 0.04, hi + 0.04, lo + 0.015], color=INK2, lw=0.8, zorder=4)
            ax.text((c0 + c1) / 2, hi + 0.05, f"+{pts:.1f} pts", ha="center", va="bottom", fontsize=8,
                    color=INK if b == worst else INK2, fontweight="semibold" if b == worst else "normal")
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels([f"{BIN_LABEL[b]}{'  (worst)' if b == worst else ''}\nn = {int(d.loc[b, 'n_gt'])}"
                            for b in d.index])
        ax.tick_params(axis="x", length=0)
        ax.grid(False, axis="x")
        r = v["classes"][cls]
        ax.set_title(f"{cls.capitalize()}: worst bin {BIN_LABEL[worst]}, margin {r['margin_points']:.1f} pts "
                     f"({'≥' if r['passed'] else '<'} {margin})")
    axes[0].set_ylabel("Recall at IoU 0.5")
    from matplotlib.patches import Patch

    fig.legend([Patch(color=ORDINAL_BLUE[k]) for k in range(3)], [lab for _, lab in series],
               loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Go/no-go: {v['verdict']}. Margin = max achievable recall − recall at τ = {t25} in the worst "
                 f"weather bin; rule needs ≥ {margin} points for person and car",
                 x=0.02, ha="left", fontsize=10.5, color=INK, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return save(fig, cfg, "fig2_go_no_go")


# ---------------------------------------------------------------- figure 3
def fig3(cfg: Cfg, summary_name: str = "crc_summary.csv", out_name: str = "fig3_main", title_note: str = ""):
    s = pd.read_csv(results_path(cfg, summary_name))
    s = s[s.cls == "all"]
    policies = ["oracle_bin", "global", "default_0.25"]
    order = bins(cfg) + ["all"]
    alphas = cfg.alphas
    fig, axes = plt.subplots(len(alphas), 2, figsize=(11, 3.4 * len(alphas)), squeeze=False)
    dx = 0.22
    for r, alpha in enumerate(alphas):
        d = s[s.alpha == alpha]
        for col, (metric, ylab) in enumerate([("image_loss", "Test image-level miss fraction"),
                                               ("fppi", "Test FPPI (log scale)")]):
            ax = axes[r, col]
            for k, p in enumerate(policies):
                dp = d[d.policy == p].set_index("bin").reindex(order)
                x = np.arange(len(order)) + (k - 1) * dx
                y, e = dp[f"{metric}_mean"].to_numpy(), dp[f"{metric}_std"].to_numpy()
                inf = dp["infeasible_frac"].fillna(0).to_numpy() if p != "default_0.25" else np.zeros(len(order))
                ax.errorbar(x, y, yerr=e, fmt="none", ecolor=SERIES[k], elinewidth=1.2, capsize=0, zorder=3)
                for xi, yi, fi in zip(x, y, inf):
                    hollow = fi >= 0.5
                    ax.plot(xi, yi, marker=MARKERS[p], ms=7, ls="none", zorder=4,
                            mfc=SURFACE if hollow else SERIES[k], mec=SERIES[k] if hollow else SURFACE, mew=1.4)
                if metric == "image_loss" and p == "oracle_bin":
                    n_seeds = int(dp.n_seeds.max())
                    for xi, fi, b in zip(x, inf, order):
                        if fi > 0 and b != "all":
                            ax.text(xi, 0.015, f"infeasible\n{round(fi * n_seeds)}/{n_seeds}", ha="center",
                                    va="bottom", fontsize=6.5, color=INK2)
            if metric == "image_loss":
                ax.axhline(alpha, color=CRITICAL, lw=1.0, zorder=2)
                ax.text(len(order) - 0.45, alpha, f"α = {alpha}", color=INK, fontsize=8, va="bottom", ha="right")
                ax.set_ylim(0, max(0.55, alpha + 0.1))
            else:
                ax.set_yscale("log")
                ax.set_ylim(0.1, 150)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([BIN_LABEL[b] for b in order])
            ax.axvline(len(order) - 1.5, color=AXIS, lw=0.8)
            ax.tick_params(axis="x", length=0)
            ax.grid(False, axis="x")
            ax.set_ylabel(ylab)
            ax.set_title(f"α = {alpha}: {'miss fraction (controlled quantity)' if metric == 'image_loss' else 'false-positive cost'}")
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker=MARKERS[p], color=SERIES[k], mfc=SERIES[k], mec=SURFACE, ms=7, ls="none",
                      label=POLICY_LABEL[p]) for k, p in enumerate(policies)]
    handles.append(Line2D([], [], marker="o", color=SERIES[0], mfc=SURFACE, mec=SERIES[0], ms=7, ls="none",
                          label="hollow: no feasible threshold in ≥ half the seeds (τ̂ set to the 0.001 floor)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.955), fontsize=8)
    n_seeds = int(s.n_seeds.max())
    fig.suptitle(f"Conformal thresholds on DAWN (YOLOv8s): held-out test halves, mean ± 1 sd over {n_seeds} "
                 f"stratified splits{title_note}",
                 x=0.02, ha="left", fontsize=10.5, color=INK, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, cfg, out_name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke)
    style()
    for f in (fig1, fig2, fig3):
        for p in f(cfg):
            print(p)


if __name__ == "__main__":
    main()
