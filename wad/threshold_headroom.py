"""EXPLORATORY: how much could a per-image threshold predictor gain, at best?
STATUS: background for the threshold exploration; the method is the dial, wad/dial.py.

For the run 2 primary detector (YOLOv8x @1280, grouped vehicles; DAWN clean labels), trace the
best achievable trade-off between the mean image-level miss fraction (the CRC loss) and false
positives per image (FPPI), in-sample, for three ways of setting thresholds:

  global        one threshold for every image
  per-weather   one threshold per weather (x time of day) label, chosen with the labels known
  per-image     the best threshold for each image, chosen with its ground truth known (oracle)

Each frontier is traced with a Lagrangian weight lam: every unit (image, bin, or all images)
takes the grid threshold minimising lam * misses + FP. No calibration/test split: these are upper
bounds on what a threshold policy of each kind could reach, not achievable results. A learned
threshold predictor sits somewhere between per-weather and the per-image oracle.

Run: python -m wad.threshold_headroom   -> results/threshold/headroom.csv, frontier.csv, fig12
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wad.config import REPO, tau_grid
from wad.curves import ALL, Cache

OUT = REPO / "results" / "threshold"
TARGETS = (0.1, 0.2, 0.3)
LAMS = np.geomspace(1e-3, 1e4, 600)


def frontier(L: np.ndarray, F: np.ndarray, M: np.ndarray, units: np.ndarray, n_gt: np.ndarray) -> pd.DataFrame:
    """L, F, M: (n_img, n_tau) image miss fraction, FP count, matched count; units: group id per image.
    For each lam, each unit takes the grid index minimising lam * sum(L) + sum(F) (largest tau on ties)."""
    u, inv = np.unique(units, return_inverse=True)
    SL = np.zeros((len(u), L.shape[1]))
    SF = np.zeros_like(SL)
    np.add.at(SL, inv, L)
    np.add.at(SF, inv, F)
    rows = []
    n = L.shape[0]
    for lam in LAMS:
        cost = lam * SL + SF
        j = cost.shape[1] - 1 - np.argmin(cost[:, ::-1], axis=1)  # largest tau among ties
        ji = j[inv]
        r = np.arange(n)
        miss, fp, tp = L[r, ji], F[r, ji], M[r, ji]
        rows.append(dict(lam=lam, image_miss=miss.mean(), fppi=fp.sum() / n, recall=tp.sum() / n_gt.sum(),
                         precision=tp.sum() / max(tp.sum() + fp.sum(), 1)))
    return pd.DataFrame(rows)


def at_target(fr: pd.DataFrame, target: float) -> dict:
    ok = fr[fr.image_miss <= target + 1e-12]
    if ok.empty:
        return dict(fppi=np.nan, recall=np.nan, precision=np.nan, image_miss=np.nan)
    b = ok.loc[ok.fppi.idxmin()]
    return dict(fppi=b.fppi, recall=b.recall, precision=b.precision, image_miss=b.image_miss)


def main():
    from wad.run2 import bdd_cfg, dawn_cfg, load_run2

    OUT.mkdir(parents=True, exist_ok=True)
    r2 = load_run2()
    fronts, rows = [], []
    for ds, v in [("BDD100K", bdd_cfg(r2, "v8x_1280", "group")), ("DAWN", dawn_cfg(r2, "v8x_1280", "group", "clean"))]:
        c = Cache(v)
        taus = tau_grid(v)
        L, F, M, n_gt = c.loss_grid(taus, ALL), c.fp_grid(taus, ALL), c.matched_grid(taus, ALL), c.n_gt(ALL)
        subsets = {"all": np.ones(c.n_img, bool)}
        if ds == "BDD100K":
            night = pd.Series(c.weather).str.endswith("night").to_numpy()
            subsets.update({"day": ~night, "night": night})
        for sub, m in subsets.items():
            for pol, units in (("global", np.zeros(c.n_img, int)), ("per-weather", c.weather), ("per-image (oracle)", np.arange(c.n_img))):
                fr = frontier(L[m], F[m], M[m], units[m], n_gt[m]).assign(dataset=ds, images=sub, policy=pol)
                fronts.append(fr)
                for t in TARGETS:
                    rows.append(dict(dataset=ds, images=sub, policy=pol, target_miss=t, n_images=int(m.sum()), **at_target(fr, t)))
    pd.concat(fronts).to_csv(OUT / "frontier.csv", index=False, float_format="%.6g")
    h = pd.DataFrame(rows)
    h.to_csv(OUT / "headroom.csv", index=False, float_format="%.6g")
    figure(pd.concat(fronts))
    return h


def figure(fr: pd.DataFrame):
    import matplotlib.pyplot as plt

    from wad.config import Cfg
    from wad.figures import INK2, SERIES, save, style

    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ds in zip(axes, ("BDD100K", "DAWN")):
        for pol, col in zip(("global", "per-weather", "per-image (oracle)"), SERIES):
            d = fr[(fr.dataset == ds) & (fr.images == "all") & (fr.policy == pol)].sort_values("image_miss")
            ax.plot(d.image_miss, d.fppi, color=col, lw=1.8, label=pol)
        ax.axvline(0.2, color=INK2, lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_xlim(0, 0.6)
        ax.set_ylim(0.01, 300)
        ax.set_xlabel("mean image miss fraction (CRC loss)")
        ax.set_ylabel("false positives per image")
        ax.set_title(f"{ds}: best achievable trade-off (in-sample)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return save(fig, Cfg(dict(paths=Cfg(dict(results=OUT)))), "fig12_threshold_headroom")


if __name__ == "__main__":
    h = main()
    pd.set_option("display.width", 200)
    print(h.round(3).to_string(index=False))
