"""EXPLORATORY: what lowering the threshold buys, and at what cost, per condition.
STATUS: background for the dial; the method is the dial, wad/dial.py.

User's hypothesis (2026-10-12): night and low-visibility images have more room to lower the
threshold than daytime ones. Tested on the run 2 primary (YOLOv8x @1280, grouped matching; DAWN
clean labels), in-sample (descriptive, nothing is fitted or calibrated):

  bands      for each step down in threshold (0.5 -> 0.25 -> 0.1 -> 0.05 -> 0.02 -> 0.01 -> 0.001):
             objects recovered and false positives admitted per image, the precision of what is
             newly admitted ("marginal precision") and false positives paid per recovered object
  optimum    per condition, the threshold minimising lam * misses + false positives for the same
             dial lam (1, 3, 10, 30, 100), with recall, precision, FPPI and mean image miss there
  scores     the detector's confidence on what it finds: quartiles of the matched score of each
             ground-truth object, the share found only below 0.25 and the share never found
95% intervals: bootstrap over near-duplicate groups within the condition (500 draws, seed 0).

Run: python -m wad.threshold_budget   -> results/threshold/budget_{bands,optimum,scores}.csv, fig14
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wad.config import tau_grid
from wad.curves import ALL, Cache
from wad.config import THRESH_RESULTS as RESULTS

EDGES = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.001]
LAMS = [1, 3, 10, 30, 100]
N_BOOT = 500
ORDER = ["clear", "cloudy", "rain", "snow", "clear_night", "rain_night", "snow_night", "day (all weather)",
         "night (all weather)", "fog", "fog: foggy", "fog: haze", "fog: mist", "rain (DAWN)", "sand", "snow (DAWN)"]


def configs():
    from wad.run2 import bdd_cfg, dawn_cfg, load_run2

    r2 = load_run2()
    return [("BDD100K", bdd_cfg(r2, "v8x_1280", "group")), ("DAWN", dawn_cfg(r2, "v8x_1280", "group", "clean"))]


def conditions(ds, c: Cache) -> dict:
    w = c.weather
    if ds == "BDD100K":
        g = {b: w == b for b in ["clear", "cloudy", "rain", "snow", "clear_night", "rain_night", "snow_night"]}
        night = pd.Series(w).str.endswith("night").to_numpy()
        g["day (all weather)"], g["night (all weather)"] = ~night, night
        return g
    sub = pd.Series(c.image_ids).str.split("/").str[1].str.extract(r"^([a-z]+)")[0].to_numpy()
    g = {"fog": w == "fog", "rain (DAWN)": w == "rain", "sand": w == "sand", "snow (DAWN)": w == "snow"}
    for s in ("foggy", "haze", "mist"):
        g[f"fog: {s}"] = (w == "fog") & (sub == s)
    return g


def per_image(c: Cache, taus: np.ndarray):
    """Per image at each edge threshold: matched objects, false positives; plus n_gt and loss grid."""
    e = np.array([taus[np.argmin(np.abs(taus - t))] for t in EDGES])  # descending
    asc = e[::-1]  # the counting helpers need ascending thresholds
    return e, c.matched_grid(asc, ALL)[:, ::-1], c.fp_grid(asc, ALL)[:, ::-1], c.n_gt(ALL)


def band_stats(M, F, n_gt, rows):
    m, f, n = M[rows].sum(0), F[rows].sum(0), n_gt[rows].sum()
    n_img = rows.sum() if rows.dtype == bool else len(rows)
    dtp, dfp = np.diff(m), np.diff(f)
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(recall=m / n, fppi=f / n_img, d_recall=dtp / n, d_fppi=dfp / n_img,
                    marginal_precision=dtp / (dtp + dfp), fp_per_recovered=dfp / dtp)


def boot_rows(groups: np.ndarray, rng) -> np.ndarray:
    g = np.unique(groups)
    pick = rng.choice(g, len(g), replace=True)
    by = pd.Series(np.arange(len(groups))).groupby(groups).apply(np.array)
    return np.concatenate([by[x] for x in pick])


def optimum(L, F, M, n_gt, rows, taus):
    out = []
    SL, SF, SM = L[rows].sum(0), F[rows].sum(0), M[rows].sum(0)
    n_img = len(rows) if rows.dtype != bool else rows.sum()
    for lam in LAMS:
        miss = n_gt[rows].sum() - SM
        cost = lam * miss + SF
        j = len(cost) - 1 - int(np.argmin(cost[::-1]))
        out.append(dict(lam=lam, tau=float(taus[j]), recall=SM[j] / n_gt[rows].sum(), fppi=SF[j] / n_img,
                        precision=SM[j] / max(SM[j] + SF[j], 1), image_miss=SL[j] / n_img))
    return out


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    bands, opt, scores = [], [], []
    for ds, v in configs():
        c = Cache(v)
        taus = tau_grid(v)
        e, Me, Fe, n_gt = per_image(c, taus)
        L, Fg, Mg = c.loss_grid(taus, ALL), c.fp_grid(taus, ALL), c.matched_grid(taus, ALL)
        groups = c.images.group.to_numpy()
        gt_score = c.gt_score  # best matched score per GT object (-inf if never matched)
        for name, m in conditions(ds, c).items():
            idx = np.nonzero(m)[0]
            st = band_stats(Me, Fe, n_gt, idx)
            rng = np.random.default_rng(0)
            boots = [band_stats(Me, Fe, n_gt, idx[boot_rows(groups[idx], rng)]) for _ in range(N_BOOT)]
            for k in range(len(EDGES) - 1):
                mp = np.array([b["marginal_precision"][k] for b in boots])
                fr = np.array([b["fp_per_recovered"][k] for b in boots])
                bands.append(dict(dataset=ds, condition=name, images=len(idx), band=f"{EDGES[k]:g} -> {EDGES[k + 1]:g}",
                                  recall_before=st["recall"][k], recall_after=st["recall"][k + 1],
                                  recall_gain_pts=100 * st["d_recall"][k], fppi_added=st["d_fppi"][k],
                                  marginal_precision=st["marginal_precision"][k],
                                  marginal_precision_lo=np.nanquantile(mp, 0.025), marginal_precision_hi=np.nanquantile(mp, 0.975),
                                  fp_per_recovered=st["fp_per_recovered"][k],
                                  fp_per_recovered_lo=np.nanquantile(fr, 0.025), fp_per_recovered_hi=np.nanquantile(fr, 0.975)))
            base = optimum(L, Fg, Mg, n_gt, idx, taus)
            rng = np.random.default_rng(0)
            bt = [optimum(L, Fg, Mg, n_gt, idx[boot_rows(groups[idx], rng)], taus) for _ in range(N_BOOT)]
            for i, r in enumerate(base):
                ts = np.array([b[i]["tau"] for b in bt])
                opt.append(dict(dataset=ds, condition=name, images=len(idx), **r,
                                tau_lo=np.quantile(ts, 0.025), tau_hi=np.quantile(ts, 0.975)))
            gi = np.isin(c.gt_img, idx)
            s = gt_score[gi]
            found = s[np.isfinite(s)]
            scores.append(dict(dataset=ds, condition=name, objects=int(gi.sum()),
                               matched_score_q25=np.quantile(found, 0.25), matched_score_median=np.median(found),
                               matched_score_q75=np.quantile(found, 0.75), share_found_at_025=float((s >= 0.25).mean()),
                               share_only_below_025=float(((s < 0.25) & (s >= 0.001)).mean()),
                               share_never=float((~np.isfinite(s) | (s < 0.001)).mean()),
                               fp_per_image_above_025=float(c.fp_grid(np.array([0.25]), ALL)[idx].mean()),
                               fp_per_image_005_to_025=float((c.fp_grid(np.array([0.05]), ALL) - c.fp_grid(np.array([0.25]), ALL))[idx].mean())))
    B, O, S = pd.DataFrame(bands), pd.DataFrame(opt), pd.DataFrame(scores)
    B.to_csv(RESULTS / "budget_bands.csv", index=False, float_format="%.6g")
    O.to_csv(RESULTS / "budget_optimum.csv", index=False, float_format="%.6g")
    S.to_csv(RESULTS / "budget_scores.csv", index=False, float_format="%.6g")
    figure(B, O)
    return B, O, S


def figure(B: pd.DataFrame, O: pd.DataFrame):
    import matplotlib.pyplot as plt

    from wad.config import Cfg
    from wad.figures import INK2, ORDINAL_BLUE, SERIES, save, style

    style()
    show = [("BDD100K", "day (all weather)", ORDINAL_BLUE[1], "-", "BDD100K day"),
            ("BDD100K", "clear_night", "#0b0b0b", "-", "BDD100K clear night"),
            ("BDD100K", "rain_night", SERIES[1], "-", "BDD100K rain night"),
            ("DAWN", "fog", SERIES[2], "--", "DAWN fog"), ("DAWN", "rain (DAWN)", SERIES[3], "--", "DAWN rain")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    bands = list(pd.unique(B.band))
    x = np.arange(len(bands))
    wbar = 0.8 / len(show)
    for i, (ds, cond, col, ls, lab) in enumerate(show):
        d = B[(B.dataset == ds) & (B.condition == cond)].set_index("band").loc[bands]
        axes[0].bar(x - 0.4 + (i + 0.5) * wbar, d.recall_gain_pts, wbar, color=col, label=lab)
        axes[1].plot(x, d.fp_per_recovered, color=col, ls=ls, marker="o", ms=4, lw=1.6, label=lab)
        axes[1].fill_between(x, d.fp_per_recovered_lo, d.fp_per_recovered_hi, color=col, alpha=0.12, lw=0)
    for ax in axes[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=30, fontsize=7.5)
        ax.set_xlabel("threshold lowered from -> to")
    axes[0].set_ylabel("recall gained (percentage points)")
    axes[0].set_title("What lowering the threshold recovers")
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[0].grid(False, axis="x")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("false positives per recovered object")
    axes[1].set_title("What each recovered object costs")
    ax = axes[2]
    for ds, cond, col, ls, lab in show:
        d = O[(O.dataset == ds) & (O.condition == cond)]
        ax.plot(d.lam, d.tau, color=col, ls=ls, marker="o", ms=4, lw=1.6, label=lab)
    ax.axhline(0.25, color=INK2, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dial λ: false positives accepted per avoided miss")
    ax.set_ylabel("best threshold for the condition")
    ax.set_title("Best threshold per condition, same dial")
    fig.tight_layout()
    return save(fig, Cfg(dict(paths=Cfg(dict(results=RESULTS)))), "fig14_threshold_budget")


if __name__ == "__main__":
    B, O, S = main()
    pd.set_option("display.width", 250)
    print(S.round(3).to_string(index=False))
    print(O[O.lam == 10].round(3).to_string(index=False))
    print(B[B.band.isin(["0.25 -> 0.1", "0.1 -> 0.05", "0.05 -> 0.02"])][["dataset", "condition", "band", "recall_gain_pts", "fppi_added",
                                                                       "marginal_precision", "fp_per_recovered"]].round(3).to_string(index=False))
