"""Evaluation of the learned per-image threshold with a cost dial (PREREG_threshold.md).
STATUS (2026-10-13): did not beat one threshold; kept so its NOTES.md numbers reproduce. The
method is the dial (wad/dial.py): one threshold or per detection.

On BDD100K val and DAWN (never used in training), with the project's 200 calibration/test halves:
  learned, calibrated     CRC picks the dial lam on the calibration half so the mean image miss is
                          <= alpha (one lam for all images; and one lam per weather bin)
  learned, direct preset  lam = 1, 3, 10, 30 false positives per avoided miss (no calibration)
  comparators             one global CRC threshold, per-weather CRC thresholds, v1 quartiles
Reported on test halves, overall and per condition. Criteria P1-P3 in criteria.csv.

Run: python -m wad.thresh_eval   -> results/threshold/{policies,criteria,quality,frontier_learned}.csv,
     threshold.md, figures/fig13_learned_threshold.{png,pdf}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from wad.config import tau_grid
from wad.crc import fit_threshold, make_splits
from wad.curves import ALL, Cache
from wad.thresh_net import CACHE, K_STEP, RESULTS

ALPHAS = (0.1, 0.2, 0.3)
PRESETS = {"precision-first (lam 1)": 1.0, "lam 3": 3.0, "balanced (lam 10)": 10.0, "recall-first (lam 30)": 30.0}
LAMS = np.geomspace(0.01, 1e4, 200)
N_SPLITS = 200


def configs():
    from wad.run2 import bdd_cfg, dawn_cfg, load_run2

    r2 = load_run2()
    return [("BDD100K", "bdd", bdd_cfg(r2, "v8x_1280", "group")), ("DAWN", "dawn", dawn_cfg(r2, "v8x_1280", "group", "clean"))]


def dial_index(pm: np.ndarray, pf: np.ndarray, lams: np.ndarray) -> np.ndarray:
    """(n_img, n_lam) full-grid threshold index chosen by the dial: per image and lam, the largest k
    minimising lam * miss_hat_k + fp_hat_k, mapped from the 40-point sub-grid to the 200-point grid."""
    K = pm.shape[1]
    out = np.empty((pm.shape[0], len(lams)), int)
    for j, lam in enumerate(lams):
        cost = lam * pm + pf
        out[:, j] = (K - 1 - np.argmin(cost[:, ::-1], axis=1)) * K_STEP
    return out


def crc_lam(L: np.ndarray, alpha: float) -> tuple[int, bool]:
    """Smallest lam index whose conformal bound on the calibration losses is <= alpha.
    L: (n, n_lam) image miss fractions, non-increasing along lam. Infeasible -> largest lam."""
    n = L.shape[0]
    if n == 0:
        return L.shape[1] - 1, True
    ok = np.nonzero(L.sum(0) + 1 <= alpha * (n + 1) + 1e-9)[0]
    return (int(ok.min()), False) if len(ok) else (L.shape[1] - 1, True)


def groups(ds, c: Cache) -> dict:
    w = c.weather
    g = {b: w == b for b in pd.unique(w)}
    if ds == "DAWN":
        sub = pd.Series(c.image_ids).str.split("/").str[1].str.extract(r"^([a-z]+)")[0].to_numpy()
        for s in ("foggy", "haze", "mist"):
            g[f"fog: {s}"] = (w == "fog") & (sub == s)
    else:
        night = pd.Series(w).str.endswith("night").to_numpy()
        g["day (all weather)"], g["night (all weather)"] = ~night, night
    g["all"] = np.ones(c.n_img, bool)
    return g


def prf(tp, fp, n_gt):
    r = tp / n_gt if n_gt else np.nan
    p = tp / (tp + fp) if tp + fp else np.nan
    return r, p, (2 * p * r / (p + r) if (r == r and p == p and p + r) else np.nan)


def evaluate_dataset(ds, key, v):
    from wad.visibility import load_h

    c = Cache(v)
    taus = tau_grid(v)
    floor = v.detector.conf_floor
    pr = np.load(CACHE / f"pred_{key}.npz", allow_pickle=True)
    pos = pd.Series(np.arange(len(pr["image_id"])), index=pr["image_id"]).loc[c.image_ids].to_numpy()
    pm, pf = pr["miss"][pos], pr["fp"][pos]
    assert np.allclose(pr["taus"], taus[::K_STEP])
    idx = dial_index(pm, pf, LAMS)
    assert (np.diff(idx, axis=1) <= 0).all(), "the dial must lower thresholds as lam grows"
    loss = c.loss_grid(taus, ALL)
    Ll = np.take_along_axis(loss, idx, axis=1)  # true image miss fraction per lam, non-increasing
    h1 = load_h().loc[c.image_ids].to_numpy()
    w = c.weather
    gs = groups(ds, c)
    v.splits["n_repeats"] = N_SPLITS
    splits = make_splits(v, c.images)
    pre_idx = dial_index(pm, pf, np.array(list(PRESETS.values())))
    acc, lam_rows = [], []
    for seed, sp in splits.groupby("seed"):
        cal = (sp.set_index("image_id").loc[c.image_ids].half == "cal").to_numpy()
        test = ~cal
        pols = {}
        for name, j in zip(PRESETS, range(pre_idx.shape[1])):
            pols[(name, np.nan)] = taus[pre_idx[:, j]]
        for a in ALPHAS:
            pols[("global", a)] = np.full(c.n_img, fit_threshold(loss[cal], taus, a, floor)["tau_hat"])
            t = np.full(c.n_img, np.nan)
            for b in pd.unique(w):
                t[w == b] = fit_threshold(loss[cal & (w == b)], taus, a, floor)["tau_hat"]
            pols[("per-weather", a)] = t
            q = np.digitize(h1, np.quantile(h1[cal], [0.25, 0.5, 0.75]))
            t = np.full(c.n_img, np.nan)
            for b in range(4):
                t[q == b] = fit_threshold(loss[cal & (q == b)], taus, a, floor)["tau_hat"]
            pols[("visibility v1 (quartiles)", a)] = t
            j, inf = crc_lam(Ll[cal], a)
            pols[("learned dial, calibrated", a)] = taus[idx[:, j]]
            lam_rows.append(dict(dataset=ds, alpha=a, seed=seed, policy="learned dial, calibrated", bin="all",
                                 lam=LAMS[j], infeasible=inf))
            t = np.full(c.n_img, np.nan)
            for b in pd.unique(w):
                m = w == b
                jb, infb = crc_lam(Ll[cal & m], a)
                t[m] = taus[idx[m, jb]]
                lam_rows.append(dict(dataset=ds, alpha=a, seed=seed, policy="learned dial, per-weather calibrated",
                                     bin=b, lam=LAMS[jb], infeasible=infb))
            pols[("learned dial, per-weather calibrated", a)] = t
        for (p, a), ti in pols.items():
            x = c.at_thresholds(ti, ALL)
            li = np.where(x["n_gt"] > 0, x["n_missed"] / np.maximum(x["n_gt"], 1), 0.0)
            for gname, m in gs.items():
                r_ = test & m
                n_gt = float(x["n_gt"][r_].sum())
                tp, fp = float((x["n_gt"] - x["n_missed"])[r_].sum()), float(x["n_fp"][r_].sum())
                r, prc, f1 = prf(tp, fp, n_gt)
                acc.append(dict(dataset=ds, group=gname, policy=p, alpha=a, seed=seed, recall=r, precision=prc, f1=f1,
                                fppi=fp / max(r_.sum(), 1), image_miss=li[r_].mean() if r_.any() else np.nan,
                                n_test=int(r_.sum())))
    a = pd.DataFrame(acc)
    rows = []
    for (g, p, al), d in a.groupby(["group", "policy", "alpha"], sort=False, dropna=False):
        row = dict(dataset=ds, group=g, policy=p, alpha=al, splits=d.seed.nunique(), n_test=d.n_test.mean(),
                   above_alpha=float((d.image_miss > al).mean()) if al == al else np.nan)
        for k in ("recall", "precision", "f1", "fppi", "image_miss"):
            row.update({k: d[k].mean(), f"{k}_lo": d[k].quantile(0.025), f"{k}_hi": d[k].quantile(0.975)})
        rows.append(row)
    # prediction quality and the learned frontier (all images; the model never saw them)
    k25 = int(np.argmin(np.abs(taus[::K_STEP] - 0.25)))
    tm = c.n_gt(ALL)[:, None] - c.matched_grid(taus[::K_STEP], ALL)
    tf = c.fp_grid(taus[::K_STEP], ALL)
    quality = dict(dataset=ds, tau=float(taus[::K_STEP][k25]),
                   spearman_miss=float(spearmanr(pm[:, k25], tm[:, k25])[0]),
                   spearman_fp=float(spearmanr(pf[:, k25], tf[:, k25])[0]),
                   mean_pred_miss=float(pm[:, k25].mean()), mean_true_miss=float(tm[:, k25].mean()),
                   mean_pred_fp=float(pf[:, k25].mean()), mean_true_fp=float(tf[:, k25].mean()))
    F = c.fp_grid(taus, ALL)
    M = c.matched_grid(taus, ALL)
    n_gt = c.n_gt(ALL)
    front = []
    for j, lam in enumerate(LAMS):
        ii = idx[:, j]
        r = np.arange(c.n_img)
        front.append(dict(dataset=ds, lam=lam, image_miss=loss[r, ii].mean(), fppi=F[r, ii].sum() / c.n_img,
                          recall=M[r, ii].sum() / n_gt.sum()))
    return pd.DataFrame(rows), pd.DataFrame(lam_rows), quality, pd.DataFrame(front)


def criteria(pol: pd.DataFrame, lam: pd.DataFrame) -> pd.DataFrame:
    def get(ds, g, p, a, col):
        d = pol[(pol.dataset == ds) & (pol.group == g) & (pol.policy == p) & (pol.alpha == a)]
        return float(d[col].iloc[0])

    out = []
    for crit, ds in (("P1", "BDD100K"), ("P2", "DAWN")):
        fl, fg = get(ds, "all", "learned dial, calibrated", 0.2, "fppi"), get(ds, "all", "global", 0.2, "fppi")
        ml = get(ds, "all", "learned dial, calibrated", 0.2, "image_miss")
        out.append(dict(criterion=f"{crit} {ds}, alpha 0.2", value=f"learned {fl:.2f} vs global {fg:.2f} FPPI "
                        f"({100 * (1 - fl / fg):.0f}% fewer), miss {ml:.3f}", met=bool(fl <= 0.9 * fg and ml <= 0.21)))
    m = get("BDD100K", "rain_night", "learned dial, per-weather calibrated", 0.2, "image_miss")
    f2 = get("BDD100K", "rain_night", "learned dial, per-weather calibrated", 0.2, "fppi")
    fw = get("BDD100K", "rain_night", "per-weather", 0.2, "fppi")
    out.append(dict(criterion="P3 BDD100K rain_night, per-weather calibrated dial, alpha 0.2",
                    value=f"miss {m:.3f}; {f2:.1f} vs per-weather thresholds {fw:.1f} FPPI", met=bool(m <= 0.21 and f2 < fw)))
    return pd.DataFrame(out)


def figure(front: pd.DataFrame):
    import matplotlib.pyplot as plt

    from wad.config import Cfg
    from wad.figures import INK2, SERIES, save, style

    style()
    hr = pd.read_csv(RESULTS / "frontier.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ds in zip(axes, ("BDD100K", "DAWN")):
        for pol, col, lab in (("global", SERIES[0], "one global threshold"), ("per-image (oracle)", SERIES[2], "perfect per-image (oracle)")):
            d = hr[(hr.dataset == ds) & (hr.images == "all") & (hr.policy == pol)].sort_values("image_miss")
            ax.plot(d.image_miss, d.fppi, color=col, lw=1.6, label=lab, ls="--" if "oracle" in pol else "-")
        d = front[front.dataset == ds].sort_values("image_miss")
        ax.plot(d.image_miss, d.fppi, color=SERIES[1], lw=2.2, label="learned dial (sweep of lam)")
        for name, lam in (("1", 1.0), ("10", 10.0), ("30", 30.0)):
            j = int(np.argmin(np.abs(np.log(d.lam.to_numpy()) - np.log(lam))))
            ax.plot(d.image_miss.iloc[j], d.fppi.iloc[j], "o", color=SERIES[1], ms=5)
            ax.annotate(f"lam {name}", (d.image_miss.iloc[j], d.fppi.iloc[j]), textcoords="offset points", xytext=(5, 4),
                        fontsize=7.5, color=INK2)
        ax.axvline(0.2, color=INK2, lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_xlim(0, 0.6)
        ax.set_ylim(0.01, 300)
        ax.set_xlabel("mean image miss fraction")
        ax.set_ylabel("false positives per image")
        ax.set_title(f"{ds}: learned per-image threshold vs one threshold")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return save(fig, Cfg(dict(paths=Cfg(dict(results=RESULTS)))), "fig13_learned_threshold")


def md(df):
    cell = lambda v: str(v).replace("|", "\\|")  # noqa: E731
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(map(cell, cols)) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
                     + ["| " + " | ".join(map(cell, r)) + " |" for r in df.itertuples(index=False)])


def report(pol, crit, quality, lam, diag=None):
    tl = json.loads((RESULTS / "train_log.json").read_text())
    best = min(tl["log"], key=lambda r: r["val_loss"])
    met = crit.met.map(lambda m: "met" if m else "not met")
    lines = ["# Learned per-image threshold with a cost dial", "",
             "Generated by `python -m wad.thresh_eval`. Pre-registration: `PREREG_threshold.md`. The network sees pixels only; "
             "it was trained on BDD100K train frames labelled with YOLOv8x @1280 (grouped matching) misses and false positives "
             "at 40 thresholds. Evaluation: BDD100K val and DAWN (never used in training), 200 calibration/test halves.", "",
             f"Training: {tl['n_train']} train / {tl['n_val']} validation frames, best epoch {best['epoch']}.", "",
             "## Pre-registered criteria", "", md(crit.assign(met=met)), "",
             "## Prediction quality at τ ≈ 0.25 (all evaluation images)", "", md(pd.DataFrame(quality).round(3)), "",
             *(["## Why it behaves like one threshold (α = 0.2, median calibrated λ)", "", md(diag.round(3)), ""]
               if diag is not None else []),
             "## Calibrated dial values (median λ over splits, share of splits infeasible)", "",
             md(lam.groupby(["dataset", "policy", "bin", "alpha"]).agg(lam_median=("lam", "median"),
                                                                        infeasible=("infeasible", "mean")).reset_index().round(3)), "",
             "## Policies on held-out halves", "",
             "Cells: recall / precision / F1, FPPI, mean image miss (share of splits above α). Direct presets have no α.", ""]
    for ds in ("BDD100K", "DAWN"):
        for a in list(ALPHAS) + [np.nan]:
            d = pol[(pol.dataset == ds) & ((pol.alpha == a) if a == a else pol.alpha.isna())]
            rows = []
            for g in pd.unique(d.group):
                row = {"group": g}
                for p in pd.unique(d.policy):
                    r = d[(d.group == g) & (d.policy == p)].iloc[0]
                    tail = f" ({r.above_alpha:.2f})" if a == a else ""
                    row[p] = f"{100 * r.recall:.0f}/{100 * r.precision:.0f}/{r.f1:.2f}, {r.fppi:.1f}, {r.image_miss:.3f}{tail}"
                rows.append(row)
            title = f"α = {a}" if a == a else "direct presets (no calibration)"
            lines += [f"### {ds}, {title}", "", md(pd.DataFrame(rows)), ""]
    (RESULTS / "threshold.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnose() -> pd.DataFrame:
    """Why the dial behaves like one threshold: at the median calibrated lam (alpha 0.2), how many
    distinct thresholds the learned dial gives, how they spread, and how they rank against the
    per-image oracle threshold at the same lam (true curves, ground truth known)."""
    lam = pd.read_csv(RESULTS / "dial_lams.csv")
    rows = []
    for ds, key, v in configs():
        c = Cache(v)
        taus = tau_grid(v)
        pr = np.load(CACHE / f"pred_{key}.npz", allow_pickle=True)
        pos = pd.Series(np.arange(len(pr["image_id"])), index=pr["image_id"]).loc[c.image_ids].to_numpy()
        lm = float(lam[(lam.dataset == ds) & (lam.alpha == 0.2) & (lam.bin == "all")].lam.median())
        t = taus[dial_index(pr["miss"][pos], pr["fp"][pos], np.array([lm]))[:, 0]]
        miss = c.n_gt(ALL)[:, None] - c.matched_grid(taus, ALL)
        cost = lm * miss + c.fp_grid(taus, ALL)
        to = taus[cost.shape[1] - 1 - np.argmin(cost[:, ::-1], axis=1)]
        q = lambda x: np.quantile(x, [0.05, 0.5, 0.95]).round(3).tolist()  # noqa: E731
        rows.append(dict(dataset=ds, lam=lm, distinct_learned_thresholds=int(len(np.unique(t))),
                         learned_tau_q05_q50_q95=q(t), oracle_tau_q05_q50_q95=q(to),
                         spearman_learned_vs_oracle_tau=float(spearmanr(t, to)[0])))
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "diagnosis.csv", index=False, float_format="%.4g")
    return d


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    pols, lams, qual, fronts = [], [], [], []
    for ds, key, v in configs():
        p, lm, q, f = evaluate_dataset(ds, key, v)
        pols.append(p)
        lams.append(lm)
        qual.append(q)
        fronts.append(f)
    pol, lam, front = pd.concat(pols), pd.concat(lams), pd.concat(fronts)
    pol.to_csv(RESULTS / "policies.csv", index=False, float_format="%.6g")
    lam.to_csv(RESULTS / "dial_lams.csv", index=False, float_format="%.6g")
    front.to_csv(RESULTS / "frontier_learned.csv", index=False, float_format="%.6g")
    pd.DataFrame(qual).to_csv(RESULTS / "quality.csv", index=False, float_format="%.6g")
    crit = criteria(pol, lam)
    crit.to_csv(RESULTS / "criteria.csv", index=False)
    figure(front)
    diag = diagnose()
    report(pol, crit, qual, lam, diag)
    print(crit.to_string(index=False))
    print(pd.DataFrame(qual).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
