"""THE DIAL'S ONE-THRESHOLD OPTION (decided 2026-10-13; the per-detection option is wad/per_detection.py,
both wrapped by wad/dial.py): one constant confidence threshold per dial setting, the same in every
frame and condition.

The dial lam is "false alarms accepted to avoid one miss". For each lam the threshold minimises
lam * misses + false alarms over the BDD100K train detections (21,831 images, YOLOv8x @1280, boxes
>= 0.001, grouped matching; greedy matching is threshold-consistent, so the objects found at a
threshold are the matched boxes kept). Nothing is chosen on val or DAWN: they only grade it, against
no tuning (Ultralytics' default confidence 0.25), against one threshold per weather (each fitted on
the train images of that weather) and against the condition's own in-sample best constant.

Outputs (results/threshold/):
  one_threshold_dial.csv     lam -> threshold, a 200-resample image bootstrap band, and the threshold
                             fitted on each train weather alone
  one_threshold_folds.csv    5-fold check inside train: cost on the held-out fold of the threshold
                             fitted on the other four, relative to that fold's own best
  one_threshold_eval.csv     BDD100K val per weather and group, DAWN per weather: recall, precision,
                             false alarms per image and cost for each policy at the presets
  figures/fig20_one_threshold.png
  weights/one_threshold_table.csv   the exact lam -> threshold switch points (used by wad/dial.py)
Run: python -m wad.one_threshold
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wad.config import REPO, THRESH_CACHE
from wad.config import THRESH_RESULTS as RESULTS

DIAL = np.round(np.geomspace(0.3, 300, 61), 4)  # the dial page's settings
PRESETS = (1.0, 3.0, 10.0, 30.0, 100.0)
DEFAULT = 0.25  # Ultralytics' default prediction confidence: "no tuning"
GRID = np.geomspace(1e-3, 1.0, 2001)  # candidate thresholds for the bootstrap and folds (0.35% steps)
TRAIN = THRESH_CACHE / "v8x_1280__group" / "matches_pred.parquet"
BDD_GROUPS = {"day": ["clear", "cloudy", "rain", "snow"], "night": ["clear_night", "rain_night", "snow_night"]}
DAWN_TO_TRAIN = {"fog": "fog", "rain": "rain", "snow": "snow"}  # sand has no BDD100K counterpart


# ---- fitting ---------------------------------------------------------------------------------------

def train_curve(m: pd.DataFrame | None = None):
    """From train detections: score -> chance real (isotonic), and the cumulative real / false counts
    over descending score at every rule 'keep score >= t' (one point per distinct score)."""
    from sklearn.isotonic import IsotonicRegression

    if m is None:
        m = pd.read_parquet(TRAIN, columns=["score", "tp"])
    s, y = m.score.to_numpy(), m.tp.to_numpy().astype(float)
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(s, y)
    o = np.argsort(-s, kind="stable")
    ss = s[o]
    last = np.r_[ss[1:] != ss[:-1], True]  # 'score >= t' cannot split detections with the same score: block ends only
    return iso, (ss[last], np.cumsum(y[o])[last], np.cumsum(1 - y[o])[last])


def train_threshold(curve, lam: float) -> float:
    """The one threshold for dial lam: minimises lam * misses + false alarms over the detections
    (the number of objects is fixed, so this is argmin false alarms - lam * real kept)."""
    s, ctp, cfp = curve
    return float(s[int(np.argmin(cfp - lam * ctp))])


def breakpoints(curve) -> pd.DataFrame:
    """The exact dial table. The one threshold is piecewise constant in lam: argmin false - lam * real
    over descending score lies on the lower convex hull of (real kept, false kept), and moves to the
    next hull corner when lam passes that edge's slope. Row k holds for lam_from[k] < lam <= lam_from[k+1]
    (ties go to the higher threshold, as in train_threshold)."""
    s, ctp, cfp = curve
    # a rule whose last block of equal scores added no real box is never better than the one before it
    # (for lam > 0), so only rules whose block added a real box can be hull corners (blocks can be mixed)
    cand = np.flatnonzero(np.diff(np.r_[0.0, ctp]) > 0)
    x, y = ctp[cand], cfp[cand]
    hull = []
    for i in range(len(cand)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (x[b] - x[a]) * (y[i] - y[a]) - (y[b] - y[a]) * (x[i] - x[a]) <= 0:  # b not strictly below a -> i
                hull.pop()
            else:
                break
        hull.append(i)
    h = np.array(hull)
    slopes = np.diff(y[h]) / np.diff(x[h])
    k = int((slopes <= 0).sum())  # corners left of a flat edge (more real boxes, no more false) never win for lam > 0
    h, slopes = h[k:], slopes[k:]
    return pd.DataFrame(dict(lam_from=np.r_[0.0, slopes], threshold=s[cand[h]]))


def threshold_from_table(table: pd.DataFrame, lam: float) -> float:
    i = int(np.searchsorted(table.lam_from.to_numpy(), lam, side="left")) - 1
    return float(table.threshold.iloc[max(i, 0)])


def grid_counts(score: np.ndarray, tp: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Real and false boxes with score >= GRID[k], for every k (optionally weighted)."""
    b = np.clip(np.searchsorted(GRID, score, side="right") - 1, 0, len(GRID) - 1)
    w = np.ones(len(score)) if w is None else w
    r = np.bincount(b, weights=w * tp, minlength=len(GRID))
    f = np.bincount(b, weights=w * (1 - tp), minlength=len(GRID))
    return r[::-1].cumsum()[::-1], f[::-1].cumsum()[::-1]


def grid_thresholds(r: np.ndarray, f: np.ndarray, lams) -> np.ndarray:
    return GRID[np.argmin(f[None, :] - np.asarray(lams, float)[:, None] * r[None, :], axis=1)]


def fit(m: pd.DataFrame, im: pd.DataFrame, B=200, seed=0) -> pd.DataFrame:
    lams = np.union1d(DIAL, PRESETS)
    _, curve = train_curve(m)
    out = pd.DataFrame(dict(lam=lams, threshold=[train_threshold(curve, x) for x in lams]))
    # image bootstrap
    codes, uniq = pd.factorize(m.image_id)
    s, tp = m.score.to_numpy(), m.tp.to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    boot = np.empty((B, len(lams)))
    for i in range(B):
        cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        boot[i] = grid_thresholds(*grid_counts(s, tp, cnt[codes].astype(float)), lams)
    out["boot_q05"], out["boot_q50"], out["boot_q95"] = np.quantile(boot, [0.05, 0.5, 0.95], axis=0)
    # one threshold per train weather
    for w, g in m.groupby("weather"):
        _, cw = train_curve(g)
        out[f"threshold_{w}"] = [train_threshold(cw, x) for x in lams]
    return out


def folds(m: pd.DataFrame, im: pd.DataFrame, k=5, seed=0) -> pd.DataFrame:
    ids = im.image_id.to_numpy()
    fold = dict(zip(ids, np.random.default_rng(seed).permutation(len(ids)) % k))
    f_det = m.image_id.map(fold).to_numpy()
    n_gt = dict(zip(ids, im.n_gt.to_numpy()))
    s, tp = m.score.to_numpy(), m.tp.to_numpy().astype(float)
    rows = []
    for j in range(k):
        r_in, f_in = grid_counts(s[f_det != j], tp[f_det != j])
        r_out, f_out = grid_counts(s[f_det == j], tp[f_det == j])
        g_out = sum(v for i, v in n_gt.items() if fold[i] == j)
        for lam in PRESETS:
            kin = int(np.argmin(f_in - lam * r_in))
            cost = lam * (g_out - r_out) + f_out
            rows.append(dict(fold=j, lam=lam, threshold=GRID[kin], own_best=GRID[int(np.argmin(cost))],
                             cost_vs_own_best=cost[kin] / cost.min()))
    return pd.DataFrame(rows)


# ---- grading on val / DAWN ------------------------------------------------------------------------

def eval_configs() -> dict:
    """The two evaluation sets: BDD100K val (8,090 images in the pre-registered bins) and DAWN (clean
    images), both YOLOv8x @1280 with grouped matching."""
    from wad.run2 import bdd_cfg, dawn_cfg, load_run2

    r2 = load_run2()
    return {"BDD100K": bdd_cfg(r2, "v8x_1280", "group"), "DAWN": dawn_cfg(r2, "v8x_1280", "group", "clean")}


def stats(tp_k: float, fp_k: float, n_gt: float, n_img: int, lam: float) -> dict:
    return dict(recall=tp_k / n_gt, precision=tp_k / (tp_k + fp_k) if tp_k + fp_k else 1.0, fppi=fp_k / n_img,
                cost_per_img=(lam * (n_gt - tp_k) + fp_k) / n_img)


def grade(dial: pd.DataFrame, lams) -> pd.DataFrame:
    from wad.curves import ALL, Cache
    tau = dict(zip(dial.lam, dial.threshold))
    rows = []
    for ds, cfg in eval_configs().items():
        c = Cache(cfg)
        sc, tp = c.pred.score.to_numpy(), c.pred.tp.to_numpy().astype(bool)
        img = c.pred.image_id.map(c.idx).to_numpy()
        w_det, n_gt_img = c.weather[img], c.n_gt(ALL)
        weathers = sorted(set(c.weather))
        groups = {w: [w] for w in weathers}
        if ds == "BDD100K":
            groups.update({k: [w for w in v if w in weathers] for k, v in BDD_GROUPS.items()})
        groups["all"] = weathers
        for g, ws in groups.items():
            rows_i, rows_d = np.isin(c.weather, ws), np.isin(w_det, ws)
            s_g, t_g, wd = sc[rows_d], tp[rows_d], w_det[rows_d]
            n_gt, n_img = float(n_gt_img[rows_i].sum()), int(rows_i.sum())
            o = np.argsort(-s_g, kind="stable")
            ctp, cfp = np.cumsum(t_g[o]), np.cumsum(~t_g[o])
            wcode = pd.Index(ws).get_indexer(wd)
            # the train weather each condition borrows its threshold from (none: use the one threshold)
            cols = [f"threshold_{DAWN_TO_TRAIN.get(x) if ds == 'DAWN' else x}" for x in ws]
            for lam in lams:
                t, drow = tau[lam], dial[dial.lam == lam].iloc[0]
                thr_det = np.array([drow[col] if col in drow.index else t for col in cols])[wcode]
                j = int(np.argmin(cfp - lam * ctp))
                for policy, thr, kept in (("dial threshold (train)", t, s_g >= t),
                                          ("no tuning (0.25)", DEFAULT, s_g >= DEFAULT),
                                          ("per-weather threshold (train)", np.nan, s_g >= thr_det)):
                    rows.append(dict(dataset=ds, group=g, lam=lam, policy=policy, threshold=thr, images=n_img,
                                     **stats(float(t_g[kept].sum()), float((~t_g[kept]).sum()), n_gt, n_img, lam)))
                rows.append(dict(dataset=ds, group=g, lam=lam, policy="own best constant (in-sample)",
                                 threshold=float(s_g[o][j]), images=n_img, **stats(float(ctp[j]), float(cfp[j]), n_gt, n_img, lam)))
    out = pd.DataFrame(rows)
    ref = out[out.policy == "dial threshold (train)"].set_index(["dataset", "group", "lam"]).cost_per_img
    out["cost_vs_dial"] = out.cost_per_img.to_numpy() / ref.loc[list(zip(out.dataset, out.group, out.lam))].to_numpy()
    return out


def figure(dial: pd.DataFrame, ev: pd.DataFrame):
    import matplotlib.pyplot as plt

    from wad.config import Cfg
    from wad.figures import INK, INK2, ORDINAL_BLUE, SERIES, save, style

    style()
    show = [("BDD100K", "day", ORDINAL_BLUE[1], "-", "BDD100K day"), ("BDD100K", "night", INK, "-", "BDD100K night"),
            ("BDD100K", "rain_night", SERIES[1], "-", "BDD100K rain night"), ("DAWN", "fog", SERIES[2], "--", "DAWN fog"),
            ("DAWN", "all", SERIES[3], "--", "DAWN all")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    d = dial[dial.lam.isin(DIAL)]
    ax.fill_between(d.lam, d.boot_q05, d.boot_q95, color=ORDINAL_BLUE[2], alpha=0.5, lw=0, label="bootstrap 90% band")
    ax.plot(d.lam, d.threshold, color=ORDINAL_BLUE[0], lw=2, label="one threshold (BDD100K train)")
    for w, col in (("clear", SERIES[0]), ("clear_night", INK2), ("rain_night", SERIES[1]), ("snow", SERIES[3])):
        ax.plot(d.lam, d[f"threshold_{w}"], color=col, lw=0.9, ls=":", label=f"fitted on {w.replace('_', ' ')} only")
    ax.axhline(DEFAULT, color=INK2, lw=0.8, ls="--")
    ax.text(20, DEFAULT * 1.1, "no tuning (0.25)", fontsize=7.5, color=INK2)
    ax.set(xscale="log", yscale="log", xlabel="dial λ: false alarms accepted per avoided miss", ylabel="confidence threshold",
           title="The dial's threshold")
    ax.legend(fontsize=7)
    from matplotlib.lines import Line2D

    for ax, policy, title in ((axes[1], "no tuning (0.25)", "Cost of not tuning (0.25) vs the dial"),
                              (axes[2], "per-weather threshold (train)", "Per-weather thresholds vs one threshold")):
        for ds, g, col, ls, lab in show:
            e = ev[(ev.dataset == ds) & (ev.group == g) & (ev.policy == policy) & ev.lam.isin(DIAL)]
            ax.plot(e.lam, e.cost_vs_dial, color=col, ls=ls, lw=1.6, label=lab)
            if policy != "no tuning (0.25)":
                b = ev[(ev.dataset == ds) & (ev.group == g) & (ev.policy == "own best constant (in-sample)") & ev.lam.isin(DIAL)]
                ax.plot(b.lam, b.cost_vs_dial, color=col, ls=":", lw=1.0)
        ax.axhline(1.0, color=INK2, lw=0.8)
        ax.set(xscale="log", xlabel="dial λ", ylabel="cost relative to one threshold (λ·misses + false alarms)", title=title)
    axes[1].legend(fontsize=7, loc="upper left")
    h, _ = axes[2].get_legend_handles_labels()
    h.append(Line2D([], [], color=INK2, ls=":", lw=1.0, label="dotted: condition's own best constant (in-sample)"))
    axes[2].legend(handles=h, fontsize=6.5, loc="upper center", ncol=3)
    lo, hi = axes[2].get_ylim()
    axes[2].set_ylim(lo, hi + 0.35 * (hi - lo))
    fig.tight_layout()
    return save(fig, Cfg(dict(paths=Cfg(dict(results=RESULTS)))), "fig20_one_threshold")


def main():
    m = pd.read_parquet(TRAIN, columns=["image_id", "weather", "score", "tp"])
    im = pd.read_parquet(THRESH_CACHE / "images.parquet", columns=["image_id", "weather", "n_gt"])
    dial = fit(m, im)
    _, curve = train_curve(m[["score", "tp"]])
    table = breakpoints(curve)
    probe = np.geomspace(0.05, 5000, 400)
    assert all(threshold_from_table(table, x) == train_threshold(curve, x) for x in probe), "table differs from the fit"
    table.to_csv(REPO / "weights" / "one_threshold_table.csv", index=False, float_format="%.17g")  # round-trips exactly
    fo = folds(m, im)
    ev = grade(dial, np.union1d(DIAL, PRESETS))
    dial.to_csv(RESULTS / "one_threshold_dial.csv", index=False, float_format="%.6g")
    fo.to_csv(RESULTS / "one_threshold_folds.csv", index=False, float_format="%.6g")
    ev[ev.lam.isin(PRESETS)].to_csv(RESULTS / "one_threshold_eval.csv", index=False, float_format="%.6g")
    figure(dial, ev)
    pd.set_option("display.width", 220)
    print(dial[dial.lam.isin(PRESETS)].round(4).to_string(index=False))
    print(fo.groupby("lam")[["threshold", "own_best", "cost_vs_own_best"]].agg(["mean", "max"]).round(4).to_string())
    show = ev[ev.lam.isin(PRESETS) & ev.group.isin(["day", "night", "rain_night", "all", "fog"])]
    print(show.pivot_table(index=["dataset", "group", "lam"], columns="policy", values="cost_vs_dial").round(3).to_string())


if __name__ == "__main__":
    main()
