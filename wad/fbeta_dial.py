"""THE DIAL IN F-BETA UNITS (user's request 2026-10-14): instead of lam (false alarms accepted per avoided
miss), set beta, how many times more recall counts than precision, F_beta = (1 + b^2) P R / (b^2 P + R).
The dial stays the lam dial (user's decision 2026-10-15): each beta is turned into its equivalent lam,
fitted on the BDD100K train images only, and the lam dial is applied there. With calibrated chances the
F_beta-best rule is a cut at F*/(1+b^2), F* the best F_beta reachable (Zhao et al. 2013; Lipton et al.
2014), which is the lam dial at lam = (1 + b^2) / F* - 1. F* is measured on train for each option:
  one threshold     over all train detections (the lam dial's threshold there is exactly the F_beta-best one)
  per detection     over the 5-fold out-of-fold chances of the adopted per-detection model (each from a
                    fold model that did not see that image); the cut is 1 / (1 + lam_equivalent)
BDD100K val and DAWN only grade it.

Statistical backing (results/threshold/fbeta_*.csv, figures/fig22_fbeta_dial):
  stability     200-resample image bootstrap of the train choice (threshold and cut), 90% band
  grading       recall, precision, false alarms per image, F1 and F_beta on BDD100K val per condition and
                DAWN per weather with 95% image-bootstrap intervals, against no tuning (0.25) and the
                condition's own best constant (in-sample, the most any one threshold could get there)
  dedicated     EXPLORATORY check of whether a model trained for one beta beats one calibrated model plus
                its cut: the per-detection model retrained with real boxes weighted by that beta's
                lam_equivalent (cost-sensitive learning), cut chosen the same way on its own 5-fold
                out-of-fold chances, for beta = 0.5, 1, 2
Table: weights/fbeta_table.csv (beta -> lam_equivalent per option, and the resulting threshold and cut;
401 settings from 0.25 to 4), used by wad/dial.py (keep_f = keep at the equivalent lam). Evaluation
labels as in wad/per_detection.py.

Run: python -m wad.fbeta_dial   (needs python -m wad.one_threshold and python -m wad.per_detection first)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from wad.config import REPO, THRESH_CACHE
from wad.config import THRESH_RESULTS as RESULTS
from wad.one_threshold import DEFAULT, TRAIN, eval_configs

BETAS = np.round(np.geomspace(0.25, 4, 41), 6)  # the page's settings; 0.25, 0.5, 1, 2, 4 exactly
FINE = np.geomspace(0.25, 4, 401)  # the table's settings (contains BETAS)
PRESETS_B = (0.5, 1.0, 2.0)
TABLE = REPO / "weights" / "fbeta_table.csv"


def fbeta(tp: float, fp: float, n_obj: float, beta: float) -> float:
    """F_beta from counts: (1 + b^2) TP / ((1 + b^2) TP + b^2 FN + FP), FN = objects not found."""
    b2 = beta * beta
    return (1 + b2) * tp / (b2 * n_obj + tp + fp) if tp > 0 else 0.0


class Ranked:
    """Detections sorted by a value (score or chance), cut only between distinct values: every
    'keep value >= v' rule as cumulative real and kept counts."""

    def __init__(self, value: np.ndarray, tp: np.ndarray):
        o = np.argsort(-value, kind="stable")
        v = value[o]
        last = np.r_[v[1:] != v[:-1], True]
        self.v = v[last]
        self.tp = np.cumsum(tp[o].astype(float))[last]
        self.n = np.arange(1, len(v) + 1, dtype=float)[last]

    def best(self, n_obj: float, beta: float) -> tuple[float, float]:
        """(cut, F_beta) of the rule with the highest F_beta."""
        b2 = beta * beta
        f = (1 + b2) * self.tp / (b2 * n_obj + self.n)
        j = int(np.argmax(f))
        return float(self.v[j]), float(f[j])


def train_data():
    from wad.per_detection import oof_path

    m = pd.read_parquet(TRAIN, columns=["image_id", "score", "tp"])
    oof = pd.read_parquet(oof_path())
    n_obj = float(pd.read_parquet(THRESH_CACHE / "images.parquet", columns=["n_gt"]).n_gt.sum())
    assert len(oof) == len(m) and int(oof.tp.sum()) == int(m.tp.sum()) and not oof.p.isna().any()
    return m, oof, n_obj


def table(m: pd.DataFrame, oof: pd.DataFrame, n_obj: float, betas=FINE) -> pd.DataFrame:
    """beta -> the equivalent lam of each option (from F* on train) and the lam dial's threshold / cut there."""
    from wad.one_threshold import train_curve, train_threshold

    r1, r2 = Ranked(m.score.to_numpy(), m.tp.to_numpy()), Ranked(oof.p.to_numpy(), oof.tp.to_numpy())
    _, curve = train_curve(m[["score", "tp"]])
    rows = []
    for b in betas:
        t_best, f1 = r1.best(n_obj, b)
        c_best, f2 = r2.best(n_obj, b)
        lam1, lam2 = (1 + b * b) / f1 - 1, (1 + b * b) / f2 - 1
        rows.append(dict(beta=b, one_train_f=f1, one_lam_equivalent=lam1, threshold=train_threshold(curve, lam1),
                         threshold_fbeta_best=t_best, det_train_f=f2, det_lam_equivalent=lam2, cut=1 / (1 + lam2),
                         cut_fbeta_best=c_best))
    return pd.DataFrame(rows)


def lookup(tab: pd.DataFrame, beta: float) -> pd.Series:
    """The table row for beta (nearest of 401 log-spaced settings, 0.7% apart)."""
    if not 0.25 <= beta <= 4:
        raise ValueError(f"beta must be in [0.25, 4], got {beta}")
    return tab.iloc[int(np.argmin(np.abs(np.log(tab.beta.to_numpy()) - np.log(beta))))]


def stability(m: pd.DataFrame, oof: pd.DataFrame, n_obj: float, B=200, seed=0) -> pd.DataFrame:
    """Image bootstrap of the settings the dial uses: per resample, F* of each option on a fine grid of
    thresholds / cuts (0.2% steps), the one threshold with the highest F_beta (= the lam dial's threshold
    at lam_eq) and the per-detection cut F* / (1 + beta^2) (= 1 / (1 + lam_eq))."""
    grid = np.geomspace(1e-4, 1.0, 4601)

    def counts(v, tp, w):
        b = np.clip(np.searchsorted(grid, v, side="right") - 1, 0, len(grid) - 1)
        r = np.bincount(b, weights=w * tp, minlength=len(grid))[::-1].cumsum()[::-1]
        n = np.bincount(b, weights=w, minlength=len(grid))[::-1].cumsum()[::-1]
        return r, n

    rng = np.random.default_rng(seed)
    ids, codes = np.unique(m.image_id.to_numpy(), return_inverse=True)
    codes_o = pd.Index(ids).get_indexer(oof.image_id.to_numpy())
    gt = pd.read_parquet(THRESH_CACHE / "images.parquet", columns=["image_id", "n_gt"]).set_index("image_id").n_gt
    gt_img = gt.reindex(ids).fillna(0).to_numpy()
    no_det = float(gt.sum() - gt_img.sum())  # objects in train images without any detection
    s, y, p, yo = m.score.to_numpy(), m.tp.to_numpy().astype(float), oof.p.to_numpy(), oof.tp.to_numpy().astype(float)
    picks = {k: np.empty((B, len(PRESETS_B))) for k in ("threshold", "cut")}
    for i in range(B):
        cnt = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)).astype(float)
        n_b = float(cnt @ gt_img) + no_det
        for key, (v, t, cc) in (("threshold", (s, y, codes)), ("cut", (p, yo, codes_o))):
            r, n = counts(v, t, cnt[cc])
            for j, b in enumerate(PRESETS_B):
                f = (1 + b * b) * r / (b * b * n_b + n)
                picks[key][i, j] = grid[int(np.argmax(f))] if key == "threshold" else f.max() / (1 + b * b)
    rows = []
    for j, b in enumerate(PRESETS_B):
        rows.append(dict(beta=b, **{f"{k}_{q}": float(np.quantile(picks[k][:, j], qq)) for k in picks
                                    for q, qq in (("q05", 0.05), ("q50", 0.5), ("q95", 0.95))}))
    return pd.DataFrame(rows)


def groups_of(ds: str, weathers) -> dict:
    from wad.per_detection import groups_of as g

    return g(ds, weathers)


def grade(ds: str, c, sc: np.ndarray, p: np.ndarray, tab: pd.DataFrame, betas, B=1000, seed=0) -> pd.DataFrame:
    """Per condition and beta: the F_beta dial's two options, no tuning and the condition's own best
    constant; F_beta with 95% image-bootstrap intervals."""
    from wad.curves import ALL

    tp = c.pred.tp.to_numpy().astype(bool)
    img = c.pred.image_id.map(c.idx).to_numpy()
    n_gt = c.n_gt(ALL).astype(float)
    rng = np.random.default_rng(seed)
    w_det = c.weather[img]
    rows = []
    for g, ws in groups_of(ds, c.weather).items():
        sel = np.flatnonzero(np.isin(c.weather, ws))
        Wb = np.stack([np.bincount(rng.integers(0, len(sel), len(sel)), minlength=len(sel)) for _ in range(B)]).astype(float)
        gt = n_gt[sel]
        dm = np.isin(w_det, ws)
        own = Ranked(sc[dm], tp[dm])
        for b in betas:
            r = lookup(tab, b)
            t_own, f_own = own.best(gt.sum(), b)
            pol = {"one threshold (F-beta dial)": (r.threshold, sc >= r.threshold),
                   "per detection (F-beta dial)": (r.cut, p >= r.cut),
                   "no tuning (0.25)": (DEFAULT, sc >= DEFAULT),
                   "own best constant (in-sample)": (t_own, sc >= t_own)}
            for name, (thr, kept) in pol.items():
                tk = np.bincount(img, weights=kept & tp, minlength=c.n_img)[sel]
                fk = np.bincount(img, weights=kept & ~tp, minlength=c.n_img)[sel]
                T, F, N = tk.sum(), fk.sum(), gt.sum()
                fb = fbeta(T, F, N, b)
                bt, bf, bn = Wb @ tk, Wb @ fk, Wb @ gt
                boot = (1 + b * b) * bt / (b * b * bn + bt + bf)
                rows.append(dict(dataset=ds, group=g, beta=b, policy=name, cut_or_threshold=thr, images=len(sel),
                                 recall=T / N, precision=T / max(T + F, 1), fppi=F / len(sel), f1=fbeta(T, F, N, 1.0),
                                 fbeta=fb, fbeta_ci_lo=float(np.quantile(boot, 0.025)), fbeta_ci_hi=float(np.quantile(boot, 0.975)),
                                 fbeta_vs_own_best=fb / f_own))
    return pd.DataFrame(rows)


def dedicated(tab: pd.DataFrame, evals: dict, n_obj: float, k=5, seed=0) -> pd.DataFrame:
    """EXPLORATORY: per-detection models trained for one beta (real boxes weighted by its
    lam_equivalent), cut chosen on their own k-fold out-of-fold chances, against one calibrated model
    plus its cut, on BDD100K val / DAWN (all images)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    from wad.curves import Cache
    from wad.per_detection import ADOPTED, PARAMS, TRAIN_DET, add_box_columns, cols_of, labelled_features, load
    from wad.thresh_net import cfg as train_cfg

    FULL = cols_of(ADOPTED)  # the adopted model's features, so the comparison is like for like
    ct = Cache(train_cfg())
    tr = labelled_features(ct, TRAIN_DET)
    if ADOPTED == "yolo":
        tr = add_box_columns(tr, "train", load()["pca"])
    grp = pd.read_parquet(THRESH_CACHE / "images.parquet", columns=["image_id", "group"]).set_index("image_id").group
    g = grp.loc[ct.image_ids[tr.img.to_numpy()]].to_numpy()
    ug = np.array(sorted(set(grp)))
    fold = pd.Series(np.random.default_rng(seed).permutation(len(ug)) % k, index=ug).loc[g].to_numpy()
    X, y = tr[FULL].to_numpy(), tr.tp.to_numpy()

    def fit(Xa, ya, w):
        m = HistGradientBoostingClassifier(**PARAMS, categorical_features=[FULL.index("cls")])
        return m.fit(Xa, ya, sample_weight=np.where(ya == 1, w, 1.0))

    rows = []
    for b in PRESETS_B:
        t0 = time.perf_counter()
        w = float(lookup(tab, b).det_lam_equivalent)
        oof = np.empty(len(tr))
        for j in range(k):
            m = fit(X[fold != j], y[fold != j], w)
            oof[fold == j] = m.predict_proba(X[fold == j])[:, 1]
        cut, f_train = Ranked(oof, y).best(n_obj, b)
        full = fit(X, y, w)
        for ds, (f, n_eval, p_cal) in evals.items():
            q = full.predict_proba(f[FULL].to_numpy())[:, 1]
            yy = f.tp.to_numpy().astype(bool)
            kd, kc = q >= cut, p_cal >= lookup(tab, b).cut
            rows.append(dict(beta=b, dataset=ds, positive_weight=w, cut=cut, train_oof_f=f_train,
                             f_dedicated=fbeta((kd & yy).sum(), (kd & ~yy).sum(), n_eval, b),
                             f_calibrated_model=fbeta((kc & yy).sum(), (kc & ~yy).sum(), n_eval, b)))
            rows[-1]["dedicated_vs_calibrated"] = rows[-1]["f_dedicated"] / rows[-1]["f_calibrated_model"]
        print(f"dedicated beta {b}: weight {w:.2f}, {time.perf_counter() - t0:.0f} s", flush=True)
    return pd.DataFrame(rows)


def figure(tab: pd.DataFrame, ev: pd.DataFrame):
    import matplotlib.pyplot as plt

    from wad.config import Cfg
    from wad.figures import INK, INK2, ORDINAL_BLUE, SERIES, save, style

    style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.plot(tab.beta, tab.one_lam_equivalent, color=INK2, lw=1.8, label="one threshold")
    ax.plot(tab.beta, tab.det_lam_equivalent, color=ORDINAL_BLUE[0], lw=1.8, label="per detection")
    ax.set(xscale="log", yscale="log", xlabel="β: weight on recall relative to precision",
           ylabel="equivalent λ (false alarms per avoided miss)", title="Every β setting is a λ setting (fitted on train)")
    ax.set_yticks([0.3, 1, 3, 10, 30], ["0.3", "1", "3", "10", "30"])
    ax.legend(fontsize=7.5)
    show = [("BDD100K", "day", ORDINAL_BLUE[1], "-", "BDD100K day"), ("BDD100K", "night", INK, "-", "BDD100K night"),
            ("BDD100K", "rain_night", SERIES[1], "-", "BDD100K rain night"), ("DAWN", "fog", SERIES[2], "--", "DAWN fog")]
    for ax, policy, title in ((axes[1], "one threshold (F-beta dial)", "One threshold: Fβ reached vs the most possible"),
                              (axes[2], "per detection (F-beta dial)", "Per detection: Fβ reached vs one threshold's most")):
        for ds, g, col, ls, lab in show:
            e = ev[(ev.dataset == ds) & (ev.group == g) & (ev.policy == policy)]
            ax.plot(e.beta, e.fbeta_vs_own_best, color=col, ls=ls, lw=1.6, label=lab)
        ax.axhline(1.0, color=INK2, lw=0.8)
        ax.set(xscale="log", xlabel="β", ylabel="Fβ / Fβ of the condition's own best constant (in-sample)", title=title)
        ax.legend(fontsize=7)
    from matplotlib.ticker import NullFormatter

    for ax in axes:  # plain ticks on the log axes
        ax.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    return save(fig, Cfg(dict(paths=Cfg(dict(results=RESULTS)))), "fig22_fbeta_dial")


def main():
    from wad.curves import ALL, Cache
    from wad.per_detection import ADOPTED, BOX_SET, EVAL_DET, add_box_columns, labelled_features, load, pred_path

    t0 = time.perf_counter()
    m, oof, n_obj = train_data()
    tab = table(m, oof, n_obj)
    tab.to_csv(TABLE, index=False, float_format="%.17g")
    print(f"table: {len(tab)} settings, {time.perf_counter() - t0:.0f} s", flush=True)
    st = stability(m, oof, n_obj)
    print(f"stability, {time.perf_counter() - t0:.0f} s", flush=True)
    ev, evals = [], {}
    for ds, v in eval_configs().items():
        c = Cache(v)
        key = "bdd" if ds == "BDD100K" else "dawn"
        pr = pd.read_parquet(pred_path(key))
        assert (pr.image_id.to_numpy() == c.pred.image_id.to_numpy()).all()
        ev.append(grade(ds, c, c.pred.score.to_numpy(), pr.p.to_numpy(), tab, BETAS))
        f = labelled_features(c, EVAL_DET[ds])
        if ADOPTED == "yolo":
            f = add_box_columns(f, BOX_SET[ds], load()["pca"])
        evals[ds] = (f, float(c.n_gt(ALL).sum()), pr.p.to_numpy())
        print(f"{ds}: graded, {time.perf_counter() - t0:.0f} s", flush=True)
    ev = pd.concat(ev, ignore_index=True)
    ev.to_csv(RESULTS / "fbeta_eval.csv", index=False, float_format="%.5g")
    st.to_csv(RESULTS / "fbeta_stability.csv", index=False, float_format="%.5g")
    figure(tab, ev)
    pd.set_option("display.width", 220)
    print(tab[tab.beta.round(4).isin([0.25, 0.5, 1.0, 2.0, 4.0])].round(4).to_string(index=False))
    print(st.round(4).to_string(index=False))
    e = ev[ev.beta.isin(PRESETS_B) & ev.group.isin(["all", "day", "night", "rain_night", "fog"])]
    print(e[["dataset", "group", "beta", "policy", "recall", "precision", "f1", "fbeta", "fbeta_ci_lo", "fbeta_ci_hi",
             "fbeta_vs_own_best"]].round(3).to_string(index=False))
    de = dedicated(tab, evals, n_obj)
    de.to_csv(RESULTS / "fbeta_dedicated.csv", index=False, float_format="%.5g")
    print(de.round(4).to_string(index=False))
    print(f"done, {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
