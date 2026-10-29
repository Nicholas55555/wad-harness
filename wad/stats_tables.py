"""Recall, precision, F1 and related statistics per condition and visibility group.

Detector: the run 2 primary (YOLOv8x @1280, grouped vehicle loss; DAWN with clean labels). Groups:
each condition (BDD100K weather x time of day, DAWN weather), each condition's lowest-visibility
third (EXPLORATORY visibility score, wad/visibility.py), the lowest-visibility quarter of each
dataset, and all images.

  stats_at_thresholds.csv  all images of the group at the shipped thresholds (0.25, 0.5) and at the
                           group's best-F1 threshold: recall, precision, F1, FPPI, plus max recall,
                           AP50 (all objects pooled), per-class person/car recall/precision/F1
  stats_by_policy.csv      held-out halves over 200 splits, alpha = 0.2: one global threshold,
                           per-weather (oracle) thresholds, per-visibility-quartile thresholds;
                           recall, precision, F1, FPPI, image miss fraction, share of splits above
                           alpha (means with 2.5-97.5% split intervals)

Run: python -m wad.stats_tables
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wad.config import REPO, tau_grid
from wad.crc import fit_threshold, make_splits
from wad.curves import ALL, Cache, average_precision, operating_curve

OUT = REPO / "results" / "run2"
ALPHA, N_SPLITS = 0.2, 200


def prf(tp, fp, n_gt):
    r = tp / n_gt if n_gt else np.nan
    p = tp / (tp + fp) if tp + fp else np.nan
    f = 2 * p * r / (p + r) if (p == p and r == r and p + r) else np.nan
    return r, p, f


def groups(c: Cache, h: np.ndarray) -> dict[str, np.ndarray]:
    w = c.weather
    g = {b: w == b for b in pd.unique(w)}
    for b in pd.unique(w):
        m = w == b
        g[f"{b} | lowest-visibility third"] = m & (h >= np.quantile(h[m], 2 / 3))
    g["lowest-visibility quarter"] = h >= np.quantile(h, 0.75)
    g["all"] = np.ones(c.n_img, bool)
    return g


def at_thresholds(ds: str, c: Cache, gs: dict, taus: np.ndarray) -> list[dict]:
    rows = []
    pm = c.pred.image_id.map(c.idx).to_numpy()
    tp_grid, fp_grid, ngt = c.matched_grid(taus, ALL), c.fp_grid(taus, ALL), c.n_gt(ALL)
    cnt = {t: {k: c.at_thresholds(np.full(c.n_img, t), k) for k in (ALL, "person", "car")} for t in (0.25, 0.5)}
    floor = c.at_thresholds(np.full(c.n_img, 0.001), ALL)
    for name, m in gs.items():
        n_img, n_gt = int(m.sum()), float(ngt[m].sum())
        row = dict(dataset=ds, group=name, images=n_img, objects=int(n_gt))
        for t in (0.25, 0.5):
            x = cnt[t][ALL]
            tp, fp = float((x["n_gt"] - x["n_missed"])[m].sum()), float(x["n_fp"][m].sum())
            r, p, f = prf(tp, fp, n_gt)
            row.update({f"recall@{t}": r, f"precision@{t}": p, f"f1@{t}": f, f"fppi@{t}": fp / n_img})
        tp, fp = tp_grid[m].sum(0), fp_grid[m].sum(0)
        f1 = 2 * tp / np.maximum(2 * tp + fp + (n_gt - tp), 1)
        j = int(np.argmax(f1))
        r, p, f = prf(tp[j], fp[j], n_gt)
        row.update(best_f1=f, best_f1_tau=float(taus[j]), best_f1_recall=r, best_f1_precision=p, best_f1_fppi=fp[j] / n_img)
        row["max_recall"] = float((floor["n_gt"] - floor["n_missed"])[m].sum() / n_gt)
        sel = m[pm]
        oc = operating_curve(c.pred.score.to_numpy()[sel], c.pred.tp.to_numpy()[sel], int(n_gt), n_img)
        row["ap50_pooled"] = average_precision(oc["precision"], oc["recall"])
        for k in ("person", "car"):
            x = cnt[0.25][k]
            nk = float(x["n_gt"][m].sum())
            tpk, fpk = float((x["n_gt"] - x["n_missed"])[m].sum()), float(x["n_fp"][m].sum())
            r, p, f = prf(tpk, fpk, nk)
            row.update({f"{k}_objects": int(nk), f"{k}_recall@0.25": r, f"{k}_precision@0.25": p, f"{k}_f1@0.25": f})
        rows.append(row)
    return rows


def by_policy(ds: str, v, c: Cache, gs: dict, h: np.ndarray, taus: np.ndarray) -> list[dict]:
    floor = v.detector.conf_floor
    loss = c.loss_grid(taus, ALL)
    v.splits["n_repeats"] = N_SPLITS
    splits = make_splits(v, c.images)
    w = c.weather
    acc = []
    for seed, sp in splits.groupby("seed"):
        cal = (sp.set_index("image_id").loc[c.image_ids].half == "cal").to_numpy()
        test = ~cal
        pol = {"global": np.full(c.n_img, fit_threshold(loss[cal], taus, ALPHA, floor)["tau_hat"])}
        t = np.full(c.n_img, np.nan)
        for b in pd.unique(w):
            t[w == b] = fit_threshold(loss[cal & (w == b)], taus, ALPHA, floor)["tau_hat"]
        pol["per-weather"] = t
        vb = np.digitize(h, np.quantile(h[cal], [0.25, 0.5, 0.75]))
        t = np.full(c.n_img, np.nan)
        for k in range(4):
            t[vb == k] = fit_threshold(loss[cal & (vb == k)], taus, ALPHA, floor)["tau_hat"]
        pol["per-visibility"] = t
        for p, ti in pol.items():
            x = c.at_thresholds(ti, ALL)
            li = np.where(x["n_gt"] > 0, x["n_missed"] / np.maximum(x["n_gt"], 1), 0.0)
            for name, m in gs.items():
                r_ = test & m
                n_gt = float(x["n_gt"][r_].sum())
                tp, fp = float((x["n_gt"] - x["n_missed"])[r_].sum()), float(x["n_fp"][r_].sum())
                r, pr, f = prf(tp, fp, n_gt)
                acc.append(dict(dataset=ds, group=name, policy=p, seed=seed, recall=r, precision=pr, f1=f,
                                fppi=fp / max(r_.sum(), 1), image_miss=li[r_].mean(), n_test=int(r_.sum())))
    a = pd.DataFrame(acc)
    rows = []
    for (g, p), d in a.groupby(["group", "policy"], sort=False):
        row = dict(dataset=ds, group=g, policy=p, splits=d.seed.nunique(), n_test=d.n_test.mean(),
                   above_alpha=float((d.image_miss > ALPHA).mean()))
        for k in ("recall", "precision", "f1", "fppi", "image_miss"):
            row.update({k: d[k].mean(), f"{k}_lo": d[k].quantile(0.025), f"{k}_hi": d[k].quantile(0.975)})
        rows.append(row)
    return rows


def main():
    from wad.run2 import bdd_cfg, dawn_cfg, load_run2
    from wad.visibility import load_h

    r2 = load_run2()
    hs = load_h()
    t_rows, p_rows = [], []
    for ds, v in [("BDD100K", bdd_cfg(r2, "v8x_1280", "group")), ("DAWN", dawn_cfg(r2, "v8x_1280", "group", "clean"))]:
        c = Cache(v)
        h = hs.loc[c.image_ids].to_numpy()
        gs = groups(c, h)
        taus = tau_grid(v)
        t_rows += at_thresholds(ds, c, gs, taus)
        p_rows += by_policy(ds, v, c, gs, h, taus)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(t_rows).to_csv(OUT / "stats_at_thresholds.csv", index=False, float_format="%.6g")
    pd.DataFrame(p_rows).to_csv(OUT / "stats_by_policy.csv", index=False, float_format="%.6g")
    print(OUT / "stats_at_thresholds.csv")
    print(OUT / "stats_by_policy.csv")


if __name__ == "__main__":
    main()
