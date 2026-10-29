"""EXPLORATORY: how much of the per-image oracle threshold's advantage is learnable, and how much is luck?
STATUS: evidence that a per-image threshold cannot beat the dial's options (wad/dial.py): the
per-image oracle's edge is mostly luck.

The oracle picks each image's threshold after seeing which detections turned out real. Its edge over
one threshold has two sources: (1) structure, images where a given score is systematically more or
less often real than usual (learnable in principle), and (2) luck, which particular detections at a
given score happen to be real (not learnable from anything seen beforehand).

Null model: keep every image's detections and scores (and its objects the detector never found),
but redraw whether each detection is real from the chance that a detection at that score and class
group is real, pooled over the dataset (global null) or within the image's condition (condition
null). Under the null nothing per image is learnable beyond score (global) or score + condition
(condition null), so the oracle's gain there is the luck floor. The real data's gain beyond the
floor bounds what any predictor using one frame could add.

Run: python -m wad.threshold_noise   -> results/threshold/noise_floor.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wad.config import tau_grid
from wad.curves import ALL, Cache
from wad.config import THRESH_RESULTS as RESULTS

LAMS = (1, 3, 10, 30, 100)
N_SIM = 5
GROUP = {"car": "vehicle", "truck": "vehicle", "bus": "vehicle", "bicycle": "two_wheeler", "motorcycle": "two_wheeler",
         "person": "person"}
SCORE_BINS = np.concatenate([np.geomspace(0.001, 0.1, 25), np.linspace(0.1, 1.0, 37)[1:]])


def configs():
    from wad.run2 import bdd_cfg, dawn_cfg, load_run2

    r2 = load_run2()
    return [("BDD100K", bdd_cfg(r2, "v8x_1280", "group")), ("DAWN", dawn_cfg(r2, "v8x_1280", "group", "clean"))]


def ratios(m: np.ndarray, f: np.ndarray) -> dict:
    """Oracle cost / best single threshold cost at each lam (both on the same images and grid)."""
    out = {}
    for lam in LAMS:
        cv = lam * m + f
        out[lam] = float(cv.min(1).mean() / cv.mean(0).min())
    return out


def curves_from(img: np.ndarray, score: np.ndarray, real: np.ndarray, never: np.ndarray, n_img: int, taus: np.ndarray):
    """Per image miss / FP counts at taus from detection-level real flags and never-found objects."""
    k = np.searchsorted(taus, score, side="right")
    tp = np.zeros((n_img, len(taus) + 1))
    fp = np.zeros_like(tp)
    np.add.at(tp, (img[real], k[real]), 1)
    np.add.at(fp, (img[~real], k[~real]), 1)
    TP = np.cumsum(tp[:, ::-1], axis=1)[:, ::-1][:, 1:]
    FP = np.cumsum(fp[:, ::-1], axis=1)[:, ::-1][:, 1:]
    n_gt = TP[:, 0] + never  # objects = those found at the floor + those never found
    return n_gt[:, None] - TP, FP


def main() -> pd.DataFrame:
    rows = []
    for ds, v in configs():
        c = Cache(v)
        taus = tau_grid(v)[::5]
        img = c.pred.image_id.map(c.idx).to_numpy()
        score = c.pred.score.to_numpy()
        real = c.pred.tp.to_numpy().astype(bool)
        grp = c.pred.cls.map(GROUP).to_numpy()
        cond = c.weather[img]
        # objects never matched at any score (thresholding cannot reach them)
        never = np.bincount(c.gt_img, weights=~np.isfinite(c.gt_score) | (c.gt_score < taus[0]), minlength=c.n_img)
        m_obs, f_obs = curves_from(img, score, real, never, c.n_img, taus)
        # sanity: these reproduce the evaluation curves
        assert np.allclose(m_obs, c.n_gt(ALL)[:, None] - c.matched_grid(taus, ALL))
        assert np.allclose(f_obs, c.fp_grid(taus, ALL))
        obs = ratios(m_obs, f_obs)
        sb = np.digitize(score, SCORE_BINS)
        rng = np.random.default_rng(0)
        for null, key in (("global null (score + class)", pd.Series(list(zip(grp, sb)))),
                          ("condition null (score + class + condition)", pd.Series(list(zip(grp, sb, cond))))):
            cal = pd.Series(real).groupby(key.to_numpy()).transform("mean").to_numpy()
            sims = []
            for _ in range(N_SIM):
                r = rng.random(len(cal)) < cal
                sims.append(ratios(*curves_from(img, score, r, never, c.n_img, taus)))
            for lam in LAMS:
                floor = float(np.mean([s[lam] for s in sims]))
                rows.append(dict(dataset=ds, null=null, lam=lam, oracle_real=obs[lam], oracle_null=floor,
                                 oracle_gain_real=1 - obs[lam], oracle_gain_null=1 - floor,
                                 learnable_beyond_null=floor - obs[lam],
                                 share_of_oracle_gain_that_is_luck=(1 - floor) / (1 - obs[lam])))
    d = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    d.to_csv(RESULTS / "noise_floor.csv", index=False, float_format="%.4g")
    return d


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    print(main().round(3).to_string(index=False))
