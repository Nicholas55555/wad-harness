"""Miss rate vs FPPI per weather bin and class, maximum achievable recall, the go/no-go.

Every quantity here is read off the single matching pass (wad.match):
  GT box missed at tau        <=>  match_score < tau or NaN
  prediction is a FP at tau   <=>  not tp and score >= tau
Per-image loss at tau is the image's missed fraction. For the pooled class ('all') it is
averaged over every image in the bin, images without ground truth counting 0 (the quantity
phase 4 controls). For a single class c it is averaged over the images that contain class c,
the same loss the phase 4 per-class budgets control.

Run: python -m wad.curves [--smoke]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from wad.config import CANONICAL, Cfg, bins, cache_path, load_config, results_path, tau_grid, write_json
from wad.data import load_images

ALL = "all"


class Cache:
    """Cached matches indexed for fast evaluation at any threshold, per image."""

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.images = load_images(cfg)
        self.image_ids = self.images.image_id.to_numpy()
        self.idx = {k: i for i, k in enumerate(self.image_ids)}
        self.weather = self.images.weather.to_numpy()
        mg = pd.read_parquet(cache_path(cfg, "matches_gt.parquet"))
        mp = pd.read_parquet(cache_path(cfg, "matches_pred.parquet"))
        self.gt = mg[mg.image_id.isin(self.idx)].reset_index(drop=True)
        self.gt_img = self.gt.image_id.map(self.idx).to_numpy()
        self.gt_cls = self.gt.cls.to_numpy()
        # NaN (never matched) -> -inf so that "missed at tau" is simply score < tau.
        self.gt_score = self.gt.match_score.fillna(-np.inf).to_numpy()
        fp = mp[(~mp.tp) & mp.image_id.isin(self.idx)].reset_index(drop=True)
        self.fp_img = fp.image_id.map(self.idx).to_numpy()
        self.fp_cls = fp.cls.to_numpy()
        self.fp_score = fp.score.to_numpy()
        self.pred = mp[mp.image_id.isin(self.idx)].reset_index(drop=True)
        # eval_cls: the class a prediction is scored under in per-class curves. A TP counts for the
        # class of the ground truth it matched (under grouped matching a truck box may match a car);
        # a FP counts for its predicted class. Under class-aware matching eval_cls == cls.
        gt_cls_of = dict(zip(self.gt.gt_id, self.gt.cls))
        self.pred["eval_cls"] = np.where(self.pred.tp, self.pred.gt_id.map(gt_cls_of), self.pred.cls)
        self.n_img = len(self.image_ids)

    # ---- per-image counts on a threshold grid (fast, used for fitting and curves) ----
    def _count_ge(self, img: np.ndarray, score: np.ndarray, taus: np.ndarray) -> np.ndarray:
        """M[i, j] = number of items of image i with score >= taus[j]."""
        k = np.searchsorted(taus, score, side="right")  # item counts for taus[:k]
        c = np.zeros((self.n_img, len(taus) + 1))
        np.add.at(c, (img, k), 1)
        return np.cumsum(c[:, ::-1], axis=1)[:, ::-1][:, 1:]

    def gt_mask(self, cls: str) -> np.ndarray:
        return np.ones(len(self.gt_cls), bool) if cls == ALL else self.gt_cls == cls

    def fp_mask(self, cls: str) -> np.ndarray:
        return np.ones(len(self.fp_cls), bool) if cls == ALL else self.fp_cls == cls

    def n_gt(self, cls: str = ALL) -> np.ndarray:
        m = self.gt_mask(cls)
        return np.bincount(self.gt_img[m], minlength=self.n_img).astype(float)

    def matched_grid(self, taus: np.ndarray, cls: str = ALL) -> np.ndarray:
        m = self.gt_mask(cls)
        return self._count_ge(self.gt_img[m], self.gt_score[m], taus)

    def fp_grid(self, taus: np.ndarray, cls: str = ALL) -> np.ndarray:
        m = self.fp_mask(cls)
        return self._count_ge(self.fp_img[m], self.fp_score[m], taus)

    def loss_grid(self, taus: np.ndarray, cls: str = ALL) -> np.ndarray:
        """L[i, j]: missed fraction of image i at taus[j]; 0 where the image has no GT of cls."""
        n = self.n_gt(cls)[:, None]
        miss = n - self.matched_grid(taus, cls)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, miss / np.where(n > 0, n, 1), 0.0)

    # ---- per-image counts at one threshold per image (exact, used for evaluation) ----
    def at_thresholds(self, tau_img: np.ndarray, cls: str = ALL) -> dict[str, np.ndarray]:
        """Per-image n_gt, n_missed, n_fp when image i is thresholded at tau_img[i]."""
        gm, fm = self.gt_mask(cls), self.fp_mask(cls)
        gi, fi = self.gt_img[gm], self.fp_img[fm]
        matched = np.bincount(gi, weights=(self.gt_score[gm] >= tau_img[gi]), minlength=self.n_img)
        n_fp = np.bincount(fi, weights=(self.fp_score[fm] >= tau_img[fi]), minlength=self.n_img)
        n_gt = self.n_gt(cls)
        return dict(n_gt=n_gt, n_missed=n_gt - matched, n_fp=n_fp)


def summarize(counts: dict[str, np.ndarray], rows: np.ndarray, per_class: bool) -> dict[str, float]:
    """Image-level loss, object-level miss rate and FPPI over the images selected by rows."""
    n_gt, miss, fp = counts["n_gt"][rows], counts["n_missed"][rows], counts["n_fp"][rows]
    has = n_gt > 0
    loss_i = np.where(has, miss / np.where(has, n_gt, 1), 0.0)
    # pooled: every image counts (no GT -> 0); per class: images containing the class
    image_loss = (loss_i[has].mean() if has.any() else np.nan) if per_class else (loss_i.mean() if len(rows) else np.nan)
    return dict(
        image_loss=float(image_loss),
        miss_rate=float(miss.sum() / n_gt.sum()) if n_gt.sum() > 0 else np.nan,
        fppi=float(fp.sum() / len(rows)) if len(rows) else np.nan,
        n_images=int(len(rows)),
        n_loss_images=int(has.sum()) if per_class else int(len(rows)),
        n_gt=int(n_gt.sum()),
    )


# ---------------------------------------------------------------- exact operating curves
def operating_curve(scores: np.ndarray, tp: np.ndarray, n_gt: int, n_img: int):
    """Exact (fppi, miss_rate) at every distinct score threshold, from the highest down.

    Starts at (0, 1): above the top score nothing is kept. At each distinct score the miss rate
    is the best reached at that threshold (all tied predictions kept together).
    Returns arrays fppi, miss_rate, threshold (threshold[0] = +inf), plus precision/recall.
    """
    order = np.argsort(-scores, kind="stable")
    s, t = scores[order], tp[order].astype(float)
    ctp, cfp = np.cumsum(t), np.cumsum(1 - t)
    last = np.r_[s[1:] != s[:-1], True] if len(s) else np.array([], bool)  # end of each tie block
    ctp, cfp, thr = ctp[last], cfp[last], s[last]
    recall = ctp / n_gt if n_gt > 0 else np.full(len(ctp), np.nan)
    precision = ctp / np.maximum(ctp + cfp, 1)
    fppi = np.r_[0.0, cfp / n_img]
    mr = np.r_[1.0, 1 - recall]
    return dict(fppi=fppi, miss_rate=mr, threshold=np.r_[np.inf, thr], precision=precision, recall=recall)


def mr_at_fppi(fppi: np.ndarray, mr: np.ndarray, refs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Miss rate at reference FPPI values, linear interpolation along the lower envelope.

    For each distinct FPPI the best (lowest) miss rate reached at it is used, then the curve is
    interpolated linearly in FPPI. References beyond the largest achievable FPPI are clamped to
    the curve's end (the miss rate at the conf floor) and flagged.
    """
    fp_u, inv = np.unique(fppi, return_inverse=True)
    mr_u = np.full(len(fp_u), np.inf)
    np.minimum.at(mr_u, inv, mr)
    clamped = refs > fp_u[-1]
    return np.interp(refs, fp_u, mr_u), clamped


def mr_at_fppi_step(fppi: np.ndarray, mr: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """Caltech (Dollar et al.) convention: miss rate at the largest FPPI <= reference."""
    out = []
    for f in refs:
        ok = fppi <= f
        out.append(mr[ok].min() if ok.any() else mr[0])
    return np.array(out)


def lamr(fppi, mr, cfg: Cfg):
    """Brief's log-average miss rate: arithmetic mean of the interpolated miss rate at nine FPPI
    values log-spaced in [0.01, 1]; plus the Caltech variant (geometric mean, stepwise)."""
    refs = np.geomspace(cfg.lamr.fppi_lo, cfg.lamr.fppi_hi, cfg.lamr.n_points)
    v, clamped = mr_at_fppi(fppi, mr, refs)
    step = mr_at_fppi_step(fppi, mr, refs)
    caltech = float(np.exp(np.mean(np.log(np.maximum(step, 1e-10)))))
    return float(np.mean(v)), caltech, int(clamped.sum())


def average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """VOC all-point interpolated AP (area under the monotone precision envelope).
    Callers pass curves for classes with ground truth; no predictions at all gives AP 0."""
    if len(recall) == 0:
        return 0.0
    if np.all(np.isnan(recall)):
        return np.nan
    r = np.r_[0.0, recall, recall[-1]]
    p = np.r_[1.0, precision, 0.0]
    p = np.maximum.accumulate(p[::-1])[::-1]
    i = np.nonzero(r[1:] != r[:-1])[0]
    return float(np.sum((r[i + 1] - r[i]) * p[i + 1]))


# ---------------------------------------------------------------- phase 3 driver
def bin_rows(c: Cache, b: str) -> np.ndarray:
    return np.arange(c.n_img) if b == ALL else np.nonzero(c.weather == b)[0]


def compute(cfg: Cfg) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    c = Cache(cfg)
    taus = tau_grid(cfg)
    t25, t50 = cfg.default_thresholds
    floor = cfg.detector.conf_floor
    curve_rows, summ_rows = [], []
    for b in bins(cfg) + [ALL]:
        rows = bin_rows(c, b)
        sel = np.zeros(c.n_img, bool)
        sel[rows] = True
        for cls in [ALL] + CANONICAL:
            per_class = cls != ALL
            # curves on the tau grid
            n_gt = c.n_gt(cls)[rows]
            matched = c.matched_grid(taus, cls)[rows]
            fp = c.fp_grid(taus, cls)[rows]
            has = n_gt > 0
            loss_i = np.where(has[:, None], (n_gt[:, None] - matched) / np.where(has, n_gt, 1)[:, None], 0.0)
            image_loss = (loss_i[has].mean(0) if has.any() else np.full(len(taus), np.nan)) if per_class else loss_i.mean(0)
            tot = n_gt.sum()
            miss_rate = 1 - matched.sum(0) / tot if tot > 0 else np.full(len(taus), np.nan)
            curve_rows.append(pd.DataFrame(dict(bin=b, cls=cls, tau=taus, miss_rate=miss_rate, image_loss=image_loss,
                                                fppi=fp.sum(0) / len(rows), n_images=len(rows),
                                                n_loss_images=int(has.sum()) if per_class else len(rows), n_gt=int(tot))))
            # exact operating points from every prediction in the bin
            pm = c.pred.image_id.map(c.idx).to_numpy()
            pmask = sel[pm] & (np.ones(len(pm), bool) if cls == ALL else (c.pred.eval_cls.to_numpy() == cls))
            oc = operating_curve(c.pred.score.to_numpy()[pmask], c.pred.tp.to_numpy()[pmask], int(tot), len(rows))

            def recall_at(t):
                cnt = summarize(c.at_thresholds(np.full(c.n_img, t), cls), rows, per_class)
                return 1 - cnt["miss_rate"] if tot > 0 else np.nan

            r_floor, r25, r50 = recall_at(floor), recall_at(t25), recall_at(t50)
            if tot > 0:
                lam, lam_cal, n_clamped = lamr(oc["fppi"], oc["miss_rate"], cfg)
                mr_budget, clamp_budget = mr_at_fppi(oc["fppi"], oc["miss_rate"], np.array(cfg.fppi_budgets))
            else:
                lam = lam_cal = np.nan
                n_clamped = 0
                mr_budget, clamp_budget = np.full(len(cfg.fppi_budgets), np.nan), np.zeros(len(cfg.fppi_budgets), bool)
            if cls == ALL:
                aps = []
                for k in CANONICAL:
                    km = sel[pm] & (c.pred.eval_cls.to_numpy() == k)
                    nk = int(c.n_gt(k)[rows].sum())
                    if nk > 0:
                        ock = operating_curve(c.pred.score.to_numpy()[km], c.pred.tp.to_numpy()[km], nk, len(rows))
                        aps.append(average_precision(ock["precision"], ock["recall"]))
                ap = float(np.mean(aps)) if aps else np.nan
            else:
                ap = average_precision(oc["precision"], oc["recall"]) if tot > 0 else np.nan
            row = dict(bin=b, cls=cls, n_images=len(rows), n_gt=int(tot), max_recall=r_floor,
                       **{f"recall_at_{t25}": r25, f"recall_at_{t50}": r50},
                       lamr=lam, lamr_caltech=lam_cal, lamr_n_clamped=n_clamped,
                       map50=ap, margin_points=100 * (r_floor - r25), margin_points_vs_0_5=100 * (r_floor - r50),
                       irrecoverable_points=100 * (1 - r_floor),
                       fppi_at_floor=float(oc["fppi"][-1]),
                       fppi_at_0_25=summarize(c.at_thresholds(np.full(c.n_img, t25), cls), rows, per_class)["fppi"])
            for f, v, cl in zip(cfg.fppi_budgets, mr_budget, clamp_budget):
                row[f"mr_at_fppi_{f}"] = float(v)
                row[f"mr_at_fppi_{f}_clamped"] = bool(cl)
            summ_rows.append(row)
    curves = pd.concat(curve_rows, ignore_index=True)
    summary = pd.DataFrame(summ_rows)
    return curves, summary, go_no_go(cfg, summary)


def go_no_go(cfg: Cfg, summary: pd.DataFrame) -> dict:
    g = cfg.go_no_go
    ref = g.reference_threshold
    out = dict(rule=f"GO if, in the worst weather bin (lowest maximum achievable recall) for each of "
                    f"{', '.join(g.classes)}, max recall - recall at tau={ref} >= {g.margin_points} points",
               classes={})
    ok = True
    for cls in g.classes:
        gb = g.get("bins") or [b for b in summary.bin.unique() if b != ALL]
        s = summary[(summary.cls == cls) & (summary.bin.isin(gb))].sort_values("max_recall")
        w = s.iloc[0]
        margin = 100 * (w.max_recall - w[f"recall_at_{ref}"])
        passed = bool(margin >= g.margin_points)
        ok &= passed
        out["classes"][cls] = dict(worst_bin=w.bin, n_gt=int(w.n_gt), max_recall=float(w.max_recall),
                                   recall_at_ref=float(w[f"recall_at_{ref}"]), margin_points=float(margin),
                                   margin_points_vs_0_5=float(w.margin_points_vs_0_5), passed=passed)
    out["verdict"] = "GO" if ok else "NO-GO"
    parts = [f"{cls}: worst bin {v['worst_bin']} (n_gt={v['n_gt']}), max recall {100 * v['max_recall']:.1f}% vs "
             f"{100 * v['recall_at_ref']:.1f}% at tau={ref} -> margin {v['margin_points']:.1f} pts"
             for cls, v in out["classes"].items()]
    out["line"] = f"VERDICT: {out['verdict']}. " + "; ".join(parts) + f" (threshold {g.margin_points} pts)."
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke)
    curves, summary, verdict = compute(cfg)
    curves.to_csv(results_path(cfg, "curves.csv"), index=False, float_format="%.6g")
    summary.to_csv(results_path(cfg, "summary_phase3.csv"), index=False, float_format="%.6g")
    write_json(results_path(cfg, "go_no_go.json"), verdict)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = ["bin", "cls", "n_images", "n_gt", "max_recall", "recall_at_0.25", "recall_at_0.5", "lamr", "map50",
            "margin_points", "margin_points_vs_0_5", "mr_at_fppi_0.1", "mr_at_fppi_1.0"]
    show = summary[summary.cls.isin([ALL, "person", "car"])][cols]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(verdict["line"])


if __name__ == "__main__":
    main()
