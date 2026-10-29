"""EXPLORATORY: is there learnable structure at the level of individual detections?
STATUS (2026-10-14): exploratory model comparisons (baseline, ablation, box combinations). The
supported per-detection option of the dial is wad/per_detection.py (same model and features).

The per-image oracle threshold's edge over one threshold is ~90% luck on BDD100K
(wad/threshold_noise.py): which particular detections at a given score are real. A per-image
threshold cannot resolve that; a per-detection judgement might. Here a gradient-boosted classifier
estimates, for every detection, the chance it is a real object from the detector's own outputs:
its score and class, box size / shape / position in the frame, its rank in the image, its overlap
with stronger detections (likely duplicates), the confidence-weighted overlap ("support") it gets from
every other box (summing boxes over all thresholds weights each by its score, so this is the
summed-area idea in one pass) and how crowded the image is. The decision rule keeps a
detection iff that chance >= 1 / (1 + lam), which is the expected-cost-optimal rule for the dial lam
when the chances are right. Baseline: the same model on score and class only (what one threshold
per class can use). Trained on BDD100K train detections; evaluated on BDD100K val and DAWN.

Ablation: the same context without the overlap-with-stronger-detections features (a second box on
an already-found object counts as a false positive but is harmless for driving: is the gain only
duplicate cleanup?). Permutation importance of the full model on BDD100K val.

Also two ways of combining a box with the same-group boxes that overlap it, each weighted by the
overlap (IoU): a cooperative, sigmoid-like noisy-OR (agreeing boxes add evidence; "is something
there?") and a competitive softmax share among overlapping boxes ("is this THE box?"); on their own
with one threshold, combined, as a learned sigmoid on top (logistic regression with a handful of
weights) and as extra inputs to the boosted model.

Run: python -m wad.detection_rescore   -> results/threshold/rescore_{detections,decision,importance,single_scores}.csv,
     cache/thresh/rescore_pred_{bdd,dawn}.parquet (the context model's probability per detection, in Cache order)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from wad.config import REPO, tau_grid
from wad.curves import ALL, Cache
from wad.config import THRESH_RESULTS as RESULTS
from wad.per_detection import CLS, FULL, GRP, iou_rows  # noqa: F401  (shared with the dial's per-detection option)
from wad.per_detection import labelled_features as features

LAMS = (1, 3, 10, 30, 100)
BASE = ["logit", "cls"]
DUP = ["max_iou_stronger_same_group", "n_stronger_overlapping", "max_iou_stronger_any", "support_same_group", "support_any"]
# combining a box with the same-group boxes that overlap it, weighted by the overlap w = IoU:
#   cooperative (sigmoid-like, nothing has to add up to 1): noisy-OR  1 - (1 - s_i) * prod_j (1 - w_ij s_j)
#   competitive (softmax where boxes overlap): share  e^z_i / (e^z_i + sum_j w_ij e^z_j), z = logit(score)
COMBINE = ["softmax_share", "noisy_or"]
NO_DUP = [x for x in FULL if x not in DUP]  # ablation: is the gain only duplicate-box cleanup?


def sets():
    from wad.run2 import bdd_cfg, dawn_cfg, load_run2
    from wad.thresh_net import cfg as train_cfg

    r2 = load_run2()
    return [("train", train_cfg(), "cache/thresh/det/v8x_1280/detections.parquet"),
            ("BDD100K", bdd_cfg(r2, "v8x_1280", "group"), "cache/bdd/det/v8x_1280/detections.parquet"),
            ("DAWN", dawn_cfg(r2, "v8x_1280", "group", "clean"), "cache/det/v8x_1280/detections.parquet")]


def decision(c: Cache, f: pd.DataFrame, prob: dict, taus: np.ndarray) -> list[dict]:
    """Cost per image (lam * misses + FPs) of each rule, relative to the best single raw-score threshold."""
    n_gt = c.n_gt(ALL)
    L = n_gt[:, None] - c.matched_grid(taus, ALL)
    F = c.fp_grid(taus, ALL)
    tp = f.tp.to_numpy().astype(bool)
    img = f.img.to_numpy()
    rows = []
    for lam in LAMS:
        cv = lam * L + F
        single = cv.mean(0).min()
        row = dict(lam=lam, best_single=1.0, oracle_image_threshold=cv.min(1).mean() / single)
        for name, pr in prob.items():
            keep = pr >= 1.0 / (1.0 + lam)
            tpk = np.bincount(img, weights=keep & tp, minlength=c.n_img)
            fpk = np.bincount(img, weights=keep & ~tp, minlength=c.n_img)
            row[name] = float((lam * (n_gt - tpk) + fpk).mean() / single)
        rows.append(row)
    return rows


def single_threshold(c: Cache, f: pd.DataFrame, scores: dict) -> list[dict]:
    """Best single threshold on each per-detection score (exact, in-sample), cost relative to the raw
    score's. Mean cost per image is lam * (objects - kept real) + kept false over all images, which
    depends only on totals, so one sort per score gives every cut."""
    n_obj = c.n_gt(ALL).sum()
    tp = f.tp.to_numpy().astype(bool)
    rows = []
    cum = {}
    for name, q in scores.items():
        o = np.argsort(-q, kind="stable")
        qs = q[o]
        last = np.r_[qs[1:] != qs[:-1], True]  # cut only between distinct values
        cum[name] = (np.r_[0, np.cumsum(tp[o])[last]], np.r_[0, np.cumsum(~tp[o])[last]])
    for lam in LAMS:
        best = {k: float((lam * (n_obj - t) + fp).min()) for k, (t, fp) in cum.items()}
        rows.append(dict(lam=lam, **{k: v / best["raw score"] for k, v in best.items()}))
    return rows


class _Logit:
    """Logistic regression on [logit score, logit noisy-OR, log softmax share] + class one-hot."""

    def __init__(self, lr):
        self.lr = lr
        self.coef_ = lr.coef_

    @staticmethod
    def design(X: np.ndarray) -> np.ndarray:
        s, nor, share, cls = X[:, 0], X[:, 1], X[:, 2], X[:, 3].astype(int)
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
        return np.column_stack([lg(s), lg(nor), np.log(np.clip(share, 1e-6, 1)), np.eye(6)[cls]])

    def predict_proba(self, X):
        return self.lr.predict_proba(self.design(X))


def sigmoid_on_top(tr: pd.DataFrame):
    """The user's 'sigmoid across boxes on top of YOLO, weighted, plus softmax where applicable': a
    logistic model with a handful of weights (own score, cooperative evidence, competitive share, class)."""
    from sklearn.linear_model import LogisticRegression

    cols = ["score", "noisy_or", "softmax_share", "cls"]
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(_Logit.design(tr[cols].to_numpy()), tr.tp.to_numpy())
    m = _Logit(lr)
    return m, ["logit score", "logit noisy-OR", "log softmax share"] + cols[3:]


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss, roc_auc_score

    t0 = time.perf_counter()
    data = {}
    for name, v, dp in sets():
        c = Cache(v)
        data[name] = (c, v, features(c, dp))
        print(f"{name}: {len(data[name][2])} detections, {time.perf_counter() - t0:.0f} s", flush=True)
    tr = data["train"][2]
    models = {}
    sig = sigmoid_on_top(tr)
    models["learned sigmoid on top (score, noisy-OR, softmax share, class)"] = sig
    print(f"logistic weights: {dict(zip(sig[1], np.round(sig[0].coef_[0][:3], 3)))}", flush=True)
    for tag, cols in (("score + class (baseline)", BASE), ("score + detection context", FULL),
                      ("context without overlap features", NO_DUP),
                      ("context + softmax share + noisy-OR", FULL + COMBINE)):
        mdl = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, max_leaf_nodes=63, early_stopping=True,
                                             validation_fraction=0.1, random_state=0,
                                             categorical_features=[cols.index("cls")])
        mdl.fit(tr[cols].to_numpy(), tr.tp.to_numpy())
        models[tag] = (mdl, cols)
        print(f"trained {tag}: {mdl.n_iter_} iterations, {time.perf_counter() - t0:.0f} s", flush=True)
    det_rows, dec_rows, single_rows = [], [], []
    for ds in ("BDD100K", "DAWN"):
        c, v, f = data[ds]
        taus = tau_grid(v)
        prob = {tag: mdl.predict_proba(f[["score", "noisy_or", "softmax_share", "cls"]].to_numpy() if isinstance(mdl, _Logit)
                                       else f[cols].to_numpy())[:, 1] for tag, (mdl, cols) in models.items()}
        # keep the full context model's probability for every detection (used by wad.dial_demo)
        out = pd.DataFrame(dict(image_id=c.image_ids[f.img.to_numpy()], score=f.score.to_numpy(), tp=f.tp.to_numpy(),
                                p_context=prob["score + detection context"]))
        out.to_parquet(REPO / "cache" / "thresh" / f"rescore_pred_{'bdd' if ds == 'BDD100K' else 'dawn'}.parquet", index=False)
        band = (f.score >= 0.05) & (f.score < 0.5)
        for tag, pr in prob.items():
            det_rows.append(dict(dataset=ds, model=tag, log_loss=log_loss(f.tp, pr), auc=roc_auc_score(f.tp, pr),
                                 auc_score_band_005_05=roc_auc_score(f.tp[band], pr[band]),
                                 mean_predicted=float(pr.mean()), share_real=float(f.tp.mean())))
        dec_rows += [dict(dataset=ds, **r) for r in decision(c, f, prob, taus)]
        # the combinations on their own, as scores with one threshold (best in-sample, like the raw score)
        alone = {"raw score": f.score.to_numpy(),
                 "competitive: score x weighted softmax share": f.score.to_numpy() * f.softmax_share.to_numpy(),
                 "cooperative: weighted noisy-OR": f.noisy_or.to_numpy(),
                 "both: weighted noisy-OR x softmax share": f.noisy_or.to_numpy() * f.softmax_share.to_numpy()}
        for r in single_threshold(c, f, alone):
            single_rows.append(dict(dataset=ds, **r))
    D, C = pd.DataFrame(det_rows), pd.DataFrame(dec_rows)
    S = pd.DataFrame(single_rows)
    S.to_csv(RESULTS / "rescore_single_scores.csv", index=False, float_format="%.4g")
    print(S.round(3).to_string(index=False))
    # which features carry the full model: permutation importance (log-loss increase) on BDD100K val,
    # decision-relevant detections (score 0.05-0.5), 200,000 sampled
    from sklearn.inspection import permutation_importance

    f = data["BDD100K"][2]
    band = f[(f.score >= 0.05) & (f.score < 0.5)]
    band = band.sample(min(200000, len(band)), random_state=0)
    mdl, cols = models["score + detection context"]
    pi = permutation_importance(mdl, band[cols].to_numpy(), band.tp.to_numpy(), scoring="neg_log_loss", n_repeats=3,
                                random_state=0)
    imp = pd.DataFrame(dict(feature=cols, log_loss_increase=pi.importances_mean)).sort_values("log_loss_increase", ascending=False)
    imp.to_csv(RESULTS / "rescore_importance.csv", index=False, float_format="%.5g")
    print(imp.round(5).to_string(index=False))
    RESULTS.mkdir(parents=True, exist_ok=True)
    D.to_csv(RESULTS / "rescore_detections.csv", index=False, float_format="%.4g")
    C.to_csv(RESULTS / "rescore_decision.csv", index=False, float_format="%.4g")
    return D, C


if __name__ == "__main__":
    D, C = main()
    pd.set_option("display.width", 220)
    print(D.round(4).to_string(index=False))
    print(C.round(3).to_string(index=False))
