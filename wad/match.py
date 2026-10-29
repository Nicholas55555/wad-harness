"""Greedy IoU matching, one pass per image -> per-GT match score, per-prediction TP/FP.

Why one pass serves every threshold: predictions are visited in descending score order, and
the decision for a prediction depends only on the predictions visited before it (which ground
truth they already took). Keeping only predictions with score >= tau keeps a prefix of that
order (all ties at the cut are kept or dropped together), so greedy matching on the truncated
list makes exactly the same decisions as the first part of the full pass. Hence:
  - a GT is matched at threshold tau  <=>  its match_score >= tau
  - a prediction is a TP at tau       <=>  tp and score >= tau
Tested on synthetic boxes in tests/test_match.py.

Run: python -m wad.match [--smoke]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from wad.config import Cfg, cache_path, det_path, load_config
from wad.data import load_gt, load_images


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (n,4) and b (m,4), xyxy in continuous pixel coordinates."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def score_order(scores: np.ndarray) -> np.ndarray:
    """Descending score; ties broken by position (stable), so the order is deterministic."""
    return np.argsort(-np.asarray(scores, dtype=float), kind="stable")


def greedy_match(pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls, iou_thr: float = 0.5):
    """Class-aware greedy matching by score.

    For each prediction in descending score order, take the still-unmatched GT of the same
    class with the highest IoU; match if that IoU >= iou_thr, else the prediction is a FP.

    Returns (tp[n] bool, pred_gt[n] int (-1 if FP), pred_iou[n] float, gt_score[m] float (NaN if
    never matched)), all indexed like the inputs.
    """
    pred_scores = np.asarray(pred_scores, dtype=float)
    pred_cls = np.asarray(pred_cls)
    gt_cls = np.asarray(gt_cls)
    n, m = len(pred_scores), len(gt_cls)
    tp = np.zeros(n, dtype=bool)
    pred_gt = np.full(n, -1, dtype=int)
    pred_iou = np.zeros(n, dtype=float)
    gt_score = np.full(m, np.nan)
    if n == 0 or m == 0:
        return tp, pred_gt, pred_iou, gt_score
    iou = iou_matrix(pred_boxes, gt_boxes)
    iou[pred_cls[:, None] != gt_cls[None, :]] = -1.0  # class-aware: other classes never match
    taken = np.zeros(m, dtype=bool)
    for p in score_order(pred_scores):
        row = np.where(taken, -1.0, iou[p])
        g = int(np.argmax(row))  # first index wins IoU ties
        if row[g] >= iou_thr:
            taken[g] = True
            tp[p] = True
            pred_gt[p] = g
            pred_iou[p] = row[g]
            gt_score[g] = pred_scores[p]
        elif row[g] > 0:
            pred_iou[p] = row[g]
    return tp, pred_gt, pred_iou, gt_score


def match_all(cfg: Cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    images = load_images(cfg)
    gt = load_gt(cfg)
    det = pd.read_parquet(det_path(cfg, "detections.parquet"))
    det = det[det.image_id.isin(set(images.image_id))]
    weather = images.set_index("image_id").weather
    gt_by = {k: v for k, v in gt.groupby("image_id")}
    det_by = {k: v for k, v in det.groupby("image_id")}
    gt_rows, pred_rows = [], []
    box = ["x1", "y1", "x2", "y2"]
    # match_groups (run 2 grouped loss): a prediction may match any ground truth of its group;
    # outputs keep the original classes.
    groups = cfg.get("match_groups") or {}
    to_group = {c: g for g, members in groups.items() for c in members}
    key = (lambda a: np.array([to_group.get(x, x) for x in a], dtype=object)) if to_group else (lambda a: a)
    for image_id in images.image_id:
        g = gt_by.get(image_id, gt.iloc[:0])
        d = det_by.get(image_id, det.iloc[:0])
        tp, pred_gt, pred_iou, gt_score = greedy_match(
            d[box].to_numpy(), d.score.to_numpy(), key(d.cls.to_numpy()),
            g[box].to_numpy(), key(g.cls.to_numpy()), cfg.match_iou,
        )
        gt_ids = g.gt_id.to_numpy()
        pred_gt_id = np.full(len(tp), -1, dtype=int)
        pred_gt_id[tp] = gt_ids[pred_gt[tp]]
        gt_rows.append(pd.DataFrame(dict(gt_id=gt_ids, image_id=image_id, weather=weather[image_id],
                                         cls=g.cls.to_numpy(), match_score=gt_score)))
        pred_rows.append(pd.DataFrame(dict(image_id=image_id, weather=weather[image_id], cls=d.cls.to_numpy(),
                                           score=d.score.to_numpy(), tp=tp, iou=pred_iou, gt_id=pred_gt_id)))
    mg = pd.concat(gt_rows, ignore_index=True)
    mp = pd.concat(pred_rows, ignore_index=True)
    mg.to_parquet(cache_path(cfg, "matches_gt.parquet"), index=False)
    mp.to_parquet(cache_path(cfg, "matches_pred.parquet"), index=False)
    return mg, mp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke)
    mg, mp = match_all(cfg)
    print(f"{len(mg)} GT boxes, {mg.match_score.notna().mean():.3f} matched at the conf floor; "
          f"{len(mp)} predictions, {mp.tp.sum()} TP")


if __name__ == "__main__":
    main()
