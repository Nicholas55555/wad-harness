import numpy as np
import pytest

from wad.match import greedy_match, iou_matrix


def box(x, y, w=10, h=10):
    return [x, y, x + w, y + h]


def test_iou_known_values():
    a = np.array([box(0, 0)])
    b = np.array([box(0, 0), box(5, 0), box(20, 20), box(0, 0, 10, 20)])
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 50 / 150, 0.0, 0.5])


def test_higher_score_takes_the_ground_truth_lower_is_fp():
    gt = np.array([box(0, 0)])
    pred = np.array([box(1, 0), box(0, 0)])  # the second is the better box but scores lower
    tp, pred_gt, _, gt_score = greedy_match(pred, [0.9, 0.6], ["car", "car"], gt, ["car"])
    assert tp.tolist() == [True, False]
    assert pred_gt.tolist() == [0, -1]
    assert gt_score.tolist() == [0.9]


def test_high_score_wrong_class_box_does_not_steal_ground_truth():
    gt = np.array([box(0, 0)])
    pred = np.array([box(0, 0), box(0, 1)])
    tp, _, _, gt_score = greedy_match(pred, [0.95, 0.4], ["person", "car"], gt, ["car"])
    assert tp.tolist() == [False, True]  # the person box is a FP; the car box still matches
    assert gt_score.tolist() == [0.4]


def test_prediction_takes_highest_iou_unmatched_gt():
    gt = np.array([box(0, 0), box(3, 0)])
    pred = np.array([box(3, 0), box(0, 0)])
    tp, pred_gt, _, gt_score = greedy_match(pred, [0.9, 0.8], ["car"] * 2, gt, ["car"] * 2)
    assert pred_gt.tolist() == [1, 0]
    np.testing.assert_allclose(gt_score, [0.8, 0.9])


def test_taken_gt_falls_through_to_next_best():
    # Both predictions overlap GT0 best; the second must fall back to GT1 if IoU >= 0.5.
    gt = np.array([box(0, 0), box(4, 0)])
    pred = np.array([box(1, 0), box(1, 0)])
    tp, pred_gt, _, _ = greedy_match(pred, [0.9, 0.8], ["car"] * 2, gt, ["car"] * 2)
    assert pred_gt.tolist() == [0, 1]  # IoU(box(1,0), box(4,0)) = 70/130 >= 0.5


def test_iou_exactly_half_matches():
    gt = np.array([box(0, 0, 10, 20)])
    pred = np.array([box(0, 0, 10, 10)])
    tp, *_ = greedy_match(pred, [0.5], ["car"], gt, ["car"], iou_thr=0.5)
    assert tp.tolist() == [True]


def test_empty_inputs():
    tp, pred_gt, _, gt_score = greedy_match(np.zeros((0, 4)), [], [], np.array([box(0, 0)]), ["car"])
    assert len(tp) == 0 and np.isnan(gt_score).all()
    tp, _, _, gt_score = greedy_match(np.array([box(0, 0)]), [0.3], ["car"], np.zeros((0, 4)), [])
    assert tp.tolist() == [False] and len(gt_score) == 0


def random_image(rng, n_gt=8, n_pred=25, classes=("car", "person", "truck")):
    centers = rng.uniform(0, 200, size=(n_gt, 2))
    gt = np.c_[centers, centers + rng.uniform(10, 40, size=(n_gt, 2))]
    gt_cls = rng.choice(classes, size=n_gt)
    # predictions: jittered copies of GT boxes (some wrong class) plus clutter
    src = rng.integers(0, n_gt, size=n_pred)
    jit = rng.normal(0, 6, size=(n_pred, 4))
    pred = gt[src] + jit
    pred[:, 2:] = np.maximum(pred[:, 2:], pred[:, :2] + 1)
    clutter = rng.random(n_pred) < 0.3
    pred[clutter] = np.c_[rng.uniform(0, 200, (clutter.sum(), 2)), np.zeros((clutter.sum(), 2))]
    pred[clutter, 2:] = pred[clutter, :2] + rng.uniform(5, 40, (clutter.sum(), 2))
    pred_cls = np.where(rng.random(n_pred) < 0.8, gt_cls[src], rng.choice(classes, size=n_pred))
    scores = np.round(rng.random(n_pred), 2)  # rounding forces score ties
    return pred, scores, pred_cls, gt, gt_cls


@pytest.mark.parametrize("seed", range(40))
def test_truncate_and_rematch_equals_filtering_single_pass(seed):
    rng = np.random.default_rng(seed)
    pred, scores, pred_cls, gt, gt_cls = random_image(rng)
    tp, _, _, gt_score = greedy_match(pred, scores, pred_cls, gt, gt_cls)
    for tau in np.r_[0.0, np.unique(scores), 0.333, 1.01]:
        keep = scores >= tau
        tp_t, _, _, gt_score_t = greedy_match(pred[keep], scores[keep], pred_cls[keep], gt, gt_cls)
        # same TP/FP decisions on the kept predictions
        assert tp_t.tolist() == tp[keep].tolist()
        # same set of matched GT, at the same scores
        matched_full = np.where(gt_score >= tau, gt_score, np.nan)
        np.testing.assert_array_equal(np.isnan(gt_score_t), np.isnan(matched_full))
        np.testing.assert_allclose(gt_score_t[~np.isnan(gt_score_t)], matched_full[~np.isnan(matched_full)])
