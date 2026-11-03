import numpy as np
import pandas as pd
import pytest

from wad.crc import fit_threshold
from wad.crc_lambda import chance_cache
from wad.curves import ALL, Cache
from wad.dial_ablation import subset_images


def toy_cache() -> Cache:
    """Two images: 'a' has two objects (matched at 0.9 and 0.3) and one false box (0.5); 'b' has one object no
    box ever matched and one false box (0.2)."""
    c = Cache.__new__(Cache)
    c.image_ids = np.array(["a", "b"])
    c.idx = {"a": 0, "b": 1}
    c.n_img = 2
    c.weather = np.array(["x", "x"])
    c.gt = pd.DataFrame(dict(gt_id=[10, 11, 12], image_id=["a", "a", "b"], cls=["car", "car", "person"],
                             match_score=[0.9, 0.3, np.nan]))
    c.gt_img, c.gt_cls = np.array([0, 0, 1]), c.gt.cls.to_numpy()
    c.gt_score = c.gt.match_score.fillna(-np.inf).to_numpy()
    c.pred = pd.DataFrame(dict(image_id=["a", "a", "a", "b"], score=[0.9, 0.5, 0.3, 0.2], tp=[True, False, True, False],
                               gt_id=[10, -1, 11, -1], cls=["car", "car", "car", "person"]))
    tp = c.pred.tp.to_numpy()
    c.fp_img, c.fp_cls = np.array([0, 1]), c.pred.cls.to_numpy()[~tp]
    c.fp_score = c.pred.score.to_numpy()[~tp]
    return c


def test_chance_equal_to_score_reproduces_the_baseline_exactly():
    c = toy_cache()
    cp = chance_cache(c, c.pred.score.to_numpy())
    grid = np.linspace(0.001, 0.99, 50)
    assert np.array_equal(cp.loss_grid(grid, ALL), c.loss_grid(grid, ALL))
    assert np.array_equal(cp.fp_grid(grid, ALL), c.fp_grid(grid, ALL))
    cut = np.array([0.4, 0.1])
    for k, v in c.at_thresholds(cut, ALL).items():
        assert np.array_equal(cp.at_thresholds(cut, ALL)[k], v)


def test_chances_go_to_the_matched_objects_and_false_boxes():
    c = toy_cache()
    p = np.array([0.8, 0.05, 0.6, 0.02])
    cp = chance_cache(c, p)
    assert cp.gt_score.tolist() == [0.8, 0.6, -np.inf]  # object 12 was never matched
    assert cp.fp_score.tolist() == [0.05, 0.02]
    # at cut 0.7 image a keeps one of its two objects; image b misses its only object
    assert cp.loss_grid(np.array([0.7]), ALL)[:, 0].tolist() == [0.5, 1.0]
    # the original cache is untouched
    assert c.gt_score.tolist()[:2] == [0.9, 0.3]


def test_loss_is_monotone_in_the_cut_so_crc_applies():
    rng = np.random.default_rng(0)
    c = toy_cache()
    cp = chance_cache(c, rng.random(4))
    grid = np.sort(rng.random(30))
    L = cp.loss_grid(grid, ALL)
    assert (np.diff(L, axis=1) >= 0).all()  # non-decreasing in the cut = non-increasing in lambda
    f = fit_threshold(L, grid, 0.9, 0.001)
    assert f["tau_hat"] in grid or f["infeasible"]


def test_chance_cache_rejects_misaligned_chances():
    c = toy_cache()
    with pytest.raises(ValueError):
        chance_cache(c, np.array([0.5, 0.5]))


@pytest.mark.parametrize("frac,seed", [(0.1, 0), (0.25, 1), (0.5, 2)])
def test_training_subsets_keep_near_duplicate_groups_whole(frac, seed):
    rng = np.random.default_rng(seed)
    groups = pd.Series(rng.integers(0, 400, 1000))
    keep = subset_images(groups, frac, seed)
    assert frac <= keep.mean() <= frac + 0.05
    for g, m in pd.Series(keep).groupby(groups):
        assert m.all() or not m.any()
