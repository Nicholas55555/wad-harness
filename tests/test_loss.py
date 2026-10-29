import numpy as np

from conftest import write_synthetic_cache
from wad.config import CANONICAL, tau_grid
from wad.curves import ALL, Cache


def test_loss_non_decreasing_for_every_cached_image(real_cfg):
    c = Cache(real_cfg)
    # the grid, plus every distinct score in the cache (the only places the loss can change)
    taus = np.unique(np.r_[tau_grid(real_cfg), c.gt_score[np.isfinite(c.gt_score)], 0.0, 1.0])
    for cls in [ALL] + CANONICAL:
        L = c.loss_grid(taus, cls)
        assert np.all(np.diff(L, axis=1) >= -1e-12), cls
        assert L.min() >= 0 and L.max() <= 1


def test_empty_ground_truth_images_have_zero_loss(real_cfg):
    c = Cache(real_cfg)
    taus = tau_grid(real_cfg)
    empty = c.n_gt(ALL) == 0
    assert empty.any(), "DAWN has one image without ground truth (fog/haze-012)"
    assert np.all(c.loss_grid(taus, ALL)[empty] == 0)


def test_loss_values_on_synthetic_cache(synthetic_cfg):
    write_synthetic_cache(
        synthetic_cfg.paths.cache,
        images=[dict(image_id="fog/a", weather="fog"), dict(image_id="fog/b", weather="fog"),
                dict(image_id="fog/empty", weather="fog")],
        gt=[dict(image_id="fog/a", cls="car", match_score=0.9), dict(image_id="fog/a", cls="car", match_score=0.2),
            dict(image_id="fog/a", cls="person", match_score=np.nan), dict(image_id="fog/b", cls="car", match_score=0.5)],
        pred=[dict(image_id="fog/a", cls="car", score=0.3, tp=False)],
    )
    c = Cache(synthetic_cfg)
    taus = np.array([0.1, 0.3, 0.6, 0.95])
    L = c.loss_grid(taus, ALL)
    np.testing.assert_allclose(L[c.idx["fog/a"]], [1 / 3, 2 / 3, 2 / 3, 1.0])
    np.testing.assert_allclose(L[c.idx["fog/b"]], [0, 0, 1, 1])
    np.testing.assert_allclose(L[c.idx["fog/empty"]], [0, 0, 0, 0])
    np.testing.assert_allclose(c.loss_grid(taus, "car")[c.idx["fog/a"]], [0, 0.5, 0.5, 1])
    np.testing.assert_allclose(c.fp_grid(taus, ALL)[c.idx["fog/a"]], [1, 1, 0, 0])
