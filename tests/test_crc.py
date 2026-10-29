import numpy as np
import pandas as pd
import pytest

from wad.config import load_config, tau_grid
from wad.crc import crc_bound, fit_threshold, make_splits, split_once

FLOOR = 0.001


def step_losses(scores, taus):
    """One GT per image, matched at `score` (NaN = never): L_i(tau) = 1[score < tau]."""
    s = np.nan_to_num(np.asarray(scores, float), nan=-np.inf)
    return (s[:, None] < taus[None, :]).astype(float)


@pytest.mark.parametrize("seed", range(25))
def test_rule_returns_largest_feasible_tau(seed):
    rng = np.random.default_rng(seed)
    taus = np.linspace(0.001, 0.99, 200)
    n = int(rng.integers(5, 300))
    # images with several GT boxes, a mix of matched scores and never-matched objects
    losses = np.zeros((n, len(taus)))
    for i in range(n):
        k = int(rng.integers(0, 6))
        if k == 0:
            continue
        s = np.where(rng.random(k) < 0.15, -np.inf, rng.beta(2, 2, k))
        losses[i] = (s[:, None] < taus[None, :]).mean(0)
    alpha = float(rng.uniform(0.05, 0.6))
    f = fit_threshold(losses, taus, alpha, FLOOR)
    bound = crc_bound(losses.mean(0), n)
    feasible = taus[losses.sum(0) + 1 <= alpha * (n + 1) + 1e-9]
    if len(feasible):
        assert not f["infeasible"]
        assert f["tau_hat"] == feasible.max()  # brute-force max over the whole grid
        # calibration-half inequality holds exactly at tau_hat
        j = int(np.nonzero(taus == f["tau_hat"])[0][0])
        assert n / (n + 1) * losses[:, j].mean() + 1 / (n + 1) <= alpha + 1e-9
        assert f["cal_bound"] <= alpha + 1e-9
        # and the next grid point up is infeasible (nothing larger qualifies)
        if j + 1 < len(taus):
            assert bound[j + 1] > alpha - 1e-9
    else:
        assert f["infeasible"] and f["tau_hat"] == FLOOR


def test_known_tau_hat_on_synthetic_miss_curve():
    # nine images, one object each, matched at scores 0.1, 0.2, ..., 0.9.
    # bound(tau) = (#{s_i < tau} + 1) / 10, so alpha = 0.3 allows two misses: tau_hat = 0.3 exactly.
    scores = np.round(np.arange(1, 10) / 10, 10)
    taus = np.round(np.linspace(0.05, 0.95, 19), 10)  # contains 0.1, 0.15, ..., 0.95
    for alpha, expected in [(0.3, 0.3), (0.5, 0.5), (0.25, 0.2), (1.0, 0.95)]:
        f = fit_threshold(step_losses(scores, taus), taus, alpha, FLOOR)
        assert f["tau_hat"] == pytest.approx(expected), alpha
        assert not f["infeasible"]


def test_exact_tie_at_alpha_is_feasible():
    # n = 79 images, 15 missed at tau = 0.5: (15 + 1) / 80 = 0.2 exactly, but in floating point
    # n/(n+1)*R + 1/(n+1) evaluates just above 0.2. The rule must still accept tau = 0.5.
    taus = np.array([0.25, 0.5, 0.95])  # at 0.95 every object is missed
    losses = step_losses(np.r_[np.full(15, 0.4), np.full(64, 0.9)], taus)
    assert not (79 / 80 * losses[:, 1].mean() + 1 / 80 <= 0.2)  # the naive float comparison fails
    f = fit_threshold(losses, taus, 0.2, FLOOR)
    assert f["tau_hat"] == 0.5 and not f["infeasible"]


def test_infeasible_when_reserve_alone_exceeds_alpha():
    taus = np.linspace(0.001, 0.99, 50)
    losses = np.zeros((9, len(taus)))  # perfect detector, but 1/(n+1) = 0.1 > alpha
    f = fit_threshold(losses, taus, 0.05, FLOOR)
    assert f["infeasible"] and f["tau_hat"] == FLOOR


def test_infeasible_when_loss_does_not_vanish_at_floor():
    taus = np.linspace(0.001, 0.99, 50)
    losses = step_losses([np.nan] * 30 + [0.9] * 70, taus)  # 30% never proposed
    f = fit_threshold(losses, taus, 0.2, FLOOR)
    assert f["infeasible"] and f["tau_hat"] == FLOOR
    f = fit_threshold(losses, taus, 0.35, FLOOR)
    assert not f["infeasible"]


def test_no_calibration_images_is_infeasible():
    f = fit_threshold(np.zeros((0, 10)), np.linspace(0.001, 0.9, 10), 0.1, FLOOR)
    assert f["infeasible"] and f["n_cal"] == 0


def test_fits_on_real_cache_satisfy_calibration_inequality(real_cfg):
    fits = pd.read_csv(real_cfg.paths.results / "crc_fits.csv")
    ok = fits[~fits.infeasible]
    assert (ok.cal_bound <= ok.alpha + 1e-9).all()
    bad = fits[fits.infeasible]
    assert (bad.tau_hat == real_cfg.detector.conf_floor).all()
    assert (bad.cal_bound > bad.alpha).all()


def synthetic_images():
    rows = []
    for b, n in [("fog", 31), ("rain", 20), ("sand", 17)]:
        rows += [dict(image_id=f"{b}/{i:03d}", weather=b, group=len(rows) + i) for i in range(n)]
    im = pd.DataFrame(rows)
    im["group"] = np.arange(len(im))
    # three near-duplicate groups, one spanning two weather bins
    im.loc[im.image_id.isin(["fog/000", "fog/001", "fog/002"]), "group"] = 1000
    im.loc[im.image_id.isin(["rain/005", "rain/006"]), "group"] = 1001
    im.loc[im.image_id.isin(["fog/010", "sand/003"]), "group"] = 1002
    return im


def test_splits_are_disjoint_stratified_and_keep_groups_together():
    im = synthetic_images()
    for seed in range(20):
        cal = split_once(im, 0.5, seed)
        for b, g in im.assign(cal=cal).groupby("weather"):
            assert g.cal.sum() <= int(np.floor(len(g) * 0.5))
            assert g.cal.sum() >= int(np.floor(len(g) * 0.5)) - 3  # groups can leave a slot or two
        for gid, g in im.assign(cal=cal).groupby("group"):
            assert g.cal.nunique() == 1, f"group {gid} split across halves"
    splits = make_splits(load_config(), im)
    for seed, s in splits.groupby("seed"):
        assert s.image_id.is_unique and set(s.half) == {"cal", "test"}
    # different seeds give different splits; the same seed gives the same split
    assert not np.array_equal(split_once(im, 0.5, 0), split_once(im, 0.5, 1))
    assert np.array_equal(split_once(im, 0.5, 3), split_once(im, 0.5, 3))
