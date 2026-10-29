"""The chosen method: one threshold per dial setting (wad/one_threshold.py)."""
import numpy as np
import pandas as pd

from wad import one_threshold as ot


def _toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.uniform(0.001, 1.0, n)
    tp = (rng.uniform(size=n) < score ** 1.5).astype(float)  # higher score, more often real
    return pd.DataFrame(dict(score=score, tp=tp))


def test_threshold_minimises_the_dial_cost():
    m = _toy()
    _, curve = ot.train_curve(m)
    s, tp = m.score.to_numpy(), m.tp.to_numpy()
    for lam in (0.5, 1, 3, 10, 30):
        t = ot.train_threshold(curve, lam)
        cost = lambda x: lam * tp[s < x].sum() + (1 - tp[s >= x]).sum()  # noqa: E731
        brute = min(cost(x) for x in np.r_[s, 2.0])
        assert cost(t) <= brute + 1e-9


def test_threshold_falls_as_the_dial_accepts_more_false_alarms():
    _, curve = ot.train_curve(_toy())
    taus = [ot.train_threshold(curve, lam) for lam in (0.3, 1, 3, 10, 30, 100)]
    assert all(a >= b for a, b in zip(taus, taus[1:]))


def test_threshold_sits_where_the_chance_of_being_real_is_one_in_one_plus_lambda():
    iso, curve = ot.train_curve(_toy(20000))
    for lam in (1, 3, 10):
        t = ot.train_threshold(curve, lam)
        assert abs(iso.predict([t])[0] - 1 / (1 + lam)) < 0.06


def test_grid_counts_match_direct_counts():
    m = _toy(500)
    s, tp = m.score.to_numpy(), m.tp.to_numpy()
    r, f = ot.grid_counts(s, tp)
    for k in (0, 300, 1000, 1700, 2000):
        # a box counts at GRID[k] when its grid bin is >= k, i.e. score >= GRID[k] up to float ties
        assert r[k] == tp[s >= ot.GRID[k]].sum()
        assert f[k] == (1 - tp[s >= ot.GRID[k]]).sum()


def test_weighted_grid_counts_equal_repeating_images():
    m = _toy(300)
    s, tp = m.score.to_numpy(), m.tp.to_numpy()
    w = np.random.default_rng(1).integers(0, 3, len(s)).astype(float)
    r, f = ot.grid_counts(s, tp, w)
    rep = np.repeat(np.arange(len(s)), w.astype(int))
    r2, f2 = ot.grid_counts(s[rep], tp[rep])
    assert np.allclose(r, r2) and np.allclose(f, f2)


def test_switch_point_table_equals_the_argmin_at_every_lambda():
    m = _toy(5000, seed=3)
    _, curve = ot.train_curve(m)
    table = ot.breakpoints(curve)
    for lam in np.geomspace(0.05, 500, 300):
        assert ot.threshold_from_table(table, lam) == ot.train_threshold(curve, lam)
