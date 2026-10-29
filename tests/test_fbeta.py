"""The dial in F-beta units (wad/fbeta_dial.py, wad/dial.py keep_f)."""
import numpy as np
import pytest

from wad import dial
from wad import fbeta_dial as fd


def test_fbeta_from_counts_matches_precision_recall_form():
    tp, fp, n = 60.0, 40.0, 80.0
    p, r = tp / (tp + fp), tp / n
    for b in (0.5, 1, 2):
        assert abs(fd.fbeta(tp, fp, n, b) - (1 + b * b) * p * r / (b * b * p + r)) < 1e-12
    assert fd.fbeta(0, 5, 10, 1) == 0


def test_best_cut_is_the_brute_force_maximum_and_falls_with_beta():
    rng = np.random.default_rng(0)
    v = np.round(rng.uniform(0, 1, 3000), 3)  # ties on purpose
    tp = rng.uniform(size=v.size) < v ** 2
    n_obj = tp.sum() + 150  # some objects never detected
    cuts = []
    for b in (0.25, 0.5, 1, 2, 4):
        cut, f = fd.Ranked(v, tp).best(n_obj, b)
        brute = max(fd.fbeta((tp & (v >= c)).sum(), (~tp & (v >= c)).sum(), n_obj, b) for c in np.unique(v))
        assert abs(f - brute) < 1e-12 and abs(fd.fbeta((tp & (v >= cut)).sum(), (~tp & (v >= cut)).sum(), n_obj, b) - f) < 1e-12
        cuts.append(cut)
    assert all(a >= b for a, b in zip(cuts, cuts[1:]))  # more weight on recall -> a lower cut


def test_lookup_rejects_out_of_range_beta():
    import pandas as pd

    tab = pd.DataFrame(dict(beta=[0.25, 1.0, 4.0], threshold=[0.5, 0.2, 0.05]))
    assert fd.lookup(tab, 1.02).threshold == 0.2
    with pytest.raises(ValueError):
        fd.lookup(tab, 5)


@pytest.mark.skipif(not dial.F_TABLE.exists(), reason="run python -m wad.fbeta_dial first")
def test_fbeta_dial_table_is_monotone_and_used_by_keep_f():
    t = dial._f_table()
    assert np.all(np.diff(t.threshold.to_numpy()) <= 0) and np.all(np.diff(t.cut.to_numpy()) <= 0)
    # every beta is a lam setting (lam = (1 + beta^2) / F* - 1): positive and rising with beta
    assert np.all(t.det_lam_equivalent > 0) and np.all(np.diff(t.det_lam_equivalent.to_numpy()) > 0)
    import pandas as pd

    det = pd.DataFrame(dict(image_id=["a", "a"], cls=["car", "person"], score=[0.6, 0.02], x1=[0, 50], y1=[0, 0],
                            x2=[40, 60], y2=[30, 30]))
    assert list(dial.keep_f(det, 1.0)) == list(det.score.to_numpy() >= dial.f_setting(1.0).threshold)
