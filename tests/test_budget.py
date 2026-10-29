"""Threshold budget and graded response helpers."""
import numpy as np

from wad import graded_response as gr
from wad import threshold_budget as tb


class _StubCache:
    """matched_grid / fp_grid like wad.curves.Cache: counts of items with score >= tau, and
    (as the real helper does) only correct for ascending thresholds."""

    def __init__(self, gt_scores, fp_scores):
        self.g, self.f = gt_scores, fp_scores

    def _grid(self, s, taus):
        assert np.all(np.diff(taus) > 0), "thresholds must be ascending"
        return np.array([[np.sum(row >= t) for t in taus] for row in s], float)

    def matched_grid(self, taus, cls):
        return self._grid(self.g, taus)

    def fp_grid(self, taus, cls):
        return self._grid(self.f, taus)

    def n_gt(self, cls):
        return np.array([len(r) for r in self.g], float)


def test_band_counts_grow_as_the_threshold_falls():
    rng = np.random.default_rng(0)
    c = _StubCache([rng.random(20) for _ in range(5)], [rng.random(30) ** 3 for _ in range(5)])
    taus = np.geomspace(0.001, 0.99, 200)
    e, M, F, n = tb.per_image(c, taus)
    assert np.all(np.diff(e) < 0)  # edges run from high to low threshold
    assert np.all(np.diff(M, axis=1) >= 0) and np.all(np.diff(F, axis=1) >= 0)
    st = tb.band_stats(M, F, n, np.arange(5))
    assert np.all(st["d_recall"] >= 0) and st["recall"][-1] <= 1


def test_stop_threshold_is_the_lowest_meeting_the_false_stop_cap():
    taus = np.linspace(0.01, 0.99, 99)
    fp = np.tile(np.linspace(10, 0, 99), (4, 1))  # false positives per image fall with tau
    t = gr.tau_stop(fp, 4, taus, cap=1.0)
    j = int(np.argmin(np.abs(taus - t)))
    assert fp[:, j].mean() <= 1.0 and fp[:, j - 1].mean() > 1.0
    assert gr.tau_stop(fp, 0, taus, 1.0) == taus[-1]


def test_pairwise_iou_is_symmetric_with_unit_diagonal():
    from wad.detection_rescore import iou_rows

    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], float)
    m = iou_rows(b)
    assert np.allclose(m, m.T) and np.allclose(np.diag(m), 1)
    assert abs(m[0, 1] - 25 / 175) < 1e-9 and m[0, 2] == 0


def test_confidence_maps_max_sum_and_noisy_or():
    from wad.summed_map import maps

    b = np.array([[10, 10, 50, 50], [14, 12, 52, 48], [200, 100, 220, 120]], float)
    m = maps(b, np.array([0.3, 0.2, 0.05]), 45, 80, 0.25)
    assert m["max (= ordinary threshold)"][5, 5] == 0.3
    assert abs(m["sum of confidences"][5, 5] - 0.5) < 1e-12
    assert abs(m["noisy-OR"][5, 5] - (1 - 0.7 * 0.8)) < 1e-12
    assert abs(m["sum of confidences"][27, 52] - 0.05) < 1e-12 and m["max (= ordinary threshold)"][0, 79] == 0
