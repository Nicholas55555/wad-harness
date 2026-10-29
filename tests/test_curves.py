import numpy as np
import pandas as pd
import pytest

from conftest import write_synthetic_cache
from wad.config import CANONICAL
from wad.curves import ALL, Cache, average_precision, lamr, mr_at_fppi, operating_curve, summarize


def test_trivial_values_at_floor_and_one_on_real_cache(real_cfg):
    c = Cache(real_cfg)
    floor = real_cfg.detector.conf_floor
    assert c.pred.score.min() >= floor and c.pred.score.max() < 1.0
    rows = np.arange(c.n_img)
    for cls in [ALL] + CANONICAL:
        per_class = cls != ALL
        top = summarize(c.at_thresholds(np.full(c.n_img, 1.0), cls), rows, per_class)
        assert top["fppi"] == 0.0 and top["miss_rate"] == 1.0
        assert top["image_loss"] == pytest.approx(1.0 if per_class else (c.n_gt(ALL) > 0).mean())
        bot = summarize(c.at_thresholds(np.full(c.n_img, floor), cls), rows, per_class)
        n_fp = ((~c.pred.tp) & ((c.pred.cls == cls) if per_class else True)).sum()
        n_tp = (c.pred.tp & ((c.pred.cls == cls) if per_class else True)).sum()
        assert bot["fppi"] == pytest.approx(n_fp / c.n_img)
        assert bot["miss_rate"] == pytest.approx(1 - n_tp / c.n_gt(cls).sum())


def test_curves_csv_trivial_values(real_cfg):
    cur = pd.read_csv(real_cfg.paths.results / "curves.csv")
    s = pd.read_csv(real_cfg.paths.results / "summary_phase3.csv")
    first = cur[cur.tau == cur.tau.min()].set_index(["bin", "cls"])
    assert cur.tau.min() == pytest.approx(real_cfg.detector.conf_floor)
    for r in s.itertuples():
        if r.n_gt > 0:
            assert 1 - first.loc[(r.bin, r.cls), "miss_rate"] == pytest.approx(r.max_recall)
            assert first.loc[(r.bin, r.cls), "fppi"] == pytest.approx(r.fppi_at_floor)
    # every curve is monotone: miss rate up, FPPI down as tau rises
    for _, g in cur.groupby(["bin", "cls"]):
        g = g.sort_values("tau")
        assert (np.diff(g.miss_rate.dropna()) >= -1e-12).all()
        assert (np.diff(g.fppi) <= 1e-12).all()


def test_synthetic_counts_at_floor_and_one(synthetic_cfg):
    write_synthetic_cache(
        synthetic_cfg.paths.cache,
        images=[dict(image_id="rain/a", weather="rain"), dict(image_id="rain/b", weather="rain")],
        gt=[dict(image_id="rain/a", cls="car", match_score=0.8), dict(image_id="rain/b", cls="car", match_score=np.nan)],
        pred=[dict(image_id="rain/a", cls="car", score=0.8, tp=True), dict(image_id="rain/a", cls="car", score=0.002, tp=False),
              dict(image_id="rain/b", cls="person", score=0.4, tp=False)],
    )
    c = Cache(synthetic_cfg)
    rows = np.arange(2)
    lo = summarize(c.at_thresholds(np.full(2, 0.001), ALL), rows, False)
    hi = summarize(c.at_thresholds(np.full(2, 1.0), ALL), rows, False)
    assert lo == dict(image_loss=0.5, miss_rate=0.5, fppi=1.0, n_images=2, n_loss_images=2, n_gt=2)
    assert hi["fppi"] == 0 and hi["miss_rate"] == 1 and hi["image_loss"] == 1


def test_perfect_detector_has_ap_one_and_zero_lamr(real_cfg):
    oc = operating_curve(np.array([0.9, 0.8, 0.7]), np.array([True, True, True]), n_gt=3, n_img=3)
    assert average_precision(oc["precision"], oc["recall"]) == pytest.approx(1.0)
    lam, _, n_clamped = lamr(oc["fppi"], oc["miss_rate"], real_cfg)
    assert lam == pytest.approx(0.0) and n_clamped == 9  # never reaches FPPI 0.01: clamped to its end


def test_ap_and_interpolation_known_values():
    # TP, FP, TP over 2 GT: precision 1, 1/2, 2/3 at recall 1/2, 1/2, 1 -> AP = 0.5*1 + 0.5*2/3
    oc = operating_curve(np.array([0.9, 0.8, 0.7]), np.array([True, False, True]), n_gt=2, n_img=1)
    assert average_precision(oc["precision"], oc["recall"]) == pytest.approx(0.5 + 1 / 3)
    v, clamped = mr_at_fppi(oc["fppi"], oc["miss_rate"], np.array([0.0, 0.5, 1.0, 2.0]))
    np.testing.assert_allclose(v, [0.5, 0.25, 0.0, 0.0])
    assert clamped.tolist() == [False, False, False, True]
