"""Phase 4 ablations (b) per-condition recalibration and (d) other detectors: helpers and paths."""
import numpy as np

from wad.dial_recalibration import ece, iso, near


def test_ece_zero_when_calibrated_in_every_bin():
    p = np.repeat([0.003, 0.15, 0.6], 1000)
    y = np.concatenate([np.r_[np.ones(3), np.zeros(997)], np.r_[np.ones(150), np.zeros(850)], np.r_[np.ones(600), np.zeros(400)]])
    assert ece(p, y) < 1e-12


def test_ece_is_the_weighted_gap():
    p = np.full(100, 0.25)
    y = np.r_[np.ones(50), np.zeros(50)]
    assert np.isclose(ece(p, y), 0.25)


def test_isotonic_map_is_monotone_and_fixes_understated_chances():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.3, 20_000)
    y = rng.uniform(size=p.size) < 2 * p  # the model under-states: true chance is twice p
    q = iso(p, y).predict(p)
    order = np.argsort(p)
    assert (np.diff(q[order]) >= -1e-12).all()
    assert ece(q, y) < ece(p, y) / 5


def test_near_counts_the_window_around_the_cut():
    p = np.array([0.05, 0.08, 0.09, 0.12, 0.2])
    y = np.array([0, 1, 1, 0, 1])
    n, mean_chance, share = near(p, y, 10)  # cut 1/11, window [0.0606, 0.1364]
    assert n == 3 and np.isclose(mean_chance, (0.08 + 0.09 + 0.12) / 3) and np.isclose(share, 2 / 3)


def test_box_feature_paths_keep_the_adopted_detector_where_it_was():
    from wad import yolo_box_features as ybf

    assert ybf.out_dir() == ybf.OUT and ybf.detections_file("train") == ybf.SETS["train"][1]
    assert ybf.detections_file("bdd") == ybf.SETS["bdd"][1] and ybf.detections_file("dawn") == ybf.SETS["dawn"][1]
    assert ybf.out_dir("v8s_640") == ybf.OUT / "v8s_640"


def test_detector_ablation_never_writes_into_run2_or_adopted_caches():
    from wad.config import REPO
    from wad.detector_ablation import CACHE, DETS, configs

    for d in DETS:
        cf = configs(d)
        for key, v in cf.items():
            assert CACHE in v.paths.cache.parents, (d, key)
            assert v.detector.name == DETS[d]["name"] and v.detector.imgsz == DETS[d]["imgsz"]
            assert "v8x_1280" not in str(v.paths.detections)
        assert cf["DAWN"].paths.detections == REPO / "cache" / "det" / d  # never run 1's CPU pass (cache/)
