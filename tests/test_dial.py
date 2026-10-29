"""The dial's two options: per-detection features and rule (wad/per_detection.py) and the shared entry
point (wad/dial.py)."""
import numpy as np
import pandas as pd
import pytest

from wad import dial
from wad import per_detection as pdet


def _det():
    # image a: a car, a weaker near-copy of it (likely duplicate) and a person; image b: one truck
    return pd.DataFrame(dict(image_id=["a", "a", "a", "b"], cls=["car", "car", "person", "truck"],
                             score=[0.9, 0.3, 0.05, 0.4],
                             x1=[10, 11, 100, 0], y1=[10, 10, 50, 0], x2=[50, 51, 120, 64], y2=[40, 40, 90, 36]))


def _sizes():
    return pd.DataFrame(dict(width=[200, 128], height=[100, 72]), index=["a", "b"])


def test_features_flag_the_weaker_copy_and_rank_within_images():
    f = pdet.features_raw(_det(), _sizes())
    assert list(np.round(np.expm1(f.log_rank), 6)) == [0, 1, 2, 0]
    assert f.max_iou_stronger_same_group.iloc[1] > 0.9 and f.n_stronger_overlapping.iloc[1] == 1
    assert f.max_iou_stronger_same_group.iloc[0] == 0 and f.max_iou_stronger_same_group.iloc[2] == 0  # person: other group
    assert np.isclose(f.x_center.iloc[3], 32 / 128) and np.isclose(f.y_bottom.iloc[3], 36 / 72)


def test_features_do_not_depend_on_the_order_of_images():
    d = _det()
    f1 = pdet.features_raw(d, _sizes())
    d2 = pd.concat([d.iloc[3:], d.iloc[:3]], ignore_index=True)
    f2 = pdet.features_raw(d2, _sizes())
    assert np.allclose(f1[pdet.FULL].to_numpy(), pd.concat([f2.iloc[1:], f2.iloc[:1]])[pdet.FULL].to_numpy())


def test_features_need_each_images_rows_together():
    d = _det().iloc[[0, 3, 1, 2]].reset_index(drop=True)
    with pytest.raises(ValueError):
        pdet.features_raw(d, _sizes())


def test_per_detection_rule_is_the_expected_cost_cut_and_nested_in_lambda():
    chance = np.linspace(0, 1, 101)
    for lam in (0.5, 1, 9, 99):
        k = pdet.keep(chance, lam)
        assert np.all(k == (chance >= 1 / (1 + lam)))  # keep iff lam * P(real) >= 1 - P(real)
    assert np.all(pdet.keep(chance, 30) >= pdet.keep(chance, 10))  # a higher lam keeps everything a lower one kept
    assert dial.cut(9) == 0.1


def test_calibration_of_perfect_chances_is_near_zero_error():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200000)
    _, ece = pdet.calibration(p, rng.uniform(size=p.size) < p)
    assert ece < 0.01


def test_dial_rejects_bad_modes_settings_and_inputs():
    with pytest.raises(ValueError):
        dial.keep(_det(), 10, mode="per image")
    with pytest.raises(ValueError):
        dial.keep(_det(), 10, mode="per detection")  # needs the image sizes
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            dial.cut(bad)
    d = _det()
    d.loc[0, "cls"] = "Car"
    with pytest.raises(ValueError):
        pdet.features_raw(d, _sizes())  # unknown class name
    with pytest.raises(ValueError):
        pdet.features_raw(_det(), _sizes().iloc[:1])  # an image without its size


def test_features_accept_sizes_with_an_image_id_column_and_empty_frames():
    f1 = pdet.features_raw(_det(), _sizes())
    f2 = pdet.features_raw(_det(), _sizes().rename_axis("image_id").reset_index())
    assert np.allclose(f1[pdet.FULL].to_numpy(), f2[pdet.FULL].to_numpy())
    assert len(pdet.features_raw(_det().iloc[:0], _sizes())) == 0


@pytest.mark.skipif(not dial.TABLE.exists(), reason="run python -m wad.one_threshold first")
def test_one_threshold_table_is_exactly_the_fit_at_any_lambda():
    from wad.one_threshold import train_curve, train_threshold

    t = dial._table()
    assert np.all(np.diff(t.threshold.to_numpy()) < 0) and np.all(np.diff(t.lam_from.to_numpy()) > 0)
    _, curve = train_curve()
    for lam in np.r_[np.geomspace(0.07, 3000, 60), 1.0, 10.0, 30.0, 8.4551, 13.72]:
        assert dial.threshold(lam) == train_threshold(curve, lam)
    k = dial.keep(_det(), 10, mode="one")
    assert list(k) == list(_det().score.to_numpy() >= dial.threshold(10))


@pytest.mark.skipif(not (pdet.MODEL.exists() and (pdet.THRESH_CACHE / "yolo_box" / "bdd.done").exists()),
                    reason="run python -m wad.yolo_box_features and python -m wad.per_detection first")
def test_per_detection_option_on_raw_detections_reproduces_the_graded_chances():
    from wad import yolo_box_features

    det_all = pd.read_parquet(pdet.REPO / pdet.EVAL_DET["BDD100K"])
    cached = pd.read_parquet(pdet.pred_path("bdd"))
    iid = cached.image_id.iloc[0]
    rows = np.flatnonzero(det_all.image_id.to_numpy() == iid)
    det = det_all.iloc[rows].reset_index(drop=True)  # raw detections of one frame, as a user would pass them
    sizes = pd.read_parquet(pdet.REPO / "cache/bdd/images.parquet").set_index("image_id")[["width", "height"]]
    bundle = pdet.load()
    box = None
    if bundle.get("features") == "yolo":
        b = yolo_box_features.load("bdd")
        box = {k: np.asarray(v[rows]) for k, v in b.items()}
    p = pdet.score(det, sizes, bundle, box)
    assert np.allclose(p, cached[cached.image_id == iid].p.to_numpy(), atol=1e-12)
    k10 = dial.keep(det, 10, mode="per detection", sizes=sizes, box=box)
    k100 = dial.keep(det, 100, mode="per detection", sizes=sizes, box=box)
    assert np.array_equal(k10, p >= 1 / 11) and np.all(k100 >= k10)


@pytest.mark.skipif(not pdet.model_path("context").exists(), reason="run python -m wad.per_detection --features context first")
def test_model_one_runs_on_new_detections_without_per_box_features():
    p = pdet.score(_det(), _sizes(), pdet.load(pdet.model_path("context")))
    assert len(p) == 4 and np.all((p >= 0) & (p <= 1)) and p[0] >= 1 / 11  # a lone 0.9 car is kept at lam 10


def test_box_columns_from_head_outputs():
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(0)
    n = 50
    cls = rng.uniform(0, 1, (n, 80)).astype(np.float16)
    logits = rng.normal(size=(n, 4, 16))
    dfl = (np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)).reshape(n, 64)
    neck = rng.normal(size=(n, 320))
    pca = PCA(n_components=pdet.N_PC, random_state=0).fit(rng.normal(size=(200, 320)))
    b = pdet.box_columns(cls, dfl, neck, np.zeros(n, np.uint8), pca)
    assert list(b.columns) == pdet.YOLO_COLS and len(b) == n and not b.isna().any().any()
    c32 = cls.astype(np.float32)
    assert np.allclose(b.p_car, c32[:, 2]) and np.allclose(b.p_truck, c32[:, 7]) and np.all(b.p_top_margin >= 0)
    assert np.all((b.dfl_ent_left >= 0) & (b.dfl_ent_left <= np.log(16) + 1e-6))
    sharp = np.zeros((1, 4, 16)); sharp[..., 3] = 1  # a certain box edge has zero entropy
    assert pdet.box_columns(cls[:1], sharp.reshape(1, 64), neck[:1], np.zeros(1), pca).dfl_ent_mean.iloc[0] < 1e-6


def test_model_two_needs_the_per_box_features():
    class _M:
        def predict_proba(self, X):
            return np.tile([0.5, 0.5], (len(X), 1))
    bundle = dict(model=_M(), cols=pdet.FULL + pdet.YOLO_COLS, features="yolo", pca=None)
    with pytest.raises(ValueError):
        pdet.score(_det(), _sizes(), bundle)
