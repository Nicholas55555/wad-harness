"""O3 version C: synthesis targets, data split, model geometry and real-image preprocessing.

Everything runs on the CPU (the synthesis is device-agnostic), so these tests also run while the GPU
is busy or absent.
"""
import math

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from wad import vis_estimator as ve  # noqa: E402

CPU = "cpu"


def test_far_depth_is_smooth_and_monotone():
    d = torch.linspace(1, 80, 400).repeat(5, 1)
    out = ve.far_depth(d)[2]
    assert torch.all(out[1:] >= out[:-1] - 1e-3)
    assert float(ve.far_depth(torch.full((20, 20), 20.0))[10, 10]) == pytest.approx(20.0, rel=1e-4)
    assert 400 < float(ve.far_depth(torch.full((20, 20), 70.0))[10, 10]) < 600


def _scene():
    rng = np.random.default_rng(0)
    frame = torch.as_tensor(rng.integers(0, 255, (ve.H, ve.W, 3), dtype=np.uint8))
    yy = torch.arange(ve.H, dtype=torch.float32)[:, None].expand(ve.H, ve.W)
    depth = (5 + 70 * yy.flip(0) / ve.H).half()  # far at the top, near at the bottom
    boxes = torch.tensor([[100, 120, 180, 200], [300, 150, 340, 190], [10, 100, 13, 103]], dtype=torch.float32)
    return frame, depth, boxes


@pytest.mark.parametrize("version", [2, 3])
def test_synthetic_targets_follow_their_definitions(version):
    syn = ve.Synth(device=CPU, version=version)
    g = torch.Generator().manual_seed(0)
    frame, depth, boxes = _scene()
    seen_spray = seen_dark = False
    for _ in range(60):
        img, t = syn.sample(frame, depth, boxes, g)
        assert img.shape == (3, ve.H, ve.W) and float(img.min()) >= 0 and float(img.max()) <= 1
        assert t["map"].shape == ve.GRID
        mor_map = 10 ** t["map"]
        road = float(torch.quantile(mor_map[ve.GRID[0] - 5:].flatten(), 0.10))
        v_light = min(max(2000 * 10 ** (t["log_k"] / 2), ve.V_MIN), ve.V_MAX)
        assert t["log_veff"] == pytest.approx(math.log10(min(road, v_light)), abs=1e-4)
        # spray only removes visibility: no cell sees farther than the global veil allows
        assert float(t["map"].max()) <= t["log_mor"] + 1e-4 or t["log_mor"] >= math.log10(ve.V_MAX) - 1e-6
        assert t["log_k"] <= 0
        seen_spray |= float(t["map"].min()) < t["log_mor"] - 0.05
        seen_dark |= t["log_k"] < 0
    assert seen_spray and seen_dark


def test_synthesis_is_reproducible_with_the_same_seed():
    syn = ve.Synth(device=CPU)
    frame, depth, boxes = _scene()
    out = []
    for _ in range(2):
        g = torch.Generator().manual_seed(12345)
        out.append([syn.sample(frame, depth, boxes, g) for _ in range(8)])
    for (i1, t1), (i2, t2) in zip(*out):
        assert torch.equal(i1, i2) and t1["log_veff"] == t2["log_veff"] and torch.equal(t1["map"], t2["map"])


def test_flip_mirrors_image_and_map_together():
    syn = ve.Synth(device=CPU)
    frame, depth, boxes = _scene()
    frames = frame.numpy()[None].repeat(16, 0)
    frames[:, :, : ve.W // 2] //= 4  # left half dark: the image's orientation is visible
    depth_ = depth.numpy()[None].repeat(16, 0)
    bx = np.zeros((16, ve.MAX_VEH, 4), np.float32)
    bx[:, 0] = [20, 150, 120, 240]  # one large vehicle on the left: spray lowers the left map cells
    nb = np.ones(16, np.int16)
    g = torch.Generator().manual_seed(3)
    x, y, m = ve.make_batch(syn, frames, depth_, bx, nb, np.arange(16), g)
    left_dark = x[:, :, :, : x.shape[-1] // 2].mean((1, 2, 3)) < x[:, :, :, x.shape[-1] // 2:].mean((1, 2, 3))
    assert 0 < int(left_dark.sum()) < 16  # some samples flipped, some not
    sprayed = (m[:, -2:, :].amin((1, 2)) < m[:, -2:, :].amax((1, 2)) - 0.05)
    assert int(sprayed.sum()) >= 1  # seed 3 draws spray in 2 of 16 samples
    for i in np.nonzero(sprayed.numpy())[0]:
        lowest_col = int(m[i, -2:, :].amin(0).argmin())
        assert (lowest_col < ve.GRID[1] // 2) == bool(left_dark[i])


def test_model_feature_grid_matches_the_map_target():
    model = ve.build_model(pretrained=False).eval()
    with torch.no_grad():
        f = model.body(torch.zeros(1, 3, *ve.INPUT))
        v, mp = model(torch.zeros(2, 3, *ve.INPUT))
    assert tuple(f.shape[-2:]) == ve.GRID
    assert v.shape == (2, 3) and tuple(mp.shape) == (2, *ve.GRID)


@pytest.mark.parametrize("size,expect", [((1280, 720), (1280, 720)), ((1600, 1000), (1600, 900)),
                                         ((1000, 1000), (1000, 562)), ((3000, 600), (1067, 600))])
def test_real_images_are_centre_cropped_to_16_9(size, expect):
    from PIL import Image

    im = Image.new("RGB", size)
    seen = {}
    orig = Image.Image.resize

    def spy(self, s, *a, **k):
        seen["size"] = self.size
        return orig(self, s, *a, **k)

    Image.Image.resize = spy
    try:
        arr = ve.to_working(im)
    finally:
        Image.Image.resize = orig
    assert arr.shape == (ve.H, ve.W, 3)
    assert abs(seen["size"][0] - expect[0]) <= 1 and abs(seen["size"][1] - expect[1]) <= 1


@pytest.mark.skipif(not (ve.CACHE / "bases.csv").exists(), reason="training cache not built")
def test_base_frames_are_clear_daytime_train_frames_disjoint_from_evaluation():
    bases = pd.read_csv(ve.CACHE / "bases.csv")
    again = ve.base_names()
    assert list(again.name) == list(bases.name) and list(again.split) == list(bases.split)
    attr = pd.read_parquet(ve.REPO / "cache" / "o3" / "train_attributes.parquet").set_index("name")
    a = attr.loc[bases.name]
    assert (a.weather == "clear").all() and (a.timeofday == "daytime").all()
    assert 0.07 < (bases.split == "val").mean() < 0.13
    evaluation = set()
    for p in ("cache/images.parquet", "cache/bdd/images.parquet"):
        im = pd.read_parquet(ve.REPO / p)
        evaluation |= set(im.path.map(lambda s: s.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]))
    assert not set(bases.name) & evaluation


def test_v3_night_is_dark_and_has_a_physical_light_level():
    syn = ve.Synth(device=CPU, version=3)
    g = torch.Generator().manual_seed(4)
    frame, depth, boxes = _scene()
    J = (frame.float() / 255).permute(2, 0, 1) ** 2.2
    dm = depth.float()
    for _ in range(10):
        L, k = syn._night(J, dm, ve.far_depth(dm), boxes, g)
        img = L.clamp(0, 1) ** (1 / 2.2)
        assert 3e-4 <= k <= 0.2
        assert float(img.mean()) < 0.35  # night frames stay dark after auto-exposure


@pytest.mark.skipif(not (torch.cuda.is_available() and ve.WEIGHTS.exists()), reason="needs the GPU and trained weights")
def test_predict_images_keeps_row_order_and_writes_per_version(tmp_path, monkeypatch):
    im = pd.read_parquet(ve.REPO / "cache" / "images.parquet")
    im = im[im.keep].sort_values("image_id").head(3).reset_index(drop=True)
    monkeypatch.setattr(ve, "paths", lambda version: dict(weights=ve.WEIGHTS, results=tmp_path, preds=tmp_path))
    d = ve.predict_images(im, "toy", batch=2, version=3)
    assert list(d.image_id) == list(im.image_id) and list(d.weather) == list(im.weather)
    assert (tmp_path / "pred_toy.parquet").exists()
    assert np.allclose(d.h_v3, -d.log_veff)
