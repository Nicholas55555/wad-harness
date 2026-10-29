"""Learned per-image threshold (PREREG_threshold.md): monotone outputs, the dial, CRC over the dial."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wad import thresh_eval as te  # noqa: E402
from wad import thresh_net as tn  # noqa: E402


def test_network_outputs_are_monotone_in_the_threshold():
    model = tn.build_model(40, pretrained=False).eval()
    with torch.no_grad():
        miss, fp = model(torch.randn(3, 3, 64, 112))
    assert miss.shape == fp.shape == (3, 40)
    assert (miss.diff(dim=1) >= 0).all() and (fp.diff(dim=1) <= 0).all() and (miss > 0).all() and (fp > 0).all()


def _curves(n=50, k=40, seed=0):
    rng = np.random.default_rng(seed)
    pm = np.cumsum(rng.random((n, k)), axis=1)  # rising misses
    pf = np.cumsum(rng.random((n, k))[:, ::-1], axis=1)[:, ::-1] * 10  # falling false positives
    return pm, pf


def test_dial_lowers_thresholds_as_misses_get_more_expensive():
    pm, pf = _curves()
    idx = te.dial_index(pm, pf, te.LAMS)
    assert (np.diff(idx, axis=1) <= 0).all()
    assert (idx % tn.K_STEP == 0).all() and idx.max() <= 39 * tn.K_STEP


def test_dial_breaks_ties_towards_the_higher_threshold():
    pm = np.zeros((1, 40))  # no misses predicted anywhere: the highest threshold is optimal
    pf = np.zeros((1, 40))
    assert te.dial_index(pm, pf, np.array([1.0]))[0, 0] == 39 * tn.K_STEP


def test_crc_over_the_dial_picks_the_smallest_feasible_lam():
    L = np.tile(np.linspace(0.5, 0.0, 10), (99, 1))  # loss falls with lam
    j, inf = te.crc_lam(L, 0.2)
    ok = L.sum(0) + 1 <= 0.2 * 100 + 1e-9
    assert not inf and j == int(np.nonzero(ok)[0].min())
    j, inf = te.crc_lam(np.ones((10, 5)), 0.2)
    assert inf and j == 4


@pytest.mark.parametrize("size", [(1280, 720), (1600, 1000), (1000, 1000), (3000, 600)])
def test_inputs_are_cropped_to_16_9(size):
    from PIL import Image

    assert tn.to_input(Image.new("RGB", size)).shape == (tn.IN_H, tn.IN_W, 3)


def test_validation_split_keeps_near_duplicate_groups_together():
    groups = np.repeat(np.arange(500), 3)
    v = tn.split_train_val(groups)
    for g in range(500):
        assert len(set(v[groups == g])) == 1
    assert 0.07 < v.mean() < 0.13


def test_yolo_feature_pooling_shape_and_values():
    from wad import thresh_yolo as ty

    maps = [torch.ones(1, 3, 8, 16), torch.full((1, 5, 4, 8), 2.0), torch.arange(2 * 4 * 8, dtype=torch.float32).view(1, 2, 4, 8)]
    f = ty.pool(maps)
    assert f.shape == (2 * (3 + 5 + 2) + 2 * ty.GRID[0] * ty.GRID[1],)
    # layout per level: channel means, then channel maxima; last level: means at 16-17, maxima 31 and 63 at 18-19
    assert f[0] == 1 and f[6] == 2 and f[18] == 31 and f[19] == 63


def test_decision_heads_lower_thresholds_as_misses_get_more_expensive(tmp_path, monkeypatch):
    import math

    from wad import thresh_yolo as ty

    head = ty.build_head(6, 40)
    w = tmp_path / "h.pt"
    torch.save(dict(state_dict=head.state_dict(), mu=np.zeros(6, np.float32), sd=np.ones(6, np.float32), lmu=0.0,
                    lsd=1.0, n_in=6, k=40, taus=np.linspace(0.001, 0.9, 40)), w)
    monkeypatch.setattr(ty, "WEIGHTS", w)
    idx = ty.policy(np.random.default_rng(0).normal(size=(30, 6)).astype(np.float16), np.geomspace(0.1, 1000, 25), w)
    assert idx.shape == (30, 25) and (np.diff(idx, axis=1) <= 0).all() and 0 <= idx.min() and idx.max() <= 39
    assert math.isfinite(float(idx.mean()))


def test_structured_head_curves_and_dial(tmp_path, monkeypatch):
    from wad import thresh_yolo as ty

    head = ty.CurveHead.build(6, 40).eval()
    with torch.no_grad():
        m, f = head(torch.randn(5, 6))
    assert (m.diff(dim=1) >= 0).all() and (f.diff(dim=1) <= 0).all()
    w = tmp_path / "c.pt"
    torch.save(dict(state_dict=head.state_dict(), mu=np.zeros(6, np.float32), sd=np.ones(6, np.float32), lmu=0.0,
                    lsd=1.0, n_in=6, k=40, taus=np.linspace(0.001, 0.9, 40), head="curves"), w)
    idx = ty.policy(np.random.default_rng(1).normal(size=(20, 6)).astype(np.float16), np.geomspace(0.1, 1000, 30), w)
    assert idx.shape == (20, 30) and (np.diff(idx, axis=1) <= 0).all() and 0 <= idx.min() and idx.max() <= 39
