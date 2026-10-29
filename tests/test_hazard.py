import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import wad.hazard as hazard

FORBIDDEN_MODULES = {"wad.detect", "wad.match", "wad.curves", "wad.crc", "wad.hazard_crc", "wad.extras",
                     "ultralytics", "torch"}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names |= {f"{node.module}.{a.name}" for a in node.names}
    return names


@pytest.mark.parametrize("module", ["hazard.py", "hazard_cnn.py", "visibility.py", "vis_estimator.py", "vis_dev.py"])
def test_hazard_imports_no_detection_code(module):
    path = Path(hazard.__file__).parent / module
    mods = imported_modules(path)
    bad = {m for m in mods for f in FORBIDDEN_MODULES - {"torch"} if m == f or m.startswith(f + ".")}
    if module in ("hazard.py", "visibility.py"):
        bad |= {m for m in mods if m == "torch" or m.startswith("torch.")}
    assert not bad, f"wad/{module} imports detection code: {bad}"
    src = path.read_text(encoding="utf-8")
    assert "detections.parquet" not in src and "matches_" not in src and "ultralytics" not in src


def test_image_features_takes_only_an_image_path():
    params = list(inspect.signature(hazard.image_features).parameters)
    assert params == ["path"]


def test_features_on_synthetic_images(tmp_path):
    rng = np.random.default_rng(0)
    sharp = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    flat = np.full((240, 320, 3), 200, dtype=np.uint8)  # bright, uniform: fog-like
    Image.fromarray(sharp).save(tmp_path / "sharp.png")
    Image.fromarray(flat).save(tmp_path / "flat.png")
    fs, ff = hazard.image_features(str(tmp_path / "sharp.png")), hazard.image_features(str(tmp_path / "flat.png"))
    assert ff["rms_contrast"] == pytest.approx(0) and ff["edge_energy"] == pytest.approx(0)
    assert ff["saturation"] == pytest.approx(0) and ff["dark_channel"] == pytest.approx(200 / 255)
    assert fs["rms_contrast"] > 10 * max(ff["rms_contrast"], 1e-3) and fs["edge_energy"] > 1e-3
    # mirror invariance: DAWN's EXIF mirror tags cannot move h
    Image.fromarray(sharp[:, ::-1]).save(tmp_path / "mirror.png")
    fm = hazard.image_features(str(tmp_path / "mirror.png"))
    for k in hazard.FEATURES:
        assert fm[k] == pytest.approx(fs[k], rel=1e-9, abs=1e-12)


def test_pca_sign_convention():
    rng = np.random.default_rng(1)
    z = rng.normal(size=500)
    x = pd.DataFrame(dict(dark_channel=z + 0.1 * rng.normal(size=500), rms_contrast=-z, edge_energy=-z,
                          saturation=-z + 0.1 * rng.normal(size=500)))
    pca = hazard.fit_pca(x)
    h = hazard.apply_pca(x, pca)
    assert np.corrcoef(h, x.rms_contrast)[0, 1] < 0 and np.corrcoef(h, x.edge_energy)[0, 1] < 0
