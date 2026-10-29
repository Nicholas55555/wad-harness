import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wad.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def real_cfg():
    """Config over the real cache; tests that need it skip when the cache has not been built."""
    cfg = load_config()
    needed = ["images.parquet", "matches_gt.parquet", "matches_pred.parquet", "detections.parquet"]
    if not all((cfg.paths.cache / n).exists() for n in needed):
        pytest.skip("real cache not built (run python scripts/run_all.py first)")
    return cfg


def write_synthetic_cache(cache_dir: Path, images: list[dict], gt: list[dict], pred: list[dict]) -> None:
    """images: image_id, weather; gt: image_id, cls, match_score; pred: image_id, cls, score, tp."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    im = pd.DataFrame(images)
    im["path"] = ""
    im["width"] = im["height"] = 100
    im["keep"] = True
    im["group"] = np.arange(len(im))
    g = pd.DataFrame(gt, columns=["image_id", "cls", "match_score"])
    im["n_gt"] = im.image_id.map(g.groupby("image_id").size()).fillna(0).astype(int)
    im.to_parquet(cache_dir / "images.parquet", index=False)
    w = im.set_index("image_id").weather
    g = g.assign(gt_id=np.arange(len(g)), weather=g.image_id.map(w))
    g.to_parquet(cache_dir / "matches_gt.parquet", index=False)
    p = pd.DataFrame(pred, columns=["image_id", "cls", "score", "tp"])
    p = p.assign(weather=p.image_id.map(w), iou=0.0, gt_id=-1)
    p["tp"] = p.tp.astype(bool)
    p.to_parquet(cache_dir / "matches_pred.parquet", index=False)


@pytest.fixture
def synthetic_cfg(tmp_path):
    cfg = load_config()
    for k in ("cache", "data_cache", "detections"):
        cfg.paths[k] = tmp_path / "cache"
    cfg.paths["results"] = tmp_path / "results"
    return cfg
