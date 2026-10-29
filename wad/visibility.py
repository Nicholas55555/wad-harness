"""EXPLORATORY: an objective, pixel-only visibility hazard score (not pre-registered).

h = -(z(log edge energy) + z(RMS contrast)) / 2, where z standardises with the mean and sd of the
BDD100K *train* frames used for O3 (disjoint from every evaluation image). Higher h = less contrast
and less fine detail visible: fog, night, downpour veiling, spray. It is a measurement of the frame,
not a weather label, and uses no detector output (features come from wad.hazard.image_features).

Known blind spot: night rain, where glare, wet-road reflections and droplet bokeh add contrast and
detail, so h rates those frames clearer than they are.

Run: python -m wad.visibility   -> cache/visibility/features.parquet, cache/visibility/h.parquet
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from wad.hazard import image_features
from wad.imageio import load_rgb

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "cache" / "visibility"


def features(path: str) -> dict:
    f = image_features(path)
    im = load_rgb(path, "apply")
    s = 640 / max(im.size)
    g = np.asarray(im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BILINEAR).convert("L"), float) / 255
    f["brightness"] = float(g.mean())
    return f


def _table(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    with ProcessPoolExecutor(8) as ex:
        out = pd.DataFrame(list(ex.map(features, [r[3] for r in rows], chunksize=32)))
    for k, col in enumerate(["dataset", "image_id", "weather"]):
        out.insert(k, col, [r[k] for r in rows])
    return out


def build() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, p in [("DAWN", "cache/images.parquet"), ("BDD100K", "cache/bdd/images.parquet")]:
        im = pd.read_parquet(REPO / p)
        rows += [(ds, r.image_id, r.weather, r.path) for r in im[im.keep].itertuples()]
    sub = pd.read_csv(REPO / "cache" / "o3" / "train_subset.csv")
    rows += [("BDD100K-train", f"train/{n}", c, f"data/BDD100K/images/100k/train/{n}.jpg") for n, c in zip(sub.name, sub.cls)]
    f = _table(rows)
    f.to_parquet(OUT / "features.parquet", index=False)
    tr = f[f.dataset == "BDD100K-train"]
    le = np.log(tr.edge_energy + 1e-6)
    stats = dict(mu_log_edge=float(le.mean()), sd_log_edge=float(le.std()), mu_contrast=float(tr.rms_contrast.mean()),
                 sd_contrast=float(tr.rms_contrast.std()), n_train=int(len(tr)))
    (OUT / "standardisation.json").write_text(json.dumps(stats, indent=1))
    ev = f[f.dataset != "BDD100K-train"].copy()
    ev["h"] = score(ev, stats)
    ev[["dataset", "image_id", "weather", "h"]].to_parquet(OUT / "h.parquet", index=False)
    return ev


def score(f: pd.DataFrame, stats: dict) -> np.ndarray:
    ze = (np.log(f.edge_energy + 1e-6) - stats["mu_log_edge"]) / stats["sd_log_edge"]
    zc = (f.rms_contrast - stats["mu_contrast"]) / stats["sd_contrast"]
    return (-(ze + zc) / 2).to_numpy()


def load_h() -> pd.Series:
    return pd.read_parquet(OUT / "h.parquet").set_index("image_id").h


if __name__ == "__main__":
    ev = build()
    print(ev.groupby(["dataset", "weather"]).h.describe()[["count", "25%", "50%", "75%"]].round(2))
