"""Hazard scalar h from pixels only (phase 6a, O3 version A).

Four per-image statistics, each a function of the image file alone:
  dark_channel   mean of the dark channel: per-pixel minimum over RGB, then a 15x15 minimum
                 filter (He et al.); high in fog and haze, where airlight lifts every channel
  rms_contrast   standard deviation of the grey image in [0, 1]
  edge_energy    variance of the Laplacian of the grey image
  saturation     mean HSV saturation in [0, 1]
Images are first resized so the longer side is 640 px (the detector's input scale), so edge
energy is comparable across DAWN's mixed resolutions. All four are invariant to a horizontal
flip, so the EXIF mirror tags in DAWN do not affect them.

h is the first principal component of the four standardised features, signed so that higher h
means lower contrast and lower edge energy (more degraded).

This module never reads detector output: it imports nothing from wad.detect, wad.match,
wad.curves or wad.crc, and image_features() takes only an image path (tests/test_hazard.py).

Run: python -m wad.hazard [--smoke]   -> cache/hazard.parquet, cache/hazard_pca.json
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from wad.imageio import load_rgb

FEATURES = ["dark_channel", "rms_contrast", "edge_energy", "saturation"]
LONG_SIDE = 640
PATCH = 15


def image_features(path: str) -> dict[str, float]:
    im = load_rgb(path, "apply")
    s = LONG_SIDE / max(im.size)
    im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BILINEAR)
    rgb = np.asarray(im, dtype=np.float64) / 255.0
    dark = ndimage.minimum_filter(rgb.min(axis=2), size=PATCH, mode="nearest")
    grey = np.asarray(im.convert("L"), dtype=np.float64) / 255.0
    sat = np.asarray(im.convert("HSV"), dtype=np.float64)[:, :, 1] / 255.0
    return dict(dark_channel=float(dark.mean()), rms_contrast=float(grey.std()),
                edge_energy=float(ndimage.laplace(grey).var()), saturation=float(sat.mean()))


def fit_pca(x: pd.DataFrame) -> dict:
    """Standardise, take the first principal component, sign it so h rises as contrast and edge
    energy fall. Returns the transform as plain numbers (recorded in cache/hazard_pca.json)."""
    mu, sd = x.mean(), x.std(ddof=0)
    z = ((x - mu) / sd).to_numpy()
    w, v = np.linalg.eigh(np.cov(z, rowvar=False))
    pc = v[:, np.argmax(w)]
    load = dict(zip(x.columns, pc))
    if load["rms_contrast"] + load["edge_energy"] > 0:
        pc = -pc
    return dict(mean=mu.to_dict(), std=sd.to_dict(), loadings=dict(zip(x.columns, pc.tolist())),
                explained_variance_ratio=float(w.max() / w.sum()))


def apply_pca(x: pd.DataFrame, pca: dict) -> np.ndarray:
    cols = list(pca["loadings"])
    z = (x[cols] - pd.Series(pca["mean"])[cols]) / pd.Series(pca["std"])[cols]
    return z.to_numpy() @ np.array([pca["loadings"][c] for c in cols])


def main(argv=None):
    from wad.config import cache_path, load_config, write_json
    from wad.data import load_images

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke)
    images = load_images(cfg)
    rows = [dict(image_id=r.image_id, weather=r.weather, **image_features(r.path)) for r in images.itertuples()]
    hz = pd.DataFrame(rows)
    # Fitted on every kept image without labels or detector output: a symmetric function of
    # the whole pool, so it does not distinguish calibration from test images.
    pca = fit_pca(hz[FEATURES])
    hz["h"] = apply_pca(hz[FEATURES], pca)
    hz.to_parquet(cache_path(cfg, "hazard.parquet"), index=False)
    write_json(cache_path(cfg, "hazard_pca.json"), pca)
    print(hz.groupby("weather")[FEATURES + ["h"]].mean().to_string(float_format=lambda v: f"{v:.4f}"))
    print("loadings:", {k: round(v, 3) for k, v in pca["loadings"].items()},
          "explained:", round(pca["explained_variance_ratio"], 3))


if __name__ == "__main__":
    main()
