"""The one way pixels are read in this repo (detector input, hashes, hazard features).

DAWN mixes RGB, CMYK and greyscale JPEGs, and 40 files carry an EXIF orientation tag of 2
(horizontal mirror). OpenCV, and therefore Ultralytics when given a path, silently applies
that tag; the annotations may or may not have been drawn in that frame. Reading every image
through this module with one explicit policy (config `data.exif_orientation`) keeps the
detector, the boxes and the hashes in the same pixel frame.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXIF_ORIENTATION_TAG = 274
REPO = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    """Paths in the cache are relative to the repo root; make them work from any directory."""
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def exif_orientation(path: str | Path) -> int:
    with Image.open(resolve(path)) as im:
        return int(im.getexif().get(EXIF_ORIENTATION_TAG) or 1)


def load_rgb(path: str | Path, exif: str = "ignore") -> Image.Image:
    """RGB PIL image. exif='apply' honours the orientation tag; 'ignore' keeps stored pixels."""
    if exif not in ("apply", "ignore"):
        raise ValueError(f"exif must be 'apply' or 'ignore', got {exif!r}")
    with Image.open(resolve(path)) as im:
        im.load()
        if exif == "apply":
            im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def load_bgr(path: str | Path, exif: str = "ignore") -> np.ndarray:
    """HxWx3 uint8 BGR array (the layout Ultralytics expects for numpy input)."""
    rgb = np.asarray(load_rgb(path, exif))
    return np.ascontiguousarray(rgb[:, :, ::-1])
