"""The dial in one place: set lam, the false alarms you accept to avoid one missed object, and pick
how it is applied. Both options are fitted on BDD100K train and graded on BDD100K val and DAWN; a
setting means the same in both (the expected-cost-best choice for lam * misses + false alarms).

  "one"            one confidence threshold for every detection: keep iff YOLOv8x's score >=
                   threshold(lam) (wad/one_threshold.py; exact switch points in
                   weights/one_threshold_table.csv, so any lam > 0 gives the fitted threshold)
  "per detection"  keep iff the per-detection model's chance the box is real >= 1 / (1 + lam)
                   (wad/per_detection.py; since 2026-10-15 model 2, weights/per_detection_yolo.joblib,
                   which also reads YOLOv8x's own per-box features: pass box=, see below)

Presets are lam values (PRESETS); the results tables also grade lam = 3.

The same dial in F-beta units (keep_f): set beta, how many times more recall counts than precision; it is
turned into its equivalent lam = (1 + beta^2) / F* - 1 (F* the best F_beta of that option on the train
images; wad/fbeta_dial.py, table weights/fbeta_table.csv, beta from 0.25 to 4; F_PRESETS) and the lam dial
is applied there.

Inputs must come from the detector as it was run for fitting (wad/detect.py with the run 2 config):
YOLOv8x at imgsz 1280, confidence floor 0.001, NMS IoU 0.5, at most 300 boxes per image, classes
restricted to car, truck, bus, bicycle, motorcycle, person inside predict, EXIF orientation applied.
Other settings (e.g. Ultralytics' default NMS IoU 0.7) shift both options without warning.
  det    DataFrame with image_id, cls (one of the six names above), score, x1, y1, x2, y2 (pixels),
         each image's rows consecutive
  sizes  width and height in pixels by image_id (index or column); only the per-detection option
         needs them
  box    YOLOv8x's per-box head outputs for det's rows (cls, dfl, neck, level); the per-detection
         option's model 2 needs them. Detector() runs YOLOv8x once and returns det, box and the size:

    from wad import dial
    from wad.yolo_box_features import Detector
    det, box, (w, h) = Detector()("frame.jpg", image_id="frame")
    sizes = pd.DataFrame(dict(width=[w], height=[h]), index=["frame"])
    kept = dial.keep(det, dial.PRESETS["balanced"], mode="per detection", sizes=sizes, box=box)
    kept = dial.keep_f(det, 2.0, mode="per detection", sizes=sizes, box=box)   # the same dial set by F2
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd

from wad.config import REPO

PRESETS = {"precision first": 1.0, "balanced": 10.0, "recall first": 30.0, "catch everything": 100.0}
F_PRESETS = {"F0.5 (precision counts double)": 0.5, "F1": 1.0, "F2 (recall counts double)": 2.0}
MODES = ("one", "per detection")
TABLE = REPO / "weights" / "one_threshold_table.csv"
F_TABLE = REPO / "weights" / "fbeta_table.csv"


def _check(lam: float) -> float:
    lam = float(lam)
    if not (lam > 0 and math.isfinite(lam)):
        raise ValueError(f"lam must be a positive number (false alarms accepted per avoided miss), got {lam}")
    return lam


@lru_cache(maxsize=1)
def _table() -> pd.DataFrame:
    return pd.read_csv(TABLE, float_precision="round_trip")  # thresholds exactly as fitted


def threshold(lam: float) -> float:
    """The one threshold for dial lam (exactly the threshold fitted on BDD100K train)."""
    from wad.one_threshold import threshold_from_table

    return threshold_from_table(_table(), _check(lam))


def cut(lam: float) -> float:
    """The per-detection cut: the chance of being real a detection needs to be kept."""
    return 1.0 / (1.0 + _check(lam))


@lru_cache(maxsize=1)
def _model() -> dict:
    from wad.per_detection import load

    return load()  # the adopted per-detection model (wad.per_detection.ADOPTED)


def keep(det: pd.DataFrame, lam: float, mode: str = "one", sizes: pd.DataFrame | None = None,
         box: dict | None = None) -> np.ndarray:
    """Boolean mask over det's rows: the detections kept at dial lam. box: YOLOv8x's per-box features for
    det's rows, needed when the adopted per-detection model uses them (wad.yolo_box_features.Detector)."""
    lam = _check(lam)
    if mode == "one":
        return det.score.to_numpy() >= threshold(lam)
    if mode == "per detection":
        if sizes is None:
            raise ValueError("the per-detection option needs the image sizes (width, height by image_id)")
        from wad.per_detection import keep as keep_by_chance
        from wad.per_detection import score

        return keep_by_chance(score(det, sizes, _model(), box), lam)
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


@lru_cache(maxsize=1)
def _f_table() -> pd.DataFrame:
    return pd.read_csv(F_TABLE, float_precision="round_trip")


def f_setting(beta: float) -> pd.Series:
    """The train-fitted setting for beta: threshold, cut, train F_beta and the equivalent lam per option."""
    from wad.fbeta_dial import lookup

    return lookup(_f_table(), float(beta))


def lam_for(beta: float, mode: str = "one") -> float:
    """The lam setting equivalent to F_beta for this option (fitted on train)."""
    r = f_setting(beta)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return float(r.one_lam_equivalent if mode == "one" else r.det_lam_equivalent)


def keep_f(det: pd.DataFrame, beta: float, mode: str = "one", sizes: pd.DataFrame | None = None,
           box: dict | None = None) -> np.ndarray:
    """Boolean mask over det's rows: the F_beta dial at beta, i.e. the lam dial at beta's equivalent lam."""
    return keep(det, lam_for(beta, mode), mode, sizes, box)
