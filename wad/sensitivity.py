"""Detector-configuration sensitivity (beyond the brief): the same released COCO weights run
with a different, fixed inference setting, through the same matching, curves and CRC code.
Nothing is trained. Results go to cache/sensitivity/<name>/ and results/sensitivity/<name>/
and are reported as a sensitivity analysis, never as the primary result.

  yolov8s_1280   YOLOv8s at imgsz 1280 instead of 640: DAWN images are mostly 1280 px wide, so
                 this removes the 2x downscale. Tests whether the never-detected objects (mostly
                 under 32 px at 640) are a resolution limit rather than an architectural one.

YOLOv8x (the SeqCRC paper's model) is not run here: the brief reserves it for a GPU.

Run: python -m wad.sensitivity [--variant yolov8s_1280]
"""
from __future__ import annotations

import argparse
import copy
import shutil

from wad.config import Cfg, load_config, results_path, write_json

VARIANTS = {"yolov8s_1280": {"imgsz": 1280}}


def variant_cfg(cfg: Cfg, name: str) -> Cfg:
    v = copy.deepcopy(cfg)
    v.detector.update(VARIANTS[name])
    v.paths["cache"] = cfg.paths.cache / "sensitivity" / name
    v.paths["results"] = cfg.paths.results / "sensitivity" / name
    v.paths.cache.mkdir(parents=True, exist_ok=True)
    for f in ("images.parquet", "gt.parquet"):
        shutil.copyfile(cfg.paths.cache / f, v.paths.cache / f)
    return v


def compare_to_primary(cfg: Cfg, v: Cfg):
    """Every box matched in one setting but not the other, with its size in original pixels and,
    for boxes lost in the variant, why the variant misses it (wad.extras.miss_reasons)."""
    import numpy as np
    import pandas as pd

    from wad import extras
    from wad.data import load_gt

    g0 = pd.read_parquet(cfg.paths.cache / "matches_gt.parquet")[["gt_id", "match_score"]]
    g1 = pd.read_parquet(v.paths.cache / "matches_gt.parquet")[["gt_id", "image_id", "weather", "cls", "match_score"]]
    m = g1.merge(g0, on="gt_id", suffixes=("_variant", "_primary"))
    m["status"] = np.select([m.match_score_primary.notna() & m.match_score_variant.isna(),
                             m.match_score_primary.isna() & m.match_score_variant.notna()],
                            ["lost", "gained"], "same")
    gt = load_gt(cfg).set_index("gt_id")
    m["side_px"] = np.sqrt((gt.x2 - gt.x1) * (gt.y2 - gt.y1)).reindex(m.gt_id).to_numpy()
    _, _, reasons = extras.miss_reasons(v)
    m = m.merge(reasons[["gt_id", "reason", "other"]].rename(columns={"reason": "variant_reason"}), on="gt_id", how="left")
    d = pd.read_parquet(v.paths.cache / "detections.parquet")
    n = d.groupby("image_id").size()
    m["variant_max_det_binds"] = m.image_id.isin(set(n[n >= v.detector.max_det].index))
    return m[m.status != "same"].sort_values("gt_id").reset_index(drop=True)


def run(cfg: Cfg, name: str, detect: bool = True) -> dict:
    from wad import crc, curves, detect as det, extras, match

    v = variant_cfg(cfg, name)
    if detect:
        det.run(v)
    match.match_all(v)
    cur, summ, verdict = curves.compute(v)
    cur.to_csv(results_path(v, "curves.csv"), index=False, float_format="%.6g")
    summ.to_csv(results_path(v, "summary_phase3.csv"), index=False, float_format="%.6g")
    write_json(results_path(v, "go_no_go.json"), verdict)
    out = crc.run(v)
    crc.write(v, out)
    ma = extras.min_feasible_alpha(v)
    ma.to_csv(results_path(v, "min_feasible_alpha.csv"), index=False, float_format="%.6g")
    sz = extras.misses_by_size(v)
    sz.to_csv(results_path(v, "misses_by_size.csv"), index=False, float_format="%.6g")
    compare_to_primary(cfg, v).to_csv(results_path(v, "lost_gained.csv"), index=False, float_format="%.6g")
    print(verdict["line"])
    return dict(verdict=verdict, summary=summ, crc=out["summary"], min_alpha=ma)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--variant", default="yolov8s_1280", choices=list(VARIANTS))
    ap.add_argument("--from-cache", action="store_true", help="reuse the variant's cached detections")
    args = ap.parse_args(argv)
    run(load_config(args.config), args.variant, detect=not args.from_cache)


if __name__ == "__main__":
    main()
