"""Per-frame cost of the dial's per-detection option next to the detector (Phase 4, runtime).

Times wad.per_detection.score (features from the frame's detections and YOLOv8x's per-box outputs, then the
boosted trees) one frame at a time, as a deployed system would call it, on 300 random BDD100K val frames with
their cached detections and per-box outputs, and the one-threshold option for comparison. CPU, one process.
The detector's own time (YOLOv8x @1280 on the RTX 3050 Ti) is in results/run2/run2.md (0.152 s per frame).

Writes results/ablation/dial_runtime.csv. Run: python -m wad.dial_runtime
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from wad.config import REPO

OUT = REPO / "results" / "ablation" / "dial_runtime.csv"


def main(n=300, seed=0):
    from wad import yolo_box_features as ybf
    from wad.per_detection import keep, load, score

    bundle = load()
    det = pd.read_parquet(REPO / "cache/bdd/det/v8x_1280/detections.parquet")
    det["row"] = np.arange(len(det))
    sizes = pd.read_parquet(REPO / "cache/bdd/images.parquet").set_index("image_id")[["width", "height"]]
    box = ybf.load("bdd")
    ids = np.random.default_rng(seed).choice(det.image_id.unique(), n, replace=False)
    by = dict(tuple(det[det.image_id.isin(ids)].groupby("image_id")))
    rows = []
    for iid in ids:
        d = by[iid].reset_index(drop=True)
        r = d.row.to_numpy()
        b = {k: np.asarray(box[k][r]) for k in ("cls", "dfl", "neck", "level")}
        t0 = time.perf_counter()
        p = score(d, sizes, bundle, b)
        k = keep(p, 10.0)
        t1 = time.perf_counter()
        k1 = d.score.to_numpy() >= 0.0541  # option A at lambda 10
        t2 = time.perf_counter()
        rows.append(dict(image_id=iid, boxes=len(d), per_detection_ms=1e3 * (t1 - t0), one_threshold_ms=1e3 * (t2 - t1),
                         kept_per_detection=int(k.sum()), kept_one_threshold=int(k1.sum())))
    r = pd.DataFrame(rows)
    r.to_csv(OUT, index=False, float_format="%.4g")
    print(f"{n} frames, {r.boxes.median():.0f} boxes per frame (median): per detection {r.per_detection_ms.median():.1f} ms "
          f"median, {r.per_detection_ms.quantile(0.95):.1f} ms p95; one threshold {r.one_threshold_ms.median():.3f} ms")


if __name__ == "__main__":
    main()
