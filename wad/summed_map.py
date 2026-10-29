"""EXPLORATORY: the summed-confidence method vs ordinary thresholding, judged as a planner would.
STATUS (2026-10-13): no better than thresholding; kept so its NOTES.md numbers reproduce. The
method is the dial (wad/dial.py): one threshold or per detection.

From every cached detection with score >= 0.001 (YOLOv8x @1280), three maps per image on a grid of
4x4-pixel cells (images scaled to 320 cells wide):
  max       each cell takes the strongest box covering it; cutting this map at tau is exactly
            ordinary thresholding (the union of boxes with score >= tau)
  sum       each box adds its confidence over its area (the user's summing method)
  noisy-OR  1 - prod(1 - score) over the boxes covering a cell (probabilistic, non-competitive sum)
Cut each map at a level h -> an "occupied" mask. Scored at the region level:
  object found   >= half of the object's box cells are occupied (any class: a planner needs "something")
  false alarm    a connected occupied region (8-connected) overlapping no object box; also the share
                 of the image occupied outside every object box
Every method is swept over its own levels; methods are compared at equal object recall.

Run: python -m wad.summed_map   -> results/threshold/summed_map_{curves,at_recall}.csv, fig18
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import ndimage

from wad.config import REPO
from wad.config import THRESH_RESULTS as RESULTS

CELLS_W = 320
FLOOR = 0.001
METHODS = {"max (= ordinary threshold)": np.geomspace(FLOOR, 0.99, 60),
           "sum of confidences": np.geomspace(FLOOR, 6.0, 60),
           "noisy-OR": np.geomspace(FLOOR, 0.999, 60)}
RECALLS = (0.6, 0.7, 0.75, 0.8, 0.85)


def configs():
    from wad.run2 import bdd_cfg, dawn_cfg, load_run2

    r2 = load_run2()
    return [("BDD100K", bdd_cfg(r2, "v8x_1280", "group"), "cache/bdd/det/v8x_1280/detections.parquet", "cache/bdd/gt.parquet"),
            ("DAWN", dawn_cfg(r2, "v8x_1280", "group", "clean"), "cache/det/v8x_1280/detections.parquet", "cache/gt.parquet")]


def cells(b: np.ndarray, s: float, gh: int, gw: int):
    """Box (x1, y1, x2, y2) in pixels -> cell index ranges (at least one cell)."""
    x1, y1, x2, y2 = b * s
    c1, r1 = int(np.clip(np.floor(x1), 0, gw - 1)), int(np.clip(np.floor(y1), 0, gh - 1))
    c2, r2 = int(np.clip(np.ceil(x2), c1 + 1, gw)), int(np.clip(np.ceil(y2), r1 + 1, gh))
    return r1, r2, c1, c2


def maps(boxes: np.ndarray, scores: np.ndarray, gh: int, gw: int, s: float):
    mx, sm, lnot = np.zeros((gh, gw)), np.zeros((gh, gw)), np.zeros((gh, gw))
    for b, sc in zip(boxes, scores):
        r1, r2, c1, c2 = cells(b, s, gh, gw)
        mx[r1:r2, c1:c2] = np.maximum(mx[r1:r2, c1:c2], sc)
        sm[r1:r2, c1:c2] += sc
        lnot[r1:r2, c1:c2] += np.log1p(-min(sc, 1 - 1e-9))
    return {"max (= ordinary threshold)": mx, "sum of confidences": sm, "noisy-OR": 1 - np.exp(lnot)}


def groups_of(ds: str, weather: str) -> list[str]:
    if ds == "BDD100K":
        return ["all", "night" if weather.endswith("night") else "day"] + (["rain_night"] if weather == "rain_night" else [])
    return ["all", weather]


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    from wad.curves import Cache

    t0 = time.perf_counter()
    acc = {}  # (dataset, group, method) -> dict of per-level accumulators

    def slot(key, n):
        if key not in acc:
            acc[key] = dict(found=np.zeros(n), objects=0, false_regions=np.zeros(n), false_area=np.zeros(n), images=0)
        return acc[key]

    for ds, v, det_p, gt_p in configs():
        c = Cache(v)
        im = c.images.set_index("image_id")
        det = pd.read_parquet(REPO / det_p)
        det = det[det.image_id.isin(c.idx) & (det.score >= FLOOR)]
        gt = pd.read_parquet(REPO / gt_p)
        gt = gt[gt.image_id.isin(c.idx)]
        det_by, gt_by = dict(list(det.groupby("image_id"))), dict(list(gt.groupby("image_id")))
        box = ["x1", "y1", "x2", "y2"]
        for n_done, iid in enumerate(c.image_ids):
            w, h = float(im.loc[iid, "width"]), float(im.loc[iid, "height"])
            s = CELLS_W / w
            gw, gh = CELLS_W, max(1, int(round(h * s)))
            d = det_by.get(iid)
            g = gt_by.get(iid)
            gb = g[box].to_numpy(float) if g is not None else np.zeros((0, 4))
            G = np.zeros((gh, gw), bool)
            obj_cells = []
            for b in gb:
                r1, r2, c1, c2 = cells(b, s, gh, gw)
                G[r1:r2, c1:c2] = True
                obj_cells.append((r1, r2, c1, c2))
            mp = maps(d[box].to_numpy(float), d.score.to_numpy(), gh, gw, s) if d is not None else \
                {k: np.zeros((gh, gw)) for k in METHODS}
            for method, levels in METHODS.items():
                m = mp[method]
                # object found at h  <=>  its ceil(n/2)-th largest cell value >= h
                kth = []
                for r1, r2, c1, c2 in obj_cells:
                    vals = np.sort(m[r1:r2, c1:c2].ravel())[::-1]
                    kth.append(vals[int(np.ceil(len(vals) / 2)) - 1])
                kth = np.array(kth)
                found = (kth[:, None] >= levels[None, :]).sum(0) if len(kth) else np.zeros(len(levels))
                outside = m[~G]
                false_area = (outside[:, None] >= levels[None, :]).sum(0) / m.size if outside.size else np.zeros(len(levels))
                false_regions = np.zeros(len(levels))
                for j, lev in enumerate(levels):
                    occ = m >= lev
                    if not occ.any():
                        break  # levels ascend: nothing occupied from here on
                    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
                    if n:
                        hit = np.bincount(lab[G], minlength=n + 1)[1:]
                        false_regions[j] = int((hit == 0).sum())
                for grp in groups_of(ds, c.weather[c.idx[iid]]):
                    a = slot((ds, grp, method), len(levels))
                    a["found"] += found
                    a["objects"] += len(kth)
                    a["false_regions"] += false_regions
                    a["false_area"] += false_area
                    a["images"] += 1
            if (n_done + 1) % 2000 == 0:
                print(f"  {ds} {n_done + 1}/{c.n_img} images, {time.perf_counter() - t0:.0f} s", flush=True)
    rows = []
    for (ds, grp, method), a in acc.items():
        for j, lev in enumerate(METHODS[method]):
            rows.append(dict(dataset=ds, group=grp, method=method, level=lev, images=a["images"], objects=a["objects"],
                             object_recall=a["found"][j] / max(a["objects"], 1),
                             false_regions_per_image=a["false_regions"][j] / a["images"],
                             false_area_share=a["false_area"][j] / a["images"]))
    C = pd.DataFrame(rows)
    out = []
    for (ds, grp, method), d in C.groupby(["dataset", "group", "method"], sort=False):
        for r in RECALLS:
            ok = d[d.object_recall >= r]
            out.append(dict(dataset=ds, group=grp, method=method, target_recall=r,
                            false_regions_per_image=ok.false_regions_per_image.min() if len(ok) else np.nan,
                            false_area_share=ok.false_area_share.min() if len(ok) else np.nan))
    A = pd.DataFrame(out)
    RESULTS.mkdir(parents=True, exist_ok=True)
    C.to_csv(RESULTS / "summed_map_curves.csv", index=False, float_format="%.5g")
    A.to_csv(RESULTS / "summed_map_at_recall.csv", index=False, float_format="%.5g")
    figure(C)
    return C, A


def figure(C: pd.DataFrame):
    import matplotlib.pyplot as plt

    from wad.config import Cfg
    from wad.figures import SERIES, save, style

    style()
    panels = [("BDD100K", "day", "BDD100K day"), ("BDD100K", "night", "BDD100K night"), ("BDD100K", "rain_night", "BDD100K rain night"),
              ("DAWN", "fog", "DAWN fog")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, (ds, grp, title) in zip(axes, panels):
        for (method, col) in zip(METHODS, SERIES):
            d = C[(C.dataset == ds) & (C.group == grp) & (C.method == method)].sort_values("object_recall")
            ax.plot(d.object_recall, d.false_regions_per_image, color=col, lw=1.8, label=method)
        ax.set_yscale("log")
        ax.set_xlabel("objects found (≥ half covered)")
        ax.set_title(title)
        ax.set_xlim(0.4, 1.0)
    axes[0].set_ylabel("false alarm regions per image")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return save(fig, Cfg(dict(paths=Cfg(dict(results=RESULTS)))), "fig18_summed_map")


if __name__ == "__main__":
    C, A = main()
    pd.set_option("display.width", 220)
    p = A.pivot_table(index=["dataset", "group", "target_recall"], columns="method", values="false_regions_per_image", sort=False)
    print("false alarm regions per image at equal object recall\n", p.round(2).to_string())
    p = A.pivot_table(index=["dataset", "group", "target_recall"], columns="method", values="false_area_share", sort=False)
    print("share of the image wrongly occupied at equal object recall\n", p.round(4).to_string())
