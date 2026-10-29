"""Every box the detector produces, at the usual threshold and down to essentially zero (exploratory).
STATUS: illustration figures (fig16, fig17, fig19) for the threshold exploration.

For one example image per condition (BDD100K val clear day, clear night, rain night; DAWN fog),
YOLOv8x @1280 is re-run with the confidence cut-off at 1e-6 and the box cap raised to 10,000 (the
cache stops at 0.001 and 300 boxes). Panels: (a) tau = 0.25 and (b) tau = 0.05, real boxes green,
false red, objects not found dashed yellow (grouped matching, IoU 0.5); (c) every box, coloured by
confidence; (d) confidence summed over all boxes (each box adds its score over its area: the
"summed area" view), with the objects outlined.

Run: python -m wad.show_boxes   -> results/threshold/figures/fig16_all_boxes_<condition>.png and
     fig17_low_thresholds_<condition>.png (the same images at 0.25, 0.01, 0.005, 0.001; --low: only these);
     --summed: fig19_summed_map_<condition>.png, the summed-confidence method from the boxes >= 0.001
     (needs python -m wad.summed_map first)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wad.config import REPO
from wad.config import THRESH_RESULTS as RESULTS

NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
GROUP = {"car": "vehicle", "truck": "vehicle", "bus": "vehicle", "bicycle": "two_wheeler", "motorcycle": "two_wheeler",
         "person": "person"}
EXAMPLES = [("BDD100K", "clear", "Clear · day"), ("BDD100K", "clear_night", "Clear · night"),
            ("BDD100K", "rain_night", "Rain · night"), ("DAWN", "fog", "Fog (DAWN)")]


def pick(ds: str, weather: str, seed=0) -> pd.Series:
    """A random image of the condition with 5-20 labelled objects (seed 0)."""
    im = pd.read_parquet(REPO / ("cache/bdd/images.parquet" if ds == "BDD100K" else "cache/images.parquet"))
    im = im[im.keep & (im.weather == weather) & im.n_gt.between(5, 20)].sort_values("image_id")
    return im.iloc[np.random.default_rng(seed).integers(len(im))]


def run_all_boxes(model, path: str):
    from wad.imageio import load_bgr

    res = model.predict(source=load_bgr(path, "apply"), imgsz=1280, conf=1e-6, iou=0.5, max_det=10000,
                        classes=list(NAMES), device="cuda:0", verbose=False)[0]
    return pd.DataFrame(dict(cls=[NAMES[int(c)] for c in res.boxes.cls.cpu().numpy()], score=res.boxes.conf.cpu().numpy(),
                             **dict(zip(["x1", "y1", "x2", "y2"], res.boxes.xyxy.cpu().numpy().T))))


def figure(ds, weather, label, r, d, gt):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from wad.config import Cfg
    from wad.figures import save, style
    from wad.imageio import load_rgb
    from wad.match import greedy_match

    style()
    img = np.asarray(load_rgb(r.path, "apply"))
    box = ["x1", "y1", "x2", "y2"]
    key = lambda a: np.array([GROUP[x] for x in a], dtype=object)  # noqa: E731
    tp, _, _, gscore = greedy_match(d[box].to_numpy(), d.score.to_numpy(), key(d.cls.to_numpy()),
                                    gt[box].to_numpy(), key(gt.cls.to_numpy()), 0.5)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4))
    n_gt = len(gt)
    for ax, tau in zip(axes[0], (0.25, 0.05)):
        ax.imshow(img)
        keep = d.score.to_numpy() >= tau
        for (x1, y1, x2, y2), t in zip(d[box].to_numpy()[keep], tp[keep]):
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.3, ec="#1baf7a" if t else "#d03b3b"))
        found = np.nan_to_num(gscore, nan=-1) >= tau
        for (x1, y1, x2, y2) in gt[box].to_numpy()[~found]:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.2, ec="#eda100", ls="--"))
        ax.set_title(f"τ = {tau}: {keep.sum()} boxes, {int((tp & keep).sum())} real, {int((keep & ~tp).sum())} false; "
                     f"{int(found.sum())}/{n_gt} objects found", fontsize=9)
    ax = axes[1, 0]
    ax.imshow(img)
    cmap = plt.get_cmap("viridis")
    order = np.argsort(d.score.to_numpy())
    for i in order:
        x1, y1, x2, y2 = d[box].to_numpy()[i]
        s = d.score.to_numpy()[i]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=0.5 + 1.5 * s, ec=cmap(min(1.0, s / 0.5)),
                               alpha=0.25 + 0.75 * min(1.0, s / 0.25)))
    ever = int(np.isfinite(gscore).sum())
    ax.set_title(f"everything (τ ≈ 0): {len(d)} boxes, colour = confidence (yellow ≥ 0.5); {ever}/{n_gt} objects have a box",
                 fontsize=9)
    ax = axes[1, 1]
    heat = np.zeros(img.shape[:2])
    for (x1, y1, x2, y2), s in zip(d[box].to_numpy().round().astype(int), d.score.to_numpy()):
        heat[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] += s
    ax.imshow(img.mean(2), cmap="gray", alpha=0.45)
    hm = ax.imshow(heat, cmap="magma", alpha=0.75)
    for (x1, y1, x2, y2) in gt[box].to_numpy():
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.0, ec="#86b6ef"))
    ax.set_title("confidence summed over all boxes (labelled objects in blue)", fontsize=9)
    fig.colorbar(hm, ax=ax, fraction=0.03, pad=0.01)
    for a in axes.flat:
        a.set_axis_off()
    fig.suptitle(f"{label} — {r.image_id}", x=0.01, ha="left", fontsize=10.5, fontweight="semibold")
    fig.tight_layout()
    tag = weather if ds == "BDD100K" else f"dawn_{weather}"
    return save(fig, Cfg(dict(paths=Cfg(dict(results=RESULTS)))), f"fig16_all_boxes_{tag}")


LOW_TAUS = (0.25, 0.01, 0.005, 0.001)


def figure_low(ds, weather, label, r, d, gt, taus=LOW_TAUS):
    """The same image at a reference threshold and three very low ones: real boxes green, false boxes
    thin red, objects with no box at that threshold dashed yellow."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from wad.config import Cfg
    from wad.figures import save, style
    from wad.imageio import load_rgb
    from wad.match import greedy_match

    style()
    img = np.asarray(load_rgb(r.path, "apply"))
    box = ["x1", "y1", "x2", "y2"]
    key = lambda a: np.array([GROUP[x] for x in a], dtype=object)  # noqa: E731
    tp, _, _, gscore = greedy_match(d[box].to_numpy(), d.score.to_numpy(), key(d.cls.to_numpy()),
                                    gt[box].to_numpy(), key(gt.cls.to_numpy()), 0.5)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4))
    for ax, tau in zip(axes.flat, taus):
        ax.imshow(img)
        keep = d.score.to_numpy() >= tau
        order = np.argsort(d.score.to_numpy())  # weak boxes first, so strong ones are drawn on top
        for i in order[keep[order]]:
            x1, y1, x2, y2 = d[box].to_numpy()[i]
            real = tp[i]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.6 if real else 0.6,
                                   ec="#1baf7a" if real else "#d03b3b", alpha=1.0 if real else 0.7))
        found = np.nan_to_num(gscore, nan=-1) >= tau
        for (x1, y1, x2, y2) in gt[box].to_numpy()[~found]:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.3, ec="#eda100", ls="--"))
        ax.set_title(f"τ = {tau:g}: {keep.sum()} boxes ({int((tp & keep).sum())} real, {int((keep & ~tp).sum())} false); "
                     f"{int(found.sum())}/{len(gt)} objects found", fontsize=9)
        ax.set_axis_off()
    fig.suptitle(f"{label} — {r.image_id}   (green real, red false, yellow dashed = object without a box)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="semibold")
    fig.tight_layout()
    tag = weather if ds == "BDD100K" else f"dawn_{weather}"
    return save(fig, Cfg(dict(paths=Cfg(dict(results=RESULTS)))), f"fig17_low_thresholds_{tag}")


def main(low_only=False):
    from wad.detect import load_model
    from wad.thresh_net import cfg

    model = load_model(cfg())
    rows = []
    for ds, weather, label in EXAMPLES:
        r = pick(ds, weather)
        gt = pd.read_parquet(REPO / ("cache/bdd/gt.parquet" if ds == "BDD100K" else "cache/gt.parquet"))
        gt = gt[gt.image_id == r.image_id]
        d = run_all_boxes(model, r.path)
        figure_low(ds, weather, label, r, d, gt)
        if low_only:
            continue
        figure(ds, weather, label, r, d, gt)
        rows.append(dict(dataset=ds, condition=weather, image_id=r.image_id, objects=len(gt), boxes_all=len(d),
                         boxes_ge_0001=int((d.score >= 0.001).sum()), boxes_ge_005=int((d.score >= 0.05).sum()),
                         boxes_ge_025=int((d.score >= 0.25).sum()), lowest_score=float(d.score.min())))
    out = pd.DataFrame(rows)
    if len(out):  # --low draws figures only; keep the summary table of the full run
        out.to_csv(RESULTS / "all_boxes_examples.csv", index=False, float_format="%.4g")
    return out


SUMMED_GROUP = {"clear": "day", "clear_night": "night", "rain_night": "rain_night", "fog": "fog"}


def level_at_recall(ds: str, group: str, method: str, recall=0.8) -> float:
    """The map level that finds >= recall of the condition's objects with the fewest false-alarm regions
    (from wad.summed_map's sweep over every image of the condition)."""
    c = pd.read_csv(RESULTS / "summed_map_curves.csv")
    d = c[(c.dataset == ds) & (c.group == group) & (c.method == method) & (c.object_recall >= recall)]
    return float(d.loc[d.false_regions_per_image.idxmin(), "level"])


def figure_summed(ds, weather, label, r, d, gt, recall=0.8):
    """The summed-confidence method on one image, from the boxes with score >= 0.001: the boxes, the
    summed map, the map cut at the condition's 80%-recall level, and ordinary thresholding at its own
    80%-recall level (regions touching an object green, false-alarm regions red, missed objects
    dashed yellow)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy import ndimage

    from wad.config import Cfg
    from wad.figures import save, style
    from wad.imageio import load_rgb
    from wad.summed_map import CELLS_W, cells, maps

    style()
    img = np.asarray(load_rgb(r.path, "apply"))
    h, w = img.shape[:2]
    s = CELLS_W / w
    gw, gh = CELLS_W, max(1, int(round(h * s)))
    box = ["x1", "y1", "x2", "y2"]
    d = d[d.score >= 0.001]
    mp = maps(d[box].to_numpy(float), d.score.to_numpy(), gh, gw, s)
    G = np.zeros((gh, gw), bool)
    for b in gt[box].to_numpy(float):
        r1, r2, c1, c2 = cells(b, s, gh, gw)
        G[r1:r2, c1:c2] = True
    ext = (0, gw / s, gh / s, 0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4))
    ax = axes[0, 0]
    ax.imshow(img)
    cmap = plt.get_cmap("viridis")
    for i in np.argsort(d.score.to_numpy()):
        x1, y1, x2, y2 = d[box].to_numpy()[i]
        sc = d.score.to_numpy()[i]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=0.5 + 1.5 * sc, ec=cmap(min(1.0, sc / 0.5)),
                               alpha=0.3 + 0.7 * min(1.0, sc / 0.25)))
    ax.set_title(f"input: {len(d)} boxes with confidence ≥ 0.001 (colour = confidence, yellow ≥ 0.5)", fontsize=9)
    ax = axes[0, 1]
    ax.imshow(img.mean(2), cmap="gray", alpha=0.45)
    from matplotlib.colors import LogNorm

    sm = mp["sum of confidences"]
    hm = ax.imshow(np.ma.masked_less(sm, 1e-3), cmap="magma", alpha=0.85, extent=ext,
                   norm=LogNorm(vmin=1e-3, vmax=max(1e-2, float(sm.max()))))  # log: weak-box piles stay visible
    for (x1, y1, x2, y2) in gt[box].to_numpy():
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.0, ec="#86b6ef"))
    fig.colorbar(hm, ax=ax, fraction=0.03, pad=0.01)
    ax.set_title("summed confidence, log scale (each box adds its score over its area); real objects in blue", fontsize=9)
    for ax, method, name in ((axes[1, 0], "sum of confidences", "summed map"),
                             (axes[1, 1], "max (= ordinary threshold)", "ordinary threshold")):
        lev = level_at_recall(ds, SUMMED_GROUP[weather], method, recall)
        occ = mp[method] >= lev
        lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
        touches = np.bincount(lab[G], minlength=n + 1) > 0
        rgba = np.zeros((gh, gw, 4))
        rgba[(lab > 0) & touches[lab]] = (0.106, 0.686, 0.478, 0.55)  # region touching an object
        rgba[(lab > 0) & ~touches[lab]] = (0.816, 0.231, 0.231, 0.65)  # false-alarm region
        ax.imshow(img)
        ax.imshow(rgba, extent=ext)
        found = 0
        for b in gt[box].to_numpy(float):
            r1, r2, c1, c2 = cells(b, s, gh, gw)
            ok = occ[r1:r2, c1:c2].mean() >= 0.5
            found += ok
            x1, y1, x2, y2 = b
            if not ok:
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.3, ec="#eda100", ls="--"))
        n_false = int((~touches[1:]).sum())
        ax.set_title(f"{name} cut at {lev:.3g} (finds 80% of {SUMMED_GROUP[weather]} objects overall): "
                     f"{found}/{len(gt)} found here, {n_false} false-alarm regions", fontsize=9)
    for a in axes.flat:
        a.set_axis_off()
    fig.suptitle(f"{label} — {r.image_id}   summed-confidence method vs ordinary threshold, boxes ≥ 0.001",
                 x=0.01, ha="left", fontsize=10.5, fontweight="semibold")
    fig.tight_layout()
    tag = weather if ds == "BDD100K" else f"dawn_{weather}"
    return save(fig, Cfg(dict(paths=Cfg(dict(results=RESULTS)))), f"fig19_summed_map_{tag}")


def main_summed():
    """Figures of the summed-confidence method from the cached detections (score >= 0.001); no GPU."""
    for ds, weather, label in EXAMPLES:
        r = pick(ds, weather)
        gt = pd.read_parquet(REPO / ("cache/bdd/gt.parquet" if ds == "BDD100K" else "cache/gt.parquet"))
        gt = gt[gt.image_id == r.image_id]
        det = pd.read_parquet(REPO / ("cache/bdd/det/v8x_1280/detections.parquet" if ds == "BDD100K"
                                      else "cache/det/v8x_1280/detections.parquet"))
        figure_summed(ds, weather, label, r, det[det.image_id == r.image_id], gt)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    import sys

    if "--summed" in sys.argv:
        main_summed()
    else:
        out = main(low_only="--low" in sys.argv)
        if len(out):
            print(out.to_string(index=False))
