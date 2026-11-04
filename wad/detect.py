"""Run the fixed detector exactly once over every kept image -> cache/detections.parquet.

One row per predicted box: image_id, cls, score, x1, y1, x2, y2 in the image's own pixel
coordinates (Ultralytics maps letterboxed boxes back to the original frame). Images are read
through wad.imageio so the detector sees the same pixel frame the annotations use.

Inference runs one image per forward pass: with mixed image sizes Ultralytics letterboxes a
batch to a common square, so a box's score would depend on which images happened to share its
batch. One image per pass keeps every detection a function of its own image only, which also
makes resume exact. Finished images are written in atomic shards (rows + an index of the image
ids the shard covers, including images with zero detections); a re-run skips them.

Run: python -m wad.detect [--smoke] [--overlays]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from wad.config import Cfg, coco_to_name, det_path, load_config, results_path, write_json
from wad.data import load_gt, load_images
from wad.imageio import load_bgr, load_rgb

DET_COLUMNS = ["image_id", "cls", "score", "x1", "y1", "x2", "y2"]


def pick_device(cfg: Cfg | None = None) -> str:
    """detector.device: 'cpu', 'cuda:0' or 'auto' (GPU when torch sees one). Run 1 pinned 'cpu'."""
    import torch

    want = (cfg.detector.get("device", "auto") if cfg is not None else "auto")
    if want == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if want.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"detector.device = {want} but torch sees no CUDA device")
    return want


def ensure_weights(cfg: Cfg) -> Path:
    """Released COCO weights, downloaded once into weights/ by Ultralytics' own asset fetcher."""
    from ultralytics.utils.downloads import attempt_download_asset

    path = cfg.paths.weights / cfg.detector.name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        attempt_download_asset(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(cfg: Cfg):
    from ultralytics import RTDETR, YOLO

    # RT-DETR (Phase 4 ablation d) needs its own predictor: set prediction, no NMS (nms_iou and max_det unused)
    cls = RTDETR if cfg.detector.name.startswith("rtdetr") else YOLO
    return cls(str(ensure_weights(cfg)))


def predict_one(model, cfg: Cfg, image_id: str, path: str, device: str) -> tuple[pd.DataFrame, tuple[int, int]]:
    names = coco_to_name(cfg)
    arr = load_bgr(path, cfg.data.exif_orientation)
    res = model.predict(
        source=arr,
        imgsz=cfg.detector.imgsz,
        conf=cfg.detector.conf_floor,  # 0.001: keep nearly everything
        iou=cfg.detector.nms_iou,  # 0.5, to match the paper
        max_det=cfg.detector.max_det,
        classes=list(cfg.classes.values()),
        device=device,
        verbose=False,
    )[0]
    b = res.boxes
    xyxy = b.xyxy.cpu().numpy().astype(float)
    df = pd.DataFrame(
        dict(
            image_id=image_id,
            cls=[names[int(c)] for c in b.cls.cpu().numpy()],
            score=b.conf.cpu().numpy().astype(float),
            x1=xyxy[:, 0], y1=xyxy[:, 1], x2=xyxy[:, 2], y2=xyxy[:, 3],
        ),
        columns=DET_COLUMNS,
    )
    return df, tuple(int(v) for v in res.orig_shape)


def fingerprint(cfg: Cfg) -> dict:
    """Everything that changes a detection. Shards made under another fingerprint are never reused."""
    return dict(detector={k: v for k, v in cfg.detector.items() if k != "device"}, classes=dict(cfg.classes),
                exif=cfg.data.exif_orientation,
                weights_sha256=sha256_file(ensure_weights(cfg)), device=pick_device(cfg))


def clear(cfg: Cfg) -> None:
    import shutil

    shutil.rmtree(shard_dir(cfg), ignore_errors=True)
    for name in ("detections.parquet", "detections.meta.json", "detection_sanity.json", "exif_check.json"):
        det_path(cfg, name).unlink(missing_ok=True)


def shard_dir(cfg: Cfg) -> Path:
    d = det_path(cfg, "det_shards")
    d.mkdir(parents=True, exist_ok=True)
    return d


def shard_indexes(cfg: Cfg, fp: dict) -> list[dict]:
    out = []
    for p in sorted(shard_dir(cfg).glob("shard_*.index.json")):
        idx = json.loads(p.read_text())
        if idx.get("fingerprint") != json.loads(json.dumps(fp)):
            raise RuntimeError(f"{p.name} was made with a different detector configuration; "
                               "rerun with --force to clear the detection cache")
        out.append(idx | {"_file": p.name})
    return out


def done_ids(cfg: Cfg, fp: dict) -> set[str]:
    return {i for x in shard_indexes(cfg, fp) for i in x["image_ids"]}


def write_shard(cfg: Cfg, k: int, rows: list[pd.DataFrame], index: dict) -> None:
    d = shard_dir(cfg)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=DET_COLUMNS)
    tmp = d / f"shard_{k:05d}.parquet.tmp"
    df.to_parquet(tmp, index=False)
    tmp.replace(d / f"shard_{k:05d}.parquet")
    # The index is written last: its presence is what marks the shard complete.
    write_json(d / f"shard_{k:05d}.index.json", index)


def run(cfg: Cfg, shard_size: int = 16, limit: int | None = None) -> None:
    images = load_images(cfg)
    fp = fingerprint(cfg)
    done = done_ids(cfg, fp)
    todo = images[~images.image_id.isin(done)]
    if limit is not None:
        todo = todo.head(limit)
    device = fp["device"]
    print(f"{len(done)} images already cached, {len(todo)} to run on {device}")
    if len(todo):
        model = load_model(cfg)
        k = len(list(shard_dir(cfg).glob("shard_*.index.json")))
        t0 = t_shard = time.perf_counter()
        rows, index = [], dict(image_ids=[], orig_shape={}, seconds={}, fingerprint=fp)
        for i, r in enumerate(todo.itertuples(index=False)):
            t = time.perf_counter()
            df, shape = predict_one(model, cfg, r.image_id, r.path, device)
            if shape != (r.height, r.width):
                raise ValueError(f"{r.image_id}: detector saw {shape}, annotations say {(r.height, r.width)}")
            rows.append(df)
            index["image_ids"].append(r.image_id)
            index["orig_shape"][r.image_id] = shape
            index["seconds"][r.image_id] = time.perf_counter() - t
            if len(index["image_ids"]) == shard_size or i == len(todo) - 1:
                index["wall_s"] = time.perf_counter() - t_shard  # elapsed for this shard (after model load)
                write_shard(cfg, k, rows, index)
                k += 1
                t_shard = time.perf_counter()
                rows, index = [], dict(image_ids=[], orig_shape={}, seconds={}, fingerprint=fp)
                print(f"  {i + 1}/{len(todo)} images, {time.perf_counter() - t0:.0f} s", flush=True)
    merge(cfg, fp)


def merge(cfg: Cfg, fp: dict) -> pd.DataFrame:
    """Concatenate complete shards into detections.parquet (+ meta) once every image is done."""
    images = load_images(cfg)
    d = shard_dir(cfg)
    idx = shard_indexes(cfg, fp)
    covered = [i for x in idx for i in x["image_ids"]]
    missing = set(images.image_id) - set(covered)
    if missing:
        print(f"{len(missing)} images not yet detected; detections.parquet not written")
        return None
    if len(covered) != len(set(covered)):
        raise RuntimeError("an image appears in two shards")
    parts = [pd.read_parquet(d / x["_file"].replace(".index.json", ".parquet")) for x in idx]
    det = pd.concat(parts, ignore_index=True)
    det = det[det.image_id.isin(set(images.image_id))]
    # Canonical row order: by image, then descending score (ties by class and box), so the
    # matching tie-break is deterministic and two runs give identical files.
    det = det.assign(_neg=-det.score).sort_values(["image_id", "_neg", "cls", "x1", "y1", "x2", "y2"])
    det = det.drop(columns="_neg").reset_index(drop=True)
    det.to_parquet(det_path(cfg, "detections.parquet"), index=False)

    import torch
    import ultralytics

    weights = ensure_weights(cfg)
    seconds = {k: v for x in idx for k, v in x["seconds"].items()}
    meta = dict(
        model_file=cfg.detector.name,
        model_sha256=sha256_file(weights),
        ultralytics_version=ultralytics.__version__,
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        processor=platform.processor(),
        imgsz=cfg.detector.imgsz,
        conf_floor=cfg.detector.conf_floor,
        nms_iou=cfg.detector.nms_iou,
        max_det=cfg.detector.max_det,
        classes=dict(cfg.classes),
        exif_orientation=cfg.data.exif_orientation,
        device=fp["device"],
        n_images=int(images.shape[0]),
        n_detections=int(len(det)),
        wall_time_s=round(float(sum(x.get("wall_s", 0.0) for x in idx)), 1),  # detection loop incl. shard writes
        inference_s=round(float(sum(seconds.values())), 1),  # per-image read + predict, summed
        mean_s_per_image=round(float(np.mean(list(seconds.values()))), 3),
    )
    write_json(det_path(cfg, "detections.meta.json"), meta)
    sanity(cfg, det, images)
    return det


def sanity(cfg: Cfg, det: pd.DataFrame, images: pd.DataFrame) -> dict:
    per_img = det.groupby("image_id").size().reindex(images.image_id, fill_value=0)
    zero = sorted(per_img[per_img == 0].index)
    s = dict(
        n_images=int(len(images)),
        n_zero_detection_images=len(zero),
        zero_detection_images=zero,
        frac_zero_detection=round(len(zero) / len(images), 4),
        dets_per_image=dict(min=int(per_img.min()), median=float(per_img.median()), max=int(per_img.max()),
                            n_at_max_det=int((per_img >= cfg.detector.max_det).sum())),
        score_deciles={f"p{q}": round(float(v), 4) for q, v in zip(range(0, 101, 10), np.percentile(det.score, range(0, 101, 10)))},
        class_fraction={k: round(float(v), 4) for k, v in det.cls.value_counts(normalize=True).items()},
        n_det_score_ge_0_25=int((det.score >= 0.25).sum()),
    )
    write_json(det_path(cfg, "detection_sanity.json"), s)
    print(json.dumps({k: v for k, v in s.items() if k != "zero_detection_images"}, indent=1))
    if s["frac_zero_detection"] > 0.05:
        raise SystemExit(f"STOP: {s['frac_zero_detection']:.1%} of images have zero detections at the floor (>5%)")
    return s


def exif_frame_check(cfg: Cfg) -> dict:
    """Which pixel frame are the annotations in? Run the detector on every image whose EXIF tag
    is not identity, once with the tag applied and once ignored, and compare recall at IoU 0.5.
    Ten untagged images serve as a control (both frames are identical there)."""
    import copy

    from wad.match import greedy_match

    images = load_images(cfg)
    gt = load_gt(cfg)
    tagged = images[~images.exif_orientation.isin([0, 1])]
    control = images[images.exif_orientation == 1].head(10)
    model, device = load_model(cfg), pick_device(cfg)
    box = ["x1", "y1", "x2", "y2"]
    out = {}
    for name, sel in [("tagged", tagged), ("control", control)]:
        for frame in ("apply", "ignore"):
            c = copy.deepcopy(cfg)
            c.data["exif_orientation"] = frame
            n = m_floor = m25 = 0
            for r in sel.itertuples():
                d, _ = predict_one(model, c, r.image_id, r.path, device)
                g = gt[gt.image_id == r.image_id]
                s = greedy_match(d[box].to_numpy(), d.score.to_numpy(), d.cls.to_numpy(),
                                 g[box].to_numpy(), g.cls.to_numpy(), cfg.match_iou)[3]
                n += len(g)
                m_floor += int(np.sum(~np.isnan(s)))
                m25 += int(np.sum(s >= 0.25))
            out[f"{name}/{frame}"] = dict(n_images=int(len(sel)), n_gt=n, max_recall=m_floor / max(n, 1),
                                          recall_at_0_25=m25 / max(n, 1))
    write_json(det_path(cfg, "exif_check.json"), out)
    return out


def overlays(cfg: Cfg, image_ids: list[str], min_score: float = 0.25) -> list[Path]:
    """GT (green) and detections at score >= min_score (red, labelled) for a visual check."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt

    images = load_images(cfg).set_index("image_id")
    gt = load_gt(cfg)
    det = pd.read_parquet(det_path(cfg, "detections.parquet"))
    out = []
    for image_id in image_ids:
        im = load_rgb(images.loc[image_id, "path"], cfg.data.exif_orientation)
        fig, ax = plt.subplots(figsize=(10, 10 * im.size[1] / im.size[0]))
        ax.imshow(im)
        for r in gt[gt.image_id == image_id].itertuples():
            ax.add_patch(mp.Rectangle((r.x1, r.y1), r.x2 - r.x1, r.y2 - r.y1, fill=False, ec="lime", lw=2))
        for r in det[(det.image_id == image_id) & (det.score >= min_score)].itertuples():
            ax.add_patch(mp.Rectangle((r.x1, r.y1), r.x2 - r.x1, r.y2 - r.y1, fill=False, ec="red", lw=1.2, ls="--"))
            ax.text(r.x1, r.y1 - 2, f"{r.cls} {r.score:.2f}", color="red", fontsize=7)
        ax.set_title(f"{image_id}: GT (green), detections score >= {min_score} (red)")
        ax.axis("off")
        p = results_path(cfg, f"figures/smoke/overlay_{image_id.replace('/', '_')}.png")
        if cfg.smoke_mode:  # smoke overlays go where the brief asks, not under results/smoke/
            p = cfg.paths.results.parent / "figures" / "smoke" / p.name
            p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        out.append(p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--overlays", nargs="*", default=None, help="image ids to draw (default: two per smoke run)")
    ap.add_argument("--force", action="store_true", help="discard cached detections and run the detector again")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke)
    if args.force:
        clear(cfg)
    run(cfg)
    if not args.smoke and not det_path(cfg, "exif_check.json").exists():
        print("exif frame check:", exif_frame_check(cfg))
    if args.smoke or args.overlays is not None:
        ids = args.overlays or list(load_images(cfg).groupby("weather").image_id.first().iloc[:2])
        for p in overlays(cfg, ids):
            print("overlay:", p)


if __name__ == "__main__":
    main()
