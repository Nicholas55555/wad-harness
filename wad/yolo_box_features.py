"""YOLOv8x's own per-box features for every cached detection (user's request 2026-10-15), for the
per-detection option's model (wad/per_detection.py).

Each image is re-run through YOLOv8x @1280 exactly as for detection (wad/detect.py settings: confidence
floor 0.001, NMS IoU 0.5, at most 300 boxes, the six classes, EXIF orientation applied), and Ultralytics'
NMS reports which anchor each kept box came from (return_idxs). At that anchor the detection head gives:
  cls    the 80 class probabilities (sigmoid of the class logits): how sure, and which other classes compete
  dfl    the four box-side distributions (softmax over 16 bins each, YOLOv8's DFL): how sure the box edges are
  neck   the neck feature vector at that anchor (the level's channels mean-reduced to 320, as Ultralytics'
         own get_obj_feats does for tracking)
  level  0 / 1 / 2 (stride 8 / 16 / 32)
The re-run must reproduce the cached detections (count, class, score, box); rows are stored in the order
of the set's detections.parquet. Images whose detections include exact score ties can come out in another
order and are matched box by box (class, score, box); the log counts those images and any row left
unmatched (2026-10-15: 38 of 30.9k images re-matched, 0 rows unmatched).

Outputs: cache/thresh/yolo_box/{train,bdd,dawn}_{cls,dfl,neck,level}.npy (float16 / uint8), with a
.done file per set for resuming. Run: python -m wad.yolo_box_features [--sets train bdd dawn]
(GPU, about 85 minutes for all three sets).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from wad.config import REPO, THRESH_CACHE

OUT = THRESH_CACHE / "yolo_box"
SETS = {"train": ("cache/thresh/images.parquet", "cache/thresh/det/v8x_1280/detections.parquet"),
        "bdd": ("cache/bdd/images.parquet", "cache/bdd/det/v8x_1280/detections.parquet"),
        "dawn": ("cache/images.parquet", "cache/det/v8x_1280/detections.parquet")}
PARTS = {"cls": (80, np.float16), "dfl": (64, np.float16), "neck": (320, np.float16), "level": (None, np.uint8)}


def paths(name: str) -> dict:
    return {k: OUT / f"{name}_{k}.npy" for k in PARTS}


def load(name: str) -> dict:
    """The features of a set, rows in the order of its detections.parquet."""
    done = OUT / f"{name}.done"
    n_img = int(done.read_text()) if done.exists() else 0
    want = pd.read_parquet(REPO / SETS[name][1], columns=["image_id"]).image_id.nunique()
    assert n_img == want, f"{name}: extraction incomplete ({n_img} of {want} images)"
    return {k: np.load(p, mmap_mode="r") for k, p in paths(name).items()}


class Capture:
    """Patches Ultralytics' NMS to also return the kept anchors and hooks the detection head."""

    def __init__(self):
        import ultralytics.utils.nms as unms

        # always wrap Ultralytics' own NMS, also when an earlier Capture already patched it
        orig = getattr(unms.non_max_suppression, "wad_original", unms.non_max_suppression)
        self.unms, self.orig, self.out, self.idxs = unms, orig, None, None

        def nms(*a, **k):
            k["return_idxs"] = True
            kept, idxs = self.orig(*a, **k)
            self.idxs = idxs
            return kept

        nms.wad_original = orig
        unms.non_max_suppression = nms

    def hook(self, module, inp, out):
        self.out = out


def per_box(cap: Capture) -> dict:
    """Head outputs at the kept anchors of the last image."""
    import torch

    y, raw = cap.out
    idx = cap.idxs[0].long()
    feats = raw["feats"]
    sizes = [f.shape[2] * f.shape[3] for f in feats]
    starts = np.cumsum([0] + sizes)
    cls = raw["scores"][0][:, idx].sigmoid().T  # (k, 80)
    dfl = raw["boxes"][0][:, idx].view(4, 16, -1).softmax(1).permute(2, 0, 1).reshape(-1, 64)  # (k, 64)
    neck = torch.empty((len(idx), 320), device=cls.device)
    level = torch.empty(len(idx), dtype=torch.uint8)
    for l, f in enumerate(feats):
        sel = (idx >= starts[l]) & (idx < starts[l + 1])
        if not sel.any():
            continue
        pos = idx[sel] - starts[l]
        v = f[0].reshape(f.shape[1], -1)[:, pos].T  # (k_l, C)
        neck[sel] = v.reshape(len(pos), 320, f.shape[1] // 320).mean(-1)  # channel groups, as get_obj_feats
        level[sel.cpu()] = l
    return dict(cls=cls.float().cpu().numpy(), dfl=dfl.float().cpu().numpy(), neck=neck.float().cpu().numpy(),
                level=level.numpy())


class Detector:
    """YOLOv8x run as the dial expects (the settings the dial was fitted with), returning the detections
    together with their per-box features: what the per-detection model needs on new images.

        det, box, (w, h) = Detector()(path_or_bgr_array, image_id="frame")
        sizes = pd.DataFrame(dict(width=[w], height=[h]), index=["frame"])
        kept = wad.dial.keep(det, 10, mode="per detection", sizes=sizes, box=box)
    """

    def __init__(self, device: str = "cuda:0"):
        from wad.detect import coco_to_name, load_model
        from wad.thresh_yolo import detector_cfg

        self.v = detector_cfg()
        self.names = coco_to_name(self.v)
        self.model = load_model(self.v)
        self.cap = Capture()
        self.args = dict(imgsz=self.v.detector.imgsz, conf=self.v.detector.conf_floor, iou=self.v.detector.nms_iou,
                         max_det=self.v.detector.max_det, classes=list(self.v.classes.values()), device=device, verbose=False)
        self.hooked = False

    def __call__(self, image, image_id: str = "image"):
        from wad.imageio import load_bgr

        src = load_bgr(image, self.v.data.exif_orientation) if not isinstance(image, np.ndarray) else image
        r = self.model.predict(source=src, **self.args)[0]
        if not self.hooked:
            self.model.predictor.model.model.model[-1].register_forward_hook(self.cap.hook)
            self.hooked = True
            r = self.model.predict(source=src, **self.args)[0]
        b = r.boxes
        xyxy = b.xyxy.cpu().numpy().astype(float)
        det = pd.DataFrame(dict(image_id=image_id, cls=[self.names[int(k)] for k in b.cls.cpu().numpy()],
                                score=b.conf.cpu().numpy().astype(float), x1=xyxy[:, 0], y1=xyxy[:, 1], x2=xyxy[:, 2], y2=xyxy[:, 3]))
        return det, per_box(self.cap), (src.shape[1], src.shape[0])


def extract(name: str, flush_every=500) -> None:
    from wad.detect import coco_to_name, load_model
    from wad.imageio import load_bgr
    from wad.thresh_yolo import detector_cfg

    OUT.mkdir(parents=True, exist_ok=True)
    v = detector_cfg()
    names = coco_to_name(v)
    im = pd.read_parquet(REPO / SETS[name][0]).set_index("image_id")
    det = pd.read_parquet(REPO / SETS[name][1])
    ids = det.image_id.to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    assert len(starts) == len(set(ids)), f"{name}: each image's detections must be consecutive"
    ends = np.r_[starts[1:], len(det)]
    order = ids[starts]
    done_f = OUT / f"{name}.done"
    start = int(done_f.read_text()) if done_f.exists() else 0
    if start >= len(order):
        print(f"{name}: complete ({len(order)} images)", flush=True)
        return
    (OUT / f"{name}.ids.txt").write_text("".join(f"{x}\n" for x in order), encoding="utf-8")
    arr = {}
    for k, (w, dt) in PARTS.items():
        shape = (len(det),) if w is None else (len(det), w)
        p = paths(name)[k]
        arr[k] = np.load(p, mmap_mode="r+") if start > 0 else np.lib.format.open_memmap(p, mode="w+", dtype=dt, shape=shape)
    model = load_model(v)
    cap = Capture()
    args = dict(imgsz=v.detector.imgsz, conf=v.detector.conf_floor, iou=v.detector.nms_iou, max_det=v.detector.max_det,
                classes=list(v.classes.values()), device="cuda:0", verbose=False)
    hooked, mismatch, unmatched, t0 = False, 0, 0, time.perf_counter()
    for i in range(start, len(order)):
        iid, a, z = order[i], starts[i], ends[i]
        src = load_bgr(im.loc[iid, "path"], v.data.exif_orientation)
        r = model.predict(source=src, **args)[0]
        if not hooked:  # the predictor builds (and fuses) its model on the first call
            model.predictor.model.model.model[-1].register_forward_hook(cap.hook)
            hooked = True
            r = model.predict(source=src, **args)[0]
        f = per_box(cap)
        c = det.iloc[a:z]
        got_cls = np.array([names[int(k)] for k in r.boxes.cls.cpu().numpy()])
        same = (len(c) == len(r.boxes) and (c.cls.to_numpy() == got_cls).all()
                and np.abs(c.score.to_numpy() - r.boxes.conf.cpu().numpy()).max() < 1e-5
                and np.abs(c[["x1", "y1", "x2", "y2"]].to_numpy() - r.boxes.xyxy.cpu().numpy()).max() < 1e-2)
        if same:
            take = np.arange(len(c))
        else:  # match box by box (class, score, box)
            mismatch += 1
            got_s, got_b = r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()
            take = np.full(len(c), -1)
            for j, row in enumerate(c.itertuples()):
                ok = np.flatnonzero((got_cls == row.cls) & (np.abs(got_s - row.score) < 1e-4)
                                    & (np.abs(got_b - [row.x1, row.y1, row.x2, row.y2]).max(1) < 0.5))
                if len(ok):
                    take[j] = ok[0]
            unmatched += int((take < 0).sum())
        for k in PARTS:
            block = np.full((len(c),) + arr[k].shape[1:], np.nan if k != "level" else 255, dtype=np.float64)
            block[take >= 0] = f[k][take[take >= 0]]
            arr[k][a:z] = block.astype(arr[k].dtype)
        if (i + 1) % flush_every == 0 or i == len(order) - 1:
            for x in arr.values():
                x.flush()
            done_f.write_text(str(i + 1))
            print(f"  {name} {i + 1}/{len(order)}, {time.perf_counter() - t0:.0f} s, images re-matched box by box: {mismatch}, "
                  f"rows unmatched: {unmatched}", flush=True)
    del arr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    for name in ap.parse_args(argv).sets:
        extract(name)


if __name__ == "__main__":
    main()
