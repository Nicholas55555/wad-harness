"""O3 version C: a trained visibility estimator (PREREG_visibility.md).

A ResNet-18 learns to read visibility from a single camera frame. Labels are physics-defined, not
weather tags: clear-daytime BDD100K *train* frames are degraded synthetically with known
parameters (scattering veil from metric depth, rain streaks, windshield drops, wheel spray around
labelled vehicles, darkness with glare and wet-road reflections) and the network regresses
log10 MOR, log10 light level, a 7x13 visibility map and the effective visibility V_eff.

Hazard: h_v2 = -log10(predicted V_eff). Like wad/hazard.py, nothing here reads detector output
(vehicle boxes for spray come from ground-truth labels of training frames only).

  python -m wad.vis_estimator select    # frame lists (extra clear-day bases; v3 night development set)
  python -m wad.vis_estimator prepare   # base frames, metric depth, vehicle boxes -> cache/vis2/
  python -m wad.vis_estimator boxes     # only re-derive vehicle boxes from labels (amendment 3)
  python -m wad.vis_estimator train     # weights/visibility_resnet18.pt, results/vis2/train_log.json
  python -m wad.vis_estimator predict   # cache/vis2/pred_{bdd,dawn}.parquet
  add --version 3 to train / predict / dev_check for the night-relighting version (cache/vis3, results/vis3)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from wad.imageio import load_rgb

REPO = Path(__file__).resolve().parents[1]
BDD = REPO / "data" / "BDD100K"
CACHE = REPO / "cache" / "vis2"
RESULTS = REPO / "results" / "vis2"
WEIGHTS = REPO / "weights" / "visibility_resnet18.pt"


def paths(version: int) -> dict:
    """Per-version outputs. The training cache (base frames, depth, boxes) is shared in cache/vis2."""
    if version == 2:
        return dict(weights=WEIGHTS, results=RESULTS, preds=CACHE)
    return dict(weights=REPO / "weights" / f"visibility_v{version}_resnet18.pt",
                results=REPO / "results" / f"vis{version}", preds=REPO / "cache" / f"vis{version}")
H, W = 252, 448  # working size (16:9)
INPUT = (224, 400)
GRID = (7, 13)  # ResNet-18 layer4 at 224x400
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
FAR_START, FAR_SCALE = 40.0, 12.0  # smooth extension of the ~80 m depth ceiling (see amendment 2)
MOR_CLEAR, V_MIN, V_MAX = 10000.0, 30.0, 10000.0
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
VEHICLES = ("car", "truck", "bus")
MAX_VEH = 48  # every labelled vehicle box (at most 46 in a base frame); amendment 3
DARK_LUMA = 52.0  # base frames darker than this are night scenes mis-tagged daytime; amendment 3


# ---------------------------------------------------------------- frame selection
def select() -> None:
    """Write the frame lists this module needs (labels only; fetch the images with
    scripts/get_bdd100k.py --no-val --train-list FILE):
      cache/vis2/extra_clear_names.txt  3,000 clear-daytime BDD100K train frames outside the O3 subset (seed 1)
      cache/vis3/dev_night.csv (+ _names.txt)  v3 development set: 300 clear, 50 rainy, 50 snowy train
                                                night frames (seed 2)"""
    from wad.hazard_cnn import train_attributes

    a = train_attributes()
    o3 = set(pd.read_csv(REPO / "cache" / "o3" / "train_subset.csv").name)
    pool = sorted(set(a[(a.weather == "clear") & (a.timeofday == "daytime")].name) - o3)
    extra = sorted(np.random.default_rng(1).choice(pool, 3000, replace=False))
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "extra_clear_names.txt").write_text("".join(f"{n}\n" for n in extra), encoding="utf-8")
    rng = np.random.default_rng(2)
    rows = []
    for w, n in (("clear", 300), ("rainy", 50), ("snowy", 50)):
        night = sorted(a[(a.timeofday == "night") & (a.weather == w)].name)
        rows += [dict(name=x, weather=w) for x in rng.choice(night, n, replace=False)]
    dev = REPO / "cache" / "vis3"
    dev.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dev / "dev_night.csv", index=False)
    (dev / "dev_night_names.txt").write_text("".join(f"{r['name']}\n" for r in rows), encoding="utf-8")


# ---------------------------------------------------------------- base frames
def base_names() -> pd.DataFrame:
    sub = pd.read_csv(REPO / "cache" / "o3" / "train_subset.csv")
    names = list(sub[sub.cls == "clear"].name) + [l.strip() for l in open(CACHE / "extra_clear_names.txt") if l.strip()]
    df = pd.DataFrame(dict(name=sorted(set(names))))
    rng = np.random.default_rng(0)
    df["split"] = np.where(rng.random(len(df)) < 0.1, "val", "train")
    return df


def vehicle_boxes(name: str) -> np.ndarray:
    d = json.loads((BDD / "labels" / "100k" / "train" / f"{name}.json").read_text(encoding="utf-8"))
    out = []
    for fr in d["frames"]:
        for o in fr["objects"]:
            if o["category"] in VEHICLES and "box2d" in o:
                b = o["box2d"]
                out.append([b["x1"] * W / 1280, b["y1"] * H / 720, b["x2"] * W / 1280, b["y2"] * H / 720])
    b = np.array(out, dtype=np.float32).reshape(-1, 4)
    b = b[((b[:, 3] - b[:, 1]) > 0) & ((b[:, 2] - b[:, 0]) > 0)]
    return b[np.argsort(-(b[:, 3] - b[:, 1]), kind="stable")][:MAX_VEH]


def prepare_boxes(names) -> None:
    """Vehicle boxes of every base frame -> boxes.npy, nbox.npy (labels only, no depth needed)."""
    boxes = np.zeros((len(names), MAX_VEH, 4), np.float32)
    nbox = np.zeros(len(names), np.int16)
    for i, name in enumerate(names):
        b = vehicle_boxes(name)
        boxes[i, :len(b)] = b
        nbox[i] = len(b)
    np.save(CACHE / "boxes.npy", boxes)
    np.save(CACHE / "nbox.npy", nbox)


def base_luma(frames) -> np.ndarray:
    """Mean Rec. 601 luma (0-255) of each cached base frame."""
    w = np.array([0.299, 0.587, 0.114], np.float32)
    return np.array([float((np.asarray(f, dtype=np.float32) @ w).mean()) for f in frames])


def prepare() -> None:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    CACHE.mkdir(parents=True, exist_ok=True)
    df = base_names()
    df.to_csv(CACHE / "bases.csv", index=False)
    n = len(df)
    frames = np.lib.format.open_memmap(CACHE / "frames.npy", mode="w+", dtype=np.uint8, shape=(n, H, W, 3))
    depth = np.lib.format.open_memmap(CACHE / "depth.npy", mode="w+", dtype=np.float16, shape=(n, H, W))
    proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).cuda().eval()
    t0 = time.perf_counter()
    for i, name in enumerate(df.name):
        im = load_rgb(BDD / "images" / "100k" / "train" / f"{name}.jpg", "apply")
        frames[i] = np.asarray(im.resize((W, H), Image.BILINEAR))
        x = proc(images=im, return_tensors="pt").to("cuda")
        with torch.no_grad():
            d = model(**x).predicted_depth[None]
        d = torch.nn.functional.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        depth[i] = d.clamp(1, 80).cpu().numpy().astype(np.float16)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n} frames, {time.perf_counter() - t0:.0f} s", flush=True)
    frames.flush()
    depth.flush()
    prepare_boxes(df.name)
    print(f"prepared {n} base frames in {time.perf_counter() - t0:.0f} s")


# ---------------------------------------------------------------- synthesis (GPU)
def far_depth(d):
    """The depth model saturates near 80 m. Beyond 40 m, extend depth smoothly and monotonically
    (70 m -> ~490 m, 80 m -> ~1,130 m) so far content fades progressively instead of jumping to full
    airlight, then smooth slightly in log space to avoid jagged sky boundaries."""
    import torch

    d = torch.where(d <= FAR_START, d, FAR_START * torch.exp((d - FAR_START) / FAR_SCALE))
    ld = torch.log(d)[None, None]
    k = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32, device=d.device)
    k = (k[:, None] * k[None, :]) / 256
    ld = torch.nn.functional.conv2d(torch.nn.functional.pad(ld, (2, 2, 2, 2), mode="replicate"), k[None, None])
    return torch.exp(ld[0, 0])



class Synth:
    """Draws degradations per the pre-registration and returns images and physics targets."""

    def __init__(self, device="cuda", version=2):
        import torch

        self.t = torch
        self.dev = device
        self.version = version
        yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                                torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
        self.yy, self.xx = yy, xx

    def _u(self, g, lo, hi, n=None):
        t = self.t
        return lo + (hi - lo) * t.rand(n or 1, generator=g, device=self.dev)

    def _logu(self, g, lo, hi):
        return float(10 ** self._u(g, math.log10(lo), math.log10(hi)))

    def _smooth_noise(self, g, scale=16):
        t = self.t
        z = t.rand(1, 1, H // scale + 2, W // scale + 2, generator=g, device=self.dev)
        return t.nn.functional.interpolate(z, size=(H, W), mode="bicubic", align_corners=False)[0, 0].clamp(0, 1)

    def _line_kernel(self, length, angle):
        t = self.t
        k = int(length) | 1
        ker = t.zeros(k, k, device=self.dev)
        c = k // 2
        for s in t.linspace(-c, c, 2 * k, device=self.dev):
            x = int(round(c + float(s) * math.sin(angle)))
            y = int(round(c + float(s) * math.cos(angle)))
            if 0 <= x < k and 0 <= y < k:
                ker[y, x] = 1.0
        return ker / ker.sum()

    def _glow(self, g, cx, cy, sc, inten, refl, sh=None, halo=(0.05, 0.2), sh_range=(8, 20), streak=(0.2, 0.6)):
        """A light source: saturated core, faint halo, optional reflection streak on the road below.
        Defaults are v2's; v3 night passes smaller halos and weaker streaks."""
        t = self.t
        sh = float(self._u(g, *sh_range)) if sh is None else sh
        r2 = (self.yy - cy) ** 2 + (self.xx - cx) ** 2
        blob = inten * t.exp(-r2 / (2 * sc * sc)) + float(self._u(g, *halo)) * inten * t.exp(-r2 / (2 * sh * sh))
        if refl:
            y0 = max(cy, 0.55 * H)  # reflections lie on the road surface
            sx = float(self._u(g, 1.5, 4.0))
            line = t.exp(-((self.xx - cx) ** 2) / (2 * sx * sx)) * (self.yy > y0).float()
            blob = blob + float(self._u(g, *streak)) * line * t.exp(-(self.yy - y0).clamp(min=0) / (0.3 * H))
        return blob

    def _dim(self, J, k, g, n_lights):
        """v2 darkness (v3: dusk): radiance x k x partial auto-gain, noise, then light sources (number
        drawn from [n_lights[0], n_lights[1]) after the noise; draw order identical to the trained v2)."""
        t = self.t
        gain = min(k ** -0.5, 40.0)  # partial auto-exposure
        J = J * k * gain
        J = J + t.randn(J.shape, generator=g, device=self.dev) * (0.01 * J.clamp(min=0).sqrt() * gain ** 0.5 + 0.004)
        nl = int(self._u(g, *n_lights))
        light = t.zeros(3, H, W, device=self.dev)
        refl = float(self._u(g, 0, 1)) < 0.5
        for _ in range(nl):
            head = float(self._u(g, 0, 1)) < 0.5  # headlights / tail lights near the road, else street lights
            cy = float(self._u(g, 0.5 * H, 0.75 * H)) if head else float(self._u(g, 0.15 * H, 0.5 * H))
            cx = float(self._u(g, 0, W))
            sc = float(self._u(g, 1.5, 4.0))
            sh = float(self._u(g, 8, 20))
            inten = float(self._u(g, 1.0, 3.0))
            col = [(1.0, 0.75, 0.4), (1.0, 1.0, 1.0), (1.0, 0.15, 0.1)][int(self._u(g, 0, 2.999))]
            light = light + self._glow(g, cx, cy, sc, inten, refl, sh)[None] * t.tensor(col, device=self.dev).view(3, 1, 1)
        return J + light

    def _night(self, J, depth_m, d_ext, boxes, g):
        """v3 night relighting (PREREG_visibility.md, version 3). J: linear daytime radiance, read as
        albedo x daylight. Returns the exposed night image (linear, before the sRGB curve) and k."""
        t = self.t
        F = t.nn.functional
        yy, xx = self.yy, self.xx
        dev = self.dev
        sodium = t.tensor([1.0, 0.75, 0.45], device=dev).view(3, 1, 1)
        white = t.tensor([1.0, 1.0, 0.95], device=dev).view(3, 1, 1)
        bluish = t.tensor([0.75, 0.85, 1.0], device=dev).view(3, 1, 1)
        amb_tint = sodium if float(self._u(g, 0, 1)) < 0.5 else bluish
        a = self._logu(g, 0.0005, 0.01)  # ambient: moon, sky glow, diffuse urban light
        Es = t.full((H, W), a, device=dev)  # illumination relative to daylight (for k)
        E = a * amb_tint.expand(3, H, W)  # coloured illumination (for the image)
        lamps = []
        if float(self._u(g, 0, 1)) < 0.9:  # ego headlights: inverse-square fall-off on the scene
            hz = float(self._u(g, 0.42, 0.55)) * H
            below = t.sigmoid((yy - hz) / 3)
            th = (xx - W / 2) / (W / 2)
            sig = float(self._u(g, 0.25, 0.45))
            e10 = float(self._u(g, 0.01, 0.1))  # illumination at 10 m on axis
            Eh = e10 * 100 * t.exp(-0.5 * (th / sig) ** 2) / depth_m.clamp(min=3) ** 2 * below
            Es = Es + Eh
            E = E + Eh[None] * white
        if float(self._u(g, 0, 1)) < 0.6:  # street lights: pools on the ground, lamps above
            tint = sodium if float(self._u(g, 0, 1)) < 0.5 else white
            for _ in range(int(self._u(g, 2, 9))):
                cx = float(self._u(g, 0, W))
                gy = float(self._u(g, 0.55 * H, 0.95 * H))
                amp = self._logu(g, 0.003, 0.05)
                sx = float(self._u(g, 15, 60)) * gy / H
                pool = amp * t.exp(-((xx - cx) ** 2 / (2 * sx * sx) + (yy - gy) ** 2 / (2 * (0.45 * sx) ** 2)))
                Es = Es + pool
                E = E + pool[None] * tint
                lamps.append((cx, float(self._u(g, 0.08 * H, 0.45 * H)), tint))
        drive = yy >= (GRID[0] - 5) / GRID[0] * H
        zone = drive & (depth_m >= 10) & (depth_m <= 60)
        k = float(Es[zone].mean()) if int(zone.sum()) >= 50 else float(Es[drive].median())
        k = min(max(k, 3e-4), 1.0)
        L = J * E
        sky = t.sigmoid((t.log(d_ext) - math.log(300.0)) / 0.25)[None]
        L = L * (1 - sky) + sky * (a * float(self._u(g, 0.3, 1.5))) * amb_tint
        # dashcam auto-exposure: night frames stay dark
        lum = float((0.2126 * L[0] + 0.7152 * L[1] + 0.0722 * L[2]).mean())
        target = float(self._u(g, 0.03, 0.15)) ** 2.2
        gain = min(max(target / max(lum, 1e-8), 1.0), 200.0)
        L = L * gain
        gn = gain ** 0.5 / 4
        L = L + t.randn(L.shape, generator=g, device=dev) * (0.01 * L.clamp(min=0).sqrt() * gn + 0.002 * gn)
        if float(self._u(g, 0, 1)) < 0.5:  # in-camera denoising
            L = F.avg_pool2d(F.pad(L[None], (1, 1, 1, 1), mode="replicate"), 3, stride=1)[0]
        refl = float(self._u(g, 0, 1)) < 0.25  # wet road
        small = dict(halo=(0.02, 0.08), sh_range=(3, 8), streak=(0.05, 0.2))
        light = t.zeros(3, H, W, device=dev)
        for cx, cy, tint in lamps:
            light = light + self._glow(g, cx, cy, float(self._u(g, 1.5, 4.0)), float(self._u(g, 1.0, 3.0)), refl, **small)[None] * tint
        red = t.tensor([1.0, 0.15, 0.1], device=dev).view(3, 1, 1)
        for b in boxes:  # head / tail lights of the labelled vehicles
            x1, y1, x2, y2 = [float(v) for v in b]
            w, h = x2 - x1, y2 - y1
            if h < 6:
                continue
            col = red if float(self._u(g, 0, 1)) < 0.7 else white
            sc = min(max(0.05 * w, 1.0), 4.0)
            inten = float(self._u(g, 0.6, 2.0))
            r = refl and float(self._u(g, 0, 1)) < 0.3  # only some vehicle lights mirror on the road
            for cx in (x1 + 0.2 * w, x2 - 0.2 * w):
                light = light + self._glow(g, cx, y1 + 0.55 * h, sc, inten, r, **small)[None] * col
        colours = [(1.0, 0.75, 0.4), (1.0, 1.0, 1.0), (1.0, 0.15, 0.1), (0.2, 1.0, 0.5)]
        for _ in range(int(self._u(g, 0, 10))):  # signals, windows, signs
            col = t.tensor(colours[int(self._u(g, 0, 3.999))], device=dev).view(3, 1, 1)
            blob = self._glow(g, float(self._u(g, 0, W)), float(self._u(g, 0.05 * H, 0.6 * H)),
                              float(self._u(g, 1.0, 2.5)), float(self._u(g, 0.5, 2.0)), False, **small)
            light = light + blob[None] * col
        return L + light, k

    def sample(self, frame_u8, depth, boxes, g):
        """One degraded sample. frame_u8 (H,W,3) uint8 tensor, depth (H,W) metres, boxes (n,4)."""
        t = self.t
        F = t.nn.functional
        J = (frame_u8.float() / 255).permute(2, 0, 1) ** 2.2  # linear light
        d = far_depth(depth.float())
        tgt = {}
        # scattering veil
        if float(self._u(g, 0, 1)) < 0.75:
            mor = self._logu(g, 30, 5000)
            beta = 3.912 / mor
            a = float(self._u(g, 0.65, 0.95))
            tint = t.tensor([1.0, 0.85, 0.6] if float(self._u(g, 0, 1)) < 0.15 else [1.0, 1.0, 1.0], device=self.dev)
            A = ((a * tint) ** 2.2).view(3, 1, 1)
            tr = t.exp(-beta * d)[None]
            J = J * tr + A * (1 - tr)
        else:
            mor, beta = MOR_CLEAR, 3.912 / MOR_CLEAR
        beta_field = t.full((H, W), beta, device=self.dev)
        # rain: spray, streaks, windshield drops
        if float(self._u(g, 0, 1)) < 0.35:
            r = float(self._u(g, 0, 1))
            if float(self._u(g, 0, 1)) < 0.6 and len(boxes):
                tex = 0.5 + 0.5 * self._smooth_noise(g, 8)
                for b in boxes:
                    x1, y1, x2, y2 = [float(v) for v in b]
                    w, h = x2 - x1, y2 - y1
                    cx, cy, rx, ry = (x1 + x2) / 2, y2 - 0.25 * h, 0.7 * w, 0.5 * h
                    e = t.exp(-2 * (((self.xx - cx) / rx) ** 2 + ((self.yy - cy) / ry) ** 2)) * tex
                    bs = 3.912 / self._logu(g, 15, 150)
                    ts = t.exp(-bs * 5.0 * e)[None]  # ~5 m of plume along the line of sight
                    As = (0.8 ** 2.2)
                    J = J * ts + As * (1 - ts)
                    beta_field = beta_field + bs * e
            n_dots = int(400 + 1600 * r)
            dots = t.zeros(1, 1, H, W, device=self.dev)
            iy = t.randint(0, H, (n_dots,), generator=g, device=self.dev)
            ix = t.randint(0, W, (n_dots,), generator=g, device=self.dev)
            # max over colliding indices: deterministic on CUDA, unlike plain index assignment
            dots.view(-1).scatter_reduce_(0, iy * W + ix, self._u(g, 0.5, 1.0, n_dots), reduce="amax")
            ker = self._line_kernel((10 + 30 * float(self._u(g, 0, 1))) * (0.5 + r), math.radians(float(self._u(g, -20, 20))))
            streak = F.conv2d(dots, ker[None, None], padding=ker.shape[0] // 2)[0, 0]
            streak = streak / streak.max().clamp(min=1e-6)
            J = (J + (0.15 + 0.35 * r) * streak[None] * 0.6).clamp(0, 1)
            if float(self._u(g, 0, 1)) < 0.5:
                nd = int(self._u(g, 20, 150))
                cy_ = self._u(g, 0, H, nd)[:, None, None]
                cx_ = self._u(g, 0, W, nd)[:, None, None]
                rr = self._u(g, 2, 12, nd)[:, None, None]
                dist = (((self.yy[None] - cy_) ** 2 + (self.xx[None] - cx_) ** 2).sqrt() / rr).min(0).values
                mask = (1 - dist).clamp(0, 1)[None] ** 0.5
                blurred = F.avg_pool2d(F.pad(J[None], (6, 6, 6, 6), mode="replicate"), 13, stride=1)[0]
                J = J * (1 - 0.8 * mask) + (blurred * 1.15).clamp(0, 1) * 0.8 * mask
        # darkness, then glare and wet-road reflections in the exposed image
        k = 1.0
        if self.version == 2:
            if float(self._u(g, 0, 1)) < 0.35:
                k = self._logu(g, 0.003, 0.3)
                J = self._dim(J, k, g, (3, 20))
        elif float(self._u(g, 0, 1)) < 0.40:
            if float(self._u(g, 0, 1)) < 0.75:
                J, k = self._night(J, depth.float(), d, boxes, g)
            else:  # dusk: v2's uniform dimming, fewer lights
                k = self._logu(g, 0.03, 0.5)
                J = self._dim(J, k, g, (0, 6))
        img = J.clamp(0, 1) ** (1 / 2.2)
        # targets
        cell = F.adaptive_avg_pool2d(beta_field[None, None], GRID)[0, 0]
        mor_map = (3.912 / cell).clamp(V_MIN, V_MAX)
        road = mor_map[GRID[0] - 5:]  # bottom five of seven rows: the driving zone
        mor_road = float(t.quantile(road.flatten(), 0.10))
        v_light = float(np.clip(2000.0 * k ** 0.5, V_MIN, V_MAX))
        tgt["log_mor"] = math.log10(min(max(mor, V_MIN), V_MAX))
        tgt["log_k"] = math.log10(k)
        tgt["log_veff"] = math.log10(min(mor_road, v_light))
        tgt["map"] = t.log10(mor_map)
        return img, tgt

    def camera(self, imgs, g, train=True):
        """JPEG 40-95, slight blur, flip (flip also mirrors the target map)."""
        import torchvision

        t = self.t
        out, flips = [], []
        for im in imgs:
            u8 = (im.clamp(0, 1) * 255).round().to(t.uint8).cpu()
            q = int(self._u(g, 40, 95))
            u8 = torchvision.io.decode_jpeg(torchvision.io.encode_jpeg(u8, quality=q)).to(self.dev).float() / 255
            if float(self._u(g, 0, 1)) < 0.3:
                u8 = t.nn.functional.avg_pool2d(t.nn.functional.pad(u8[None], (1, 1, 1, 1), mode="replicate"), 3, stride=1)[0]
            f = train and float(self._u(g, 0, 1)) < 0.5
            out.append(u8.flip(-1) if f else u8)
            flips.append(f)
        return t.stack(out), flips


# ---------------------------------------------------------------- model
def build_model(pretrained=True):
    import torch.nn as nn
    import torchvision

    class VisNet(nn.Module):
        def __init__(self):
            super().__init__()
            r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.body = nn.Sequential(*list(r.children())[:-2])
            self.heads = nn.Linear(512, 3)  # log_mor, log_k, log_veff
            self.map = nn.Conv2d(512, 1, 1)

        def forward(self, x):
            f = self.body(x)
            v = self.heads(f.mean((2, 3)))
            m = nn.functional.interpolate(self.map(f), size=GRID, mode="bilinear", align_corners=False)[:, 0]
            return v, m

    return VisNet()


def normalise(x):
    import torch

    x = torch.nn.functional.interpolate(x, size=INPUT, mode="bilinear", align_corners=False)
    m = torch.as_tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.as_tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def make_batch(syn, frames, depth, boxes, nbox, idx, g, train=True):
    import torch

    imgs, tg = [], []
    for i in idx:
        im, t_ = syn.sample(torch.as_tensor(np.asarray(frames[i])).to(syn.dev), torch.as_tensor(np.asarray(depth[i])).to(syn.dev),
                            torch.as_tensor(boxes[i, :nbox[i]]).to(syn.dev), g)
        imgs.append(im)
        tg.append(t_)
    x, flips = syn.camera(imgs, g, train)
    y = torch.tensor([[t_["log_mor"], t_["log_k"], t_["log_veff"]] for t_ in tg], device=syn.dev, dtype=torch.float32)
    m = torch.stack([t_["map"].flip(-1) if f else t_["map"] for t_, f in zip(tg, flips)])
    return normalise(x), y, m


def train(epochs=20, batch=32, lr=3e-4, version=2) -> dict:
    import torch

    torch.manual_seed(0)
    bases = pd.read_csv(CACHE / "bases.csv")
    frames = np.load(CACHE / "frames.npy", mmap_mode="r")
    depth = np.load(CACHE / "depth.npy", mmap_mode="r")
    boxes, nbox = np.load(CACHE / "boxes.npy"), np.load(CACHE / "nbox.npy")
    bright = base_luma(frames) >= DARK_LUMA  # amendment 3: drop night scenes tagged daytime
    tr = np.nonzero((bases.split.to_numpy() == "train") & bright)[0]
    va = np.nonzero((bases.split.to_numpy() == "val") & bright)[0]
    syn = Synth(version=version)
    out_paths = paths(version)
    model = build_model().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(tr) / batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)  # per step, no warm-up
    g = torch.Generator(device="cuda").manual_seed(0)
    rng = np.random.default_rng(0)
    # fixed validation set: two seeded degradations per validation base frame
    gv = torch.Generator(device="cuda").manual_seed(12345)
    val = [make_batch(syn, frames, depth, boxes, nbox, va[i:i + batch], gv, train=False)
           for _ in range(2) for i in range(0, len(va), batch)]
    log, best = [], (float("inf"), None)
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        order = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(order), batch):
            x, y, m = make_batch(syn, frames, depth, boxes, nbox, order[i:i + batch], g)
            v, mp = model(x)
            # equal weights: log MOR, log k, log V_eff and the map each enter as one mean L1 term
            loss = (v - y).abs().mean(0).sum() + (mp - m).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(x)
        model.eval()
        err = []
        with torch.no_grad():
            for x, y, m in val:
                v, mp = model(x)
                err.append(torch.cat([(v - y).abs(), (mp - m).abs().mean((1, 2))[:, None]], 1).cpu().numpy())
        err = np.concatenate(err)
        mae = dict(log_mor=float(err[:, 0].mean()), log_k=float(err[:, 1].mean()), log_veff=float(err[:, 2].mean()),
                   map=float(err[:, 3].mean()))
        vloss = float(err.mean(0).sum())
        log.append(dict(epoch=ep + 1, train_loss=tot / len(order), val_mae=mae, val_loss=vloss,
                        seconds=time.perf_counter() - t0))
        print(json.dumps(log[-1]), flush=True)
        if vloss < best[0]:
            best = (vloss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    out_paths["weights"].parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=best[1], grid=GRID, input=INPUT, version=version), out_paths["weights"])
    out_paths["results"].mkdir(parents=True, exist_ok=True)
    out = dict(log=log, best_val_loss=best[0], n_train=int(len(tr)), n_val=int(len(va)),
               n_dark_excluded=int((~bright).sum()),
               prereg="PREREG_visibility.md (amendment 3)" if version == 2 else
               f"PREREG_visibility.md (Version {version}; not blind: designed after v2's results)")
    out["version"] = version
    (out_paths["results"] / "train_log.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ---------------------------------------------------------------- prediction on real images
def to_working(im, crop=True):
    """Real image -> the 448x252 working frame. crop=True (primary, amendment 3) centre-crops to 16:9
    first, the geometry of every training frame; crop=False stretches (DAWN sensitivity check)."""
    from PIL import Image

    if crop:
        w, h = im.size
        if w * H > h * W:  # wider than 16:9
            nw = round(h * W / H)
            im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
        elif w * H < h * W:  # taller than 16:9
            nh = round(w * H / W)
            im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    return np.asarray(im.resize((W, H), Image.BILINEAR))


def load_model(version=2):
    import torch

    ck = torch.load(paths(version)["weights"], map_location="cuda")
    model = build_model(pretrained=False).cuda().eval()
    model.load_state_dict(ck["state_dict"])
    return model


def predict_arrays(model, arr: np.ndarray):
    """(n, H, W, 3) uint8 working frames -> (n, 3) [log MOR, log k, log V_eff] and (n, 7, 13) log map."""
    import torch

    x = torch.as_tensor(arr).cuda().permute(0, 3, 1, 2).float() / 255
    with torch.no_grad():
        v, mp = model(normalise(x))
    return v, mp


def predict_images(images: pd.DataFrame, name: str, batch=64, crop=True, version=2) -> pd.DataFrame:
    import torch

    model = load_model(version)
    rows = []
    files = [REPO / p for p in images.path]
    for i in range(0, len(files), batch):
        arr = np.stack([to_working(load_rgb(p, "apply"), crop) for p in files[i:i + batch]])
        v, mp = predict_arrays(model, arr)
        mor_road = torch.quantile((10 ** mp[:, GRID[0] - 5:]).flatten(1), 0.10, dim=1)
        for j in range(len(v)):
            rows.append(dict(log_mor=float(v[j, 0]), log_k=float(v[j, 1]), log_veff=float(v[j, 2]),
                             log_mor_road_map=float(torch.log10(mor_road[j]))))
    df = pd.DataFrame(rows)
    df.insert(0, "image_id", images.image_id.to_numpy())
    df.insert(1, "weather", images.weather.to_numpy())
    df[f"h_v{version}"] = -df.log_veff
    out = paths(version)["preds"]
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"pred_{name}{'' if crop else '_stretch'}.parquet", index=False)
    return df


def dev_check(version=3, n_day=400) -> dict:
    """Version 3 development check (PREREG_visibility.md): predicted light level on the 400 BDD100K
    *train* night frames of cache/vis3/dev_night.csv vs 400 train daytime frames that are not base
    frames (cloudy, from the O3 subset). Never uses BDD100K val or DAWN."""
    from sklearn.metrics import roc_auc_score

    model = load_model(version)
    night = pd.read_csv(REPO / "cache" / "vis3" / "dev_night.csv").name.tolist()
    sub = pd.read_csv(REPO / "cache" / "o3" / "train_subset.csv")
    bases = set(pd.read_csv(CACHE / "bases.csv").name)
    day = [n for n in sub[sub.cls == "cloudy"].name if n not in bases][:n_day]
    out = {}
    for tag, names in (("night", night), ("day", day)):
        ks = []
        for i in range(0, len(names), 64):
            arr = np.stack([to_working(load_rgb(BDD / "images" / "100k" / "train" / f"{n}.jpg", "apply")) for n in names[i:i + 64]])
            ks.append(predict_arrays(model, arr)[0][:, 1].cpu().numpy())
        out[tag] = np.concatenate(ks)
    res = dict(version=version, n_night=len(out["night"]), n_day=len(out["day"]),
               median_k_night=float(10 ** np.median(out["night"])), median_k_day=float(10 ** np.median(out["day"])),
               auc_night_vs_day=float(roc_auc_score(np.r_[np.ones(len(out["night"])), np.zeros(len(out["day"]))],
                                                    -np.r_[out["night"], out["day"]])))
    r = paths(version)["results"]
    r.mkdir(parents=True, exist_ok=True)
    (r / "dev_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["select", "prepare", "boxes", "train", "predict", "dev_check"])
    ap.add_argument("--version", type=int, default=2, choices=[2, 3], help="2: as pre-registered; 3: night relighting")
    args = ap.parse_args(argv)
    if args.command == "select":
        select()
    elif args.command == "prepare":
        prepare()
    elif args.command == "boxes":
        prepare_boxes(pd.read_csv(CACHE / "bases.csv").name)
    elif args.command == "train":
        train(version=args.version)
    elif args.command == "dev_check":
        print(json.dumps(dev_check(args.version), indent=1))
    else:
        for name, p in [("dawn", "cache/images.parquet"), ("bdd", "cache/bdd/images.parquet")]:
            im = pd.read_parquet(REPO / p)
            im = im[im.keep].sort_values("image_id").reset_index(drop=True)
            d = predict_images(im, name, version=args.version)
            print(name, d.groupby("weather")[["log_mor", "log_k", "log_veff"]].median().round(2).to_string())
            if name == "dawn":  # sensitivity: stretch instead of crop (BDD100K is 16:9, unaffected)
                predict_images(im, name, crop=False, version=args.version)


if __name__ == "__main__":
    main()
