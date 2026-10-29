"""A learned per-image threshold with a cost dial (PREREG_threshold.md).
STATUS (2026-10-13): did not beat one threshold; kept so its NOTES.md numbers reproduce. The
method is the dial (wad/dial.py): one threshold or per detection.

One network reads an image and predicts, at 40 thresholds, how many ground-truth objects the
detector (YOLOv8x @1280, grouped matching) would miss and how many false positives it would
produce. A dial lam ("false positives accepted to avoid one miss") then gives each image the
threshold minimising lam * misses + false positives. Detector outputs on BDD100K *train* images
are used only as labels; at test time the model sees pixels only.

  python -m wad.thresh_net select    # cache/thresh/train_names.txt (+ fetch with scripts/get_bdd100k.py)
  python -m wad.thresh_net scan      # cache/thresh/images.parquet, gt.parquet
  python -m wad.thresh_net detect    # YOLOv8x @1280 on the training frames (GPU, ~55 min)
  python -m wad.thresh_net labels    # matching + per-image miss / FP counts at 40 thresholds
  python -m wad.thresh_net train     # weights/thresh_resnet18.pt, results/threshold/train_log.json
  python -m wad.thresh_net predict   # cache/thresh/pred_{bdd,dawn}.npz (BDD100K val, DAWN)
  python -m wad.thresh_eval          # results/threshold/ evaluation
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time

import numpy as np
import pandas as pd

from wad.config import REPO, THRESH_CACHE, THRESH_RESULTS, tau_grid

CACHE = THRESH_CACHE
RESULTS = THRESH_RESULTS
WEIGHTS = REPO / "weights" / "thresh_resnet18.pt"
BDD = REPO / "data" / "BDD100K"
IN_H, IN_W = 288, 512
K_STEP = 5  # every 5th point of the 200-point grid -> 40 thresholds
DET_ID, LOSS = "v8x_1280", "group"
DRAW = {("clear", "night"): 4000, ("rainy", "night"): 1950, ("snowy", "night"): 1950, ("*", "dawn/dusk"): 2000,
        ("foggy", "*"): None, ("rainy", "daytime"): None, ("snowy", "daytime"): None}  # None = all remaining
TRAIN_BINS = {
    "clear": {"weather": ["clear"], "timeofday": ["daytime"]},
    "cloudy": {"weather": ["overcast", "partly cloudy"], "timeofday": ["daytime"]},
    "rain": {"weather": ["rainy"], "timeofday": ["daytime"]},
    "snow": {"weather": ["snowy"], "timeofday": ["daytime"]},
    "clear_night": {"weather": ["clear"], "timeofday": ["night"]},
    "rain_night": {"weather": ["rainy"], "timeofday": ["night"]},
    "snow_night": {"weather": ["snowy"], "timeofday": ["night"]},
    "dusk": {"weather": ["clear", "overcast", "partly cloudy", "rainy", "snowy", "undefined"], "timeofday": ["dawn/dusk"]},
    "fog": {"weather": ["foggy"], "timeofday": ["daytime", "night", "dawn/dusk"]},
}


def taus_k(cfg) -> np.ndarray:
    return tau_grid(cfg)[::K_STEP]


# ---------------------------------------------------------------- data
def select() -> list[str]:
    """Training frames: every train frame on disk + a stratified draw (seed 3) of the remaining pool."""
    from wad.hazard_cnn import train_attributes

    a = train_attributes()
    on_disk = sorted(p.stem for p in (BDD / "images" / "100k" / "train").glob("*.jpg"))
    have = set(on_disk)
    rng = np.random.default_rng(3)
    extra = []
    for (w, t), n in DRAW.items():
        m = np.ones(len(a), bool)
        if w != "*":
            m &= (a.weather == w).to_numpy()
        if t != "*":
            m &= (a.timeofday == t).to_numpy()
        pool = sorted(set(a[m].name) - have - set(extra))
        take = pool if n is None or n >= len(pool) else list(rng.choice(pool, n, replace=False))
        extra += sorted(take)
    names = sorted(have | set(extra))
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "train_names.txt").write_text("".join(f"{n}\n" for n in names), encoding="utf-8")
    (CACHE / "fetch_names.txt").write_text("".join(f"{n}\n" for n in sorted(set(extra))), encoding="utf-8")
    return names


def cfg():
    """BDD100K train subset, the run 2 primary detector and loss, its own cache."""
    from wad.run2 import bdd_cfg, load_run2

    v = copy.deepcopy(bdd_cfg(load_run2(), DET_ID, LOSS))
    v.data["split"] = "train"
    v.data["names_file"] = str(CACHE / "train_names.txt")
    v.data["bins"] = copy.deepcopy(TRAIN_BINS)
    v.paths["data_cache"] = CACHE
    v.paths["detections"] = CACHE / "det" / DET_ID
    v.paths["cache"] = CACHE / f"{DET_ID}__{LOSS}"
    v.paths["results"] = RESULTS / "train_set"
    v["tag"], v["dataset_name"] = f"train_{DET_ID}__{LOSS}", "bdd_train"
    return v


def scan() -> None:
    from wad import data
    from wad.config import data_path, write_json

    v = cfg()
    images, gt, report = data.scan(v)
    images.to_parquet(data_path(v, "images.parquet"), index=False)
    gt.to_parquet(data_path(v, "gt.parquet"), index=False)
    write_json(data_path(v, "data_report.json"), report)
    print(images[images.keep].groupby("weather").size().to_string())


def labels() -> None:
    """Match, then per-image (miss count, FP count) at the 40 thresholds for the training set."""
    from wad import match
    from wad.curves import ALL, Cache

    v = cfg()
    match.match_all(v)
    c = Cache(v)
    t = taus_k(v)
    n = c.n_gt(ALL)
    miss = n[:, None] - c.matched_grid(t, ALL)
    fp = c.fp_grid(t, ALL)
    np.savez_compressed(CACHE / "labels.npz", image_id=c.image_ids, path=c.images.path.to_numpy(), weather=c.weather,
                        group=c.images.group.to_numpy(), n_gt=n, miss=miss.astype(np.float32), fp=fp.astype(np.float32),
                        taus=t)
    print(f"{c.n_img} training images; mean misses at tau 0.25 ~ {miss[:, np.searchsorted(t, 0.25)].mean():.2f}, "
          f"mean FP ~ {fp[:, np.searchsorted(t, 0.25)].mean():.2f}")


# ---------------------------------------------------------------- images
def to_input(im) -> np.ndarray:
    """Centre-crop to 16:9, resize to 512x288, uint8 HWC."""
    from PIL import Image

    w, h = im.size
    if w * IN_H > h * IN_W:
        nw = round(h * IN_W / IN_H)
        im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    elif w * IN_H < h * IN_W:
        nh = round(w * IN_H / IN_W)
        im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    return np.asarray(im.resize((IN_W, IN_H), Image.BILINEAR))


def frames(paths, out) -> np.ndarray:
    """Cache input frames (uint8, n x 288 x 512 x 3) as a memory map."""
    from wad.imageio import load_rgb

    side = out.with_name(out.stem + ".paths.txt")  # the exact image order of the cache
    want = "".join(f"{p}\n" for p in paths)
    if out.exists() and side.exists() and side.read_text(encoding="utf-8") == want:
        return np.load(out, mmap_mode="r")
    tmp = out.with_name(out.stem + ".partial.npy")  # renamed only when complete
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(paths), IN_H, IN_W, 3))
    t0 = time.perf_counter()
    for i, p in enumerate(paths):
        arr[i] = to_input(load_rgb(REPO / p, "apply"))
        if (i + 1) % 2000 == 0:
            print(f"  frames {i + 1}/{len(paths)}, {time.perf_counter() - t0:.0f} s", flush=True)
    arr.flush()
    del arr
    tmp.replace(out)
    side.write_text(want, encoding="utf-8")
    return np.load(out, mmap_mode="r")


# ---------------------------------------------------------------- model
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model(k: int, pretrained=True):
    import torch.nn as nn
    import torchvision

    class ThreshNet(nn.Module):
        def __init__(self):
            super().__init__()
            r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            self.body = nn.Sequential(*list(r.children())[:-1])
            self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 2 * k))
            self.k = k

        def forward(self, x):
            z = self.head(self.body(x).flatten(1))
            sp = nn.functional.softplus
            miss = sp(z[:, : self.k]).cumsum(1)  # non-decreasing in tau
            fp = sp(z[:, self.k:]).flip(1).cumsum(1).flip(1)  # non-increasing in tau
            return miss, fp

    return ThreshNet()


def to_tensor(batch_u8, flip=None):
    import torch

    x = torch.as_tensor(np.asarray(batch_u8)).cuda().permute(0, 3, 1, 2).contiguous().float() / 255  # NCHW: much faster here
    if flip is not None:
        x = torch.where(torch.as_tensor(flip, device="cuda").view(-1, 1, 1, 1), x.flip(-1), x)
    m = torch.as_tensor(MEAN, device="cuda").view(1, 3, 1, 1)
    s = torch.as_tensor(STD, device="cuda").view(1, 3, 1, 1)
    return (x - m) / s


def poisson_nll(mu, y, w):
    """Mean over images of sum_k w_k (mu - y log mu)."""
    import torch

    mu = mu.clamp(min=1e-6)
    return ((mu - y * torch.log(mu)) * w).sum(1).mean()


def split_train_val(groups: np.ndarray) -> np.ndarray:
    """True for validation: 10% of near-duplicate groups, seed 0."""
    g = np.unique(groups)
    rng = np.random.default_rng(0)
    val_groups = set(rng.choice(g, max(1, round(0.1 * len(g))), replace=False))
    return np.array([x in val_groups for x in groups])


def train(epochs=15, batch=32, lr=3e-4) -> dict:
    import torch

    torch.manual_seed(0)
    lab = np.load(CACHE / "labels.npz", allow_pickle=True)
    X = frames(lab["path"], CACHE / "frames_train.npy")
    miss, fp = lab["miss"], lab["fp"]
    k = miss.shape[1]
    is_val = split_train_val(lab["group"])
    tr, va = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]
    w_m = torch.as_tensor(1.0 / (1.0 + miss[tr].mean(0)), device="cuda")
    w_f = torch.as_tensor(1.0 / (1.0 + fp[tr].mean(0)), device="cuda")
    torch.backends.cudnn.benchmark = True
    model = build_model(k).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    steps = epochs * math.ceil(len(tr) / batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = np.random.default_rng(0)
    M, Fp = torch.as_tensor(miss, device="cuda"), torch.as_tensor(fp, device="cuda")

    def loss_of(idx, flip=None):
        # mixed precision for the network (4 GB GPU); the Poisson loss in fp32
        with torch.autocast("cuda", dtype=torch.float16):
            pm, pf = model(to_tensor(X[np.sort(idx)], flip))  # (channels_last is 7x slower on this GPU)
        j = torch.as_tensor(np.sort(idx), device="cuda")
        return poisson_nll(pm.float(), M[j], w_m) + poisson_nll(pf.float(), Fp[j], w_f)

    log, best = [], (float("inf"), None)
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        order = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            loss = loss_of(idx, rng.random(len(idx)) < 0.5)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            vl = sum(float(loss_of(va[i:i + 64])) * len(va[i:i + 64]) for i in range(0, len(va), 64)) / len(va)
        log.append(dict(epoch=ep + 1, train_loss=tot / len(tr), val_loss=vl, seconds=time.perf_counter() - t0))
        print(json.dumps(log[-1]), flush=True)
        if vl < best[0]:
            best = (vl, {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()})
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=best[1], k=k, taus=lab["taus"], input=(IN_H, IN_W)), WEIGHTS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = dict(log=log, best_val_loss=best[0], n_train=int(len(tr)), n_val=int(len(va)), prereg="PREREG_threshold.md")
    (RESULTS / "train_log.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def predict() -> None:
    """Predicted miss / FP curves for BDD100K val and DAWN (every kept image)."""
    import torch

    ck = torch.load(WEIGHTS, map_location="cuda", weights_only=False)
    model = build_model(ck["k"], pretrained=False).cuda().eval()
    model.load_state_dict(ck["state_dict"])
    for name, p in [("dawn", "cache/images.parquet"), ("bdd", "cache/bdd/images.parquet")]:
        im = pd.read_parquet(REPO / p)
        im = im[im.keep].sort_values("image_id").reset_index(drop=True)
        X = frames(im.path.to_numpy(), CACHE / f"frames_{name}.npy")
        pm, pf = [], []
        with torch.no_grad():
            for i in range(0, len(im), 64):
                a, b = model(to_tensor(X[i:i + 64]))
                pm.append(a.cpu().numpy())
                pf.append(b.cpu().numpy())
        np.savez_compressed(CACHE / f"pred_{name}.npz", image_id=im.image_id.to_numpy(), weather=im.weather.to_numpy(),
                            miss=np.concatenate(pm), fp=np.concatenate(pf), taus=ck["taus"])
        print(name, len(im), "images predicted")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["select", "scan", "detect", "labels", "train", "predict"])
    args = ap.parse_args(argv)
    if args.command == "select":
        n = select()
        print(f"{len(n)} training frames listed; fetch the new ones with "
              f"scripts/get_bdd100k.py --no-val --train-list cache/thresh/fetch_names.txt")
    elif args.command == "scan":
        scan()
    elif args.command == "detect":
        from wad import detect

        detect.run(cfg())
    elif args.command == "labels":
        labels()
    elif args.command == "train":
        train()
    else:
        predict()


if __name__ == "__main__":
    main()
