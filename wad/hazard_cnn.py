"""O3 version B: a learned camera hazard estimator (PREREG_o3.md).

A ResNet-18 (ImageNet weights) with a 5-way weather head (clear, cloudy, rain, snow, fog), trained
on daytime BDD100K *train* images only. Its hazard scalar is h_B = p(rain) + p(snow) + p(fog).
Like wad/hazard.py, this module reads pixels and weather tags only: it imports nothing from the
detection side (tests/test_hazard.py) and never sees a DAWN or BDD100K-val image while training.

  python -m wad.hazard_cnn select    # cache/o3/train_subset.csv (+ list for scripts/get_bdd100k.py)
  python -m wad.hazard_cnn train     # weights/hazard_resnet18.pt, results/o3/train_log.json
  python -m wad.hazard_cnn predict   # cache/o3/h_dawn.parquet, cache/o3/h_bdd_val.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wad.imageio import load_rgb

REPO = Path(__file__).resolve().parents[1]
BDD = REPO / "data" / "BDD100K"
CACHE = REPO / "cache" / "o3"
RESULTS = REPO / "results" / "o3"
WEIGHTS = REPO / "weights" / "hazard_resnet18.pt"
CLASSES = ["clear", "cloudy", "rain", "snow", "fog"]
TAG_TO_CLASS = {"clear": "clear", "overcast": "cloudy", "partly cloudy": "cloudy", "rainy": "rain",
                "snowy": "snow", "foggy": "fog"}
PER_CLASS, SEED, HOLDOUT = 1500, 0, 0.10
STORE = (252, 448)  # stored frame size (16:9); training crops are resized to INPUT
INPUT = (224, 400)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def train_attributes() -> pd.DataFrame:
    """Weather and time-of-day tags of every BDD100K train image (cached once; 70k small JSONs)."""
    p = CACHE / "train_attributes.parquet"
    if p.exists():
        return pd.read_parquet(p)
    rows = []
    for f in sorted((BDD / "labels" / "100k" / "train").glob("*.json")):
        a = json.loads(f.read_text(encoding="utf-8")).get("attributes", {})
        rows.append(dict(name=f.stem, weather=a.get("weather"), timeofday=a.get("timeofday")))
    df = pd.DataFrame(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df


def select() -> pd.DataFrame:
    a = train_attributes()
    a = a[(a.timeofday == "daytime") & a.weather.isin(TAG_TO_CLASS)].assign(cls=lambda d: d.weather.map(TAG_TO_CLASS))
    rng = np.random.default_rng(SEED)
    parts = []
    for c in CLASSES:
        pool = a[a.cls == c].sort_values("name")
        take = pool if len(pool) <= PER_CLASS else pool.iloc[np.sort(rng.choice(len(pool), PER_CLASS, replace=False))]
        parts.append(take)
    sub = pd.concat(parts, ignore_index=True)
    # stratified hold-out for model selection
    sub["split"] = "train"
    for c in CLASSES:
        idx = sub.index[sub.cls == c].to_numpy()
        hold = rng.choice(idx, max(1, int(round(HOLDOUT * len(idx)))), replace=False)
        sub.loc[hold, "split"] = "holdout"
    CACHE.mkdir(parents=True, exist_ok=True)
    sub.to_csv(CACHE / "train_subset.csv", index=False)
    (CACHE / "train_subset_names.txt").write_text("\n".join(sub.name) + "\n", encoding="utf-8")
    print(sub.groupby(["cls", "split"]).size().unstack(fill_value=0).to_string())
    print("pool sizes:", a.cls.value_counts().to_dict())
    return sub


def frame(path: str | Path) -> np.ndarray:
    """The whole frame resized to STORE (squashing non-16:9 images), uint8 RGB."""
    im = load_rgb(path, "apply").resize((STORE[1], STORE[0]))
    return np.asarray(im, dtype=np.uint8)


def store(paths: list[Path], dst: Path) -> np.ndarray:
    if dst.exists():
        arr = np.load(dst, mmap_mode="r")
        if arr.shape[0] == len(paths):
            return arr
    arr = np.stack([frame(p) for p in paths])
    np.save(dst, arr)
    return np.load(dst, mmap_mode="r")


def model(n_classes: int = len(CLASSES), pretrained: bool = True):
    import torch.nn as nn
    import torchvision

    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def to_input(batch_u8, train: bool, gen=None):
    """uint8 NHWC batch -> normalised NCHW float tensor at INPUT size (random crop + flip if train)."""
    import torch
    import torch.nn.functional as F

    x = torch.as_tensor(np.ascontiguousarray(batch_u8)).cuda().permute(0, 3, 1, 2).float() / 255.0
    if train:
        n, _, h, w = x.shape
        out = []
        for i in range(n):
            s = float(torch.empty(1).uniform_(0.6, 1.0, generator=gen)) if gen is not None else 1.0
            ch, cw = max(8, int(round(h * s ** 0.5))), max(8, int(round(w * s ** 0.5)))
            y0 = int(torch.randint(0, h - ch + 1, (1,), generator=gen))
            x0 = int(torch.randint(0, w - cw + 1, (1,), generator=gen))
            crop = x[i:i + 1, :, y0:y0 + ch, x0:x0 + cw]
            if bool(torch.randint(0, 2, (1,), generator=gen)):
                crop = crop.flip(-1)
            out.append(F.interpolate(crop, size=INPUT, mode="bilinear", align_corners=False))
        x = torch.cat(out)
    else:
        x = F.interpolate(x, size=INPUT, mode="bilinear", align_corners=False)
    mean = torch.as_tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.as_tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def train(epochs: int = 8, batch: int = 32, lr: float = 1e-4) -> dict:
    import time

    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    sub = pd.read_csv(CACHE / "train_subset.csv")
    paths = [BDD / "images" / "100k" / "train" / f"{n}.jpg" for n in sub.name]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} training images missing; run scripts/get_bdd100k.py --no-val "
                         f"--train-list {CACHE / 'train_subset_names.txt'}")
    t0 = time.perf_counter()
    X = store(paths, CACHE / "train_frames.npy")
    y = sub.cls.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    tr, ho = np.nonzero(sub.split.to_numpy() == "train")[0], np.nonzero(sub.split.to_numpy() == "holdout")[0]
    print(f"frames ready in {time.perf_counter() - t0:.0f} s; train {len(tr)}, hold-out {len(ho)}", flush=True)

    m = model().cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    counts = np.bincount(y[tr], minlength=len(CLASSES))
    w = (1.0 / counts[y[tr]])
    w = w / w.sum()
    gen = torch.Generator().manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    log, best = [], (-1.0, None)
    for ep in range(epochs):
        m.train()
        order = rng.choice(tr, size=len(tr), replace=True, p=w)  # class-balanced sampling
        tot = 0.0
        for i in range(0, len(order), batch):
            idx = np.sort(order[i:i + batch])
            xb = to_input(X[idx], train=True, gen=gen)
            yb = torch.as_tensor(y[idx]).cuda()
            opt.zero_grad()
            loss = loss_fn(m(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        probs = predict_frames(m, X[ho])
        pred = probs.argmax(1)
        recall = [float((pred[y[ho] == k] == k).mean()) for k in range(len(CLASSES))]
        bacc = float(np.mean(recall))
        log.append(dict(epoch=ep + 1, train_loss=tot / len(order), holdout_balanced_acc=bacc,
                        holdout_recall=dict(zip(CLASSES, recall)), seconds=time.perf_counter() - t0))
        print(json.dumps(log[-1]), flush=True)
        if bacc > best[0]:
            best = (bacc, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()})
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=best[1], classes=CLASSES, input=INPUT, store=STORE), WEIGHTS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = dict(log=log, best_holdout_balanced_acc=best[0], n_train=int(len(tr)), n_holdout=int(len(ho)),
               per_class=sub.groupby("cls").size().to_dict(), prereg="PREREG_o3.md")
    (RESULTS / "train_log.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def predict_frames(m, frames: np.ndarray, batch: int = 64) -> np.ndarray:
    import torch

    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), batch):
            out.append(torch.softmax(m(to_input(frames[i:i + batch], train=False)), 1).cpu().numpy())
    return np.concatenate(out)


def load_trained():
    import torch

    ck = torch.load(WEIGHTS, map_location="cuda")
    m = model(len(ck["classes"]), pretrained=False).cuda()
    m.load_state_dict(ck["state_dict"])
    return m


def predict_images(images: pd.DataFrame, name: str) -> pd.DataFrame:
    m = load_trained()
    paths = [REPO / p for p in images.path]
    X = store(paths, CACHE / f"frames_{name}.npy")
    probs = predict_frames(m, np.asarray(X))
    df = pd.DataFrame(probs, columns=[f"p_{c}" for c in CLASSES])
    df.insert(0, "image_id", images.image_id.to_numpy())
    df.insert(1, "weather", images.weather.to_numpy())
    df["pred"] = np.array(CLASSES)[probs.argmax(1)]
    df["h"] = df.p_rain + df.p_snow + df.p_fog
    df.to_parquet(CACHE / f"h_{name}.parquet", index=False)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["select", "train", "predict"])
    args = ap.parse_args(argv)
    if args.command == "select":
        select()
    elif args.command == "train":
        train()
    else:
        dawn = pd.read_parquet(REPO / "cache" / "images.parquet")
        dawn = dawn[dawn.keep].sort_values("image_id").reset_index(drop=True)
        d = predict_images(dawn, "dawn")
        print(pd.crosstab(d.weather, d.pred).to_string())
        bdd = pd.read_parquet(REPO / "cache" / "bdd" / "images.parquet")
        bdd = bdd[bdd.keep].sort_values("image_id").reset_index(drop=True)
        b = predict_images(bdd, "bdd_val")
        print(pd.crosstab(b.weather, b.pred).to_string())


if __name__ == "__main__":
    main()
