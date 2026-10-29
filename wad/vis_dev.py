"""Development diagnostics for the visibility estimator's night rendering (not evaluation).

  gap   v2's synthetic darkness vs real night, the diagnosis that motivated version 3. It looked at
        8 BDD100K *val* images per condition (clear day, clear night, rain night; seed 0) with their
        v2 light-level predictions; no detection result is used. -> results/vis2/night_gap.csv,
        results/vis2/figures/night_gap.jpg
  luma  the final version-3 tuning check: 200 seeded synthetic night renders (validation base
        frames) vs the first 200 development night frames (BDD100K *train*, cache/vis3/dev_night.csv).
        -> results/vis3/dev_luma.csv, results/vis3/figures/night_preview.jpg

Regions: whole frame, sky = top quarter, road = bottom quarter of the 448x252 working frame; luma
is Rec. 601 on 0-255. Run: python -m wad.vis_dev {gap,luma}
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from wad import vis_estimator as ve
from wad.imageio import load_rgb

W601 = np.array([0.299, 0.587, 0.114], np.float32)


def luma_stats(a: np.ndarray) -> dict:
    y = a.astype(np.float32) @ W601
    return dict(mean=float(y.mean()), sky=float(y[: ve.H // 4].mean()), road=float(y[-ve.H // 4:].mean()))


def _sheet(rows, path, tw=224, th=126):
    from PIL import Image, ImageDraw

    S = Image.new("RGB", (8 * tw, len(rows) * (th + 30)), "white")
    dr = ImageDraw.Draw(S)
    for r, (name, arr, labs) in enumerate(rows):
        dr.text((3, r * (th + 30) + 2), name, fill="black")
        for c, (a, lab) in enumerate(zip(arr[:8], labs[:8])):
            S.paste(Image.fromarray(a).resize((tw, th)), (c * tw, r * (th + 30) + 16))
            dr.text((c * tw + 2, r * (th + 30) + 16 + th), lab, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    S.save(path, quality=85)


def _bases():
    bases = pd.read_csv(ve.CACHE / "bases.csv")
    va = np.nonzero(bases.split.to_numpy() == "val")[0]
    fr = np.load(ve.CACHE / "frames.npy", mmap_mode="r")
    dp = np.load(ve.CACHE / "depth.npy", mmap_mode="r")
    return va, fr, dp, np.load(ve.CACHE / "boxes.npy"), np.load(ve.CACHE / "nbox.npy")


def gap():
    import torch

    va, fr, dp, bx, nb = _bases()
    model = ve.load_model(2)
    syn, g = ve.Synth(version=2), torch.Generator(device="cuda").manual_seed(5)
    synth, i = [], 0
    while len(synth) < 8:  # darkness-only v2 samples (no veil), as rendered for training
        j = va[i % len(va)]
        i += 1
        img, t = syn.sample(torch.as_tensor(np.asarray(fr[j])).cuda(), torch.as_tensor(np.asarray(dp[j])).cuda(),
                            torch.as_tensor(bx[j, :nb[j]]).cuda(), g)
        if t["log_k"] < -1.0 and t["log_mor"] >= 3.99:
            synth.append(((img.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8), t["log_k"]))
    im = pd.read_parquet(ve.REPO / "cache" / "bdd" / "images.parquet")
    im = im[im.keep]
    rng = np.random.default_rng(0)
    groups = [("synthetic darkness (v2)", [a for a, _ in synth])]
    for w in ("clear", "clear_night", "rain_night"):
        groups.append((f"BDD100K val {w}", [ve.to_working(load_rgb(ve.REPO / p, "apply"))
                                            for p in rng.choice(im[im.weather == w].path.to_numpy(), 8, replace=False)]))
    rows, sheet = [], []
    for name, arr in groups:
        k = 10 ** ve.predict_arrays(model, np.stack(arr))[0][:, 1].cpu().numpy()
        for a, kk in zip(arr, k):
            rows.append(dict(group=name, predicted_k_v2=float(kk), **luma_stats(a)))
        sheet.append((name, arr, [f"pred k {x:.3f}" for x in k]))
    df = pd.DataFrame(rows)
    df.to_csv(ve.RESULTS / "night_gap.csv", index=False, float_format="%.4g")
    _sheet(sheet, ve.RESULTS / "figures" / "night_gap.jpg")
    return df.groupby("group", sort=False).agg(["median", "min", "max"]).round(2)


def luma(n=200):
    import torch

    va, fr, dp, bx, nb = _bases()
    syn, g = ve.Synth(version=3), torch.Generator(device="cuda").manual_seed(21)
    rows, synth = [], []
    for i in range(n):
        j = va[i % len(va)]
        J = (torch.as_tensor(np.asarray(fr[j])).cuda().float() / 255).permute(2, 0, 1) ** 2.2
        dm = torch.as_tensor(np.asarray(dp[j])).cuda().float()
        L, k = syn._night(J, dm, ve.far_depth(dm), torch.as_tensor(bx[j, :nb[j]]).cuda(), g)
        a = ((L.clamp(0, 1) ** (1 / 2.2)).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        synth.append(a)
        rows.append(dict(source="synthetic night (v3)", k=k, **luma_stats(a)))
    dev = pd.read_csv(ve.REPO / "cache" / "vis3" / "dev_night.csv").name[:n]
    real = [ve.to_working(load_rgb(ve.BDD / "images" / "100k" / "train" / f"{x}.jpg", "apply")) for x in dev]
    rows += [dict(source="real night (BDD100K train, development)", k=np.nan, **luma_stats(a)) for a in real]
    df = pd.DataFrame(rows)
    out = ve.paths(3)["results"]
    df.to_csv(out / "dev_luma.csv", index=False, float_format="%.4g")
    _sheet([("synthetic night (v3)", synth[:8], [f"k {x:.4f}" for x in df.k[:8]]),
            ("synthetic night (v3)", synth[8:16], [f"k {x:.4f}" for x in df.k[8:16]]),
            ("real BDD100K train night", real[:8], ["development"] * 8),
            ("real BDD100K train night", real[8:16], ["development"] * 8)], out / "figures" / "night_preview.jpg")
    return df.groupby("source", sort=False)[["mean", "sky", "road", "k"]].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["gap", "luma"])
    a = ap.parse_args()
    print((gap if a.command == "gap" else luma)().to_string())
