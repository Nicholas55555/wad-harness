"""Decision-focused per-image threshold from the detector's own confidence profile
(PREREG_threshold.md, version 2 + amendment). Departs from the brief at the user's decision: the
detector's outputs on the image are the model's only inputs (no pixels, no scene measurements).
STATUS (2026-10-13): did not beat one threshold; kept so its NOTES.md numbers reproduce. The
method is the dial (wad/dial.py): one threshold or per detection.

An MLP reads an image's features and a dial lam ("false positives accepted to avoid one miss") and
picks one of 40 thresholds. It is trained on what its pick costs, lam * missed objects + false
positives, across lam, so the trade-off stays tunable after training.

  python -m wad.thresh_dfl features   # cache/thresh/dfl_features_{train,bdd,dawn}.parquet
  python -m wad.thresh_dfl train      # weights/thresh_dfl.pt, results/threshold/dfl/train_log.json
  python -m wad.thresh_dfl check      # results/threshold/dfl/{calibration_by_condition,decision_cost}.csv
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd

from wad.config import REPO
from wad.thresh_net import CACHE, RESULTS, split_train_val

OUT = RESULTS / "dfl"
GROUPS = {"vehicle": ("car", "truck", "bus"), "two_wheeler": ("bicycle", "motorcycle"), "person": ("person",)}
BANDS = [(0.25, 1.01), (0.05, 0.25), (0.01, 0.05)]
LAM_LO, LAM_HI = 0.1, 1000.0
VAL_LAMS = (0.3, 1, 3, 10, 30, 100, 300)
SETS = {"train": ("cache/thresh/images.parquet", "cache/thresh/det/v8x_1280/detections.parquet"),
        "bdd": ("cache/bdd/images.parquet", "cache/bdd/det/v8x_1280/detections.parquet"),
        "dawn": ("cache/images.parquet", "cache/det/v8x_1280/detections.parquet")}
WEIGHTS = REPO / "weights" / "thresh_dfl.pt"


def taus40() -> np.ndarray:
    from wad.thresh_net import cfg, taus_k

    return taus_k(cfg())


# ---------------------------------------------------------------- features
def det_features(det: pd.DataFrame, images: pd.DataFrame, taus: np.ndarray) -> pd.DataFrame:
    ids = images.image_id.to_numpy()
    pos = pd.Series(np.arange(len(ids)), index=ids)
    det = det[det.image_id.isin(pos.index)]
    img = pos.loc[det.image_id].to_numpy()
    score = det.score.to_numpy()
    h_img = images.set_index("image_id").height.loc[det.image_id].to_numpy().astype(float)
    rel_h = (det.y2 - det.y1).to_numpy() / h_img
    bottom = det.y2.to_numpy() / h_img
    out = {}
    K = len(taus)

    def counts_ge(mask):
        k = np.searchsorted(taus, score[mask], side="right")
        c = np.zeros((len(ids), K + 1))
        np.add.at(c, (img[mask], k), 1)
        return np.cumsum(c[:, ::-1], axis=1)[:, ::-1][:, 1:]

    for g, classes in list(GROUPS.items()) + [("all", sum(GROUPS.values(), ()))]:
        cg = np.log1p(counts_ge(det.cls.isin(classes).to_numpy()))
        for k in range(K):
            out[f"n_{g}_ge_{k:02d}"] = cg[:, k]
    top = np.zeros((len(ids), 10))
    order = np.lexsort((-score, img))
    s_sorted, i_sorted = score[order], img[order]
    first = np.searchsorted(i_sorted, np.arange(len(ids)))
    for r in range(10):
        j = first + r
        ok = (j < len(i_sorted)) & (i_sorted[np.minimum(j, len(i_sorted) - 1)] == np.arange(len(ids)))
        top[ok, r] = s_sorted[j[ok]]
    for r in range(10):
        out[f"top_score_{r}"] = top[:, r]
    df = pd.DataFrame(out, index=ids)
    for lo, hi in BANDS:
        m = (score >= lo) & (score < hi)
        g = pd.DataFrame(dict(i=img[m], lh=np.log(np.clip(rel_h[m], 1e-3, None)), small=rel_h[m] < 0.03, bottom=bottom[m]))
        agg = g.groupby("i").agg(lh=("lh", "median"), small=("small", "mean"), bottom=("bottom", "mean"))
        tag = f"{lo:g}_{hi:g}"
        for col, fill in (("lh", np.log(1e-3)), ("small", 0.0), ("bottom", 0.0)):
            v = np.full(len(ids), fill)
            v[agg.index.to_numpy()] = agg[col].to_numpy()
            df[f"band_{tag}_{col}"] = v
    df["log_n_det"] = np.log1p(np.bincount(img, minlength=len(ids)))
    return df


def features() -> None:
    taus = taus40()
    for name, (im_p, det_p) in SETS.items():
        t0 = time.perf_counter()
        im = pd.read_parquet(REPO / im_p)
        im = im[im.keep].sort_values("image_id").reset_index(drop=True)
        df = det_features(pd.read_parquet(REPO / det_p), im, taus)
        df.insert(0, "weather", im.weather.to_numpy())
        df.index.name = "image_id"
        df.reset_index().to_parquet(CACHE / f"dfl_features_{name}.parquet", index=False)
        print(f"{name}: {len(df)} images x {df.shape[1] - 1} features, {time.perf_counter() - t0:.0f} s", flush=True)


def columns(all_cols) -> list[str]:
    return [c for c in all_cols if c.startswith(("n_", "top_score_", "band_", "log_n_det"))]


# ---------------------------------------------------------------- model
def build_mlp(n_in: int, k: int):
    import torch.nn as nn

    return nn.Sequential(nn.Linear(n_in + 1, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 256), nn.ReLU(),
                         nn.Dropout(0.1), nn.Linear(256, k))


def norm_const(Mbar, Fbar, lam):
    """Mean cost of the best single threshold on the training set at each lam."""
    import torch

    return torch.min(lam[:, None] * Mbar[None] + Fbar[None], dim=1).values


def train(epochs=40, batch=512, lr=3e-4, temp=0.05) -> dict:
    import torch

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lab = np.load(CACHE / "labels.npz", allow_pickle=True)
    f = pd.read_parquet(CACHE / "dfl_features_train.parquet").set_index("image_id").loc[lab["image_id"]]
    cols = columns(f.columns)
    X = f[cols].to_numpy(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = np.clip((X - mu) / sd, -5, 5)
    m, fp = lab["miss"].astype(np.float32), lab["fp"].astype(np.float32)
    is_val = split_train_val(lab["group"])
    tr, va = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]
    Xt, Mt, Ft = (torch.as_tensor(a, device=dev) for a in (X, m, fp))
    Mbar, Fbar = Mt[tr].mean(0), Ft[tr].mean(0)
    lmu, lsd = (math.log(LAM_LO) + math.log(LAM_HI)) / 2, (math.log(LAM_HI) - math.log(LAM_LO)) / math.sqrt(12)
    model = build_mlp(len(cols), m.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(tr) / batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator(device=dev).manual_seed(0)
    rng = np.random.default_rng(0)

    def logits(idx, loglam):
        z = (loglam - lmu) / lsd
        return model(torch.cat([Xt[idx], z[:, None]], 1))

    def val_cost():
        model.eval()
        out = []
        with torch.no_grad():
            j = torch.as_tensor(va, device=dev)
            for lam in VAL_LAMS:
                ll = torch.full((len(va),), math.log(lam), device=dev)
                k = logits(j, ll).argmax(1)
                c = lam * Mt[j, k] + Ft[j, k]
                out.append(float(c.mean() / norm_const(Mbar, Fbar, torch.tensor([float(lam)], device=dev))[0]))
        model.train()
        return float(np.mean(out)), out

    log, best = [], (float("inf"), None)
    t0 = time.perf_counter()
    for ep in range(epochs):
        order = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(order), batch):
            idx = torch.as_tensor(order[i:i + batch], device=dev)
            loglam = math.log(LAM_LO) + (math.log(LAM_HI) - math.log(LAM_LO)) * torch.rand(len(idx), generator=g, device=dev)
            lam = loglam.exp()
            lo = logits(idx, loglam)
            p = torch.softmax(lo, 1)
            c = lam[:, None] * Mt[idx] + Ft[idx]
            nc = norm_const(Mbar, Fbar, lam)
            cost = (p * c).sum(1) / nc
            # soft oracle: thresholds within a few percent of the image's best cost share the target
            q = torch.softmax(-(c - c.min(1, keepdim=True).values) / nc[:, None] / temp, 1)
            ce = -(q * torch.log_softmax(lo, 1)).sum(1)
            loss = cost.mean() + ce.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        vc, per = val_cost()
        log.append(dict(epoch=ep + 1, train_cost=tot / len(tr), val_cost=vc, val_cost_per_lam=per, seconds=time.perf_counter() - t0))
        if vc < best[0]:
            best = (vc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    torch.save(dict(state_dict=best[1], cols=cols, mu=mu, sd=sd, lmu=lmu, lsd=lsd, k=m.shape[1], taus=lab["taus"]), WEIGHTS)
    b = min(log, key=lambda r: r["val_cost"])
    print(f"best epoch {b['epoch']}, validation cost {b['val_cost']:.3f} (1 = best single threshold)", flush=True)
    out = dict(log=log, best_epoch=b["epoch"], best_val_cost=b["val_cost"], n_train=int(len(tr)), n_val=int(len(va)),
               n_features=len(cols), prereg="PREREG_threshold.md (version 2 + amendments 1-2: confidence only, soft-oracle loss)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train_log.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def policy(feats: pd.DataFrame, lams: np.ndarray) -> np.ndarray:
    """(n_img, len(lams)) index on the 40-threshold grid; lams ascending; non-increasing along lams."""
    import torch

    assert np.all(np.diff(lams) > 0)
    ck = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model = build_mlp(len(ck["cols"]), ck["k"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    X = torch.as_tensor(np.clip((feats[ck["cols"]].to_numpy(np.float32) - ck["mu"]) / ck["sd"], -5, 5).astype(np.float32))
    out = np.empty((len(X), len(lams)), int)
    with torch.no_grad():
        for j, lam in enumerate(lams):
            z = (math.log(min(max(lam, LAM_LO), LAM_HI)) - ck["lmu"]) / ck["lsd"]
            out[:, j] = model(torch.cat([X, torch.full((len(X), 1), z)], 1)).argmax(1).numpy()
    return np.minimum.accumulate(out, axis=1)


def check() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Why per-image thresholds cannot beat one threshold in expectation here, and how the model does.
    (1) calibration: share of detections that are real objects by confidence band and condition;
    (2) decision cost on all BDD100K val / DAWN images at fixed lam, relative to the best single
    threshold chosen on those images (in-sample, favours the single threshold slightly), with the
    per-image oracle and the model's agreement with it."""
    from wad.config import tau_grid
    from wad.curves import ALL, Cache
    from wad.thresh_eval import configs

    bins = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    cal, rows = [], []
    for ds, key, v in configs():
        c = Cache(v)
        p = c.pred.copy()
        w = p.image_id.map(dict(zip(c.image_ids, c.weather)))
        if ds == "BDD100K":
            cond = np.where(w.str.endswith("night"), "night", "day")
            cond = np.where(w == "rain_night", "rain_night", cond)
        else:
            cond = "DAWN " + w
        p["condition"] = cond
        p["band"] = pd.cut(p.score, bins, right=False).astype(str)
        t = p.groupby(["condition", "band"], observed=True).agg(share_real=("tp", "mean"), detections=("tp", "size")).reset_index()
        cal.append(t.assign(dataset=ds))
        taus = tau_grid(v)[::5]
        m = c.n_gt(ALL)[:, None] - c.matched_grid(taus, ALL)
        f = c.fp_grid(taus, ALL)
        feats = pd.read_parquet(CACHE / f"dfl_features_{key}.parquet").set_index("image_id").loc[c.image_ids]
        lams = np.array([1, 3, 10, 30, 100.0])
        idx = policy(feats, lams)
        r = np.arange(c.n_img)
        for j, lam in enumerate(lams):
            cv = lam * m + f
            n = cv.mean(0).min()
            o = cv.argmin(1)
            rows.append(dict(dataset=ds, lam=lam, model=cv[r, idx[:, j]].mean() / n, best_single=1.0, oracle=cv.min(1).mean() / n,
                             spearman_vs_oracle=pd.Series(idx[:, j]).corr(pd.Series(o), method="spearman"),
                             within_one_step_of_oracle=float((np.abs(idx[:, j] - o) <= 1).mean()),
                             distinct_thresholds=int(len(np.unique(idx[:, j])))))
    cal, dec = pd.concat(cal), pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    cal.to_csv(OUT / "calibration_by_condition.csv", index=False, float_format="%.4g")
    dec.to_csv(OUT / "decision_cost.csv", index=False, float_format="%.4g")
    return cal, dec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["features", "train", "check"])
    a = ap.parse_args(argv)
    if a.command == "check":
        cal, dec = check()
        print(dec.round(3).to_string(index=False))
    else:
        features() if a.command == "features" else train()


if __name__ == "__main__":
    main()
