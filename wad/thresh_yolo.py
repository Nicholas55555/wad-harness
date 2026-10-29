"""Predict the oracle threshold from YOLOv8's own view of the image (PREREG_threshold.md, version 3
+ amendment).
STATUS (2026-10-13): did not beat one threshold; kept so its NOTES.md numbers reproduce. The
method is the dial (wad/dial.py): one threshold or per detection.

The input is what YOLOv8 is trained on, read the way YOLOv8 reads it: each image goes through the
same preparation as detection (EXIF orientation, letterbox to 1280, 0-1 scaling) and YOLOv8x's
frozen backbone and neck; the three feature maps it hands to its detection head are pooled into
one vector per image. A small head takes that vector and the dial lam ("false positives accepted
to avoid one miss") and is trained to pick the threshold at which the detector does best on that
image (cost lam * misses + false positives; soft-oracle target).

  python -m wad.thresh_yolo extract   # cache/thresh/yolo_feats_{train,bdd,dawn}.npy (GPU, ~80 min, resumable)
  python -m wad.thresh_yolo train     # weights/thresh_yolo.pt, results/threshold/yolo/train_log.json
  python -m wad.thresh_yolo check     # results/threshold/yolo/decision_cost.csv (E1-E3; versions 1-3 compared)
  add --pilot to any of them: 1,000 random train, 1,000 random BDD100K val and all DAWN images (results/threshold/yolo/pilot)
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

OUT = RESULTS / "yolo"
WEIGHTS = REPO / "weights" / "thresh_yolo.pt"
SETS = {"train": "cache/thresh/images.parquet", "bdd": "cache/bdd/images.parquet", "dawn": "cache/images.parquet"}
LAM_LO, LAM_HI = 0.1, 1000.0
VAL_LAMS = (0.3, 1, 3, 10, 30, 100, 300)
GRID = (3, 4)


def images(name: str) -> pd.DataFrame:
    im = pd.read_parquet(REPO / SETS[name])
    return im[im.keep].sort_values("image_id").reset_index(drop=True)


def detector_cfg():
    from wad.thresh_net import cfg

    return cfg()


# ---------------------------------------------------------------- feature extraction
class Grab:
    """Forward pre-hook on YOLOv8's Detect layer: keeps the three neck feature maps it receives."""

    def __init__(self):
        self.x = None

    def __call__(self, module, args):
        self.x = [t.detach() for t in args[0]]


def pool(maps) -> np.ndarray:
    import torch

    parts = []
    for t in maps:  # (1, C, H, W) at strides 8, 16, 32
        t = t.float()
        parts += [t.mean((2, 3)).flatten(), t.amax((2, 3)).flatten()]
    parts.append(torch.nn.functional.adaptive_avg_pool2d(maps[-1].float(), GRID).flatten())
    return torch.cat(parts).cpu().numpy().astype(np.float16)


PILOT = {"train": 1000, "bdd": 1000, "dawn": None}  # images per set for the pilot (None = all), seed 0


def subset(name: str, n: int | None) -> pd.DataFrame:
    """n images drawn at random (seed 0) from a set, in image_id order; all of them if n is None.
    The sets are sorted by condition, so the literal first n would be a single condition."""
    im = images(name)
    if n is None or n >= len(im):
        return im
    pick = np.random.default_rng(0).choice(len(im), n, replace=False)
    return im.iloc[np.sort(pick)].reset_index(drop=True)


def extract(names=("bdd", "dawn", "train"), flush_every=500, prefix="yolo_feats", sizes=None) -> None:
    from wad.detect import load_model
    from wad.imageio import load_bgr

    v = detector_cfg()
    model = load_model(v)
    grab = Grab()
    hooked = False
    for name in names:
        im = subset(name, (sizes or {}).get(name))
        out, done_f, ids_f = (CACHE / f"{prefix}_{name}.npy", CACHE / f"{prefix}_{name}.done",
                              CACHE / f"{prefix}_{name}.ids.txt")
        start = int(done_f.read_text()) if done_f.exists() and out.exists() else 0
        if start >= len(im):
            print(f"{name}: complete ({len(im)} images)", flush=True)
            continue
        ids_f.write_text("".join(f"{x}\n" for x in im.image_id), encoding="utf-8")
        arr = None if start == 0 else np.load(out, mmap_mode="r+")
        t0 = time.perf_counter()
        for i in range(start, len(im)):
            r = im.iloc[i]
            model.predict(source=load_bgr(r.path, v.data.exif_orientation), imgsz=v.detector.imgsz,
                          conf=v.detector.conf_floor, iou=v.detector.nms_iou, max_det=v.detector.max_det,
                          classes=list(v.classes.values()), device="cuda:0", verbose=False)
            if not hooked:  # the predictor builds (and fuses) its model on the first call
                model.predictor.model.model.model[-1].register_forward_pre_hook(grab)
                hooked = True
                model.predict(source=load_bgr(r.path, v.data.exif_orientation), imgsz=v.detector.imgsz,
                              conf=v.detector.conf_floor, iou=v.detector.nms_iou, max_det=v.detector.max_det,
                              classes=list(v.classes.values()), device="cuda:0", verbose=False)
            f = pool(grab.x)
            if arr is None:
                arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(len(im), len(f)))
            arr[i] = f
            if (i + 1) % flush_every == 0 or i == len(im) - 1:
                arr.flush()
                done_f.write_text(str(i + 1))
                print(f"  {name} {i + 1}/{len(im)}, {time.perf_counter() - t0:.0f} s", flush=True)
        del arr


def load_feats(name: str, prefix: str = "yolo_feats") -> tuple[np.ndarray, list[str]]:
    ids = (CACHE / f"{prefix}_{name}.ids.txt").read_text(encoding="utf-8").splitlines()
    assert int((CACHE / f"{prefix}_{name}.done").read_text()) == len(ids), f"{prefix}_{name}: extraction incomplete"
    if prefix == "yolo_feats":
        assert ids == list(images(name).image_id)
    elif prefix == "yolo_pilot":
        assert ids == list(subset(name, PILOT[name]).image_id), f"{prefix}_{name}: ids differ from the pilot draw"
    return np.load(CACHE / f"{prefix}_{name}.npy"), ids


# ---------------------------------------------------------------- head
def build_head(n_in: int, k: int):
    import torch.nn as nn

    return nn.Sequential(nn.Linear(n_in + 1, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 256), nn.ReLU(),
                         nn.Dropout(0.2), nn.Linear(256, k))


class CurveHead:
    """Structured head: features -> per-image curves of missed objects m_k (rising with tau) and false
    positives f_k (falling); the dial picks argmin_k lam * m_k + f_k, so a larger lam can only lower
    the threshold (by construction, not by learning)."""

    @staticmethod
    def build(n_in: int, k: int):
        import torch.nn as nn

        class _H(nn.Module):
            def __init__(self):
                super().__init__()
                self.k = k
                self.net = nn.Sequential(nn.Linear(n_in, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 256), nn.ReLU(),
                                         nn.Dropout(0.2), nn.Linear(256, 2 * k))

            def forward(self, x):
                z = self.net(x)
                sp = nn.functional.softplus
                return sp(z[:, :self.k]).cumsum(1), sp(z[:, self.k:]).flip(1).cumsum(1).flip(1)

        return _H()


def train(epochs=40, batch=512, lr=3e-4, temp=0.05, prefix="yolo_feats", weights=WEIGHTS, out_dir=OUT,
          head_type="logits") -> dict:
    import torch

    from wad.thresh_dfl import norm_const

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lab = dict(np.load(CACHE / "labels.npz", allow_pickle=True))
    X, ids = load_feats("train", prefix)
    rows = pd.Series(np.arange(len(lab["image_id"])), index=lab["image_id"]).loc[ids].to_numpy()
    for key in ("image_id", "group", "miss", "fp"):
        lab[key] = lab[key][rows]
    assert list(lab["image_id"]) == ids
    X = X.astype(np.float32)
    m, fp = lab["miss"].astype(np.float32), lab["fp"].astype(np.float32)
    is_val = split_train_val(lab["group"])
    tr, va = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6  # fit split only
    X = np.clip((X - mu) / sd, -5, 5)
    Xt, Mt, Ft = (torch.as_tensor(a, device=dev) for a in (X, m, fp))
    Mbar, Fbar = Mt[tr].mean(0), Ft[tr].mean(0)
    lmu, lsd = (math.log(LAM_LO) + math.log(LAM_HI)) / 2, (math.log(LAM_HI) - math.log(LAM_LO)) / math.sqrt(12)
    head = (CurveHead.build(X.shape[1], m.shape[1]) if head_type == "curves" else build_head(X.shape[1], m.shape[1])).to(dev)
    w_m = 1.0 / (1.0 + Mt[tr].mean(0))
    w_f = 1.0 / (1.0 + Ft[tr].mean(0))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * math.ceil(len(tr) / batch))
    g = torch.Generator(device=dev).manual_seed(0)
    rng = np.random.default_rng(0)

    def logits(idx, loglam, nc=None):
        if head_type == "curves":  # minus the predicted cost, scaled so a few % of cost spreads the choice
            pm, pf = head(Xt[idx])
            ch = loglam.exp()[:, None] * pm + pf
            scale = nc if nc is not None else norm_const(Mbar, Fbar, loglam.exp())
            return -(ch - ch.min(1, keepdim=True).values) / scale[:, None] / temp
        return head(torch.cat([Xt[idx], ((loglam - lmu) / lsd)[:, None]], 1))

    def val_cost():
        head.eval()
        out = []
        with torch.no_grad():
            j = torch.as_tensor(va, device=dev)
            for lam in VAL_LAMS:
                k = logits(j, torch.full((len(va),), math.log(lam), device=dev)).argmax(1)
                c = lam * Mt[j, k] + Ft[j, k]
                own = (lam * Mt[j] + Ft[j]).mean(0).min()  # the validation images' own best single threshold
                out.append(float(c.mean() / own))
        head.train()
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
            nc = norm_const(Mbar, Fbar, lam)
            lo = logits(idx, loglam, nc)
            c = lam[:, None] * Mt[idx] + Ft[idx]
            cost = (torch.softmax(lo, 1) * c).sum(1) / nc
            q = torch.softmax(-(c - c.min(1, keepdim=True).values) / nc[:, None] / temp, 1)
            loss = cost.mean() - (q * torch.log_softmax(lo, 1)).sum(1).mean()
            if head_type == "curves":  # weak anchor so the curves keep their meaning (counts)
                from wad.thresh_net import poisson_nll

                pm, pf = head(Xt[idx])
                loss = loss + 0.1 * (poisson_nll(pm, Mt[idx], w_m) + poisson_nll(pf, Ft[idx], w_f))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(cost.detach().mean()) * len(idx)
        vc, per = val_cost()
        log.append(dict(epoch=ep + 1, train_cost=tot / len(tr), val_cost=vc, val_cost_per_lam=per, seconds=time.perf_counter() - t0))
        if vc < best[0]:
            best = (vc, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
    torch.save(dict(state_dict=best[1], mu=mu, sd=sd, lmu=lmu, lsd=lsd, n_in=X.shape[1], k=m.shape[1], taus=lab["taus"],
                    head=head_type), weights)
    b = min(log, key=lambda r: r["val_cost"])
    out = dict(log=log, best_epoch=b["epoch"], best_val_cost=b["val_cost"], n_train=int(len(tr)), n_val=int(len(va)),
               n_features=int(X.shape[1]), features=prefix, head=head_type, batch=batch, epochs=epochs,
               prereg="PREREG_threshold.md (version 3 + amendment: YOLOv8x features)")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_log.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"best epoch {b['epoch']}, validation cost {b['val_cost']:.3f} (1 = the validation images' best single threshold)",
          flush=True)
    return out


def policy(X16: np.ndarray, lams: np.ndarray, weights=WEIGHTS) -> np.ndarray:
    """(n_img, len(lams)) index on the 40-threshold grid; lams ascending; non-increasing along lams."""
    import torch

    assert np.all(np.diff(lams) > 0)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    curves = ck.get("head") == "curves"
    head = CurveHead.build(ck["n_in"], ck["k"]) if curves else build_head(ck["n_in"], ck["k"])
    head.load_state_dict(ck["state_dict"])
    head.eval()
    X = torch.as_tensor(np.clip((X16.astype(np.float32) - ck["mu"]) / ck["sd"], -5, 5))
    out = np.empty((len(X), len(lams)), int)
    if curves:
        from wad.thresh_eval import dial_index

        with torch.no_grad():
            pm, pf = head(X)
        return dial_index(pm.numpy(), pf.numpy(), lams) // 5
    with torch.no_grad():
        for j, lam in enumerate(lams):
            z = (math.log(min(max(lam, LAM_LO), LAM_HI)) - ck["lmu"]) / ck["lsd"]
            out[:, j] = head(torch.cat([X, torch.full((len(X), 1), z)], 1)).argmax(1).numpy()
    return np.minimum.accumulate(out, axis=1)


def check(prefix="yolo_feats", weights=WEIGHTS, out_dir=OUT) -> pd.DataFrame:
    """E1-E3: decision cost at fixed lam on BDD100K val / DAWN (vs the best single threshold on the same
    images and the per-image oracle), agreement with the oracle; versions 1 and 2 on the same footing."""
    from wad import thresh_dfl
    from wad.config import tau_grid
    from wad.curves import ALL, Cache
    from wad.thresh_eval import configs, dial_index

    lams = np.array([1, 3, 10, 30, 100.0])
    lab = np.load(CACHE / "labels.npz", allow_pickle=True)
    train_single = {lam: int(np.argmin(lam * lab["miss"].mean(0) + lab["fp"].mean(0))) for lam in lams}
    rows = []
    for ds, key, v in configs():
        c = Cache(v)
        taus = tau_grid(v)[::5]
        X, ids = load_feats(key, prefix)
        keep = np.isin(c.image_ids, ids)  # evaluation images with features (DAWN: the clean-label set)
        img = c.image_ids[keep]
        m = (c.n_gt(ALL)[:, None] - c.matched_grid(taus, ALL))[keep]
        f = c.fp_grid(taus, ALL)[keep]
        pos = pd.Series(np.arange(len(ids)), index=ids).loc[img].to_numpy()
        pols = {"v3: YOLOv8x features (this)": policy(X[pos], lams, weights)}
        feats = pd.read_parquet(CACHE / f"dfl_features_{key}.parquet").set_index("image_id").loc[img]
        pols["v2: confidence profile"] = thresh_dfl.policy(feats, lams)
        pr = np.load(CACHE / f"pred_{key}.npz", allow_pickle=True)
        p1 = pd.Series(np.arange(len(pr["image_id"])), index=pr["image_id"]).loc[img].to_numpy()
        pols["v1: own pixel CNN, curve prediction"] = dial_index(pr["miss"][p1], pr["fp"][p1], lams) // 5
        r = np.arange(len(img))
        for j, lam in enumerate(lams):
            cv = lam * m + f
            n = cv.mean(0).min()
            K = cv.shape[1]
            o = K - 1 - np.argmin(cv[:, ::-1], axis=1)  # oracle, ties to the largest k (the dial's rule)
            opt = cv <= cv.min(1, keepdims=True) + 1e-9  # every optimal index per image
            for name, idx in pols.items():
                near = np.zeros(len(img), bool)
                for dk in (-1, 0, 1):
                    kk = np.clip(idx[:, j] + dk, 0, K - 1)
                    near |= opt[r, kk]
                rows.append(dict(dataset=ds, images=len(img), lam=lam, policy=name,
                                 cost_vs_best_single=cv[r, idx[:, j]].mean() / n,
                                 cost_vs_train_chosen_single=cv[r, idx[:, j]].mean() / cv[:, train_single[lam]].mean(),
                                 oracle=cv.min(1).mean() / n,
                                 spearman_vs_oracle=pd.Series(idx[:, j]).corr(pd.Series(o), method="spearman"),
                                 within_one_step_of_oracle=float(near.mean()),
                                 distinct_thresholds=int(len(np.unique(idx[:, j])))))
    d = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    d.to_csv(out_dir / "decision_cost.csv", index=False, float_format="%.4g")
    t = d[(d.policy.str.startswith("v3")) & (d.lam == 10)].set_index("dataset")
    crit = pd.DataFrame([
        dict(criterion="E1 BDD100K, lam 10: cost <= 0.97x best single", value=f"{t.loc['BDD100K', 'cost_vs_best_single']:.3f}",
             met=bool(t.loc["BDD100K", "cost_vs_best_single"] <= 0.97)),
        dict(criterion="E2 DAWN, lam 10: cost <= 0.97x best single", value=f"{t.loc['DAWN', 'cost_vs_best_single']:.3f}",
             met=bool(t.loc["DAWN", "cost_vs_best_single"] <= 0.97)),
        dict(criterion="E3 BDD100K, lam 10: Spearman with oracle threshold >= 0.3",
             value=f"{t.loc['BDD100K', 'spearman_vs_oracle']:.3f}", met=bool(t.loc["BDD100K", "spearman_vs_oracle"] >= 0.3))])
    crit.to_csv(out_dir / "criteria.csv", index=False)
    return d, crit


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["extract", "train", "check"])
    ap.add_argument("--pilot", action="store_true", help="1,000 random train + 1,000 random BDD100K val images + all DAWN")
    ap.add_argument("--steps-matched", action="store_true",
                    help="pilot only: batch 64 x 120 epochs (~1,700 updates, as many as the full run) instead of 512 x 40")
    ap.add_argument("--curves", action="store_true",
                    help="structured head: per-image miss / FP curves, the dial applied analytically")
    a = ap.parse_args(argv)
    kw = dict(prefix="yolo_pilot", weights=REPO / "weights" / "thresh_yolo_pilot.pt", out_dir=OUT / "pilot") if a.pilot else {}
    tkw = {}
    if a.pilot and a.steps_matched:
        kw.update(weights=REPO / "weights" / "thresh_yolo_pilot_steps.pt", out_dir=OUT / "pilot_steps_matched")
        tkw = dict(batch=64, epochs=120)
    if a.curves:
        tag = "_curves"
        kw.update(weights=kw.get("weights", WEIGHTS).with_name(kw.get("weights", WEIGHTS).stem + tag + ".pt"),
                  out_dir=kw.get("out_dir", OUT).with_name(kw.get("out_dir", OUT).name + tag))
        tkw["head_type"] = "curves"
    if a.command == "extract":
        extract(prefix="yolo_pilot", sizes=PILOT) if a.pilot else extract()
    elif a.command == "train":
        train(**kw, **tkw)
    else:
        d, crit = check(**kw)
        pd.set_option("display.width", 220)
        print(crit.to_string(index=False))
        print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
