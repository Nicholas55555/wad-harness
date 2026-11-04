"""Ablation (b) of Phase 4 (exploratory; PREREG_crc_lambda.md lists it): per-condition recalibration of the dial's
per-detection chance p(d).

The question (doc 10, ablation b): the model under-states chances in DAWN fog (17% of the boxes near the lam = 10
cut of 9.1% are real), so the fixed-lam rule p >= 1/(1+lam) drops boxes worth keeping there. Does an isotonic map
p -> share real, fitted per condition, fix that, and what does it cost where p is already calibrated?

On run 2's 200 calibration/test splits (wad.crc.make_splits, as wad.crc_lambda), per split and condition (the seven
BDD100K val bins and the four DAWN weathers), the variants are
  raw         model 2's chance as adopted (no recalibration)
  condition   isotonic map fitted on that condition's calibration-half detections (their real / false labels,
              the same data CRC calibrates on), applied to its test half
  pooled      one isotonic map fitted on all calibration-half detections of the dataset
  train       isotonic map per weather fitted on BDD100K train's out-of-fold chances (no labels from the target
              data; DAWN fog / rain / snow use train fog / rain / snow, DAWN sand all train weathers)
  one thr.    the one threshold tau(lam) fitted on train (reference)
and each is graded on the test half at lam = 1, 3, 10, 30 with the dial's cost, lam * misses + false alarms
(misses and false alarms from the matching at the floor, as wad.per_detection.grade), plus the calibration error
(ECE, the per_detection bins) and the share real near the cut (chance within x1.5 of 1/(1+lam)).
CRC over lam needs no recalibration: its cut depends on p only through the order of p within a condition, which a
monotone map keeps (an isotonic map can only tie neighbouring boxes).

Outputs results/ablation/: dial_recalibration_splits.csv (per split), dial_recalibration.csv (summaries),
dial_recalibration_calibration.csv (ECE and near-cut shares), dial_recalibration_reliability.csv (split 0's test
half, per bin, for the figure), dial_recalibration.md and .log.
Run: python -m wad.dial_recalibration   (CPU, a few minutes)
"""
from __future__ import annotations

import copy
import sys
import time

import numpy as np
import pandas as pd

from wad.config import REPO

OUT = REPO / "results" / "ablation"
LAMS = [1.0, 3.0, 10.0, 30.0]
N_SPLITS = 200
VARIANTS = ["raw", "condition", "pooled", "train", "one threshold"]
DAWN_TO_TRAIN = {"fog": "fog", "rain": "rain", "snow": "snow"}  # sand has no BDD100K counterpart
BINS = np.array([0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0 + 1e-9])  # wad.per_detection.BINS


def iso(p: np.ndarray, y: np.ndarray):
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True).fit(p, y)


def ece(p: np.ndarray, y: np.ndarray) -> float:
    b = np.digitize(p, BINS) - 1
    n = np.bincount(b, minlength=len(BINS))
    gap = np.abs(np.bincount(b, weights=p, minlength=len(BINS)) - np.bincount(b, weights=y, minlength=len(BINS)))
    return float(gap.sum() / max(n.sum(), 1))


def near(p: np.ndarray, y: np.ndarray, lam: float) -> tuple[int, float, float]:
    cut = 1 / (1 + lam)
    m = (p >= cut / 1.5) & (p <= min(cut * 1.5, 1.0))
    return int(m.sum()), float(p[m].mean()) if m.any() else np.nan, float(y[m].mean()) if m.any() else np.nan


def train_maps() -> dict:
    """Isotonic maps per BDD100K train weather from the 5-fold out-of-fold chances (model 2), and one over all."""
    from wad.data import load_images
    from wad.per_detection import oof_path
    from wad.thresh_net import cfg as train_cfg

    oof = pd.read_parquet(oof_path("yolo"))
    im = load_images(train_cfg())
    w = oof.image_id.map(dict(zip(im.image_id, im.weather))).to_numpy()
    p, y = oof.p.to_numpy(), oof.tp.to_numpy().astype(float)
    maps = {b: iso(p[w == b], y[w == b]) for b in np.unique(w[pd.notna(w)])}
    maps["all"] = iso(p, y)
    return maps


def run_dataset(ds: str, cfg, pr: pd.DataFrame, tau: dict, tmaps: dict, log) -> tuple[list, list, list]:
    from wad.crc import make_splits
    from wad.curves import ALL, Cache

    c = Cache(cfg)
    if not (np.array_equal(pr.image_id.to_numpy(), c.pred.image_id.to_numpy())
            and np.array_equal(pr.score.to_numpy(), c.pred.score.to_numpy())):
        raise ValueError(f"{ds}: chances are not aligned with the cached predictions")
    p, y, s = pr.p.to_numpy(), pr.tp.to_numpy().astype(bool), pr.score.to_numpy()
    img = pr.image_id.map(c.idx).to_numpy()
    n_gt = c.n_gt(ALL).astype(float)
    conds = sorted(set(c.weather))
    cond_img = np.array([conds.index(w) for w in c.weather])
    cond_det = cond_img[img]
    to_train = (lambda b: b) if ds == "BDD100K" else (lambda b: DAWN_TO_TRAIN.get(b, "all"))
    p_train = np.empty_like(p)
    for k, b in enumerate(conds):
        m = cond_det == k
        p_train[m] = tmaps[to_train(b)].predict(p[m])
    v = copy.deepcopy(cfg)
    v.splits["n_repeats"] = N_SPLITS
    splits = make_splits(v, c.images)
    rows, cal_rows, rel_rows = [], [], []
    t0 = time.perf_counter()
    for seed, sp in splits.groupby("seed", sort=True):
        cal_img = (sp.set_index("image_id").loc[c.image_ids].half == "cal").to_numpy()
        cal, test = cal_img[img], ~cal_img[img]
        chance = {"raw": p, "train": p_train, "condition": np.empty_like(p)}
        chance["pooled"] = iso(p[cal], y[cal]).predict(p)
        for k in range(len(conds)):
            m = cond_det == k
            chance["condition"][m] = iso(p[m & cal], y[m & cal]).predict(p[m])
        gt_test = np.bincount(cond_img[~cal_img], weights=n_gt[~cal_img], minlength=len(conds))
        n_test = np.bincount(cond_img[~cal_img], minlength=len(conds))
        for lam in LAMS:
            cut = 1 / (1 + lam)
            kept = {x: chance[x] >= cut for x in ("raw", "condition", "pooled", "train")}
            kept["one threshold"] = s >= tau[lam]
            for x, k_ in kept.items():
                tk = np.bincount(cond_det[test], weights=(k_ & y)[test], minlength=len(conds))
                fk = np.bincount(cond_det[test], weights=(k_ & ~y)[test], minlength=len(conds))
                for k, b in enumerate(conds):
                    rows.append(dict(dataset=ds, seed=int(seed), condition=b, lam=lam, variant=x, images=int(n_test[k]),
                                     objects=float(gt_test[k]), found=float(tk[k]), false_alarms=float(fk[k]),
                                     cost=float(lam * (gt_test[k] - tk[k]) + fk[k])))
        for x in ("raw", "condition", "pooled", "train"):
            for k, b in enumerate(conds):
                m = test & (cond_det == k)
                r = dict(dataset=ds, seed=int(seed), condition=b, variant=x, ece=ece(chance[x][m], y[m]))
                for lam in (3.0, 10.0):
                    n_near, mc, sr = near(chance[x][m], y[m], lam)
                    r.update({f"n_near_{lam:g}": n_near, f"mean_chance_near_{lam:g}": mc, f"share_real_near_{lam:g}": sr})
                cal_rows.append(r)
                if seed == 0:
                    bb = np.digitize(chance[x][m], BINS) - 1
                    for j in np.unique(bb):
                        q = bb == j
                        rel_rows.append(dict(dataset=ds, condition=b, variant=x, bin_lo=BINS[j], bin_hi=min(BINS[j + 1], 1.0),
                                             n=int(q.sum()), mean_chance=float(chance[x][m][q].mean()),
                                             share_real=float(y[m][q].mean())))
        if (seed + 1) % 50 == 0:
            log(f"  {ds}: {seed + 1}/{N_SPLITS} splits, {time.perf_counter() - t0:.0f} s")
    return rows, cal_rows, rel_rows


def summarize(sp: pd.DataFrame) -> pd.DataFrame:
    """Per dataset, condition (and 'all': summed over a dataset's conditions) and lam: mean cost per test image per
    variant, and the paired per-split ratio to raw and to the one threshold (median, 95% interval, share below 1)."""
    tot = sp.groupby(["dataset", "seed", "lam", "variant"], as_index=False)[["images", "objects", "found", "false_alarms", "cost"]].sum()
    sp = pd.concat([sp, tot.assign(condition="all")], ignore_index=True)
    w = sp.pivot_table(index=["dataset", "condition", "lam", "seed"], columns="variant", values="cost")
    out = []
    for (ds, cond, lam), g in w.groupby(level=[0, 1, 2], sort=False):
        base = sp[(sp.dataset == ds) & (sp.condition == cond) & (sp.lam == lam)]
        per_img = base.groupby("variant").apply(lambda d: d.cost.sum() / d.images.sum(), include_groups=False)
        rec = base.groupby("variant").apply(lambda d: d.found.sum() / d.objects.sum(), include_groups=False)
        fa = base.groupby("variant").apply(lambda d: d.false_alarms.sum() / d.images.sum(), include_groups=False)
        for x in VARIANTS:
            r_raw, r_one = g[x] / g["raw"], g[x] / g["one threshold"]
            out.append(dict(dataset=ds, condition=cond, lam=lam, variant=x, cost_per_img=per_img[x], recall=rec[x], fppi=fa[x],
                            vs_raw_median=r_raw.median(), vs_raw_q025=r_raw.quantile(0.025), vs_raw_q975=r_raw.quantile(0.975),
                            vs_raw_share_below_1=(r_raw < 1).mean(), vs_one_median=r_one.median(),
                            vs_one_q025=r_one.quantile(0.025), vs_one_q975=r_one.quantile(0.975)))
    return pd.DataFrame(out)


def report(summ: pd.DataFrame, calib: pd.DataFrame) -> str:
    lines = ["# Ablation (b): per-condition recalibration of the dial's chance", "",
             "Exploratory (PREREG_crc_lambda.md). Run 2's 200 calibration/test splits; maps fitted on the calibration half",
             "(condition, pooled) or on BDD100K train (train); cost lam * misses + false alarms on the test half.", "",
             "## Cost at fixed lambda: recalibrated / raw (median over splits, 95% interval)", "",
             "| dataset | condition | lam | condition map | pooled map | train map | raw / one thr. |", "|---|---|---|---|---|---|---|"]
    s = summ.set_index(["dataset", "condition", "lam", "variant"])
    for (ds, cond, lam), _ in summ.groupby(["dataset", "condition", "lam"], sort=False):
        def f(x):
            r = s.loc[(ds, cond, lam, x)]
            return f"{r.vs_raw_median:.3f} ({r.vs_raw_q025:.3f}-{r.vs_raw_q975:.3f})"
        raw = s.loc[(ds, cond, lam, "raw")]
        lines.append(f"| {ds} | {cond} | {lam:g} | {f('condition')} | {f('pooled')} | {f('train')} | "
                     f"{raw.vs_one_median:.3f} ({raw.vs_one_q025:.3f}-{raw.vs_one_q975:.3f}) |")
    c = calib.groupby(["dataset", "condition", "variant"], sort=False)[
        ["ece", "mean_chance_near_10", "share_real_near_10", "mean_chance_near_3", "share_real_near_3"]].mean().reset_index()
    lines += ["", "## Calibration on the test half (mean over splits)", "",
              "| dataset | condition | variant | ECE | near lam=10 cut: mean chance / share real | near lam=3 cut |", "|---|---|---|---|---|---|"]
    for r in c.itertuples():
        lines.append(f"| {r.dataset} | {r.condition} | {r.variant} | {r.ece:.4f} | {r.mean_chance_near_10:.3f} / {r.share_real_near_10:.3f} | "
                     f"{r.mean_chance_near_3:.3f} / {r.share_real_near_3:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from wad.one_threshold import TRAIN, eval_configs, train_curve, train_threshold
    from wad.per_detection import pred_path

    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / "dial_recalibration.log", "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.perf_counter()
    _, curve = train_curve(pd.read_parquet(TRAIN, columns=["score", "tp"]))
    tau = {lam: train_threshold(curve, lam) for lam in LAMS}
    tmaps = train_maps()
    log(f"train maps for {sorted(tmaps)}, {time.perf_counter() - t0:.0f} s")
    rows, cal_rows, rel_rows = [], [], []
    for ds, cfg in eval_configs().items():
        pr = pd.read_parquet(pred_path("bdd" if ds == "BDD100K" else "dawn", "yolo"))
        a, b, r = run_dataset(ds, cfg, pr, tau, tmaps, log)
        rows += a
        cal_rows += b
        rel_rows += r
        log(f"{ds}: done, {time.perf_counter() - t0:.0f} s")
    sp, calib = pd.DataFrame(rows), pd.DataFrame(cal_rows)
    summ = summarize(sp)
    ff = "%.6g"
    sp.to_csv(OUT / "dial_recalibration_splits.csv", index=False, float_format=ff)
    summ.to_csv(OUT / "dial_recalibration.csv", index=False, float_format=ff)
    calib.to_csv(OUT / "dial_recalibration_calibration.csv", index=False, float_format=ff)
    pd.DataFrame(rel_rows).to_csv(OUT / "dial_recalibration_reliability.csv", index=False, float_format=ff)
    md = report(summ, calib)
    (OUT / "dial_recalibration.md").write_text(md, encoding="utf-8")
    log(md)
    log(f"done, {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
