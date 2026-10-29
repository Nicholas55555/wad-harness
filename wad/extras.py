"""Analyses beyond the brief, each labelled as such in results.md. None changes a pre-registered
choice; they quantify how much the pre-registered results can be trusted and why they come out
the way they do.

  go_no_go_bootstrap   cluster bootstrap (near-duplicate groups within weather bins) of the
                       go/no-go margins and verdict: how fragile is GO?
  misses_by_size       which objects are never proposed at the conf floor, by box size at the
                       detector's input resolution (the irrecoverable part of the loss)
  alpha_sweep          EXPLORATORY: the oracle-bin and global policies over a grid of budgets,
                       to show where each bin becomes feasible and what FPPI each budget costs
  robustness_seeds     the phase 4 procedure with 200 split repeats instead of 10
  near_dup_dropped     phase 4 with one image per near-duplicate group (crc --variant)

Run: python -m wad.extras [--smoke]
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd

from wad.config import CANONICAL, Cfg, bins, load_config, results_path, tau_grid, write_json
from wad.crc import fit_threshold, make_splits
from wad.crc import run as crc_run
from wad.crc import write as crc_write
from wad.curves import ALL, Cache, summarize


# ---------------------------------------------------------------- go/no-go bootstrap
def go_no_go_bootstrap(cfg: Cfg, n_boot: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    c = Cache(cfg)
    t25 = cfg.go_no_go.reference_threshold
    floor = cfg.detector.conf_floor
    margin_pts = cfg.go_no_go.margin_points
    classes = list(cfg.go_no_go.classes)
    feats = {}
    for k in classes:
        feats[k] = np.c_[c.n_gt(k),
                         c.at_thresholds(np.full(c.n_img, floor), k)["n_missed"],
                         c.at_thresholds(np.full(c.n_img, t25), k)["n_missed"]]
    rng = np.random.default_rng(seed)
    groups = c.images.group.to_numpy()
    reps = {}  # bin -> {cls: (B, 3) sums}
    point = {}
    for b in bins(cfg):
        rows = np.nonzero(c.weather == b)[0]
        g_ids, inv = np.unique(groups[rows], return_inverse=True)
        idx = rng.integers(0, len(g_ids), size=(n_boot, len(g_ids)))
        reps[b], point[b] = {}, {}
        for k in classes:
            per_group = np.zeros((len(g_ids), 3))
            np.add.at(per_group, inv, feats[k][rows])
            reps[b][k] = per_group[idx].sum(axis=1)
            point[b][k] = per_group.sum(axis=0)

    def recalls(s):
        n = np.maximum(s[..., 0], 1)
        return 1 - s[..., 1] / n, 1 - s[..., 2] / n

    rows_out, passed = [], {}
    for k in classes:
        maxr = np.stack([recalls(reps[b][k])[0] for b in bins(cfg)], axis=1)  # (B, n_bins)
        r25 = np.stack([recalls(reps[b][k])[1] for b in bins(cfg)], axis=1)
        worst = np.argmin(maxr, axis=1)
        m = 100 * (maxr - r25)
        worst_margin = m[np.arange(n_boot), worst]
        passed[k] = worst_margin >= margin_pts
        for j, b in enumerate(bins(cfg)):
            pm, p25 = recalls(point[b][k])
            rows_out.append(dict(cls=k, bin=b, n_gt=int(point[b][k][0]), margin_points=100 * (pm - p25),
                                 margin_ci_lo=np.percentile(m[:, j], 2.5), margin_ci_hi=np.percentile(m[:, j], 97.5),
                                 max_recall=pm, max_recall_ci_lo=np.percentile(maxr[:, j], 2.5),
                                 max_recall_ci_hi=np.percentile(maxr[:, j], 97.5),
                                 p_worst_bin=float(np.mean(worst == j)),
                                 p_margin_ge_threshold=float(np.mean(m[:, j] >= margin_pts))))
        rows_out.append(dict(cls=k, bin="worst", n_gt=np.nan, margin_points=np.nan,
                             margin_ci_lo=np.percentile(worst_margin, 2.5), margin_ci_hi=np.percentile(worst_margin, 97.5),
                             p_margin_ge_threshold=float(np.mean(passed[k]))))
    df = pd.DataFrame(rows_out)
    p_go = float(np.mean(np.logical_and.reduce([passed[k] for k in classes])))
    summary = dict(n_boot=n_boot, seed=seed, resampling="near-duplicate groups with replacement, within weather bin",
                   p_go=p_go, p_pass={k: float(np.mean(passed[k])) for k in classes})
    return df, summary


# ---------------------------------------------------------------- irrecoverable misses by size
SIZE_EDGES = [0, 16, 32, 64, 96, np.inf]  # sqrt(box area) after scaling the image to a 640-px long side
SIZE_LABELS = ["<16", "16-32", "32-64", "64-96", ">=96"]


def misses_by_size(cfg: Cfg) -> pd.DataFrame:
    c = Cache(cfg)
    from wad.data import load_gt

    gt = load_gt(cfg).set_index("gt_id")
    g = c.gt.join(gt[["x1", "y1", "x2", "y2", "width", "height"]], on="gt_id")
    scale = 640 / np.maximum(g.width, g.height)  # fixed 640-px reference, whatever the detector's imgsz
    side = np.sqrt((g.x2 - g.x1) * (g.y2 - g.y1)) * scale
    g = g.assign(size=pd.cut(side, SIZE_EDGES, labels=SIZE_LABELS, right=False),
                 never=~np.isfinite(c.gt_score),
                 missed_025=c.gt_score < cfg.go_no_go.reference_threshold)
    out = []
    for b in bins(cfg) + [ALL]:
        gb = g if b == ALL else g[g.weather == b]
        for cls in [ALL, "person", "car"]:
            gc = gb if cls == ALL else gb[gb.cls == cls]
            for s in SIZE_LABELS + [ALL]:
                gs = gc if s == ALL else gc[gc["size"] == s]
                out.append(dict(bin=b, cls=cls, size=s, n_gt=len(gs),
                                share_of_boxes=len(gs) / max(len(gc), 1),
                                never_proposed=gs.never.mean() if len(gs) else np.nan,
                                missed_at_0_25=gs.missed_025.mean() if len(gs) else np.nan,
                                share_of_never_proposed=gs.never.sum() / max(gc.never.sum(), 1)))
    return pd.DataFrame(out)


def miss_reasons(cfg: Cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Why is a ground-truth box never matched, even at the conf floor? For each such box, in order:
      taken        a same-class detection overlaps it at IoU >= 0.5 but was matched to another box
      wrong_class  a detection of another class overlaps it at IoU >= 0.5 (e.g. truck called car)
      loose_box    the best same-class detection reaches only 0.3 <= IoU < 0.5
      no_proposal  nothing of the right class within IoU 0.3
    """
    from wad.data import load_gt
    from wad.match import iou_matrix

    c = Cache(cfg)
    gt = load_gt(cfg).set_index("gt_id").loc[c.gt.gt_id]
    det = pd.read_parquet(cfg.paths.detections / "detections.parquet")
    det_by = {k: v for k, v in det.groupby("image_id")}
    never = ~np.isfinite(c.gt_score)
    box = ["x1", "y1", "x2", "y2"]
    rows = []
    for image_id, g in gt[never].groupby("image_id"):
        d = det_by.get(image_id)
        if d is None or len(d) == 0:
            for r in g.itertuples():
                rows.append(dict(gt_id=r.Index, image_id=image_id, weather=r.weather, cls=r.cls, reason="no_proposal",
                                 other=""))
            continue
        iou = iou_matrix(g[box].to_numpy(), d[box].to_numpy())
        same = g.cls.to_numpy()[:, None] == d.cls.to_numpy()[None, :]
        for i, r in enumerate(g.itertuples()):
            s_best = iou[i][same[i]].max(initial=0.0)
            o = np.where(~same[i], iou[i], -1)
            o_best = o.max(initial=-1)
            if s_best >= cfg.match_iou:
                reason, other = "taken", ""
            elif o_best >= cfg.match_iou:
                reason, other = "wrong_class", d.cls.to_numpy()[int(np.argmax(o))]
            elif s_best >= 0.3:
                reason, other = "loose_box", ""
            else:
                reason, other = "no_proposal", ""
            rows.append(dict(gt_id=r.Index, image_id=image_id, weather=r.weather, cls=r.cls, reason=reason,
                             other=other))
    m = pd.DataFrame(rows)
    n_gt = c.gt.groupby("weather").size().to_dict()
    n_gt[ALL] = len(c.gt)
    out = []
    for b in bins(cfg) + [ALL]:
        mb = m if b == ALL else m[m.weather == b]
        for reason in ["no_proposal", "wrong_class", "loose_box", "taken"]:
            k = int((mb.reason == reason).sum())
            out.append(dict(bin=b, reason=reason, n=k, share_of_never_matched=k / max(len(mb), 1),
                            share_of_all_gt=k / n_gt[b]))
    conf = m[m.reason == "wrong_class"].groupby(["cls", "other"]).size().rename("n").reset_index()
    return pd.DataFrame(out), conf.sort_values("n", ascending=False), m


# ---------------------------------------------------------------- exploratory alpha sweep
def alpha_sweep(cfg: Cfg, alphas=None) -> pd.DataFrame:
    """EXPLORATORY. Same splits, rule and evaluation as phase 4, over a grid of budgets."""
    alphas = np.round(np.arange(0.05, 0.501, 0.025), 3) if alphas is None else alphas
    c = Cache(cfg)
    taus = tau_grid(cfg)
    floor = cfg.detector.conf_floor
    loss = c.loss_grid(taus, ALL)
    splits = make_splits(cfg, c.images)
    out = []
    for seed, sp in splits.groupby("seed"):
        cal = (sp.set_index("image_id").loc[c.image_ids].half == "cal").to_numpy()
        test = ~cal
        for a in alphas:
            pol = {}
            t = np.full(c.n_img, np.nan)
            inf = {}
            for b in bins(cfg):
                f = fit_threshold(loss[cal & (c.weather == b)], taus, a, floor)
                t[c.weather == b] = f["tau_hat"]
                inf[b] = f["infeasible"]
            pol["oracle_bin"] = t
            f = fit_threshold(loss[cal], taus, a, floor)
            pol["global"] = np.full(c.n_img, f["tau_hat"])
            inf["global"] = f["infeasible"]
            ref = cfg.get("reference_bin")
            if ref:
                f = fit_threshold(loss[cal & (c.weather == ref)], taus, a, floor)
                pol["ref_calibrated"] = np.full(c.n_img, f["tau_hat"])
                inf["ref_calibrated"] = f["infeasible"]
            for p, ti in pol.items():
                counts = c.at_thresholds(ti, ALL)
                for b in bins(cfg) + [ALL]:
                    rows = np.nonzero(test & ((c.weather == b) if b != ALL else True))[0]
                    s = summarize(counts, rows, per_class=False)
                    out.append(dict(seed=seed, alpha=a, policy=p, bin=b, image_loss=s["image_loss"], fppi=s["fppi"],
                                    miss_rate=s["miss_rate"],
                                    tau=float(ti[rows][0]) if b != ALL or p != "oracle_bin" else np.nan,
                                    infeasible=inf.get(b if p == "oracle_bin" else p, np.nan)))
    d = pd.DataFrame(out)
    d["infeasible"] = d.infeasible.astype(float)
    d["exceeds"] = d.image_loss > d.alpha
    return d.groupby(["alpha", "policy", "bin"]).agg(
        image_loss_mean=("image_loss", "mean"), image_loss_std=("image_loss", "std"),
        fppi_mean=("fppi", "mean"), fppi_std=("fppi", "std"), miss_rate_mean=("miss_rate", "mean"),
        tau_mean=("tau", "mean"), infeasible_frac=("infeasible", "mean"),
        exceed_frac=("exceeds", "mean"), n_seeds=("seed", "nunique")).reset_index()


def min_feasible_alpha(cfg: Cfg) -> pd.DataFrame:
    """Per bin and seed: the smallest budget the rule can certify, (n * R(floor) + 1) / (n + 1)."""
    c = Cache(cfg)
    taus = tau_grid(cfg)
    loss_floor = c.loss_grid(taus[:1], ALL)[:, 0]
    out = []
    for seed, sp in make_splits(cfg, c.images).groupby("seed"):
        cal = (sp.set_index("image_id").loc[c.image_ids].half == "cal").to_numpy()
        for b in bins(cfg) + [ALL]:
            rows = cal & ((c.weather == b) if b != ALL else True)
            n = int(rows.sum())
            out.append(dict(seed=seed, bin=b, n_cal=n, min_alpha=(loss_floor[rows].sum() + 1) / (n + 1)))
    return pd.DataFrame(out).groupby("bin").agg(n_cal=("n_cal", "mean"), min_alpha_mean=("min_alpha", "mean"),
                                                min_alpha_min=("min_alpha", "min"),
                                                min_alpha_max=("min_alpha", "max")).reset_index()


def fig5(cfg: Cfg, sweep: pd.DataFrame, minalpha: pd.DataFrame):
    import matplotlib.pyplot as plt

    from wad.figures import BIN_LABEL, CRITICAL, INK, INK2, SERIES, save, style

    style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for i, b in enumerate(bins(cfg)):
        d = sweep[(sweep.policy == "oracle_bin") & (sweep.bin == b)]
        feas = d.infeasible_frac < 0.5
        for ax, col in zip(axes, ["image_loss_mean", "fppi_mean"]):
            ax.plot(d.alpha, d[col], color=SERIES[i], lw=1.6, label=BIN_LABEL[b])
            ax.plot(d.alpha[~feas], d[col][~feas], ls="none", marker="o", ms=5, mfc="white", mec=SERIES[i], mew=1.2)
            ax.plot(d.alpha[feas], d[col][feas], ls="none", marker="o", ms=5, mfc=SERIES[i], mec="white", mew=1.0)
    for ax, col in zip(axes, ["image_loss_mean", "fppi_mean"]):
        d = sweep[(sweep.policy == "global") & (sweep.bin == ALL)]
        ax.plot(d.alpha, d[col], color=INK2, lw=1.2, ls="--", label="Global CRC (pooled test)")
    a = sweep.alpha.unique()
    axes[0].plot(a, a, color=CRITICAL, lw=1.0, label="loss = α")
    axes[0].set_ylabel("Test image-level miss fraction (oracle-bin)")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Test FPPI (log scale, oracle-bin)")
    for ax in axes:
        ax.set_xlabel("Budget α")
        for x in cfg.alphas:
            ax.axvline(x, color="#c3c2b7", lw=0.8, zorder=0)
    axes[0].legend(loc="upper left", fontsize=7.5)
    txt = "smallest certifiable α (mean over splits): " + " · ".join(
        f"{BIN_LABEL[r.bin]} {r.min_alpha_mean:.3f}" for r in minalpha.itertuples() if r.bin != ALL)
    fig.suptitle("EXPLORATORY (not pre-registered): oracle-bin CRC over a grid of budgets. Hollow = infeasible in ≥ "
                 "half the splits.", x=0.02, ha="left", fontsize=10, color=INK, fontweight="semibold")
    fig.text(0.02, 0.905, txt, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return save(fig, cfg, "extra/fig5_alpha_sweep")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke)
    ff = "%.6g"

    df, summ = go_no_go_bootstrap(cfg)
    df.to_csv(results_path(cfg, "extra/go_no_go_bootstrap.csv"), index=False, float_format=ff)
    write_json(results_path(cfg, "extra/go_no_go_bootstrap.json"), summ)
    print(f"bootstrap P(GO) = {summ['p_go']:.3f}; per class {summ['p_pass']}")

    sz = misses_by_size(cfg)
    sz.to_csv(results_path(cfg, "extra/misses_by_size.csv"), index=False, float_format=ff)
    print(sz[(sz.bin == ALL) & (sz.cls == ALL)].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    mr, conf, _ = miss_reasons(cfg)
    mr.to_csv(results_path(cfg, "extra/miss_reasons.csv"), index=False, float_format=ff)
    conf.to_csv(results_path(cfg, "extra/miss_class_confusions.csv"), index=False)
    print(mr[mr.bin == ALL].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(conf.head(8).to_string(index=False))

    sweep = alpha_sweep(cfg)
    sweep.to_csv(results_path(cfg, "extra/alpha_sweep_EXPLORATORY.csv"), index=False, float_format=ff)
    ma = min_feasible_alpha(cfg)
    ma.to_csv(results_path(cfg, "extra/min_feasible_alpha.csv"), index=False, float_format=ff)
    print(ma.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if not cfg.smoke_mode:
        fig5(cfg, sweep, ma)

    # robustness: 200 split repeats (same rule, same seeds 0..199)
    cfg200 = copy.deepcopy(cfg)
    cfg200.splits["n_repeats"] = 200
    out = crc_run(cfg200)
    out["summary"].to_csv(results_path(cfg, "sensitivity/seeds200/crc_summary.csv"), index=False, float_format=ff)
    out["perclass"].to_csv(results_path(cfg, "sensitivity/seeds200/crc_perclass.csv"), index=False, float_format=ff)

    # sensitivity: one image per near-duplicate group, cross-weather groups dropped
    crc_write(cfg, crc_run(cfg, "near_dup_dropped"), "near_dup_dropped")
    print("wrote results/extra and results/sensitivity")


if __name__ == "__main__":
    main()
