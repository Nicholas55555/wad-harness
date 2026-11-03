"""CRC over the dial lam vs per-weather CRC over the threshold (Phase 4, PREREG_crc_lambda.md).

The baseline's conformal rule, applied to the dial's per-detection cut, compared with the baseline itself on
the same 200 calibration/test splits:
  baseline  per-bin CRC over the detector score tau (SeqCRC step 1 per weather bin; run 2's policy)
  ours      per-bin CRC over the cut c = 1 / (1 + lam) on the per-detection chance p(d) (model 2): the
            smallest lam whose conformal bound on the calibration half holds
Both use run 2's grid, loss and splits (wad.crc): BDD100K val and DAWN (clean labels), YOLOv8x @1280,
grouped matching. An object is found iff its matched box is kept (the matching at the floor is fixed), so
the image loss is non-decreasing in tau and non-increasing in lam, as CRC needs.

Outputs results/crc_lambda/: crc_lambda_eval.csv (per split), crc_lambda_summary.csv, crc_lambda_ratio.csv
(paired per-split false-alarm ratios), crc_lambda_min_alpha.csv, crc_lambda_criteria.csv, crc_lambda.md and
fig24_crc_lambda.png/.pdf.

Run: python -m wad.crc_lambda   (CPU, a few minutes)
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd

from wad.config import REPO, bins, tau_grid
from wad.crc import fit_threshold, make_splits
from wad.curves import ALL, Cache, summarize

OUT = REPO / "results" / "crc_lambda"
ALPHAS = [0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30]
PRIMARY_ALPHA = 0.20
N_SPLITS = 200
METHODS = {"baseline": "per-weather CRC over the score (Phase 1 baseline)", "ours": "CRC over the dial (per-detection chance)"}
RUN2 = {"BDD100K": "results/run2/bdd/v8x_1280__group/seeds200/crc_summary.csv",
        "DAWN": "results/run2/dawn/v8x_1280__group__clean/seeds200/crc_summary.csv"}


def chance_cache(c: Cache, p: np.ndarray) -> Cache:
    """A copy of c whose 'scores' are the per-detection chances: an object counts as found at cut x iff its
    matched box has p >= x, and a false box counts at cut x iff its p >= x."""
    tp = c.pred.tp.to_numpy().astype(bool)
    if len(p) != len(c.pred):
        raise ValueError("one chance per cached prediction expected")
    if not np.array_equal(c.fp_score, c.pred.score.to_numpy()[~tp]):
        raise ValueError("false-positive arrays are not the cache's predictions in order")
    pos = pd.Series(np.arange(len(c.gt)), index=c.gt.gt_id.to_numpy())
    gt_p = np.full(len(c.gt), -np.inf)
    gt_p[pos.loc[c.pred.gt_id.to_numpy()[tp]].to_numpy()] = p[tp]
    if not np.array_equal(np.isfinite(gt_p), np.isfinite(c.gt_score)):
        raise ValueError("matched objects differ between the score and the chance view")
    out = Cache.__new__(Cache)
    out.__dict__.update(c.__dict__)
    out.gt_score, out.fp_score = gt_p, np.asarray(p, float)[~tp]
    return out


def evaluate(ds: str, cfg, p: np.ndarray, alphas=ALPHAS, n_splits=N_SPLITS) -> pd.DataFrame:
    """Every split x alpha x method x scope (per bin / global), evaluated per bin on the test half."""
    c = Cache(cfg)
    caches = {"baseline": c, "ours": chance_cache(c, p)}
    grid, floor = tau_grid(cfg), cfg.detector.conf_floor
    losses = {m: cc.loss_grid(grid, ALL) for m, cc in caches.items()}
    v = copy.deepcopy(cfg)
    v.splits["n_repeats"] = n_splits
    splits = make_splits(v, c.images)
    names = bins(cfg)
    rows = []
    for seed, sp in splits.groupby("seed", sort=True):
        cal = (sp.set_index("image_id").loc[c.image_ids].half == "cal").to_numpy()
        test = ~cal
        for alpha in alphas:
            for method, cc in caches.items():
                for scope in ("per_bin", "global"):
                    cut = np.full(c.n_img, np.nan)
                    fits = {}
                    if scope == "per_bin":
                        for b in names:
                            f = fit_threshold(losses[method][cal & (c.weather == b)], grid, alpha, floor)
                            cut[c.weather == b], fits[b] = f["tau_hat"], f
                    else:
                        f = fit_threshold(losses[method][cal], grid, alpha, floor)
                        cut[:] = f["tau_hat"]
                        fits = {b: f for b in names}
                    counts = cc.at_thresholds(cut, ALL)
                    for b in names:
                        r = np.nonzero(test & (c.weather == b))[0]
                        s = summarize(counts, r, per_class=False)
                        kept_real = float((counts["n_gt"][r] - counts["n_missed"][r]).sum())
                        n_fp = float(counts["n_fp"][r].sum())
                        rows.append(dict(dataset=ds, seed=int(seed), alpha=alpha, method=method, scope=scope, bin=b,
                                         cut=fits[b]["tau_hat"], infeasible=bool(fits[b]["infeasible"]),
                                         n_cal=int(fits[b]["n_cal"]), image_loss=s["image_loss"],
                                         miss_rate=s["miss_rate"], fppi=s["fppi"], n_test=s["n_images"],
                                         n_gt=s["n_gt"], kept_real=kept_real, n_fp=n_fp))
    return pd.DataFrame(rows)


def min_alpha(ds: str, cfg, p: np.ndarray, n_splits=N_SPLITS) -> pd.DataFrame:
    """Smallest certifiable alpha per bin and method: the conformal bound at the lowest grid point,
    n/(n+1) R(floor) + 1/(n+1), averaged over the calibration halves (as run 2 reports it)."""
    c = Cache(cfg)
    caches = {"baseline": c, "ours": chance_cache(c, p)}
    grid = tau_grid(cfg)
    v = copy.deepcopy(cfg)
    v.splits["n_repeats"] = n_splits
    splits = make_splits(v, c.images)
    rows = []
    for method, cc in caches.items():
        l0 = cc.loss_grid(grid[:1], ALL)[:, 0]
        for seed, sp in splits.groupby("seed", sort=True):
            cal = (sp.set_index("image_id").loc[c.image_ids].half == "cal").to_numpy()
            for b in bins(cfg):
                m = cal & (c.weather == b)
                n = int(m.sum())
                rows.append(dict(dataset=ds, method=method, bin=b, seed=int(seed), min_alpha=(l0[m].sum() + 1) / (n + 1)))
    return pd.DataFrame(rows).groupby(["dataset", "method", "bin"], sort=False).min_alpha.mean().reset_index()


def summarize_eval(ev: pd.DataFrame) -> pd.DataFrame:
    e = ev.copy()
    e["exceeds"] = (e.image_loss > e.alpha).astype(float)
    e["recall"] = 1 - e.miss_rate
    e["precision"] = e.kept_real / (e.kept_real + e.n_fp).where(e.kept_real + e.n_fp > 0)
    e["lam"] = np.where(e.method == "ours", 1 / e.cut - 1, np.nan)
    keys = ["dataset", "alpha", "scope", "bin", "method"]
    s = e.groupby(keys, sort=False).agg(
        n_splits=("seed", "nunique"), image_loss_mean=("image_loss", "mean"), image_loss_std=("image_loss", "std"),
        exceed_frac=("exceeds", "mean"), recall_mean=("recall", "mean"), precision_mean=("precision", "mean"),
        fppi_mean=("fppi", "mean"), fppi_q025=("fppi", lambda x: x.quantile(0.025)),
        fppi_q975=("fppi", lambda x: x.quantile(0.975)), cut_mean=("cut", "mean"), cut_median=("cut", "median"),
        lam_median=("lam", "median"), infeasible_frac=("infeasible", "mean"), n_cal_mean=("n_cal", "mean"),
        n_test_mean=("n_test", "mean")).reset_index()
    return s


def paired_ratios(ev: pd.DataFrame) -> pd.DataFrame:
    """Per split, false alarms per image of ours / baseline, per bin and pooled over each dataset and all."""
    k = ["dataset", "alpha", "scope", "seed"]
    w = ev.pivot_table(index=k + ["bin"], columns="method", values="n_fp", aggfunc="sum")
    w = w.join(ev[ev.method == "baseline"].set_index(k + ["bin"]).n_test)
    w["ratio"] = w.ours / w.baseline
    per_bin = w.reset_index()
    pooled = []
    for (alpha, scope, seed), g in per_bin.groupby(["alpha", "scope", "seed"]):
        for ds, gg in list(g.groupby("dataset")) + [("both", g)]:
            pooled.append(dict(dataset=ds, alpha=alpha, scope=scope, seed=seed, bin=ALL,
                               ratio=gg.ours.sum() / gg.baseline.sum()))
    r = pd.concat([per_bin[["dataset", "alpha", "scope", "seed", "bin", "ratio"]], pd.DataFrame(pooled)], ignore_index=True)
    return r.groupby(["dataset", "alpha", "scope", "bin"], sort=False).ratio.agg(
        ratio_median="median", ratio_q025=lambda x: x.quantile(0.025), ratio_q975=lambda x: x.quantile(0.975),
        share_below_1=lambda x: (x < 1).mean()).reset_index()


def criteria(summary: pd.DataFrame, ratios: pd.DataFrame) -> pd.DataFrame:
    s = summary[(summary.alpha == PRIMARY_ALPHA) & (summary.scope == "per_bin")]
    w = s.pivot_table(index=["dataset", "bin"], columns="method", values="image_loss_mean")
    held = w[w.baseline <= PRIMARY_ALPHA + 0.01]
    c1 = bool((held.ours <= PRIMARY_ALPHA + 0.01).all())
    r = ratios[(ratios.alpha == PRIMARY_ALPHA) & (ratios.scope == "per_bin")]
    per = r[r.bin != ALL]
    n_below = int((per.ratio_median < 1).sum())
    pooled = r[(r.dataset == "both") & (r.bin == ALL)].iloc[0]
    return pd.DataFrame([
        dict(criterion="C1 validity: ours mean held-out miss <= alpha + 0.01 wherever the baseline holds (alpha 0.2)",
             value=f"{len(held)} conditions where the baseline holds; ours max mean miss {held.ours.max():.4f}", met=c1),
        dict(criterion="C2 price: median per-split FPPI ratio ours/baseline < 1 in >= 9 of 11 conditions",
             value=f"{n_below} of {len(per)}", met=n_below >= 9),
        dict(criterion="C3 pooled price: 97.5th percentile of the pooled per-split ratio < 1",
             value=f"median {pooled.ratio_median:.3f}, 95% {pooled.ratio_q025:.3f}-{pooled.ratio_q975:.3f}",
             met=bool(pooled.ratio_q975 < 1))])


def sanity(summary: pd.DataFrame) -> None:
    """The baseline arm must reproduce run 2's 200-split per-bin summary (PREREG_crc_lambda.md)."""
    for ds, path in RUN2.items():
        r2 = pd.read_csv(REPO / path)
        r2 = r2[(r2.policy == "oracle_bin") & (r2.cls == ALL) & (r2.bin != ALL)].set_index(["alpha", "bin"])
        s = summary[(summary.dataset == ds) & (summary.method == "baseline") & (summary.scope == "per_bin")]
        for row in s[s.alpha.isin(r2.index.get_level_values(0))].itertuples():
            ref = r2.loc[(row.alpha, row.bin)]
            for ours_v, ref_v, what in ((row.image_loss_mean, ref.image_loss_mean, "image loss"), (row.fppi_mean, ref.fppi_mean, "FPPI")):
                if not np.isclose(ours_v, ref_v, rtol=1e-4, atol=1e-6):
                    raise AssertionError(f"{ds} {row.bin} alpha {row.alpha}: {what} {ours_v} vs run 2 {ref_v}")
    print("sanity: the baseline arm reproduces run 2's 200-split per-bin summary")


LABEL = {"clear": "clear", "cloudy": "cloudy", "rain": "rain", "snow": "snow", "clear_night": "clear night",
         "rain_night": "rain night", "snow_night": "snow night", "fog": "fog", "sand": "sand"}


def figure(summary: pd.DataFrame, ratios: pd.DataFrame, name="fig24_crc_lambda") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wad.config import THRESH_RESULTS

    s = summary[summary.scope == "per_bin"]
    conds = [(ds, b) for ds in ("BDD100K", "DAWN") for b in s[s.dataset == ds].bin.unique()]
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.6))
    col = {"baseline": "#7B8794", "ours": "#C26A00"}
    for ax, (ds, b) in zip(axes.flat, conds):
        for m, lab in (("baseline", "baseline: CRC over τ"), ("ours", "ours: CRC over λ")):
            g = s[(s.dataset == ds) & (s.bin == b) & (s.method == m)].sort_values("alpha")
            ax.plot(g.image_loss_mean, g.fppi_mean, "-o", color=col[m], ms=3.5, lw=1.8, label=lab)
            a2 = g[np.isclose(g.alpha, PRIMARY_ALPHA)]
            ax.plot(a2.image_loss_mean, a2.fppi_mean, "o", ms=8, mfc="none", mec=col[m], mew=1.6)
        ax.set_yscale("log")
        ax.set_title(f"{ds} {LABEL.get(b, b)}", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(labelsize=8)
    ax = axes.flat[len(conds)]
    r = ratios[(ratios.scope == "per_bin") & (ratios.bin == ALL)]
    for ds, c_ in (("BDD100K", "#2F6FDB"), ("DAWN", "#2BA36B"), ("both", "#1F2933")):
        g = r[r.dataset == ds].sort_values("alpha")
        ax.plot(g.alpha, g.ratio_median, "-o", ms=3.5, color=c_, label=ds if ds != "both" else "all 11")
        ax.fill_between(g.alpha, g.ratio_q025, g.ratio_q975, color=c_, alpha=0.15)
    ax.axhline(1, color="#7B8794", lw=1, ls="--")
    ax.set_title("false alarms: ours / baseline", fontsize=10)
    ax.set_xlabel("miss budget α", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)
    for ax in axes.flat[len(conds) + 1:]:
        ax.axis("off")
    for ax in axes[-1, :3]:
        ax.set_xlabel("held-out image miss (mean over 200 splits)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("false alarms per image", fontsize=9)
    axes.flat[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Same certified miss budget, fewer false alarms? CRC over the dial λ vs per-weather CRC over the "
                 "score (α = 0.10 ... 0.30; big ring: α = 0.2)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(THRESH_RESULTS / "figures" / f"{name}.{ext}", dpi=170)
        fig.savefig(OUT / f"{name}.{ext}", dpi=170)
    plt.close(fig)


def report(summary: pd.DataFrame, ratios: pd.DataFrame, crit: pd.DataFrame, mina: pd.DataFrame) -> str:
    s = summary[(summary.alpha == PRIMARY_ALPHA) & (summary.scope == "per_bin")]
    r = ratios[(ratios.alpha == PRIMARY_ALPHA) & (ratios.scope == "per_bin")].set_index(["dataset", "bin"])
    lines = ["# CRC over the dial λ vs per-weather CRC over the score (Phase 4)", "",
             "Generated by `python -m wad.crc_lambda` (`PREREG_crc_lambda.md`). 200 calibration/test splits (run 2's), "
             "YOLOv8x @1280, grouped matching; BDD100K val and DAWN (clean labels). Cells: held-out image miss "
             "(share of splits above α) / recall / precision / false alarms per image.", "",
             f"## α = {PRIMARY_ALPHA}, per weather bin", "",
             "| condition | baseline: CRC over τ | ours: CRC over λ | ours λ̂ (median) | FA ratio ours/baseline (95%) |",
             "|---|---|---|---|---|"]
    for (ds, b), g in s.groupby(["dataset", "bin"], sort=False):
        bl, ou = g[g.method == "baseline"].iloc[0], g[g.method == "ours"].iloc[0]
        rr = r.loc[(ds, b)]

        def cell(x):
            inf = f", inf {x.infeasible_frac:.0%}" if x.infeasible_frac > 0 else ""
            return (f"{x.image_loss_mean:.3f} ({x.exceed_frac:.0%}{inf}) / {x.recall_mean:.1%} / {x.precision_mean:.1%} / "
                    f"{x.fppi_mean:.2f}")
        lines.append(f"| {ds} {LABEL.get(b, b)} | {cell(bl)} | {cell(ou)} | {ou.lam_median:.1f} | "
                     f"{rr.ratio_median:.3f} ({rr.ratio_q025:.3f}-{rr.ratio_q975:.3f}) |")
    for ds in ("BDD100K", "DAWN", "both"):
        rr = r.loc[(ds, ALL)]
        lines.append(f"| pooled: {ds if ds != 'both' else 'all 11 conditions'} | | | | {rr.ratio_median:.3f} "
                     f"({rr.ratio_q025:.3f}-{rr.ratio_q975:.3f}) |")
    lines += ["", "## Pre-registered criteria", "", "| criterion | value | verdict |", "|---|---|---|"]
    lines += [f"| {x.criterion} | {x.value} | {'met' if x.met else 'not met'} |" for x in crit.itertuples()]
    lines += ["", "## Smallest certifiable α (mean over calibration halves)", "", "| condition | baseline | ours |", "|---|---|---|"]
    w = mina.pivot_table(index=["dataset", "bin"], columns="method", values="min_alpha", sort=False)
    lines += [f"| {ds} {LABEL.get(b, b)} | {x.baseline:.3f} | {x.ours:.3f} |" for (ds, b), x in w.iterrows()]
    return "\n".join(lines) + "\n"


def main():
    import logging
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # the report prints lambda / alpha on any console
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    t0 = time.perf_counter()
    from wad.one_threshold import eval_configs
    from wad.per_detection import pred_path

    OUT.mkdir(parents=True, exist_ok=True)
    ev, mins = [], []
    for ds, cfg in eval_configs().items():
        c = Cache(cfg)
        pr = pd.read_parquet(pred_path("bdd" if ds == "BDD100K" else "dawn"))
        if not (np.array_equal(pr.image_id.to_numpy(), c.pred.image_id.to_numpy())
                and np.array_equal(pr.score.to_numpy(), c.pred.score.to_numpy())):
            raise ValueError(f"{ds}: chances are not aligned with the cached predictions")
        p = pr.p.to_numpy()
        ev.append(evaluate(ds, cfg, p))
        mins.append(min_alpha(ds, cfg, p))
        print(f"{ds}: evaluated, {time.perf_counter() - t0:.0f} s", flush=True)
    ev = pd.concat(ev, ignore_index=True)
    summary, ratios = summarize_eval(ev), paired_ratios(ev)
    sanity(summary)
    crit, mina = criteria(summary, ratios), pd.concat(mins, ignore_index=True)
    ff = "%.6g"
    ev.to_csv(OUT / "crc_lambda_eval.csv", index=False, float_format=ff)
    summary.to_csv(OUT / "crc_lambda_summary.csv", index=False, float_format=ff)
    ratios.to_csv(OUT / "crc_lambda_ratio.csv", index=False, float_format=ff)
    mina.to_csv(OUT / "crc_lambda_min_alpha.csv", index=False, float_format=ff)
    crit.to_csv(OUT / "crc_lambda_criteria.csv", index=False)
    figure(summary, ratios)
    md = report(summary, ratios, crit, mina)
    (OUT / "crc_lambda.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"done, {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
