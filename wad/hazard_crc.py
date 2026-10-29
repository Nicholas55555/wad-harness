"""Phase 6a: re-run the phase 4 oracle-bin procedure with bins predicted from the hazard
scalar h (wad.hazard) instead of the weather labels.

Per split: cut h into quartiles on the calibration half, assign test images by the same cut
points, fit one CRC threshold per predicted bin on the calibration images in it, evaluate on
the test half. Because h is a function of the image alone, each predicted bin is a fixed
region of image space and the CRC argument applies within it whatever the weather label is;
the cut points, though, are estimated from the calibration half (see NOTES.md).

Run: python -m wad.hazard_crc [--smoke]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from wad.config import Cfg, bins, cache_path, load_config, results_path, tau_grid, write_json
from wad.crc import fit_threshold, make_splits
from wad.curves import ALL, Cache, summarize


def predicted_bins(h: np.ndarray, cal: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    cuts = np.quantile(h[cal], np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(h, cuts), cuts  # 0 = least degraded ... n_bins-1 = most degraded


def run(cfg: Cfg, h_path=None, categorical: str | None = None, merge: dict | None = None) -> dict:
    """h_path: a parquet with image_id, weather, h (default: version A, cache/hazard.parquet).
    categorical: bin by this column instead of quartiles of h (e.g. the classifier's argmax class);
    merge maps category values onto bin names. Categorical bins are a fixed function of the image,
    so no cut point is estimated from the calibration half."""
    c = Cache(cfg)
    hz = pd.read_parquet(h_path or cache_path(cfg, "hazard.parquet")).set_index("image_id").loc[c.image_ids]
    h = hz.h.to_numpy()
    taus = tau_grid(cfg)
    floor = cfg.detector.conf_floor
    loss = c.loss_grid(taus, ALL)
    if categorical:
        cat = hz[categorical].map(lambda v: (merge or {}).get(v, v)).to_numpy()
        names = sorted(set(cat))
        fixed = np.array([names.index(v) for v in cat])
    else:
        fixed = None
        names = [f"h{k + 1}" for k in range(cfg.hazard.n_bins)]
    nb = len(names)
    splits = make_splits(cfg, c.images)
    fits, evals, agree, cont = [], [], [], []
    for seed, sp in splits.groupby("seed"):
        cal = (sp.set_index("image_id").loc[c.image_ids].half == "cal").to_numpy()
        test = ~cal
        pb, cuts = (fixed, np.array([])) if fixed is not None else predicted_bins(h, cal, nb)
        agree.append(dict(seed=seed, ari_all=adjusted_rand_score(c.weather, pb),
                          ari_test=adjusted_rand_score(c.weather[test], pb[test]), cuts=cuts.tolist()))
        ct = pd.crosstab(pd.Series(c.weather, name="weather"), pd.Series(np.array(names)[pb], name="pred_bin"))
        cont.append(ct.stack().rename("n").reset_index().assign(seed=seed))
        for alpha in cfg.alphas:
            pol = {}
            t = np.full(c.n_img, np.nan)
            for k, name in enumerate(names):
                f = fit_threshold(loss[cal & (pb == k)], taus, alpha, floor)
                fits.append(dict(seed=seed, alpha=alpha, policy="predicted_bin", bin=name, **f))
                t[pb == k] = f["tau_hat"]
            pol["predicted_bin"] = t
            t = np.full(c.n_img, np.nan)
            for b in bins(cfg):
                f = fit_threshold(loss[cal & (c.weather == b)], taus, alpha, floor)
                t[c.weather == b] = f["tau_hat"]
            pol["oracle_bin"] = t
            for p, ti in pol.items():
                counts = c.at_thresholds(ti, ALL)
                groups = [("pred", k, pb == k) for k in range(nb)] + \
                         [("weather", b, c.weather == b) for b in bins(cfg)] + [("all", ALL, np.ones(c.n_img, bool))]
                for kind, key, mask in groups:
                    rows = np.nonzero(test & mask)[0]
                    s = summarize(counts, rows, per_class=False)
                    evals.append(dict(seed=seed, alpha=alpha, policy=p, grouping=kind,
                                      bin=names[key] if kind == "pred" else key, image_loss=s["image_loss"],
                                      miss_rate=s["miss_rate"], fppi=s["fppi"], n_test=s["n_images"]))
    fits, evals, agree = pd.DataFrame(fits), pd.DataFrame(evals), pd.DataFrame(agree)
    e = evals.assign(exceeds=evals.image_loss > evals.alpha)
    summary = e.groupby(["alpha", "policy", "grouping", "bin"], sort=False).agg(
        n_seeds=("seed", "nunique"), image_loss_mean=("image_loss", "mean"), image_loss_std=("image_loss", "std"),
        fppi_mean=("fppi", "mean"), fppi_std=("fppi", "std"), exceed_frac=("exceeds", "mean"),
        n_test_mean=("n_test", "mean")).reset_index()
    fs = fits.groupby(["alpha", "bin"]).agg(tau_hat_mean=("tau_hat", "mean"), tau_hat_std=("tau_hat", "std"),
                                            infeasible_frac=("infeasible", "mean"),
                                            n_cal_mean=("n_cal", "mean")).reset_index()
    summary = summary.merge(fs.assign(policy="predicted_bin", grouping="pred"), how="left",
                            on=["alpha", "policy", "grouping", "bin"])
    # FPPI gap to the oracle-bin policy over all test images, per split
    allrows = evals[evals.grouping == "all"].pivot_table(index=["seed", "alpha"], columns="policy", values="fppi")
    gap = (allrows["predicted_bin"] - allrows["oracle_bin"]).rename("fppi_gap").reset_index()
    contingency = pd.concat(cont).groupby(["weather", "pred_bin"]).n.mean().unstack(fill_value=0)
    return dict(summary=summary, fits=fits, evals=evals, agreement=agree, gap=gap, contingency=contingency, h=hz)


def fig4(cfg: Cfg, out: dict):
    import matplotlib.pyplot as plt

    from wad.figures import BIN_LABEL, CRITICAL, INK, INK2, MUTED, SERIES, SURFACE, save, style

    style()
    hz = out["h"]
    s = out["summary"]
    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1])
    ax0 = fig.add_subplot(gs[:, 0])
    data = [hz[hz.weather == b].h.to_numpy() for b in bins(cfg)]
    parts = ax0.violinplot(data, showextrema=False, widths=0.75)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(SERIES[i])
        pc.set_alpha(0.25)
        pc.set_edgecolor(SERIES[i])
    for i, d in enumerate(data):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax0.plot([i + 1, i + 1], [q1, q3], color=SERIES[i], lw=3, solid_capstyle="round")
        ax0.plot(i + 1, med, marker="o", ms=6, mfc=SURFACE, mec=SERIES[i], mew=1.6)
    ax0.set_xticks(range(1, len(data) + 1))
    ax0.set_xticklabels([f"{BIN_LABEL[b]}\nn = {len(d)}" for b, d in zip(bins(cfg), data)])
    ax0.tick_params(axis="x", length=0)
    ax0.grid(False, axis="x")
    ari = out["agreement"].ari_all
    ax0.set_ylabel("Hazard scalar h (higher = lower contrast, fewer edges)")
    ax0.set_title(f"h by weather label (ARI with quartile bins {ari.mean():.2f} ± {ari.std():.2f})")
    names = [f"h{k + 1}" for k in range(cfg.hazard.n_bins)]
    for r, alpha in enumerate(cfg.alphas):
        for col, (metric, lab) in enumerate([("image_loss", "Test image-level miss fraction"),
                                              ("fppi", "Test FPPI (log scale)")]):
            ax = fig.add_subplot(gs[r, col + 1])
            ticks, ticklabels = [], []
            for k, (p, grouping, order, labels, mk, x0) in enumerate([
                    ("predicted_bin", "pred", names, names, "o", 0.0),
                    ("oracle_bin", "weather", bins(cfg), [BIN_LABEL[b] for b in bins(cfg)], "s", 4.6)]):
                d = s[(s.alpha == alpha) & (s.policy == p) & (s.grouping == grouping)].set_index("bin").reindex(order)
                x = x0 + np.arange(len(order))
                ax.errorbar(x, d[f"{metric}_mean"], yerr=d[f"{metric}_std"], fmt="none", ecolor=SERIES[k], elinewidth=1.2)
                ax.plot(x, d[f"{metric}_mean"], ls="none", marker=mk, ms=6.5, mfc=SERIES[k], mec=SURFACE, mew=1.2,
                        label=("Predicted bins h1–h4 (CRC per hazard quartile)" if k == 0
                               else "Oracle bins (CRC per weather label)"))
                ticks += list(x)
                ticklabels += labels
            ax.axvline(3.8, color="#c3c2b7", lw=0.8)
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels, fontsize=7.5)
            ax.tick_params(axis="x", length=0)
            ax.grid(False, axis="x")
            if metric == "image_loss":
                ax.axhline(alpha, color=CRITICAL, lw=1.0)
                ax.set_ylim(0, 0.4)
            else:
                ax.set_yscale("log")
                ax.set_ylim(1, 150)
            ax.set_title(f"α = {alpha}: {'miss fraction' if metric == 'image_loss' else 'false-positive cost'}",
                         fontsize=9)
            if r == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=7, bbox_to_anchor=(0, 1.0))
    fig.suptitle("Hazard-predicted bins (quartiles of h, h1 lowest to h4 highest; h is mostly a sand axis) vs weather-label bins, held-out test halves, "
                 f"mean ± 1 sd over {int(s.n_seeds.max())} splits", x=0.02, ha="left", fontsize=10, color=INK,
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, cfg, "fig4_hazard")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke)
    out = run(cfg)
    ff = "%.6g"
    out["summary"].to_csv(results_path(cfg, "hazard_summary.csv"), index=False, float_format=ff)
    out["fits"].to_csv(results_path(cfg, "hazard_fits.csv"), index=False, float_format=ff)
    out["evals"].to_csv(results_path(cfg, "hazard_eval.csv"), index=False, float_format=ff)
    out["agreement"].to_csv(results_path(cfg, "hazard_agreement.csv"), index=False, float_format=ff)
    out["gap"].to_csv(results_path(cfg, "hazard_fppi_gap.csv"), index=False, float_format=ff)
    out["contingency"].to_csv(results_path(cfg, "hazard_contingency.csv"), float_format="%.1f")
    write_json(results_path(cfg, "hazard_headline.json"), dict(
        ari_mean=float(out["agreement"].ari_all.mean()), ari_std=float(out["agreement"].ari_all.std()),
        fppi_gap=out["gap"].groupby("alpha").fppi_gap.agg(["mean", "std"]).to_dict(orient="index")))
    if not cfg.smoke_mode:
        fig4(cfg, out)
    s = out["summary"]
    pd.set_option("display.width", 250)
    print(s[s.grouping != "weather"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(out["contingency"].round(1).to_string())
    print("ARI", out["agreement"].ari_all.mean(), "FPPI gap", out["gap"].groupby("alpha").fppi_gap.mean().to_dict())


if __name__ == "__main__":
    main()
