"""EXPLORATORY: a graded response from detector confidence (user's idea, 2026-10-12).
STATUS: stop / slow tiers on raw confidence; complements the dial (wad/dial.py).

Instead of one threshold, two tiers from the detector's confidence:
  stop   score >= tau_stop: brake / stop. tau_stop is the smallest grid threshold whose
         calibration false positives per image are <= cap (cap 0.1 or 1: false stops are costly)
  slow   tau_slow <= score < tau_stop: something is possibly there; slow down. tau_slow is the CRC
         threshold that keeps the mean image miss <= alpha (0.1 or 0.2): a false slow-down is cheap
  miss   score < tau_slow
Both thresholds global (one pair for every image) or per condition (weather x time-of-day labels
known). Run 2 primary detector (YOLOv8x @1280, grouped matching; DAWN clean labels), the project's
200 calibration/test halves; everything reported on test halves.

Per condition: share of objects in each tier, mean image miss at tau_slow (share of splits above
alpha), false stops and false slow-downs per image, precision of the slow tier, and the share of
the condition's objects that only the slow tier catches (how much more the car slows there).
Detections are not filtered to the driving path, so per-image counts overstate what a planner
would react to; the comparison across conditions is the point.

Run: python -m wad.graded_response   -> results/threshold/graded_response.csv, graded_response.md, fig15
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wad.config import tau_grid
from wad.crc import fit_threshold, make_splits
from wad.curves import ALL, Cache
from wad.config import THRESH_RESULTS as RESULTS
from wad.threshold_budget import conditions, configs

CAPS = (0.1, 1.0)
ALPHAS = (0.1, 0.2)
N_SPLITS = 200


def tau_stop(fp_cal: np.ndarray, n: int, taus: np.ndarray, cap: float) -> float:
    if n == 0:
        return float(taus[-1])
    ok = np.nonzero(fp_cal.sum(0) / n <= cap)[0]
    return float(taus[ok.min()] if len(ok) else taus[-1])


def evaluate(ds, v) -> pd.DataFrame:
    c = Cache(v)
    taus = tau_grid(v)
    floor = v.detector.conf_floor
    loss, fp = c.loss_grid(taus, ALL), c.fp_grid(taus, ALL)
    w = c.weather
    conds = conditions(ds, c)
    conds["all"] = np.ones(c.n_img, bool)
    v.splits["n_repeats"] = N_SPLITS
    splits = make_splits(v, c.images)
    acc = []
    for seed, sp in splits.groupby("seed"):
        cal = (sp.set_index("image_id").loc[c.image_ids].half == "cal").to_numpy()
        test = ~cal
        for cap in CAPS:
            for a in ALPHAS:
                pols = {}
                ts, tl = tau_stop(fp[cal], int(cal.sum()), taus, cap), fit_threshold(loss[cal], taus, a, floor)["tau_hat"]
                pols["two tiers, global"] = (np.full(c.n_img, ts), np.full(c.n_img, tl))
                t_s, t_l = np.full(c.n_img, np.nan), np.full(c.n_img, np.nan)
                for b in pd.unique(w):
                    m = w == b
                    t_s[m] = tau_stop(fp[cal & m], int((cal & m).sum()), taus, cap)
                    t_l[m] = fit_threshold(loss[cal & m], taus, a, floor)["tau_hat"]
                pols["two tiers, per condition"] = (t_s, np.minimum(t_l, t_s))
                pols["one threshold (stop on everything), global"] = (np.full(c.n_img, tl), np.full(c.n_img, tl))
                for p, (s_hi, s_lo) in pols.items():
                    hi, lo = c.at_thresholds(s_hi, ALL), c.at_thresholds(s_lo, ALL)
                    li = np.where(lo["n_gt"] > 0, lo["n_missed"] / np.maximum(lo["n_gt"], 1), 0.0)
                    for name, m in conds.items():
                        r = test & m
                        n_gt = lo["n_gt"][r].sum()
                        tp_hi = (hi["n_gt"] - hi["n_missed"])[r].sum()
                        tp_lo = (lo["n_gt"] - lo["n_missed"])[r].sum()
                        fp_hi, fp_lo = hi["n_fp"][r].sum(), lo["n_fp"][r].sum()
                        n_img = max(r.sum(), 1)
                        band_tp, band_fp = tp_lo - tp_hi, fp_lo - fp_hi
                        acc.append(dict(dataset=ds, condition=name, policy=p, cap=cap, alpha=a, seed=seed,
                                        stop_share=tp_hi / n_gt, slow_only_share=band_tp / n_gt, miss_share=1 - tp_lo / n_gt,
                                        image_miss=li[r].mean(), false_stops_per_image=fp_hi / n_img,
                                        false_slowdowns_per_image=band_fp / n_img,
                                        slow_tier_precision=band_tp / (band_tp + band_fp) if band_tp + band_fp else np.nan,
                                        tau_stop=float(np.nanmedian(s_hi[r])), tau_slow=float(np.nanmedian(s_lo[r]))))
    a = pd.DataFrame(acc)
    a["above_alpha"] = (a.image_miss > a.alpha).astype(float)
    keys = ["dataset", "condition", "policy", "cap", "alpha"]
    g = a.groupby(keys, sort=False)
    out = g.mean(numeric_only=True).drop(columns="seed").reset_index()
    for k in ("image_miss", "false_stops_per_image", "slow_only_share"):
        out[f"{k}_lo"] = g[k].quantile(0.025).to_numpy()
        out[f"{k}_hi"] = g[k].quantile(0.975).to_numpy()
    return out


def md(df):
    cell = lambda v: str(v).replace("|", "\\|")  # noqa: E731
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(map(cell, cols)) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
                     + ["| " + " | ".join(map(cell, r)) + " |" for r in df.itertuples(index=False)])


def report(R: pd.DataFrame):
    lines = ["# Graded response from detector confidence (exploratory)", "",
             "Generated by `python -m wad.graded_response`. Stop tier: score ≥ τ_stop (calibration false positives per "
             "image ≤ cap). Slow tier: τ_slow ≤ score < τ_stop (τ_slow from CRC, mean image miss ≤ α). Held-out halves, "
             f"{N_SPLITS} splits. Shares are of ground-truth objects; false stops / slow-downs are false positives per image "
             "in each tier (not filtered to the driving path).", ""]
    cols = ["condition", "tau_stop", "tau_slow", "stop_share", "slow_only_share", "miss_share", "image_miss", "above_alpha",
            "false_stops_per_image", "false_slowdowns_per_image", "slow_tier_precision"]
    for (ds, cap, a), d in R.groupby(["dataset", "cap", "alpha"], sort=False):
        for p in pd.unique(d.policy):
            x = d[d.policy == p][cols].round(3)
            lines += [f"## {ds}, cap {cap:g} false stops per image, α = {a:g}: {p}", "", md(x), ""]
    (RESULTS / "graded_response.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def figure(R: pd.DataFrame):
    import matplotlib.pyplot as plt

    from wad.config import Cfg
    from wad.figures import INK2, SERIES, save, style

    style()
    show = [("BDD100K", c) for c in ["clear", "cloudy", "rain", "snow", "clear_night", "rain_night", "snow_night"]] + \
           [("DAWN", c) for c in ["rain (DAWN)", "snow (DAWN)", "sand", "fog"]]
    lab = {"clear": "Clear · day", "cloudy": "Cloudy · day", "rain": "Rain · day", "snow": "Snow · day",
           "clear_night": "Clear · night", "rain_night": "Rain · night", "snow_night": "Snow · night",
           "rain (DAWN)": "Rain (DAWN)", "snow (DAWN)": "Snow (DAWN)", "sand": "Sand (DAWN)", "fog": "Fog (DAWN)"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, p in zip(axes, ("two tiers, global", "two tiers, per condition")):
        d = R[(R.policy == p) & (R.cap == 1.0) & (R.alpha == 0.2)].set_index(["dataset", "condition"])
        y = np.arange(len(show))[::-1]
        stop = np.array([d.loc[k, "stop_share"] for k in show])
        slow = np.array([d.loc[k, "slow_only_share"] for k in show])
        miss = np.array([d.loc[k, "miss_share"] for k in show])
        ax.barh(y, stop, color=SERIES[0], label="stop (confident)")
        ax.barh(y, slow, left=stop, color=SERIES[3], label="slow down (possible)")
        ax.barh(y, miss, left=stop + slow, color="#d9d8d2", label="missed")
        for yi, k in zip(y, show):
            ax.text(1.01, yi, f"{d.loc[k, 'false_slowdowns_per_image']:.1f}", va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([lab[k[1]] for k in show])
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of objects")
        ax.set_title(f"{p} (cap 1 false stop/image, α = 0.2)", fontsize=9.5)
        ax.grid(False, axis="y")
    axes[1].text(1.01, len(show) - 0.2, "false slow-\ndowns/image", fontsize=7, color=INK2, va="bottom")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, -0.32), ncol=3)
    fig.tight_layout()
    return save(fig, Cfg(dict(paths=Cfg(dict(results=RESULTS)))), "fig15_graded_response")


def main():
    R = pd.concat([evaluate(ds, v) for ds, v in configs()])
    R.to_csv(RESULTS / "graded_response.csv", index=False, float_format="%.6g")
    report(R)
    figure(R)
    return R


if __name__ == "__main__":
    R = main()
    pd.set_option("display.width", 250)
    x = R[(R.cap == 1.0) & (R.alpha == 0.2)][["dataset", "condition", "policy", "tau_stop", "tau_slow", "stop_share", "slow_only_share",
                                              "miss_share", "image_miss", "false_stops_per_image", "false_slowdowns_per_image",
                                              "slow_tier_precision"]]
    print(x.round(3).to_string(index=False))
