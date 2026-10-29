"""Evaluation of the trained visibility estimator (PREREG_visibility.md + amendments), on real images it never saw.

S1  Spearman(h_v2, per-image miss fraction at 0.25) >= v1's, on BDD100K and on DAWN
S2  per-visibility v2 thresholds hold alpha = 0.2 on average in BDD100K rain-night and
    clear-night and in DAWN fog
S3  overall FPPI of the v2 policy <= v1's, per dataset
S4  synthetic validation MAE of log10 V_eff <= 0.15
Sanity: predicted light separates BDD100K day from night (ROC AUC); predicted MOR orders DAWN fog
below DAWN rain/snow below BDD100K clear day.

Run: python -m wad.vis_eval [--version 3]   -> results/vis{2,3}/{policies,correlations,criteria,bins,predictions}.csv, vis{2,3}.md,
     figures/fig10_visibility_estimator.{png,pdf}, figures/fig11_visibility_examples.jpg
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from wad.config import REPO, tau_grid
from wad.crc import fit_threshold, make_splits
from wad.curves import ALL, Cache

OUT = REPO / "results" / "vis2"
CACHE = REPO / "cache" / "vis2"
VERSION, V = 2, "v2"  # set by configure(); v3 = night relighting (PREREG_visibility.md, version 3)


def configure(version: int) -> None:
    global OUT, CACHE, VERSION, V
    from wad.vis_estimator import paths

    pth = paths(version)
    OUT, CACHE, VERSION, V = pth["results"], pth["preds"], version, f"v{version}"
ALPHA, N_SPLITS = 0.2, 200
BANDS = [100.0, 300.0, 1000.0]  # fixed physical bands of predicted V_eff (m): <100, [100,300), [300,1000), >=1000
BAND_NAMES = ["< 100 m", "100-300 m", "300-1000 m", ">= 1000 m"]


def prf(tp, fp, n_gt):
    r = tp / n_gt if n_gt else np.nan
    p = tp / (tp + fp) if tp + fp else np.nan
    return r, p, (2 * p * r / (p + r) if (r == r and p == p and p + r) else np.nan)


def configs():
    from wad.run2 import bdd_cfg, dawn_cfg, load_run2

    r2 = load_run2()
    return [("BDD100K", "bdd", bdd_cfg(r2, "v8x_1280", "group")), ("DAWN", "dawn", dawn_cfg(r2, "v8x_1280", "group", "clean"))]


def groups(ds: str, c: Cache, h1: np.ndarray, h2: np.ndarray, extra: dict | None = None) -> dict:
    w = c.weather
    g = {b: w == b for b in pd.unique(w)}
    if ds == "DAWN":
        sub = pd.Series(c.image_ids).str.split("/").str[1].str.extract(r"^([a-z]+)")[0].to_numpy()
        for s in ("foggy", "haze", "mist"):
            g[f"fog: {s}"] = (w == "fog") & (sub == s)
    g["lowest-visibility quarter (v1)"] = h1 >= np.quantile(h1, 0.75)
    for name, h in (extra or {}).items():
        g[f"lowest-visibility quarter ({name})"] = h >= np.quantile(h, 0.75)
    g[f"lowest-visibility quarter ({V})"] = h2 >= np.quantile(h2, 0.75)
    g["all"] = np.ones(c.n_img, bool)
    # EXPLORATORY (not pre-registered): each condition's lowest-visibility third by this version,
    # e.g. the heaviest-looking rain within DAWN rain
    for b in pd.unique(w):
        m = w == b
        g[f"{b} | lowest-visibility third by {V} (exploratory)"] = m & (h2 >= np.quantile(h2[m], 2 / 3))
    return g


def policies_table(ds, v, c, h1, h2, extra: dict | None = None) -> pd.DataFrame:
    taus = tau_grid(v)
    floor = v.detector.conf_floor
    loss = c.loss_grid(taus, ALL)
    v.splits["n_repeats"] = N_SPLITS
    splits = make_splits(v, c.images)
    w = c.weather
    gs = groups(ds, c, h1, h2, extra)
    veff = 10 ** (-h2)
    band = np.digitize(veff, BANDS)  # 0: < 100 m ... 3: >= 1000 m (a value on an edge goes to the higher band)
    acc, cal_info = [], []
    for seed, sp in splits.groupby("seed"):
        cal = (sp.set_index("image_id").loc[c.image_ids].half == "cal").to_numpy()
        test = ~cal

        def per_bin(lab, record=None):
            t = np.full(c.n_img, np.nan)
            for b in np.unique(lab):
                fit = fit_threshold(loss[cal & (lab == b)], taus, ALPHA, floor)
                t[lab == b] = fit["tau_hat"]
                if record:
                    cal_info.append(dict(dataset=ds, policy=record, bin=int(b), seed=seed, n_cal=fit["n_cal"],
                                         infeasible=fit["infeasible"], tau=fit["tau_hat"]))
            return t

        quart = lambda h: np.digitize(h, np.quantile(h[cal], [0.25, 0.5, 0.75]))  # noqa: E731
        pol = {"global": np.full(c.n_img, fit_threshold(loss[cal], taus, ALPHA, floor)["tau_hat"]),
               "per-weather": per_bin(w),
               "visibility v1 (quartiles)": per_bin(quart(h1))}
        for name, h in (extra or {}).items():
            pol[f"visibility {name} (quartiles)"] = per_bin(quart(h))
        pol[f"visibility {V} (quartiles)"] = per_bin(quart(h2), f"{V} quartiles")
        pol[f"visibility {V} (fixed bands)"] = per_bin(band, f"{V} fixed bands")
        for p, ti in pol.items():
            x = c.at_thresholds(ti, ALL)
            li = np.where(x["n_gt"] > 0, x["n_missed"] / np.maximum(x["n_gt"], 1), 0.0)
            for name, m in gs.items():
                r_ = test & m
                n_gt = float(x["n_gt"][r_].sum())
                tp, fp = float((x["n_gt"] - x["n_missed"])[r_].sum()), float(x["n_fp"][r_].sum())
                r, pr, f = prf(tp, fp, n_gt)
                acc.append(dict(dataset=ds, group=name, policy=p, seed=seed, recall=r, precision=pr, f1=f,
                                fppi=fp / max(r_.sum(), 1), image_miss=li[r_].mean() if r_.any() else np.nan,
                                n_test=int(r_.sum())))
    a = pd.DataFrame(acc)
    rows = []
    for (g, p), d in a.groupby(["group", "policy"], sort=False):
        row = dict(dataset=ds, group=g, policy=p, splits=d.seed.nunique(), n_test=d.n_test.mean(),
                   above_alpha=float((d.image_miss > ALPHA).mean()))
        for k in ("recall", "precision", "f1", "fppi", "image_miss"):
            row.update({k: d[k].mean(), f"{k}_lo": d[k].quantile(0.025), f"{k}_hi": d[k].quantile(0.975)})
        rows.append(row)
    ci = pd.DataFrame(cal_info)
    bins = ci.groupby(["dataset", "policy", "bin"]).agg(splits=("seed", "nunique"), n_cal_mean=("n_cal", "mean"),
                                                         n_cal_min=("n_cal", "min"), infeasible_share=("infeasible", "mean"),
                                                         tau_median=("tau", "median")).reset_index()
    bins["band"] = np.where(bins.policy == f"{V} fixed bands", bins.bin.map(dict(enumerate(BAND_NAMES))),
                            "quartile " + (bins.bin + 1).astype(str) + " (4 = least visible)")
    # bins seen in only some splits (e.g. a band empty in some calibration halves) show splits < N_SPLITS
    return pd.DataFrame(rows), bins


def evaluate():
    from wad.visibility import load_h

    OUT.mkdir(parents=True, exist_ok=True)
    h1_all = load_h()
    pol, corr, preds, bins = [], [], {}, []
    for ds, key, v in configs():
        c = Cache(v)
        p = pd.read_parquet(CACHE / f"pred_{key}.parquet").set_index("image_id").loc[c.image_ids]
        preds[ds] = p.assign(dataset=ds)
        h1 = h1_all.loc[c.image_ids].to_numpy()
        h2 = -p.log_veff.to_numpy()
        extra = {}
        if VERSION > 2:  # v2 alongside, as a comparator (not a criterion)
            prev = REPO / "cache" / "vis2" / f"pred_{key}.parquet"
            extra["v2"] = -pd.read_parquet(prev).set_index("image_id").loc[c.image_ids].log_veff.to_numpy()
        n = c.n_gt(ALL)
        found = np.bincount(c.gt_img, weights=c.gt_score >= 0.25, minlength=c.n_img)
        ever = np.bincount(c.gt_img, weights=c.gt_score >= 0.001, minlength=c.n_img)
        has = n > 0
        miss = 1 - found[has] / n[has]
        never = 1 - ever[has] / n[has]
        scores = [("v1 (contrast + detail)", h1)] + [(f"{n} (trained estimator)", h) for n, h in extra.items()] + \
            [(f"{V} (trained estimator)", h2), (f"{V} light only (-log k)", -p.log_k.to_numpy()),
             (f"{V} scattering only (-log MOR)", -p.log_mor.to_numpy())]
        for name, h in scores:
            corr.append(dict(dataset=ds, score=name, spearman_miss025=spearmanr(h[has], miss)[0],
                             spearman_never=spearmanr(h[has], never)[0], n_images=int(has.sum())))
        pt, bt = policies_table(ds, v, c, h1, h2, extra)
        pol.append(pt)
        bins.append(bt.assign(n_images_in_band=bt.apply(
            lambda r: int((np.digitize(10 ** (-h2), BANDS) == r.bin).sum()) if r.policy == f"{V} fixed bands" else np.nan, axis=1)))
        if ds == "DAWN":  # sensitivity (amendment 3): stretched instead of centre-cropped DAWN images
            st = pd.read_parquet(CACHE / "pred_dawn_stretch.parquet").set_index("image_id").loc[c.image_ids]
            corr.append(dict(dataset=ds, score=f"{V}, stretched images (sensitivity)",
                             spearman_miss025=spearmanr(-st.log_veff.to_numpy()[has], miss)[0],
                             spearman_never=spearmanr(-st.log_veff.to_numpy()[has], never)[0], n_images=int(has.sum()),
                             note=f"Spearman with the cropped score: {spearmanr(-st.log_veff, h2)[0]:.3f}"))
    pol = pd.concat(pol)
    bins = pd.concat(bins)
    bins.to_csv(OUT / "bins.csv", index=False, float_format="%.6g")
    corr = pd.DataFrame(corr)
    pol.to_csv(OUT / "policies.csv", index=False, float_format="%.6g")
    corr.to_csv(OUT / "correlations.csv", index=False, float_format="%.6g")
    allp = pd.concat(preds.values()).reset_index()
    allp.to_csv(OUT / "predictions.csv", index=False, float_format="%.6g")
    # criteria
    tl = json.loads((OUT / "train_log.json").read_text())
    best = min(tl["log"], key=lambda r: r["val_loss"])
    crit = []
    for ds in ("BDD100K", "DAWN"):
        cc = corr[corr.dataset == ds].set_index("score")
        me, v1 = cc.loc[f"{V} (trained estimator)", "spearman_miss025"], cc.loc["v1 (contrast + detail)", "spearman_miss025"]
        crit.append(dict(criterion=f"S1 {ds}", value=f"{V} {me:+.3f} vs v1 {v1:+.3f}", met=bool(me >= v1)))
    q = pol[pol.policy == f"visibility {V} (quartiles)"].set_index(["dataset", "group"])
    for ds, g in [("BDD100K", "rain_night"), ("BDD100K", "clear_night"), ("DAWN", "fog")]:
        m = q.loc[(ds, g), "image_miss"]
        crit.append(dict(criterion=f"S2 {ds} {g}", value=f"{m:.3f} (above alpha in {q.loc[(ds, g), 'above_alpha']:.0%})", met=bool(m <= ALPHA)))
    for ds in ("BDD100K", "DAWN"):
        a = pol[(pol.dataset == ds) & (pol.group == "all")].set_index("policy")
        f2, f1 = a.loc[f"visibility {V} (quartiles)", "fppi"], a.loc["visibility v1 (quartiles)", "fppi"]
        crit.append(dict(criterion=f"S3 {ds}", value=f"{V} {f2:.2f} vs v1 {f1:.2f} FPPI", met=bool(f2 <= f1)))
    crit.append(dict(criterion="S4", value=f"MAE log10 V_eff {best['val_mae']['log_veff']:.3f} (epoch {best['epoch']})",
                     met=bool(best["val_mae"]["log_veff"] <= 0.15)))
    b = preds["BDD100K"]
    night = b.weather.str.endswith("night").to_numpy()
    auc = roc_auc_score(night, -b.log_k.to_numpy())
    med = {f"{ds} {w}": float(10 ** preds[ds][preds[ds].weather == w].log_mor.median())
           for ds, w in [("DAWN", "fog"), ("DAWN", "rain"), ("DAWN", "snow"), ("DAWN", "sand"), ("BDD100K", "clear")]}
    crit.append(dict(criterion="sanity: light separates BDD100K night from day (ROC AUC)", value=f"{auc:.3f}", met=None))
    order = med["DAWN fog"] < min(med["DAWN rain"], med["DAWN snow"]) and max(med["DAWN rain"], med["DAWN snow"]) < med["BDD100K clear"]
    crit.append(dict(criterion="sanity: median predicted MOR (m); fog < rain, snow < BDD100K clear day?",
                     value=", ".join(f"{k} {v:.0f}" for k, v in med.items()) + f" (ordering {'holds' if order else 'does not hold'})",
                     met=None))
    crit = pd.DataFrame(crit)
    crit.to_csv(OUT / "criteria.csv", index=False)
    return pol, corr, crit, allp, bins


def figure(allp: pd.DataFrame):
    import matplotlib.pyplot as plt

    from wad.config import Cfg
    from wad.figures import INK, INK2, SERIES, SURFACE, save, style

    style()
    order = [("BDD100K", b) for b in ["clear", "cloudy", "rain", "snow", "clear_night", "rain_night", "snow_night"]] + \
            [("DAWN", b) for b in ["rain", "snow", "sand", "fog"]]
    lab = {"clear": "Clear · day", "cloudy": "Cloudy · day", "rain": "Rain · day", "snow": "Snow · day",
           "clear_night": "Clear · night", "rain_night": "Rain · night", "snow_night": "Snow · night"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (col, title) in zip(axes, [("log_veff", "Effective visibility V_eff (m)"), ("log_mor", "Scattering visibility MOR (m)"),
                                       ("log_k", "Light level k (1 = daylight)")]):
        data, names = [], []
        for ds, b in order:
            x = allp[(allp.dataset == ds) & (allp.weather == b)][col].to_numpy()
            data.append(x)
            names.append(lab.get(b, b) if ds == "BDD100K" else f"{b.capitalize()} (DAWN)")
        y = np.arange(len(data))[::-1]
        for yi, x, (ds, _) in zip(y, data, order):
            q1, m, q3 = np.percentile(x, [25, 50, 75])
            lo, hi = np.percentile(x, [5, 95])
            c = SERIES[0] if ds == "BDD100K" else SERIES[1]
            ax.plot([lo, hi], [yi, yi], color=c, lw=1, alpha=0.6)
            ax.plot([q1, q3], [yi, yi], color=c, lw=5, solid_capstyle="butt")
            ax.plot(m, yi, marker="o", ms=6, mfc=SURFACE, mec=c, mew=1.6, ls="none")
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        if col == "log_k":
            ticks, labels, lim = [-2.5, -2, -1, 0], ["0.003", "0.01", "0.1", "1"], (-2.7, 0.25)
        else:
            vals = [30, 100, 300, 1000, 3000, 10000]
            ticks, labels, lim = np.log10(vals), ["30", "100", "300", "1k", "3k", "10k"], (1.35, 4.15)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlim(*lim)
        ax.set_title(title, fontsize=9.5)
        ax.grid(False, axis="y")
    axes[0].text(0.01, -0.12, "median (dot), interquartile range (bar), 5–95% (line); blue BDD100K, orange DAWN",
                 transform=axes[0].transAxes, fontsize=7.5, color=INK2)
    fig.suptitle(f"Trained visibility estimator {V} on real images it never saw (BDD100K val, DAWN)", x=0.02, ha="left",
                 fontsize=10.5, color=INK, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    return save(fig, Cfg(dict(paths=Cfg(dict(results=OUT)))), "fig10_visibility_estimator")


def examples(allp: pd.DataFrame, per_side=3, tw=320, th=180):
    """Contact sheet: per condition, the lowest and highest predicted V_eff images (visual check)."""
    from PIL import Image, ImageDraw

    from wad.imageio import load_rgb
    from wad.vis_estimator import to_working

    paths = pd.concat([pd.read_parquet(REPO / "cache" / "images.parquet"), pd.read_parquet(REPO / "cache" / "bdd" / "images.parquet")])
    paths = paths.drop_duplicates("image_id").set_index("image_id").path
    rows = [("DAWN", "fog"), ("DAWN", "rain"), ("DAWN", "snow"), ("BDD100K", "rain_night"), ("BDD100K", "rain"), ("BDD100K", "clear")]
    lab_h = 18
    sheet = Image.new("RGB", (2 * per_side * tw + 12, len(rows) * (th + lab_h) + 20), "white")
    dr = ImageDraw.Draw(sheet)
    dr.text((4, 3), "per condition: 3 lowest predicted V_eff (left) | 3 highest (right); frames as the model sees them "
                    "(centre-cropped to 16:9)", fill="black")
    for r, (ds, w) in enumerate(rows):
        d = allp[(allp.dataset == ds) & (allp.weather == w)].sort_values("log_veff")
        pick = pd.concat([d.head(per_side), d.tail(per_side)])
        for c, row in enumerate(pick.itertuples()):
            im = Image.fromarray(to_working(load_rgb(paths.loc[row.image_id], "apply"))).resize((tw, th), Image.BILINEAR)
            x = c * tw + (12 if c >= per_side else 0)
            y = 20 + r * (th + lab_h)
            sheet.paste(im, (x, y + lab_h))
            dr.text((x + 2, y + 3), f"{ds} {w}  Veff {10 ** row.log_veff:.0f}m  MOR {10 ** row.log_mor:.0f}m  k {10 ** row.log_k:.2g}",
                    fill="black")
    p = OUT / "figures" / "fig11_visibility_examples.jpg"
    p.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(p, quality=88)
    return p


def md(df):
    cols = list(df.columns)
    cell = lambda v: str(v).replace("|", "\\|")  # noqa: E731  (group names contain '|')
    return "\n".join(["| " + " | ".join(map(cell, cols)) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
                     + ["| " + " | ".join(map(cell, r)) + " |" for r in df.itertuples(index=False)])


def dev_lines() -> list:
    """The development check on BDD100K *train* frames (wad.vis_estimator dev_check), if it was run."""
    f = OUT / "dev_check.json"
    if not f.exists():
        return []
    d = json.loads(f.read_text())
    note = ("Post hoc: run after the v2 evaluation, as the baseline for version 3." if VERSION == 2 else
            "The same night frames were used to tune the night rendering, so this is not an independent check.")
    return [f"Development check (BDD100K train frames, not evaluation data): {d['n_night']} night vs {d['n_day']} day "
            f"frames, ROC AUC of the light level {d['auc_night_vs_day']:.3f}, median k night {d['median_k_night']:.2f} vs "
            f"day {d['median_k_day']:.2f}. {note}", ""]


def report(pol, corr, crit, bins):
    tl = json.loads((OUT / "train_log.json").read_text())
    best = min(tl["log"], key=lambda r: r["val_loss"])
    met = crit.met.map(lambda m: "—" if m is None or m != m else ("met" if m else "not met"))
    head = ("Pre-registration: `PREREG_visibility.md` (three dated amendments, all made before any real-image prediction)."
            if VERSION == 2 else
            f"Pre-registration: `PREREG_visibility.md`, section 'Version {VERSION}'. Not blind: designed after v2's results "
            "(v2 read real night as daylight); development used only BDD100K train night frames. v2 is shown alongside.")
    lines = [f"# O3 version C: trained visibility estimator ({V})", "",
             f"Generated by `python -m wad.vis_eval --version {VERSION}`. {head} Training: synthetic degradations of "
             "clear-day BDD100K train frames; "
             "evaluation: BDD100K val and DAWN, never seen in training or model selection. Detector for the threshold "
             f"policies: YOLOv8x @1280, grouped vehicles (DAWN clean labels); α = {ALPHA}, {N_SPLITS} calibration/test splits.", "",
             f"Training: {tl['n_train']} train / {tl['n_val']} validation base frames ({tl.get('n_dark_excluded', 0)} dark, "
             f"mis-tagged frames excluded); best epoch {best['epoch']}; synthetic validation MAE (log10): "
             + ", ".join(f"{k} {v:.3f}" for k, v in best["val_mae"].items()) + ".", "",
             *dev_lines(),
             "By construction V_eff ≤ 2000 m (V_light = 2000 m at k = 1). The top fixed band (≥ 1000 m) holds daylight "
             "without a significant veil or spray, and any image read as no darker than k = 0.25 (dusk level); it is not "
             "'no darkness'.", "",
             "## Pre-registered criteria" + ("" if VERSION == 2 else " (pre-registered for v2, applied unchanged to v3)"), "",
             md(crit.assign(met=met)), "",
             "## Does the score track detection difficulty? (Spearman with the per-image miss fraction at τ = 0.25)", "",
             "'never' = Spearman with the share of objects never detected at any score. The light-only and scattering-only "
             "rows are post hoc breakdowns (exploratory); the stretched-image row is the sensitivity check of amendment 3. "
             "None of them is a criterion.", "",
             md(corr.round(3).fillna("")), "",
             f"## Calibration bins of the {V} policies", "",
             "Per bin over the splits: calibration images (mean, min), share of splits where no grid threshold met α "
             "(the threshold then falls to the confidence floor), median threshold.", "",
             md(bins.round(3).fillna("")), "",
             "## Threshold policies on held-out images", "",
             "Cells: recall / precision / F1, FPPI, mean held-out image miss fraction (share of splits above α). Rows marked "
             "exploratory were added after the pre-registration.", ""]
    for ds in ("BDD100K", "DAWN"):
        d = pol[pol.dataset == ds]
        rows = []
        for g in pd.unique(d.group):
            row = {"group": g}
            for p in pd.unique(d.policy):
                r = d[(d.group == g) & (d.policy == p)].iloc[0]
                row[p] = f"{100 * r.recall:.0f}/{100 * r.precision:.0f}/{r.f1:.2f}, {r.fppi:.1f}, {r.image_miss:.3f} ({r.above_alpha:.2f})"
            rows.append(row)
        lines += [f"### {ds}", "", md(pd.DataFrame(rows)), ""]
    (OUT / f"vis{VERSION}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=2, choices=[2, 3])
    configure(ap.parse_args().version)
    pol, corr, crit, allp, bins = evaluate()
    figure(allp)
    examples(allp)
    report(pol, corr, crit, bins)
    print(crit.to_string(index=False))
    print(corr.round(3).to_string(index=False))
