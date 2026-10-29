"""O3 evaluation (PREREG_o3.md): predicted hazard bins vs oracle weather bins on DAWN.

Hazard sources: version A (pixel statistics, wad/hazard.py -> cache/hazard.parquet) and version B
(the BDD100K-trained ResNet-18, wad/hazard_cnn.py -> cache/o3/h_dawn.parquet). Detector / loss
settings: run 1 (YOLOv8s @640, class, all labels), the run 2 primary (YOLOv8x @1280, group, clean)
and the best run 2 variant (YOLOv8x @640, group, clean).

Success criteria (proposal slide 7):
  S1  in every weather bin where the global threshold exceeds alpha in at least half the splits,
      the predicted-bin policy's mean held-out miss fraction in that weather bin is <= alpha;
  S2  the predicted-bin policy costs at most 5 FPPI more than oracle bins over all test images.

  python -m wad.o3_eval      -> results/o3/o3_summary.csv, results/o3/o3.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from wad import hazard_crc
from wad.config import REPO, load_config
from wad.run2 import dawn_cfg, load_run2

OUT = REPO / "results" / "o3"
SOURCES = {"A": REPO / "cache" / "hazard.parquet", "B": REPO / "cache" / "o3" / "h_dawn.parquet"}


def settings():
    r2 = load_run2()
    run1 = load_config()
    return [("run1 v8s_640/class/all", run1, REPO / "results" / "crc_summary.csv"),
            ("run2 v8x_1280/group/clean", dawn_cfg(r2, "v8x_1280", "group", "clean"),
             REPO / "results" / "run2" / "dawn" / "v8x_1280__group__clean" / "crc_summary.csv"),
            ("run2 v8x_640/group/clean", dawn_cfg(r2, "v8x_640", "group", "clean"),
             REPO / "results" / "run2" / "dawn" / "v8x_640__group__clean" / "crc_summary.csv")]


def evaluate() -> pd.DataFrame:
    rows = []
    for name, cfg, crc_summary in settings():
        glob_ = pd.read_csv(crc_summary)
        glob_ = glob_[(glob_.policy == "global") & (glob_.cls == "all")]
        for src, path in SOURCES.items():
            if not path.exists():
                continue
            out = hazard_crc.run(cfg, h_path=path)
            s = out["summary"]
            tag = f"{name.replace(' ', '_').replace('/', '-')}__{src}"
            d = OUT / tag
            d.mkdir(parents=True, exist_ok=True)
            for k in ("summary", "evals", "agreement", "gap"):
                out[k].to_csv(d / f"{k}.csv", index=False, float_format="%.6g")
            out["contingency"].to_csv(d / "contingency.csv", float_format="%.1f")
            evaluate_one(rows, name, src, cfg, out, glob_)
    # EXPLORATORY (not pre-registered): bins = the classifier's argmax class (clear and cloudy
    # merged), a fixed function of the image, instead of quartiles of h_B
    for name, cfg, crc_summary in settings():
        glob_ = pd.read_csv(crc_summary)
        glob_ = glob_[(glob_.policy == "global") & (glob_.cls == "all")]
        if not SOURCES["B"].exists():
            continue
        out = hazard_crc.run(cfg, h_path=SOURCES["B"], categorical="pred", merge={"cloudy": "clear"})
        d = OUT / f"{name.replace(' ', '_').replace('/', '-')}__B_argmax_EXPLORATORY"
        d.mkdir(parents=True, exist_ok=True)
        for k in ("summary", "evals", "agreement", "gap"):
            out[k].to_csv(d / f"{k}.csv", index=False, float_format="%.6g")
        out["contingency"].to_csv(d / "contingency.csv", float_format="%.1f")
        evaluate_one(rows, name, "B-argmax (exploratory)", cfg, out, glob_)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "o3_summary.csv", index=False, float_format="%.6g")
    return df


def evaluate_one(rows, name, src, cfg, out, glob_):
    """Score one predicted-bin analysis against S1 and S2."""
    s = out["summary"]
    for alpha in cfg.alphas:
        g = glob_[glob_.alpha == alpha].set_index("bin")
        fails = [b for b in ["fog", "rain", "sand", "snow"] if g.loc[b, "exceed_frac"] >= 0.5]
        pw = s[(s.alpha == alpha) & (s.policy == "predicted_bin") & (s.grouping == "weather")].set_index("bin")
        ow = s[(s.alpha == alpha) & (s.policy == "oracle_bin") & (s.grouping == "weather")].set_index("bin")
        pred_bins = s[(s.alpha == alpha) & (s.policy == "predicted_bin") & (s.grouping == "pred")]
        gap = out["gap"][out["gap"].alpha == alpha].fppi_gap
        s1 = all(pw.loc[b, "image_loss_mean"] <= alpha for b in fails) if fails else None
        rows.append(dict(setting=name, source=src, alpha=alpha, ari=float(out["agreement"].ari_all.mean()),
                         global_fails_in=",".join(fails) or "none",
                         pred_loss_in_failing_bins=",".join(f"{b}:{pw.loc[b, 'image_loss_mean']:.3f}" for b in fails),
                         oracle_loss_in_failing_bins=",".join(f"{b}:{ow.loc[b, 'image_loss_mean']:.3f}" for b in fails),
                         S1=s1, fppi_gap_mean=float(gap.mean()), fppi_gap_std=float(gap.std()),
                         S2=bool(gap.mean() <= 5.0),
                         pred_bins_infeasible=float(pred_bins.infeasible_frac.mean()),
                         pred_bins_max_loss=float(pred_bins.image_loss_mean.max())))



def classifier_checks() -> dict:
    out = {}
    p = REPO / "cache" / "o3" / "h_bdd_val.parquet"
    if p.exists():
        b = pd.read_parquet(p)
        tag = {"clear": "clear", "cloudy": "cloudy", "rain": "rain", "snow": "snow"}
        day = b[b.weather.isin(tag)]
        out["bdd_val_day_accuracy"] = float((day.pred == day.weather.map(tag)).mean())
        out["bdd_val_day_confusion"] = pd.crosstab(day.weather, day.pred).to_dict()
        out["bdd_val_mean_h"] = b.groupby("weather").h.mean().round(3).to_dict()
    p = REPO / "cache" / "o3" / "h_dawn.parquet"
    if p.exists():
        d = pd.read_parquet(p)
        out["dawn_pred_by_weather"] = pd.crosstab(d.weather, d.pred).to_dict()
        out["dawn_mean_h"] = d.groupby("weather").h.mean().round(3).to_dict()
    log = REPO / "results" / "o3" / "train_log.json"
    if log.exists():
        out["train"] = json.loads(log.read_text())
    (OUT / "classifier_checks.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


def report(df: pd.DataFrame, cc: dict) -> str:
    lines = ["# O3: camera hazard estimators vs oracle weather bins", "",
             "Generated by `python -m wad.o3_eval`. Pre-registration: `PREREG_o3.md`. Version A = pixel statistics "
             "(dark channel, contrast, edges, saturation; PCA); version B = ResNet-18 trained on daytime BDD100K train "
             "weather tags, h = p(rain)+p(snow)+p(fog). Bins: quartiles of h on each calibration half; 10 splits.", ""]
    if "train" in cc:
        t = cc["train"]
        lines += [f"Classifier: {t['n_train']} training and {t['n_holdout']} hold-out images "
                  f"({', '.join(f'{k} {v}' for k, v in t['per_class'].items())}); best hold-out balanced accuracy "
                  f"{t['best_holdout_balanced_acc']:.3f}."]
    if "bdd_val_day_accuracy" in cc:
        lines += [f"BDD100K val daytime, argmax class vs weather tag: accuracy {cc['bdd_val_day_accuracy']:.3f}; mean "
                  f"h per bin {cc['bdd_val_mean_h']}."]
    if "dawn_mean_h" in cc:
        lines += [f"DAWN mean h per weather: {cc['dawn_mean_h']}; argmax class by weather "
                  f"{ {w: {k: v for k, v in d.items()} for w, d in pd.DataFrame(cc['dawn_pred_by_weather']).T.to_dict().items()} }."]
    show = df.assign(S1=df.S1.map({True: "met", False: "not met", None: "n/a (global never fails)"}),
                     S2=df.S2.map({True: "met", False: "not met"}),
                     fppi_gap=[f"{m:+.2f} ± {s:.2f}" for m, s in zip(df.fppi_gap_mean, df.fppi_gap_std)])
    show = show[["setting", "source", "alpha", "ari", "global_fails_in", "pred_loss_in_failing_bins",
                 "oracle_loss_in_failing_bins", "S1", "fppi_gap", "S2", "pred_bins_infeasible"]].round(3)
    lines += ["", "S1: predicted-bin policy meets α in every weather bin where the global threshold fails; S2: FPPI gap to "
              "oracle bins ≤ 5. *pred_bins_infeasible* = share of (predicted bin, split) fits with no feasible threshold. "
              "Rows marked *B-argmax (exploratory)* were not pre-registered: they bin by the classifier's argmax class "
              "(clear and cloudy merged), a fixed function of the image.", "", md(show), ""]
    ex = OUT / "run2_v8x_640-group-clean__B_argmax_EXPLORATORY" / "summary.csv"
    if ex.exists():
        s = pd.read_csv(ex)
        pb = s[(s.policy == "predicted_bin") & (s.grouping == "pred")]
        wf = s[(s.policy == "predicted_bin") & (s.grouping == "weather") & (s.bin == "fog")].set_index("alpha")
        of = s[(s.policy == "oracle_bin") & (s.grouping == "weather") & (s.bin == "fog")].set_index("alpha")
        cont = pd.read_csv(ex.parent / "contingency.csv").set_index("weather")
        lines += ["**Reading.** Within the bins the estimator predicts, the guarantee holds as the theory says (YOLOv8x @640, "
                  "group, clean, argmax bins: held-out miss per predicted bin "
                  + "; ".join(f"α = {a}: {g.image_loss_mean.min():.3f}–{g.image_loss_mean.max():.3f}" for a, g in pb.groupby("alpha"))
                  + "). But it does not carry over to the true weather condition: on images that really are foggy the "
                  "predicted-bin policy misses " + ", ".join(f"{wf.loc[a, 'image_loss_mean']:.3f} at α = {a} (exceeds α in "
                                                              f"{wf.loc[a, 'exceed_frac']:.0%} of splits)" for a in wf.index)
                  + ", against " + ", ".join(f"{of.loc[a, 'image_loss_mean']:.3f}" for a in of.index)
                  + " for the oracle fog bin. The classifier puts "
                  f"{cont.loc['fog', 'fog']:.0f} of {cont.loc['fog'].sum():.0f} fog test-and-calibration images (mean per "
                  f"split) in its fog bin, alongside {cont.loc['sand', 'fog']:.0f} sand images; the fog bin's threshold is "
                  "set for that mixture and is too high for real fog. *A wrong bin costs false positives, not safety* holds "
                  "for the predicted bin's average, not for the hazardous condition inside it: protecting fog needs an "
                  "estimator that isolates fog (more fog training data, e.g. WEDGE or Foggy Cityscapes, or a visibility "
                  "regressor) or a conservative rule that sends uncertain images to the strictest bin.", ""]
    text = "\n".join(lines) + "\n"
    (OUT / "o3.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    df = evaluate()
    cc = classifier_checks()
    report(df, cc)
    print(OUT / "o3.md")
