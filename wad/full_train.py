"""Phase 4, training data (ablation c extended; exploratory): the per-detection model refitted on ALL BDD100K train
frames instead of the 21,831-frame weather-stratified subset the adopted (pre-registered) model was fitted on.

Nothing else changes: YOLOv8x @1280 with wad.detect's settings, grouped matching, the per-box head outputs, model 2's
68 features and settings (wad.per_detection.PARAMS), the neck PCA refitted the same way on the new training boxes.
The adopted model and every pre-registered result stay as they are; this answers the question the learning curve
left open (10 / 25 / 50 / 100% of the subset: 0.932 / 0.923 / 0.917 / 0.910 at lam = 10): does three times more
data help?

Frames: every labelled BDD100K train image (70,000). The subset's weather bins keep their names, so its 21,831
frames keep their image ids, and their detections and per-box outputs are reused (copied, then checked box by box)
instead of recomputed; three catch-all bins take the weather / time-of-day pairs the subset left out. The missing
images come from the dataset's official host (scripts/get_bdd100k.py --train-list).

Graded like the adopted model: cost vs one threshold at lam = 1, 3, 10, 30 on BDD100K val and DAWN (95% image-
bootstrap intervals), ECE, and CRC over lam vs the per-weather baseline on run 2's 200 splits at alpha 0.2.

Run: python -m wad.full_train names    # cache/thresh_full/train_names.txt, fetch_names.txt
     python scripts/get_bdd100k.py --no-val --train-list cache/thresh_full/fetch_names.txt
     python -m wad.full_train scan     # images.parquet, gt.parquet (near-duplicate groups, exact duplicates)
     python -m wad.full_train gpu      # detection + per-box outputs for the frames not in the subset
     python -m wad.full_train fit      # matching, models, grading, CRC -> results/ablation/full_train*
"""
from __future__ import annotations

import argparse
import copy
import gc
import shutil
import sys
import time

import numpy as np
import pandas as pd

from wad.config import REPO, THRESH_CACHE, cache_path, det_path

CACHE = REPO / "cache" / "thresh_full"
OUT = REPO / "results" / "ablation"
DET_ID, LOSS = "v8x_1280", "group"
LAMS = [1.0, 3.0, 10.0, 30.0]
EXTRA_BINS = {  # the weather / time-of-day pairs the subset's bins (wad.thresh_net.TRAIN_BINS) leave out
    "cloudy_night": {"weather": ["overcast", "partly cloudy"], "timeofday": ["night"]},
    "undefined_weather": {"weather": ["undefined"], "timeofday": ["daytime", "night"]},
    "undefined_time": {"weather": ["clear", "overcast", "partly cloudy", "rainy", "snowy", "foggy", "undefined"],
                       "timeofday": ["undefined"]},
}


def say(msg: str) -> None:
    print(msg, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "full_train.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def names() -> list[str]:
    """Every labelled BDD100K train frame; fetch_names.txt lists those not on disk yet."""
    from wad.hazard_cnn import train_attributes
    from wad.thresh_net import BDD

    a = train_attributes()
    have = {p.stem for p in (BDD / "images" / "100k" / "train").glob("*.jpg")}
    all_names = sorted(a.name)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "train_names.txt").write_text("".join(f"{n}\n" for n in all_names), encoding="utf-8")
    missing = [n for n in all_names if n not in have]
    (CACHE / "fetch_names.txt").write_text("".join(f"{n}\n" for n in missing), encoding="utf-8")
    say(f"{len(all_names)} train frames; {len(have)} on disk; {len(missing)} to fetch")
    return all_names


def cfg():
    """All BDD100K train frames: the subset's config (wad.thresh_net.cfg) with every frame and its own caches."""
    from wad.thresh_net import TRAIN_BINS, cfg as subset_cfg

    v = subset_cfg()
    v.data["names_file"] = str(CACHE / "train_names.txt")
    v.data["bins"] = {**copy.deepcopy(TRAIN_BINS), **copy.deepcopy(EXTRA_BINS)}
    v.paths["data_cache"] = CACHE
    v.paths["detections"] = CACHE / "det" / DET_ID
    v.paths["cache"] = v.paths["results"] = CACHE / f"{DET_ID}__{LOSS}"
    v["tag"], v["dataset_name"] = f"train_full_{DET_ID}__{LOSS}", "bdd_train_full"
    return v


def scan() -> None:
    from wad import data
    from wad.config import data_path, write_json

    v, t0 = cfg(), time.perf_counter()
    images, gt, report = data.scan(v)
    images.to_parquet(data_path(v, "images.parquet"), index=False)
    gt.to_parquet(data_path(v, "gt.parquet"), index=False)
    write_json(data_path(v, "data_report.json"), report)
    kept = images[images.keep]
    say(f"scan: {len(images)} frames, {len(kept)} kept (exact duplicates dropped), {len(gt)} objects, "
        f"{time.perf_counter() - t0:.0f} s; per bin: {kept.groupby('weather').size().to_dict()}")


def gpu() -> None:
    """Detection and per-box outputs for the frames not in the subset: the subset's detection shards are copied
    (same detector, settings and image ids, so wad.detect skips those frames), and its per-box rows are reused."""
    from wad import detect
    from wad.yolo_box_features import extract

    v, t0 = cfg(), time.perf_counter()
    dst = CACHE / "det" / DET_ID / "det_shards"
    if not dst.exists():
        shutil.copytree(THRESH_CACHE / "det" / DET_ID / "det_shards", dst)
        say(f"copied the subset's detection shards to {dst}")
    if not det_path(v, "detections.parquet").exists():
        detect.run(v)
        say(f"detection done, {time.perf_counter() - t0:.0f} s")
    extract("train_full", DET_ID, v, reuse="train")
    say(f"per-box outputs done, {time.perf_counter() - t0:.0f} s")


def fit() -> None:
    import joblib
    import sklearn

    from wad.crc_lambda import evaluate, paired_ratios, summarize_eval
    from wad.curves import ALL, Cache
    from wad.match import match_all
    from wad.one_threshold import TRAIN, eval_configs, train_curve, train_threshold
    from wad.per_detection import (BOX_SET, EVAL_DET, FULL, PARAMS, YOLO_COLS, add_box_columns, calibration, fit_pca,
                                   grade, groups_of, labelled_features, load, pred_path, versus_model1)

    v, t0 = cfg(), time.perf_counter()
    if not cache_path(v, "matches_pred.parquet").exists():
        match_all(v)
        say(f"matched, {time.perf_counter() - t0:.0f} s")
    ct = Cache(v)
    tr = labelled_features(ct, f"cache/thresh_full/det/{DET_ID}/detections.parquet")
    pca = fit_pca(box_set="train_full")  # refitted on the new training boxes, as fit_pca does for the subset
    tr = add_box_columns(tr, "train_full", pca)
    data = dict(frames=ct.n_img, objects=int(ct.n_gt(ALL).sum()), detections=len(tr), real=int(tr.tp.sum()))
    del ct  # its 10.3 M prediction rows are not needed any more
    say(f"features: {data}, {time.perf_counter() - t0:.0f} s")
    # 10.3 M detections: fit from arrays (as wad.per_detection.train_model does) and drop the feature table before
    # model 2, whose categorical preprocessing copies the whole matrix once more (5 GB)
    from sklearn.ensemble import HistGradientBoostingClassifier

    y, models = tr.tp.to_numpy(), {}
    for name, cols in (("model 1", FULL), ("model 2", FULL + YOLO_COLS)):
        X = tr[cols].to_numpy(np.float64)
        if name == "model 2":
            del tr
            gc.collect()
        models[name] = (HistGradientBoostingClassifier(**PARAMS, categorical_features=[cols.index("cls")]).fit(X, y), cols)
        del X
        gc.collect()
        say(f"{name}: {models[name][0].n_iter_} iterations, {time.perf_counter() - t0:.0f} s")
    joblib.dump(dict(model=models["model 2"][0], cols=models["model 2"][1], features="yolo", pca=pca, params=PARAMS,
                     model1=models["model 1"], sklearn=sklearn.__version__,
                     trained_on=f"all BDD100K train frames ({data['frames']}), YOLOv8x @1280 detections >= 0.001, grouped matching",
                     rule="keep iff chance >= 1 / (1 + lam)"), REPO / "weights" / "per_detection_yolo_full.joblib")
    _, curve_full = train_curve(pd.read_parquet(cache_path(v, "matches_pred.parquet"), columns=["score", "tp"]))
    _, curve_sub = train_curve(pd.read_parquet(TRAIN, columns=["score", "tp"]))
    tau_full = {lam: train_threshold(curve_full, lam) for lam in LAMS}
    tau_sub = {lam: train_threshold(curve_sub, lam) for lam in LAMS}
    adopted = load()
    ev, vs, cal, crc = [], [], [], []
    for ds, c_cfg in eval_configs().items():
        c = Cache(c_cfg)
        f = add_box_columns(labelled_features(c, EVAL_DET[ds]), BOX_SET[ds], pca)
        ps = {name: m.predict_proba(f[cols].to_numpy())[:, 1] for name, (m, cols) in models.items()}
        key = "bdd" if ds == "BDD100K" else "dawn"
        pa = pd.read_parquet(pred_path(key, "yolo"))
        if not np.array_equal(pa.image_id.to_numpy(), c.image_ids[f.img.to_numpy()]):
            raise ValueError(f"{ds}: the adopted chances are not in this order")
        p_adopted = pa.p.to_numpy()
        pd.DataFrame(dict(image_id=c.image_ids[f.img.to_numpy()], score=f.score.to_numpy(), tp=f.tp.to_numpy(),
                          p1=ps["model 1"], p2=ps["model 2"])).to_parquet(CACHE / f"pred_full_{key}.parquet", index=False)
        for ref, tau in (("own (all train)", tau_full), ("adopted (subset)", tau_sub)):
            for name, p in ps.items():
                g = grade(ds, c, f, p, tau, LAMS).assign(one_threshold_from=ref)
                g.loc[g.policy == "per detection", "policy"] = f"per detection ({name}, all train)"
                ev.append(g if name == "model 2" else g[g.policy.str.startswith("per detection")])
        vs.append(versus_model1(ds, c, f, ps["model 2"], p_adopted).rename(columns={"cost_model2_vs_model1": "cost_full_vs_adopted"}))
        y, w_det = f.tp.to_numpy().astype(bool), c.weather[f.img.to_numpy()]
        for grp, ws in groups_of(ds, c.weather).items():
            m = np.isin(w_det, ws)
            for name, p in (("model 2, all train", ps["model 2"]), ("model 2, subset (adopted)", p_adopted)):
                cal.append(dict(dataset=ds, group=grp, model=name, ece=calibration(p[m], y[m])[1]))
        crc.append(evaluate(ds, c_cfg, ps["model 2"], alphas=[0.2]))
        say(f"{ds}: graded and CRC, {time.perf_counter() - t0:.0f} s")
    ev, vs, cal, crc = (pd.concat(ev, ignore_index=True), pd.concat(vs, ignore_index=True), pd.DataFrame(cal),
                        pd.concat(crc, ignore_index=True))
    summ, rat = summarize_eval(crc), paired_ratios(crc)
    ff = "%.6g"
    ev.to_csv(OUT / "full_train_eval.csv", index=False, float_format=ff)
    vs.to_csv(OUT / "full_train_vs_adopted.csv", index=False, float_format=ff)
    cal.to_csv(OUT / "full_train_ece.csv", index=False, float_format=ff)
    summ.to_csv(OUT / "full_train_crc.csv", index=False, float_format=ff)
    rat.to_csv(OUT / "full_train_crc_ratio.csv", index=False, float_format=ff)
    pd.Series(dict(**data, iterations_model1=int(models["model 1"][0].n_iter_), iterations_model2=int(models["model 2"][0].n_iter_),
                   adopted_iterations=int(adopted["model"].n_iter_))).to_csv(OUT / "full_train_data.csv", header=False)
    say(report())
    say(f"done, {time.perf_counter() - t0:.0f} s")


def report() -> str:
    ev = pd.read_csv(OUT / "full_train_eval.csv")
    vs = pd.read_csv(OUT / "full_train_vs_adopted.csv")
    cal = pd.read_csv(OUT / "full_train_ece.csv")
    rat = pd.read_csv(OUT / "full_train_crc_ratio.csv")
    summ = pd.read_csv(OUT / "full_train_crc.csv")
    data = pd.read_csv(OUT / "full_train_data.csv", header=None, index_col=0)[1]
    lines = ["# Training on all BDD100K train frames (ablation c, extended)", "",
             f"{int(data.frames):,} frames, {int(data.objects):,} objects, {int(data.detections):,} detections "
             f"({int(data.real):,} real); model 2 stopped at {int(data.iterations_model2)} iterations (adopted, subset: "
             f"{int(data.adopted_iterations)}).", "",
             "## Model 2 on all train frames vs the adopted model 2 (21,831-frame subset): cost ratio, 95% image-bootstrap", "",
             "| dataset | group | lam | all train / subset | 95% CI |", "|---|---|---|---|---|"]
    for r in vs[vs.group.isin(["all", "day", "night", "rain_night", "fog"]) & vs.lam.isin([1, 3, 10, 30])].itertuples():
        lines.append(f"| {r.dataset} | {r.group} | {r.lam:g} | {r.cost_full_vs_adopted:.3f} | {r.ci_lo:.3f}-{r.ci_hi:.3f} |")
    lines += ["", "## Cost vs one threshold at lam = 10", "",
              "| dataset | group | one threshold from | per detection (model 2, all train) | 95% CI | model 1, all train |",
              "|---|---|---|---|---|---|"]
    e = ev[ev.lam == 10]
    for (ds, grp, ref), g in e[e.group.isin(["all", "day", "night", "fog"])].groupby(["dataset", "group", "one_threshold_from"], sort=False):
        g = g.set_index("policy")
        m2, m1 = g.loc["per detection (model 2, all train)"], g.loc["per detection (model 1, all train)"]
        lines.append(f"| {ds} | {grp} | {ref} | {m2.cost_vs_one:.3f} | {m2.ci_lo:.3f}-{m2.ci_hi:.3f} | {m1.cost_vs_one:.3f} |")
    c = cal[cal.group.isin(["all", "day", "night", "fog"])].pivot_table(index=["dataset", "group"], columns="model", values="ece")
    lines += ["", "## Calibration error (ECE, points)", "", "| dataset | group | " + " | ".join(c.columns) + " |",
              "|---|---|" + "---|" * len(c.columns)]
    lines += [f"| {ds} | {grp} | " + " | ".join(f"{100 * x:.2f}" for x in row) + " |" for (ds, grp), row in c.iterrows()]
    s = summ[summ.scope == "per_bin"].pivot_table(index=["dataset", "bin"], columns="method", values="image_loss_mean")
    held = s[s.baseline <= 0.21]
    pooled = rat[(rat.scope == "per_bin") & (rat.dataset == "both") & (rat.bin == "all")].iloc[0]
    lines += ["", "## CRC over lambda (model 2, all train) vs per-weather CRC over the score, alpha 0.2, 200 splits", "",
              f"- conditions where the baseline holds: {len(held)} of {len(s)}; ours max mean miss there {held.ours.max():.4f}",
              f"- pooled false-alarm ratio ours / baseline: {pooled.ratio_median:.3f} ({pooled.ratio_q025:.3f}-{pooled.ratio_q975:.3f}); "
              "adopted model (subset): 0.610 (0.589-0.633)"]
    md = "\n".join(lines) + "\n"
    (OUT / "full_train.md").write_text(md, encoding="utf-8")
    return md


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["names", "scan", "gpu", "fit", "report"])
    cmd = ap.parse_args(argv).command
    out = {"names": names, "scan": scan, "gpu": gpu, "fit": fit, "report": report}[cmd]()
    if cmd == "report":
        print(out)


if __name__ == "__main__":
    main()
