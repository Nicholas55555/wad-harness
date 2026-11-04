"""Ablation (d) of Phase 4 (exploratory; PREREG_crc_lambda.md lists it): does the dial transfer to other
detectors, input resolutions and detector families?

For YOLOv8x @640, YOLOv8s @640, YOLOv8s @1280 and RT-DETR-l @640 (a transformer detector with set prediction and no
NMS; it has no anchor-level head outputs, so it runs with model 1 only) the adopted pipeline is rerun with only the
detector changed:
  detection   the 21,831 BDD100K train frames, all 8,090 BDD100K val frames and the DAWN images, with wad.detect's
              settings (floor 0.001, NMS IoU 0.5, at most 300 boxes, the six classes, GPU)
  matching    grouped matching at IoU 0.5, as for YOLOv8x @1280
  per-box     the head outputs at each kept box's anchor (wad.yolo_box_features.extract)
  dial        that detector's own one threshold tau(lam) from its train curve; model 1 (20 context features)
              and model 2 (68 features, the neck PCA refitted on that detector's train boxes), the adopted
              settings (wad.per_detection.PARAMS), BDD100K train only
  grading     on BDD100K val and DAWN (clean labels): cost vs the detector's own one threshold at lam = 1, 3, 10,
              30 with 95% image-bootstrap intervals (wad.per_detection.grade), ECE, and cost per image, which the
              report sets against YOLOv8x @1280's one threshold (same images and objects for every detector)
  CRC         CRC over lam vs per-weather CRC over that detector's score on run 2's 200 splits at alpha 0.2, 0.25
              and 0.3 (wad.crc_lambda.evaluate), with paired per-split false-alarm ratios and the smallest
              certifiable alpha
Reused detections: BDD100K val YOLOv8s @640 and DAWN YOLOv8x @640 are run 2's GPU passes (same settings). DAWN
YOLOv8s @640 is detected again on the GPU: run 1's pass ran on the CPU, which a GPU re-run of the head would not
reproduce box for box. Matching caches go to cache/det_ablation/; the BDD100K val YOLOv8s @640 cache must equal
run 2's (checked). Nothing is chosen on BDD100K val or DAWN.

Outputs results/ablation/: detector_ablation_eval.csv (grading), detector_ablation_crc.csv (CRC summaries),
detector_ablation_ratio.csv, detector_ablation_min_alpha.csv, detector_ablation_data.csv, detector_ablation.md and
the logs; per-detector models in weights/det_ablation/, chances in cache/det_ablation/.
Run: python -m wad.detector_ablation gpu [--det v8x_640 ...]   (GPU: detection and per-box passes)
     python -m wad.detector_ablation fit [--det ...]           (CPU: matching, models, grading, CRC)
     python -m wad.detector_ablation report
"""
from __future__ import annotations

import argparse
import copy
import sys
import time

import numpy as np
import pandas as pd

from wad.config import REPO, THRESH_CACHE, cache_path, det_path

DETS = {"v8x_640": dict(name="yolov8x.pt", imgsz=640, label="YOLOv8x @640"),
        "v8s_640": dict(name="yolov8s.pt", imgsz=640, label="YOLOv8s @640"),
        "v8s_1280": dict(name="yolov8s.pt", imgsz=1280, label="YOLOv8s @1280"),
        # another family: a transformer detector with set prediction (no NMS, no DFL, no anchor-level neck vector),
        # so no per-box outputs: its per-detection rule and CRC use model 1 (the 20 context features)
        "rtdetr_640": dict(name="rtdetr-l.pt", imgsz=640, label="RT-DETR-l @640", per_box=False)}
ADOPTED = dict(det_id="v8x_1280", label="YOLOv8x @1280 (adopted)")
CACHE = REPO / "cache" / "det_ablation"
OUT = REPO / "results" / "ablation"
WEIGHTS = REPO / "weights" / "det_ablation"
LAMS = [1.0, 3.0, 10.0, 30.0]
ALPHAS = [0.20, 0.25, 0.30, 0.35]  # 0.35: YOLOv8s @640 cannot certify 0.30 in BDD100K rain at night
KEY = {"BDD100K": "bdd", "DAWN": "dawn"}


def say(msg: str, log: str = "detector_ablation.log") -> None:
    print(msg, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / log, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _r2(det_id: str):
    """Run 2's config with this detector's entry: GPU, DAWN detections in cache/det/<det_id>."""
    from wad.run2 import load_run2

    r2 = copy.deepcopy(load_run2())
    d = DETS[det_id]
    r2.detectors[det_id] = dict(name=d["name"], imgsz=d["imgsz"], device="cuda:0", detections=f"cache/det/{det_id}")
    return r2


def configs(det_id: str) -> dict:
    """train / BDD100K val / DAWN (clean) configs for a detector, plus 'DAWN detect' (all DAWN images, as run 2
    detected them). Matching caches live in cache/det_ablation/<set>/."""
    from wad.run2 import bdd_cfg, dawn_cfg
    from wad.thresh_net import cfg as train_cfg

    r2, d = _r2(det_id), DETS[det_id]
    t = train_cfg()
    t.detector.update(name=d["name"], imgsz=d["imgsz"], device="cuda:0")
    t.paths["detections"] = THRESH_CACHE / "det" / det_id
    t.paths["cache"] = t.paths["results"] = CACHE / "train" / f"{det_id}__group"
    t["tag"], t["det_id"] = f"train_{det_id}__group", det_id
    out = {"train": t, "BDD100K": bdd_cfg(r2, det_id, "group"), "DAWN": dawn_cfg(r2, det_id, "group", "clean"),
           "DAWN detect": dawn_cfg(r2, det_id, "class", "all")}
    for key, sub in (("BDD100K", "bdd"), ("DAWN", "dawn"), ("DAWN detect", "dawn_detect")):
        out[key].paths["cache"] = out[key].paths["results"] = CACHE / sub / out[key].tag
    return out


def det_file(det_id: str, key: str) -> str:
    from wad.yolo_box_features import detections_file

    return detections_file({"train": "train", "BDD100K": "bdd", "DAWN": "dawn"}[key], det_id)


# ---- GPU: detection and per-box passes ---------------------------------------------------------------

def gpu(det_id: str) -> None:
    import torch

    from wad import detect
    from wad.yolo_box_features import extract

    cf, t0 = configs(det_id), time.perf_counter()
    for key, name in (("train", "train"), ("BDD100K", "bdd"), ("DAWN detect", "dawn")):
        v = cf[key]
        if det_path(v, "detections.parquet").exists():
            say(f"{det_id} {name}: detections cached ({det_path(v, 'detections.parquet')})")
        else:
            say(f"{det_id} {name}: detecting ({v.detector.name} @{v.detector.imgsz})")
            detect.run(v)
            say(f"{det_id} {name}: detected, {time.perf_counter() - t0:.0f} s")
        if not DETS[det_id].get("per_box", True):
            continue
        extract(name, det_id, cf["train"])
        say(f"{det_id} {name}: per-box features done, {time.perf_counter() - t0:.0f} s")
        torch.cuda.empty_cache()


# ---- CPU: matching, models, grading, CRC ---------------------------------------------------------------

def match(det_id: str) -> dict:
    from wad.match import match_all

    cf = configs(det_id)
    for key in ("train", "BDD100K", "DAWN"):
        if not cache_path(cf[key], "matches_pred.parquet").exists():
            t = time.perf_counter()
            match_all(cf[key])
            say(f"{det_id} {key}: matched, {time.perf_counter() - t:.0f} s")
    if det_id == "v8s_640":  # same detections and loss as run 2's BDD100K val analysis
        from wad.run2 import bdd_cfg, load_run2

        old = bdd_cfg(load_run2(), "v8s_640", "group")
        try:
            for f in ("matches_gt.parquet", "matches_pred.parquet"):
                a, b = pd.read_parquet(cache_path(cf["BDD100K"], f)), pd.read_parquet(cache_path(old, f))
                pd.testing.assert_frame_equal(a, b, check_dtype=False)
            say("v8s_640 BDD100K: matching identical to run 2's cache (check passed)")
        except AssertionError as e:
            say(f"v8s_640 BDD100K: matching differs from run 2's cache (CHECK FAILED): {str(e)[:300]}")
    return cf


def fit(det_id: str) -> None:
    import joblib

    from wad.crc_lambda import evaluate, min_alpha
    from wad.curves import ALL, Cache
    from wad.one_threshold import train_curve, train_threshold
    from wad.per_detection import (BOX_SET, FULL, YOLO_COLS, add_box_columns, calibration, fit_pca, grade,
                                   groups_of, labelled_features, train_model)

    t0 = time.perf_counter()
    cf = match(det_id)
    ct = Cache(cf["train"])
    tr = labelled_features(ct, det_file(det_id, "train"))
    per_box = DETS[det_id].get("per_box", True)
    pca = fit_pca(det_id=det_id) if per_box else None
    if per_box:
        tr = add_box_columns(tr, "train", pca, det_id=det_id)
    data = [dict(detector=det_id, set="train", images=ct.n_img, objects=int(ct.n_gt(ALL).sum()), detections=len(tr),
                 real=int(tr.tp.sum()), objects_ever_detected=float(np.isfinite(ct.gt_score).mean()))]
    say(f"{det_id} train: {len(tr)} detections, {int(tr.tp.sum())} real, {time.perf_counter() - t0:.0f} s")
    models = {}
    for name, cols in (("model 1", FULL), ("model 2", FULL + YOLO_COLS))[:2 if per_box else 1]:
        models[name] = (train_model(tr, cols), cols)
        say(f"{det_id} {name}: {models[name][0].n_iter_} iterations, {time.perf_counter() - t0:.0f} s")
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(models=models, pca=pca, det_id=det_id, detector=DETS[det_id]), WEIGHTS / f"{det_id}.joblib")
    _, curve = train_curve(pd.read_parquet(cache_path(cf["train"], "matches_pred.parquet"), columns=["score", "tp"]))
    tau = {lam: train_threshold(curve, lam) for lam in LAMS}
    del tr
    ev, cal, crc, mins = [], [], [], []
    for ds in ("BDD100K", "DAWN"):
        c = Cache(cf[ds])
        f = labelled_features(c, det_file(det_id, ds))
        if per_box:
            f = add_box_columns(f, BOX_SET[ds], pca, det_id=det_id)
        ps = {name: m.predict_proba(f[cols].to_numpy())[:, 1] for name, (m, cols) in models.items()}
        pd.DataFrame(dict(image_id=c.image_ids[f.img.to_numpy()], score=f.score.to_numpy(), tp=f.tp.to_numpy(),
                          p1=ps["model 1"], p2=ps.get("model 2", np.full(len(f), np.nan)))).to_parquet(CACHE / f"pred_{det_id}_{KEY[ds]}.parquet", index=False)
        data.append(dict(detector=det_id, set=ds, images=c.n_img, objects=int(c.n_gt(ALL).sum()), detections=len(f),
                         real=int(f.tp.sum()), objects_ever_detected=float(np.isfinite(c.gt_score).mean())))
        for name, p in ps.items():
            g = grade(ds, c, f, p, tau, LAMS)
            g.loc[g.policy == "per detection", "policy"] = f"per detection ({name})"
            ev.append(g if name == list(models)[-1] else g[g.policy.str.startswith("per detection")])
            w_det, y = c.weather[f.img.to_numpy()], f.tp.to_numpy().astype(bool)
            for grp, ws in groups_of(ds, c.weather).items():
                m = np.isin(w_det, ws)
                cal.append(dict(dataset=ds, group=grp, model=name, ece=calibration(p[m], y[m])[1]))
        say(f"{det_id} {ds}: graded, {time.perf_counter() - t0:.0f} s")
        main = ps["model 2" if per_box else "model 1"]  # the chance CRC over lam cuts
        e = evaluate(ds, cf[ds], main, alphas=ALPHAS)
        crc.append(e)
        mins.append(min_alpha(ds, cf[ds], main))
        say(f"{det_id} {ds}: CRC over 200 splits, {time.perf_counter() - t0:.0f} s")
    sub = OUT / "det"
    sub.mkdir(parents=True, exist_ok=True)
    ff = "%.6g"
    pd.concat(ev, ignore_index=True).assign(detector=det_id).to_csv(sub / f"{det_id}_eval.csv", index=False, float_format=ff)
    pd.DataFrame(cal).assign(detector=det_id).to_csv(sub / f"{det_id}_ece.csv", index=False, float_format=ff)
    pd.concat(crc, ignore_index=True).assign(detector=det_id).to_csv(sub / f"{det_id}_crc_eval.csv", index=False, float_format=ff)
    pd.concat(mins, ignore_index=True).assign(detector=det_id).to_csv(sub / f"{det_id}_min_alpha.csv", index=False, float_format=ff)
    pd.DataFrame(data).to_csv(sub / f"{det_id}_data.csv", index=False, float_format=ff)
    pd.Series(tau).rename_axis("lam").rename("tau").to_csv(sub / f"{det_id}_tau.csv", float_format=ff)
    say(f"{det_id}: done, {time.perf_counter() - t0:.0f} s")


def crc(det_id: str, alphas) -> None:
    """CRC over lam vs the per-weather baseline at more alphas, from the saved chances (merged into
    results/ablation/det/<det_id>_crc_eval.csv)."""
    from wad.crc_lambda import evaluate
    from wad.curves import Cache

    cf, sub, t0 = configs(det_id), OUT / "det", time.perf_counter()
    old = pd.read_csv(sub / f"{det_id}_crc_eval.csv")
    new = []
    for ds in ("BDD100K", "DAWN"):
        pr, c = pd.read_parquet(CACHE / f"pred_{det_id}_{KEY[ds]}.parquet"), Cache(cf[ds])
        if not (np.array_equal(pr.image_id.to_numpy(), c.pred.image_id.to_numpy()) and np.allclose(pr.score, c.pred.score)):
            raise ValueError(f"{det_id} {ds}: saved chances are not aligned with the cached predictions")
        main = pr.p2 if pr.p2.notna().all() else pr.p1
        new.append(evaluate(ds, cf[ds], main.to_numpy(), alphas=alphas).assign(detector=det_id))
        say(f"{det_id} {ds}: CRC at alpha {alphas}, {time.perf_counter() - t0:.0f} s")
    keep = ~old.alpha.round(4).isin([round(a, 4) for a in alphas])
    pd.concat([old[keep]] + new, ignore_index=True).to_csv(sub / f"{det_id}_crc_eval.csv", index=False, float_format="%.6g")


# ---- report --------------------------------------------------------------------------------------------

def adopted_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """YOLOv8x @1280's grading (doc 08 / Phase 4, model 2 and model 1), CRC eval and data, for the same tables."""
    from wad.crc_lambda import OUT as CRC_OUT
    from wad.config import THRESH_RESULTS

    e2 = pd.read_csv(THRESH_RESULTS / "per_detection_yolo_eval.csv")
    e1 = pd.read_csv(THRESH_RESULTS / "per_detection_eval.csv")
    e2.loc[e2.policy == "per detection", "policy"] = "per detection (model 2)"
    e1 = e1[e1.policy == "per detection"].assign(policy="per detection (model 1)")
    ev = pd.concat([e2, e1], ignore_index=True)
    ev = ev[ev.lam.isin(LAMS)].assign(detector=ADOPTED["det_id"])
    crc = pd.read_csv(CRC_OUT / "crc_lambda_eval.csv")
    crc = crc[crc.alpha.isin(ALPHAS)].assign(detector=ADOPTED["det_id"])
    mins = pd.read_csv(CRC_OUT / "crc_lambda_min_alpha.csv").assign(detector=ADOPTED["det_id"])
    return ev, crc, mins


def adopted_data() -> pd.DataFrame:
    """The data rows of YOLOv8x @1280 (its train, BDD100K val and DAWN caches), as fit() writes them for the others."""
    from wad.curves import ALL, Cache
    from wad.one_threshold import eval_configs
    from wad.thresh_net import cfg as train_cfg

    rows = []
    for key, v in [("train", train_cfg())] + list(eval_configs().items()):
        c = Cache(v)
        rows.append(dict(detector=ADOPTED["det_id"], set=key, images=c.n_img, objects=int(c.n_gt(ALL).sum()), detections=len(c.pred),
                         real=int(c.pred.tp.sum()), objects_ever_detected=float(np.isfinite(c.gt_score).mean())))
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> list:
    cell = lambda x: f"{x:.4g}" if isinstance(x, float) else str(x)
    return (["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
            + ["| " + " | ".join(cell(x) for x in r) + " |" for r in df.itertuples(index=False)])


def report() -> str:
    from wad.crc_lambda import paired_ratios, summarize_eval

    sub = OUT / "det"
    done = [d for d in DETS if (sub / f"{d}_eval.csv").exists()]
    ev0, crc0, min0 = adopted_rows()
    ev = pd.concat([ev0] + [pd.read_csv(sub / f"{d}_eval.csv") for d in done], ignore_index=True)
    crc = pd.concat([crc0] + [pd.read_csv(sub / f"{d}_crc_eval.csv") for d in done], ignore_index=True)
    mins = pd.concat([min0] + [pd.read_csv(sub / f"{d}_min_alpha.csv") for d in done], ignore_index=True)
    ece = pd.concat([pd.read_csv(sub / f"{d}_ece.csv") for d in done], ignore_index=True) if done else pd.DataFrame()
    data = pd.concat([adopted_data()] + [pd.read_csv(sub / f"{d}_data.csv") for d in done], ignore_index=True)
    order = [ADOPTED["det_id"]] + done
    labels = {ADOPTED["det_id"]: ADOPTED["label"],
              **{k: v["label"] + ("" if v.get("per_box", True) else " (model 1 in the model 2 columns)") for k, v in DETS.items()}}
    # cost per image against YOLOv8x @1280's one threshold (same images and objects)
    ref = ev[(ev.detector == ADOPTED["det_id"]) & (ev.policy == "one threshold (train)")].set_index(["dataset", "group", "lam"]).cost_per_img
    ev["cost_vs_adopted_one"] = ev.cost_per_img.to_numpy() / ref.reindex(pd.MultiIndex.from_frame(ev[["dataset", "group", "lam"]])).to_numpy()
    summ, rat = [], []
    for d, g in crc.groupby("detector", sort=False):
        summ.append(summarize_eval(g).assign(detector=d))
        rat.append(paired_ratios(g).assign(detector=d))
    summ, rat = pd.concat(summ, ignore_index=True), pd.concat(rat, ignore_index=True)
    ff = "%.6g"
    ev.to_csv(OUT / "detector_ablation_eval.csv", index=False, float_format=ff)
    summ.to_csv(OUT / "detector_ablation_crc.csv", index=False, float_format=ff)
    rat.to_csv(OUT / "detector_ablation_ratio.csv", index=False, float_format=ff)
    mins.to_csv(OUT / "detector_ablation_min_alpha.csv", index=False, float_format=ff)
    if len(ece):
        ece.to_csv(OUT / "detector_ablation_ece.csv", index=False, float_format=ff)
    data.to_csv(OUT / "detector_ablation_data.csv", index=False, float_format=ff)

    lines = ["# Ablation (d): other detectors and resolutions", "",
             "Exploratory (PREREG_crc_lambda.md). Each detector gets its own one threshold, model 1 and model 2, fitted on",
             "BDD100K train only with the adopted settings; graded on BDD100K val (8,090 frames) and DAWN (clean labels).",
             "", "## Cost at fixed lambda", "",
             "Per-detection rule (model 2) / that detector's one threshold, with 95% image-bootstrap intervals; the last",
             "columns divide cost per image by YOLOv8x @1280's one threshold (below 1: better than the adopted detector's",
             "one threshold).", "",
             "| detector | dataset | group | lam | model 2 / own one thr. | 95% CI | model 1 / own one thr. | own one thr. / adopted one thr. | model 2 / adopted one thr. |",
             "|---|---|---|---|---|---|---|---|---|"]
    for d in order:
        for ds, grp in (("BDD100K", "all"), ("BDD100K", "day"), ("BDD100K", "night"), ("DAWN", "all"), ("DAWN", "fog")):
            for lam in (1.0, 10.0, 30.0):
                e = ev[(ev.detector == d) & (ev.dataset == ds) & (ev.group == grp) & (ev.lam == lam)].set_index("policy")
                if e.empty:
                    continue
                m1, one = e.loc["per detection (model 1)"], e.loc["one threshold (train)"]
                m2 = e.loc["per detection (model 2)"] if "per detection (model 2)" in e.index else m1
                lines.append(f"| {labels[d]} | {ds} | {grp} | {lam:g} | {m2.cost_vs_one:.3f} | {m2.ci_lo:.3f}-{m2.ci_hi:.3f} | "
                             f"{m1.cost_vs_one:.3f} | {one.cost_vs_adopted_one:.3f} | {m2.cost_vs_adopted_one:.3f} |")
    lines += ["", "## CRC over lambda vs per-weather CRC over the score (200 splits, per bin)", "",
              "| detector | alpha | conditions where the baseline holds | ours max mean miss there | pooled FA ratio, median (95%) | conditions with median ratio < 1 |",
              "|---|---|---|---|---|---|"]
    for d in order:
        for alpha in ALPHAS:
            s = summ[(summ.detector == d) & (summ.alpha == alpha) & (summ.scope == "per_bin")]
            if s.empty:
                continue
            w = s.pivot_table(index=["dataset", "bin"], columns="method", values="image_loss_mean")
            held = w[w.baseline <= alpha + 0.01]
            r = rat[(rat.detector == d) & (rat.alpha == alpha) & (rat.scope == "per_bin")]
            per = r[r.bin != "all"]
            pooled = r[(r.dataset == "both") & (r.bin == "all")].iloc[0]
            lines.append(f"| {labels[d]} | {alpha:g} | {len(held)} of {len(w)} | {held.ours.max() if len(held) else float('nan'):.3f} | "
                         f"{pooled.ratio_median:.3f} ({pooled.ratio_q025:.3f}-{pooled.ratio_q975:.3f}) | "
                         f"{int((per.ratio_median < 1).sum())} of {len(per)} |")
    lines += ["", "## Smallest certifiable alpha (mean over calibration halves)", "",
              "| detector | method | BDD100K bins (min-max) | DAWN bins (min-max) |", "|---|---|---|---|"]
    for d in order:
        for method in ("baseline", "ours"):
            m = mins[(mins.detector == d) & (mins.method == method)]
            b, w = m[m.dataset == "BDD100K"].min_alpha, m[m.dataset == "DAWN"].min_alpha
            lines.append(f"| {labels[d]} | {method} | {b.min():.3f}-{b.max():.3f} | {w.min():.3f}-{w.max():.3f} |")
    lines += ["", "## Data per detector", ""] + md_table(data)
    if len(ece):
        e = ece[ece.group.isin(["all", "day", "night", "fog"])].pivot_table(index=["detector", "dataset", "group"], columns="model", values="ece")
        lines += ["", "## Calibration error (ECE)", ""] + md_table(e.reset_index())
    md = "\n".join(lines) + "\n"
    (OUT / "detector_ablation.md").write_text(md, encoding="utf-8")
    return md


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["gpu", "fit", "crc", "report"])
    ap.add_argument("--det", nargs="+", default=list(DETS), choices=list(DETS))
    ap.add_argument("--alphas", nargs="+", type=float, default=None, help="crc: the alphas to (re)compute")
    args = ap.parse_args(argv)
    if args.command == "report":
        print(report())
        return
    for d in args.det:
        if args.command == "crc":
            crc(d, args.alphas or ALPHAS)
        else:
            (gpu if args.command == "gpu" else fit)(d)


if __name__ == "__main__":
    main()
