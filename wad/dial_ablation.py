"""Ablations of the dial's per-detection model (Phase 4, exploratory; PREREG_crc_lambda.md lists them).

Each variant refits the adopted model's gradient-boosted trees (wad.per_detection.PARAMS, BDD100K train only)
with a different input set or training subset, then grades it on BDD100K val and DAWN as in doc 08: cost relative
to the one-threshold dial with 95% image-bootstrap intervals, recall / precision / false alarms per image, and the
calibration error. Variants:
  feature groups  the adopted 68 features; without class probabilities, box-edge (DFL) entropies, neck components
                  or pyramid level; context only (model 1's 20); per-box only (48); score and class only
  training size   10 / 25 / 50% of the train images (whole near-duplicate groups, seeds 0-2), and 100%
The PCA of the neck vectors is the adopted model's, so every variant sees identical feature values. The full
variant must reproduce the adopted model's chances exactly (same data, settings and seed).

Outputs results/ablation/: dial_ablation.csv, dial_ablation_curves.csv (training and validation log-loss per
boosting iteration), dial_ablation.md, dial_ablation.log.
Run: python -m wad.dial_ablation   (CPU, about 30-40 min)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from wad.config import REPO, THRESH_CACHE
from wad.per_detection import (BOX_SET, EVAL_DET, FULL, PARAMS, TRAIN_DET, YOLO_COLS, add_box_columns, calibration,
                               grade, labelled_features, load, pred_path)

OUT = REPO / "results" / "ablation"
LAMS = [1, 10, 30]
CLS_COLS = [c for c in YOLO_COLS if c.startswith("p_")]
DFL_COLS = [c for c in YOLO_COLS if c.startswith("dfl_")]
NECK_COLS = [c for c in YOLO_COLS if c.startswith("neck_")]
VARIANTS = {
    "model 2 (adopted, 68 features)": FULL + YOLO_COLS,
    "without class probabilities (-10)": [c for c in FULL + YOLO_COLS if c not in CLS_COLS],
    "without box-edge DFL entropies (-5)": [c for c in FULL + YOLO_COLS if c not in DFL_COLS],
    "without neck components (-32)": [c for c in FULL + YOLO_COLS if c not in NECK_COLS],
    "without pyramid level (-1)": [c for c in FULL + YOLO_COLS if c != "level"],
    "context only (model 1, 20)": FULL,
    "per-box outputs only (48)": YOLO_COLS,
    "score and class only (2)": ["logit", "cls"],
}
FRACTIONS = [0.10, 0.25, 0.50]
SEEDS = [0, 1, 2]
GROUPS = [("BDD100K", "all"), ("BDD100K", "day"), ("BDD100K", "night"), ("BDD100K", "rain_night"), ("DAWN", "all"), ("DAWN", "fog")]


def fit(f: pd.DataFrame, cols: list):
    from sklearn.ensemble import HistGradientBoostingClassifier

    cat = [cols.index("cls")] if "cls" in cols else None
    m = HistGradientBoostingClassifier(**PARAMS, categorical_features=cat)
    return m.fit(f[cols].to_numpy(), f.tp.to_numpy())


def subset_images(groups: pd.Series, frac: float, seed: int) -> np.ndarray:
    """Whole near-duplicate groups in random order until frac of the images is reached (boolean per image)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.unique(groups.to_numpy()))
    sizes = groups.value_counts()
    take, n = [], 0
    for g in order:
        if n >= frac * len(groups):
            break
        take.append(g)
        n += int(sizes[g])
    return groups.isin(take).to_numpy()


def main():
    import sys

    from wad.curves import Cache
    from wad.one_threshold import TRAIN, eval_configs, train_curve, train_threshold
    from wad.thresh_net import cfg as train_cfg

    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "dial_ablation.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    t0 = time.perf_counter()
    bundle = load()  # the adopted model 2: its PCA and columns
    assert bundle["cols"] == FULL + YOLO_COLS
    pca = bundle["pca"]
    ct = Cache(train_cfg())
    tr = add_box_columns(labelled_features(ct, TRAIN_DET), "train", pca)
    tim = pd.read_parquet(THRESH_CACHE / "images.parquet", columns=["image_id", "group"]).set_index("image_id")
    groups = tim.group.loc[ct.image_ids].reset_index(drop=True)
    say(f"train: {len(tr):,} detections on {ct.n_img:,} images, {time.perf_counter() - t0:.0f} s")
    evals = {}
    for ds, v in eval_configs().items():
        c = Cache(v)
        evals[ds] = (c, add_box_columns(labelled_features(c, EVAL_DET[ds]), BOX_SET[ds], pca))
    _, curve = train_curve(pd.read_parquet(TRAIN, columns=["score", "tp"]))
    tau = {lam: train_threshold(curve, lam) for lam in LAMS}
    say(f"eval features ready, {time.perf_counter() - t0:.0f} s")

    runs = [(name, cols, 1.0, 0) for name, cols in VARIANTS.items()]
    runs += [("model 2, training subset", FULL + YOLO_COLS, frac, seed) for frac in FRACTIONS for seed in SEEDS]
    rows, curves = [], []
    for name, cols, frac, seed in runs:
        t1 = time.perf_counter()
        if frac < 1:
            keep_img = subset_images(groups, frac, seed)
            f_tr = tr[keep_img[tr.img.to_numpy()]]
        else:
            f_tr = tr
        m = fit(f_tr, cols)
        t_fit = time.perf_counter() - t1
        curves.append(pd.DataFrame(dict(variant=name, fraction=frac, seed=seed, iteration=np.arange(len(m.train_score_)),
                                        train_logloss=-m.train_score_, val_logloss=-m.validation_score_)))
        res = dict(variant=name, fraction=frac, seed=seed, n_features=len(cols), train_images=int(f_tr.img.nunique()),
                   train_detections=len(f_tr), iterations=int(m.n_iter_), fit_seconds=round(t_fit, 1),
                   final_train_logloss=float(-m.train_score_[-1]), final_val_logloss=float(-m.validation_score_[-1]))
        for ds, (c, f) in evals.items():
            p = m.predict_proba(f[cols].to_numpy())[:, 1]
            if name.startswith("model 2 (adopted"):
                ref = pd.read_parquet(pred_path("bdd" if ds == "BDD100K" else "dawn")).p.to_numpy()
                res[f"{ds}_max_abs_diff_vs_adopted"] = float(np.abs(p - ref).max())
            res[f"{ds}_ece_points"] = 100 * calibration(p, f.tp.to_numpy())[1]
            ev = grade(ds, c, f, p, tau, LAMS)
            ev = ev[ev.policy == "per detection"]
            for (dsg, grp) in GROUPS:
                if dsg != ds:
                    continue
                for lam in LAMS:
                    x = ev[(ev.group == grp) & (ev.lam == lam)].iloc[0]
                    key = f"{ds}_{grp}_lam{lam}"
                    res.update({f"{key}_cost": x.cost_vs_one, f"{key}_lo": x.ci_lo, f"{key}_hi": x.ci_hi,
                                f"{key}_recall": x.recall, f"{key}_precision": x.precision, f"{key}_fppi": x.fppi})
        rows.append(res)
        say(f"{name} [{frac:.0%}, seed {seed}]: {res['iterations']} iterations, {t_fit:.0f} s fit; BDD100K lam 10 cost "
            f"{res['BDD100K_all_lam10_cost']:.4f} ({res['BDD100K_all_lam10_lo']:.4f}-{res['BDD100K_all_lam10_hi']:.4f}), "
            f"DAWN {res['DAWN_all_lam10_cost']:.4f}; ECE {res['BDD100K_ece_points']:.2f} / {res['DAWN_ece_points']:.2f} pt"
            f"; total {time.perf_counter() - t0:.0f} s")
    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUT / "dial_ablation.csv", index=False, float_format="%.6g")
    pd.concat(curves, ignore_index=True).to_csv(OUT / "dial_ablation_curves.csv", index=False, float_format="%.6g")
    (OUT / "dial_ablation.md").write_text(report(res_df), encoding="utf-8")
    say(report(res_df))
    say(f"done, {time.perf_counter() - t0:.0f} s")
    log.close()
    sys.stdout.flush()


def report(r: pd.DataFrame) -> str:
    lines = ["# Ablations of the per-detection model (Phase 4, exploratory)", "",
             "Generated by `python -m wad.dial_ablation`. Cost relative to the one-threshold dial at lambda = 10 "
             "(95% image bootstrap); lower is better; ECE in percentage points.", "",
             "| variant | train images | features | iterations | BDD100K val | BDD100K night | DAWN | DAWN fog | ECE BDD100K / DAWN |",
             "|---|---|---|---|---|---|---|---|---|"]
    for x in r.itertuples():
        d = x._asdict()

        def cell(key):
            return f"{d[key + '_cost']:.3f} ({d[key + '_lo']:.3f}-{d[key + '_hi']:.3f})"
        lab = x.variant if x.fraction == 1 else f"{x.variant} {x.fraction:.0%}, seed {x.seed}"
        lines.append(f"| {lab} | {x.train_images:,} | {x.n_features} | {x.iterations} | {cell('BDD100K_all_lam10')} | "
                     f"{cell('BDD100K_night_lam10')} | {cell('DAWN_all_lam10')} | {cell('DAWN_fog_lam10')} | "
                     f"{x.BDD100K_ece_points:.2f} / {x.DAWN_ece_points:.2f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
