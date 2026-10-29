"""EXPLORATORY: how many false alarms a threshold makes, estimated without the image's labels.
STATUS: supports the dial (wad/dial.py): both options' false alarms can be estimated without labels.

A threshold never knows which box is false. What it can know, from labelled TRAINING images only,
is the chance a box at a given score is real (isotonic fit of score -> real on BDD100K train
detections). On an unlabelled image the expected false alarms are then sum(1 - chance real) over
the kept boxes, and the expected real objects dropped are sum(chance real) over the dropped ones.
Val labels are used only to check these estimates. Same check for the per-detection model's
probability (wad/per_detection.py, also trained on BDD100K train).

Run: python -m wad.label_free_fp   (needs python -m wad.per_detection first)
     -> results/threshold/label_free_{calibration,fp,boxes}.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from wad.config import THRESH_CACHE
from wad.one_threshold import train_curve, train_threshold
from wad.config import THRESH_RESULTS as RESULTS

LAMS = (1, 10, 30)
BDD_GROUPS = {"day": ["clear", "cloudy", "rain", "snow"], "night": ["clear_night", "rain_night", "snow_night"],
              "rain_night": ["rain_night"]}


def fit_train() -> tuple:
    """Score -> chance real, and the train-chosen single threshold per dial setting."""
    iso, curve = train_curve()
    grid = np.geomspace(1e-3, 1, 4000)
    tau, rows = {}, []
    for lam in LAMS:
        tau[lam] = train_threshold(curve, lam)
        t_cal = float(grid[np.argmax(iso.predict(grid) >= 1 / (1 + lam))])  # chance real = 1/(1+lam)
        rows.append(dict(lam=lam, threshold_chosen_on_train=tau[lam], score_where_chance_real_is_1_over_1_plus_lam=t_cal))
    cal = pd.DataFrame(rows)
    pts = pd.DataFrame(dict(score=[0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75]))
    pts["chance_real_on_train"] = iso.predict(pts.score.to_numpy())
    return iso, tau, pd.concat([cal, pts], ignore_index=True)


def check(iso, tau: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    from wad.curves import Cache
    from wad.one_threshold import eval_configs
    from wad.per_detection import pred_path

    # every evaluation image with its weather, including images without any detection
    images = {("bdd" if ds == "BDD100K" else "dawn"): (lambda c: pd.Series(c.weather, index=c.image_ids))(Cache(v))
              for ds, v in eval_configs().items()}
    rows, boxes = [], []
    for key in ("bdd", "dawn"):
        d = pd.read_parquet(pred_path(key)).rename(columns={"p": "p_det"})  # the adopted per-detection model
        d["w"] = d.image_id.str.split("/").str[0]
        d["q"] = iso.predict(d.score.to_numpy())
        boxes.append(dict(data=key, auc_score=roc_auc_score(d.tp, d.score), auc_per_detection=roc_auc_score(d.tp, d.p_det)))
        groups = BDD_GROUPS if key == "bdd" else {w: [w] for w in sorted(d.w.unique())}
        for g, ws in groups.items():
            e = d[d.w.isin(ws)]
            ids = images[key].index[images[key].isin(ws)]
            n_img = len(ids)
            assert set(e.image_id) <= set(ids)
            for lam in LAMS:
                for mode, prob, keep in (("one threshold", e.q.to_numpy(), (e.score >= tau[lam]).to_numpy()),
                                         ("per-detection", e.p_det.to_numpy(), (e.p_det >= 1 / (1 + lam)).to_numpy())):
                    k, fp = e[keep], 1 - e.tp.to_numpy()[keep]
                    per = pd.DataFrame(dict(img=k.image_id, fp=fp, efp=1 - prob[keep], n=1)).groupby("img").sum().reindex(ids, fill_value=0)
                    rows.append(dict(data=key, cond=g, lam=lam, mode=mode, images=n_img,
                                     fp_per_img=fp.sum() / n_img, est_fp_per_img=(1 - prob[keep]).sum() / n_img,
                                     dropped_real_per_img=e.tp.to_numpy()[~keep].sum() / n_img,
                                     est_dropped_real_per_img=prob[~keep].sum() / n_img,
                                     spearman_est_vs_actual_fp=spearmanr(per.efp, per.fp)[0],
                                     spearman_kept_count_vs_actual_fp=spearmanr(per.n, per.fp)[0],
                                     auc_among_kept=roc_auc_score(k.tp, prob[keep]) if 0 < fp.sum() < len(fp) else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(boxes)


def main():
    iso, tau, cal = fit_train()
    fp, boxes = check(iso, tau)
    for name, t in (("calibration", cal), ("fp", fp), ("boxes", boxes)):
        t.to_csv(RESULTS / f"label_free_{name}.csv", index=False)
    pd.set_option("display.width", 250)
    print(cal.round(4).to_string(index=False))
    print(fp.round(3).to_string(index=False))
    print(boxes.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
