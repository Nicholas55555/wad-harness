"""The interactive trade-off dial page (results/threshold/dial_demo/index.html).

The dial lam is "false alarms accepted to avoid one miss", with two options, both fitted on BDD100K
train and graded here on BDD100K val and DAWN from the cached YOLOv8x @1280 detections (score >= 0.001,
grouped matching):
  one threshold     one confidence threshold per lam, chosen on train (wad/one_threshold.py), used in
                    every condition
  per detection     keep a detection iff the per-detection model's chance it is real >= 1 / (1 + lam)
                    (wad/per_detection.py)
Every number per condition and setting comes from wad.per_detection.grade, the function behind the
adopted per-detection model's results/threshold/per_detection{,_yolo}_eval.csv (recall, precision, false
alarms per image, cost relative to one threshold with 95% image-bootstrap intervals, and no tuning =
0.25), so the page and the tables agree. Also, exactly: the condition's own best constant (in-sample), the detections kept, the
precision-recall curves, the false alarms estimated without labels, and the calibration right at the
cut (boxes within x1.5 of the threshold or cut: share real vs what the dial assumes). For the selected
example frame the page also scores the dial against the best possible on that frame with hindsight
(F1 / F-beta, precision, recall, why it is off, its F curve), and frame_gaps puts that in context across
every frame of the condition. F1 and a tunable
F-beta are computed on the page from exact counts; the F-beta dial is the lam dial at beta's equivalent
lam (wad/fbeta_dial.py, weights/fbeta_table.csv, fitted on the BDD100K train images), so picking beta
picks lam without looking at the graded images. The slider covers
lam 0.3 to 300 (61 log steps) plus the presets and lam = 3 exactly. Two example frames per condition
with every box (score, model chance, real or false) and the objects.

Run: python -m wad.dial_demo   (needs python -m wad.one_threshold and python -m wad.per_detection first)
     -> results/threshold/dial_demo/index.html
"""
from __future__ import annotations

import base64
import io
import json

import numpy as np
import pandas as pd

from wad.config import REPO, THRESH_CACHE
from wad.config import THRESH_RESULTS as RESULTS
from wad.curves import ALL, Cache
from wad.dial import PRESETS as NAMED
from wad.one_threshold import DIAL, PRESETS, TRAIN, eval_configs

OUT = RESULTS / "dial_demo"
TEMPLATE = REPO / "wad" / "assets" / "dial_demo.html"
LAMS = np.union1d(DIAL, PRESETS)  # the dial's 61 log steps plus the graded settings 1, 3, 10, 30, 100 exactly
CONDITIONS = [("BDD100K", "day", "Day", ["clear", "cloudy", "rain", "snow"]),
              ("BDD100K", "night", "Night", ["clear_night", "rain_night", "snow_night"]),
              ("BDD100K", "rain_night", "Rain at night", ["rain_night"]),
              ("DAWN", "fog", "Fog", ["fog"])]
EXAMPLE_WEATHER = {"day": "clear", "night": "clear_night", "rain_night": "rain_night", "fog": "fog"}
NEAR = 1.5  # calibration window around the threshold / cut, as in wad.per_detection.near_cuts
POLICY = {"one threshold (train)": "one", "per detection": "det", "no tuning (0.25)": "none"}


def stats(tp: float, fp: float, n_gt: float, n_img: int) -> dict:
    return dict(recall=tp / n_gt, precision=tp / (tp + fp) if tp + fp else 1.0, fppi=fp / n_img)


def near(value: np.ndarray, at: float, chance: np.ndarray, tp: np.ndarray) -> dict:
    """Boxes whose score (one threshold) or chance (per detection) is within x1.5 of the cut: how often
    they are real, against the chance the dial assumes for them."""
    m = (value >= at / NEAR) & (value <= min(at * NEAR, 1.0))
    return dict(n=int(m.sum()), expected=float(chance[m].mean()) if m.any() else None,
                real=float(tp[m].mean()) if m.any() else None)


def curves(c: Cache, rows: np.ndarray, p: np.ndarray, q: np.ndarray, tau: dict, graded: pd.DataFrame) -> dict:
    """Dial table for one condition: per lam, the three policies as graded, the condition's own best
    constant, kept counts, the no-label estimate and the calibration at the cut; plus the curves."""
    n_gt, n_img = float(c.n_gt(ALL)[rows].sum()), int(rows.sum())
    det_rows = np.isin(c.pred.image_id.map(c.idx).to_numpy(), np.flatnonzero(rows))
    tp = c.pred.tp.to_numpy().astype(bool)[det_rows]
    sc, pp, qq = c.pred.score.to_numpy()[det_rows], p[det_rows], q[det_rows]
    o1 = np.argsort(-sc, kind="stable")
    c1tp, c1fp = np.cumsum(tp[o1]), np.cumsum(~tp[o1])
    g = graded.set_index(["lam", "policy"])
    r4 = lambda v: None if v is None else round(float(v), 4)  # noqa: E731
    dial = []
    for lam in LAMS:
        row = dict(lam=float(lam), tau=float(tau[lam]), cut=float(1 / (1 + lam)))
        kept = {"one": sc >= tau[lam], "det": pp >= 1 / (1 + lam)}
        for name, key in POLICY.items():
            r = g.loc[(lam, name)]
            d = {k: r4(r[k]) for k in ("recall", "precision", "fppi", "cost_vs_one", "ci_lo", "ci_hi")}
            if key in kept:
                k = kept[key]
                assert abs(d["recall"] - (k & tp).sum() / n_gt) < 1e-4  # same detections as graded
                d.update(kept_real=int((k & tp).sum()), kept_false=int((k & ~tp).sum()),
                         est_fppi=r4((1 - (qq if key == "one" else pp)[k]).sum() / n_img))
            row[key] = d
        k0 = sc >= 0.25
        row["none"].update(tp=int((k0 & tp).sum()), fp=int((k0 & ~tp).sum()))
        for key in ("one", "det"):
            row[key].update(tp=row[key]["kept_real"], fp=row[key]["kept_false"])
        row["one"]["near"] = {k: r4(v) for k, v in near(sc, tau[lam], qq, tp).items()}
        row["det"]["near"] = {k: r4(v) for k, v in near(pp, 1 / (1 + lam), pp, tp).items()}
        j = int(np.argmin(c1fp - lam * c1tp))  # the condition's own best constant, exact (in-sample)
        best = stats(float(c1tp[j]), float(c1fp[j]), n_gt, n_img)
        cost_one = lam * (n_gt - (kept["one"] & tp).sum()) + (kept["one"] & ~tp).sum()
        best["cost_vs_one"] = (lam * (n_gt - c1tp[j]) + c1fp[j]) / cost_one
        row["best"] = {k: r4(v) for k, v in best.items()}
        row["best"].update(tp=int(c1tp[j]), fp=int(c1fp[j]))
        row["tau_best"] = float(sc[o1][j])
        dial.append(row)
    # precision-recall curves over every cut (for the chart), exact on the sorted detections
    idx1 = np.unique(np.geomspace(1, len(o1), 300).astype(int)) - 1
    pr_one = [stats(float(c1tp[i]), float(c1fp[i]), n_gt, n_img) for i in idx1]
    o = np.argsort(-pp, kind="stable")
    ctp, cfp = np.cumsum(tp[o]), np.cumsum(~tp[o])
    idx = np.unique(np.geomspace(1, len(o), 300).astype(int)) - 1
    pr_det = [stats(float(ctp[i]), float(cfp[i]), n_gt, n_img) for i in idx]
    pick = lambda L: [[round(d["recall"], 4), round(d["precision"], 4)] for d in L]  # noqa: E731

    def histo(v: np.ndarray, lo: float) -> dict:
        e = np.geomspace(lo, 1.0 + 1e-9, 61)
        return dict(edges=[float(x) for x in e[:-1]] + [1.0], real=np.histogram(np.clip(v[tp], lo, 1), e)[0].tolist(),
                    false=np.histogram(np.clip(v[~tp], lo, 1), e)[0].tolist())
    # what the per-detection cut removes (model chance, 1e-4 .. 1) and what one threshold cuts (score, 1e-3 .. 1)
    return dict(dial=dial, pr_one=pick(pr_one), pr_det=pick(pr_det), hist=histo(pp, 1e-4), hist_score=histo(sc, 1e-3),
                images=n_img, objects=int(n_gt), detections=int(len(pp)), real=int(tp.sum()), false=int((~tp).sum()))


FPOL = {"one threshold (F-beta dial)": "one", "per detection (F-beta dial)": "det", "no tuning (0.25)": "none",
        "own best constant (in-sample)": "best"}


def fcurves(c: Cache, rows: np.ndarray, p: np.ndarray, q: np.ndarray, tab: pd.DataFrame, graded: pd.DataFrame) -> list:
    """The dial in F-beta units for one condition: per beta (wad.fbeta_dial.BETAS) the four policies as
    graded by wad.fbeta_dial.grade (behind fbeta_eval.csv), plus kept counts, the no-label estimate and
    the calibration at the cut."""
    from wad.fbeta_dial import BETAS, lookup

    n_img = int(rows.sum())
    det_rows = np.isin(c.pred.image_id.map(c.idx).to_numpy(), np.flatnonzero(rows))
    tp = c.pred.tp.to_numpy().astype(bool)[det_rows]
    sc, pp, qq = c.pred.score.to_numpy()[det_rows], p[det_rows], q[det_rows]
    g = graded.set_index(["beta", "policy"])
    r4 = lambda v: None if v is None else round(float(v), 4)  # noqa: E731
    out = []
    for b in BETAS:
        t = lookup(tab, b)
        row = dict(beta=float(b), tau=float(t.threshold), cut=float(t.cut), lam_one=r4(t.one_lam_equivalent),
                   lam_det=r4(t.det_lam_equivalent), train_f_one=r4(t.one_train_f), train_f_det=r4(t.det_train_f))
        for name, key in FPOL.items():
            r = g.loc[(b, name)]
            row[key] = {k: r4(r[k]) for k in ("recall", "precision", "fppi", "f1", "fbeta", "fbeta_ci_lo", "fbeta_ci_hi",
                                              "fbeta_vs_own_best")}
            if key == "best":
                row["tau_best"] = float(r.cut_or_threshold)
        for key, kept, chance, val, at in (("one", sc >= t.threshold, qq, sc, t.threshold), ("det", pp >= t.cut, pp, pp, t.cut)):
            assert abs(row[key]["recall"] - (kept & tp).sum() / c.n_gt(ALL)[rows].sum()) < 1e-4
            row[key].update(kept_real=int((kept & tp).sum()), kept_false=int((kept & ~tp).sum()),
                            tp=int((kept & tp).sum()), fp=int((kept & ~tp).sum()),
                            est_fppi=r4((1 - chance[kept]).sum() / n_img),
                            near={k: r4(v) for k, v in near(val, at, chance, tp).items()})
        for key, kept in (("none", sc >= 0.25), ("best", sc >= row["tau_best"])):
            row[key].update(tp=int((kept & tp).sum()), fp=int((kept & ~tp).sum()))
            assert abs(fbeta_counts(row[key], c.n_gt(ALL)[rows].sum(), b) - row[key]["fbeta"]) < 2e-4
        out.append(row)
    return out


def fbeta_counts(m: dict, n_obj: float, beta: float) -> float:
    b2 = beta * beta
    return (1 + b2) * m["tp"] / (b2 * n_obj + m["tp"] + m["fp"]) if m["tp"] else 0.0


def frame_gaps(c: Cache, rows: np.ndarray, p: np.ndarray, tau: dict, ftab: pd.DataFrame) -> dict:
    """Across the condition's frames (with at least one labelled object): how far the dial's F on each
    frame is from that frame's own best F with hindsight (the best 'keep value >= t' on the frame's own
    labels). Per setting and option: 21 quantiles of the gap (dial - best), the share within 0.05, and
    the average per-frame F with the dial and with each frame's best. lam settings use F1; beta settings
    use F_beta at that beta."""
    from wad.fbeta_dial import BETAS, lookup

    n_gt_all = c.n_gt(ALL)
    imgs = np.flatnonzero(rows & (n_gt_all > 0))
    loc = np.full(c.n_img, -1)
    loc[imgs] = np.arange(len(imgs))
    img = loc[c.pred.image_id.map(c.idx).to_numpy()]
    d = img >= 0
    img, tp = img[d], c.pred.tp.to_numpy().astype(bool)[d]
    n = n_gt_all[imgs].astype(float)
    vals = {"one": c.pred.score.to_numpy()[d], "det": p[d]}
    qs = np.linspace(0, 1, 21)

    def f_of(t, k, b):
        return np.where(t > 0, (1 + b * b) * t / (b * b * n + k), 0.0)

    def best(v, betas):  # per frame, per beta: the highest F over 'keep v >= t' (keeping nothing = 0)
        o = np.lexsort((-v, img))
        vi, ti, ii = v[o], tp[o], img[o]
        start = np.r_[True, ii[1:] != ii[:-1]]
        seg = np.flatnonzero(start)
        ctp = np.cumsum(ti) - np.repeat(np.cumsum(ti)[seg] - ti[seg], np.diff(np.r_[seg, len(ti)]))
        cnt = np.arange(len(ti)) - np.repeat(seg, np.diff(np.r_[seg, len(ti)])) + 1
        end = np.r_[(vi[1:] != vi[:-1]) | (ii[1:] != ii[:-1]), True]  # cannot split equal values within a frame
        out = {}
        for b in betas:
            f = np.where(end, (1 + b * b) * ctp / (b * b * n[ii] + cnt), 0.0)
            bf = np.zeros(len(imgs))
            np.maximum.at(bf, ii, f)
            out[b] = bf
        return out

    def summary(gap, fd, fb):
        return dict(q=[round(float(x), 4) for x in np.quantile(gap, qs)], within05=round(float((gap >= -0.05).mean()), 4),
                    mean_dial=round(float(fd.mean()), 4), mean_best=round(float(fb.mean()), 4))

    out = dict(frames=int(len(imgs)), lam={}, beta={})
    for key, v in vals.items():
        bests = best(v, [1.0] + [float(b) for b in BETAS])
        lam_rows, beta_rows = [], []
        for lam in LAMS:
            k = v >= (tau[lam] if key == "one" else 1 / (1 + lam))
            t = np.bincount(img, weights=k & tp, minlength=len(imgs)); kk = np.bincount(img, weights=k, minlength=len(imgs))
            fd = f_of(t, kk, 1.0)
            lam_rows.append(summary(fd - bests[1.0], fd, bests[1.0]))
        for b in BETAS:
            r = lookup(ftab, b)
            k = v >= (r.threshold if key == "one" else r.cut)
            t = np.bincount(img, weights=k & tp, minlength=len(imgs)); kk = np.bincount(img, weights=k, minlength=len(imgs))
            fd = f_of(t, kk, float(b))
            beta_rows.append(summary(fd - bests[float(b)], fd, bests[float(b)]))
        out["lam"][key], out["beta"][key] = lam_rows, beta_rows
    return out


def train_curves(tau: dict) -> dict:
    """Recall and precision on the BDD100K train images at every dial setting, both options: the
    one threshold on all train detections, per detection on the 5-fold out-of-fold chances (each from
    the fold model that did not see that image). The page draws them (dashed) in its F-beta chart."""
    m = pd.read_parquet(TRAIN, columns=["score", "tp"])
    from wad.per_detection import oof_path

    oof = pd.read_parquet(oof_path())
    n_gt = float(pd.read_parquet(THRESH_CACHE / "images.parquet", columns=["n_gt"]).n_gt.sum())
    s, y = m.score.to_numpy(), m.tp.to_numpy().astype(bool)
    po, yo = oof.p.to_numpy(), oof.tp.to_numpy().astype(bool)
    assert len(oof) == len(m) and yo.sum() == y.sum() and not np.isnan(po).any()
    out = dict(objects=int(n_gt), one=[], det=[])
    for lam in LAMS:
        for key, kept, yy in (("one", s >= tau[lam], y), ("det", po >= 1 / (1 + lam), yo)):
            t = float(yy[kept].sum())
            out[key].append([round(t / n_gt, 5), round(t / max(int(kept.sum()), 1), 5)])
    return out


def image_b64(path: str, max_w=960) -> tuple[str, float]:
    from wad.imageio import load_rgb

    im = load_rgb(REPO / path, "apply")
    s = min(1.0, max_w / im.size[0])
    im = im.resize((round(im.size[0] * s), round(im.size[1] * s)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), s


def example(c: Cache, ds: str, iid: str, p: np.ndarray, det: pd.DataFrame) -> dict:
    sel = (c.pred.image_id == iid).to_numpy()
    pr = c.pred[sel].reset_index(drop=True)
    d = det[det.image_id == iid].reset_index(drop=True)
    assert len(d) == len(pr) and np.allclose(d.score.to_numpy(), pr.score.to_numpy())
    path = c.images.set_index("image_id").loc[iid, "path"]
    src, s = image_b64(path)
    gt = c.gt[c.gt.image_id == iid]
    gtb = pd.read_parquet(REPO / ("cache/bdd/gt.parquet" if ds == "BDD100K" else "cache/gt.parquet"))
    gtb = gtb[gtb.gt_id.isin(gt.gt_id) & (gtb.image_id == iid)].set_index("gt_id").loc[gt.gt_id]
    match = {g: i for i, g in enumerate(pr.gt_id.to_numpy()) if g >= 0}
    boxes = [[round(float(v) * s, 1) for v in (b.x1, b.y1, b.x2, b.y2)] + [round(float(b.score), 4), round(float(q), 4), int(t), b.cls]
             for b, q, t in zip(d.itertuples(), p[sel], pr.tp.to_numpy())]
    objects = [[round(float(v) * s, 1) for v in (o.x1, o.y1, o.x2, o.y2)] + [match.get(g, -1), o.cls]
               for g, o in zip(gt.gt_id.to_numpy(), gtb.itertuples())]
    return dict(id=iid, src=src, boxes=boxes, objects=objects)


def pick_images(c: Cache, weather: str, n=2) -> list[str]:
    """Random images of the condition with 5-20 labelled objects (seed 0; the first is the one used in
    the earlier example figures)."""
    from wad.show_boxes import pick

    first = pick("BDD100K" if c.cfg.dataset_name == "bdd" else "DAWN", weather).image_id
    im = c.images[(c.weather == weather) & c.images.n_gt.between(5, 20).to_numpy()].sort_values("image_id")
    rest = [x for x in im.image_id if x != first]
    extra = list(np.random.default_rng(1).choice(rest, n - 1, replace=False))
    return [first] + extra


def build() -> dict:
    from wad.fbeta_dial import BETAS
    from wad.fbeta_dial import grade as fgrade
    from wad.one_threshold import train_curve, train_threshold
    from wad.per_detection import grade, pred_path

    iso, tcurve = train_curve()
    tau = {lam: train_threshold(tcurve, lam) for lam in LAMS}
    ftab = pd.read_csv(REPO / "weights" / "fbeta_table.csv", float_precision="round_trip")
    caches = {ds: Cache(v) for ds, v in eval_configs().items()}
    probs, dets, graded, fgraded = {}, {}, {}, {}
    for ds, key in (("BDD100K", "bdd"), ("DAWN", "dawn")):
        pr = pd.read_parquet(pred_path(key))  # the adopted per-detection model
        c = caches[ds]
        assert len(pr) == len(c.pred) and (pr.image_id.to_numpy() == c.pred.image_id.to_numpy()).all()
        assert np.array_equal(pr.score.to_numpy(), c.pred.score.to_numpy())
        probs[ds] = pr.p.to_numpy()
        f = pd.DataFrame(dict(tp=c.pred.tp.to_numpy().astype(int), img=c.pred.image_id.map(c.idx).to_numpy(),
                              score=c.pred.score.to_numpy()))
        graded[ds] = grade(ds, c, f, probs[ds], tau, LAMS)  # the same grading as per_detection_eval.csv
        fgraded[ds] = fgrade(ds, c, c.pred.score.to_numpy(), probs[ds], ftab, BETAS)  # as fbeta_eval.csv
        det = pd.read_parquet(REPO / ("cache/bdd/det/v8x_1280/detections.parquet" if ds == "BDD100K"
                                      else "cache/det/v8x_1280/detections.parquet"))
        det = det[det.image_id.isin(c.idx)]
        dets[ds] = det.iloc[np.argsort(det.image_id.map(c.idx).to_numpy(), kind="stable")].reset_index(drop=True)
    conds = []
    for ds, key, label, weathers in CONDITIONS:
        c = caches[ds]
        rows = np.isin(c.weather, weathers)
        gr = graded[ds][graded[ds].group == key]
        assert gr.images.iloc[0] == rows.sum()
        q = iso.predict(c.pred.score.to_numpy())
        cur = curves(c, rows, probs[ds], q, tau, gr)
        cur["fdial"] = fcurves(c, rows, probs[ds], q, ftab, fgraded[ds][fgraded[ds].group == key])
        cur["gaps"] = frame_gaps(c, rows, probs[ds], tau, ftab)
        ex = [example(c, ds, iid, probs[ds], dets[ds]) for iid in pick_images(c, EXAMPLE_WEATHER[key])]
        conds.append(dict(key=key, label=label, dataset=ds, **cur, examples=ex))
    chance = [dict(score=x, real=round(float(iso.predict([x])[0]), 4)) for x in (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75)]
    presets = [dict(label=k[0].upper() + k[1:], lam=v) for k, v in NAMED.items()]
    from wad.dial import F_PRESETS

    fpresets = [dict(label=k, beta=v) for k, v in F_PRESETS.items()]
    return dict(lams=[float(x) for x in LAMS], presets=presets, conditions=conds, chance=chance, default=0.25,
                train=train_curves(tau), betas=[float(b) for b in BETAS], fpresets=fpresets)


def main():
    data = build()
    OUT.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    (OUT / "index.html").write_text(html, encoding="utf-8")
    (OUT / "data_summary.json").write_text(json.dumps({c["label"]: {k: c[k] for k in ("images", "objects")} for c in data["conditions"]},
                                                      indent=1), encoding="utf-8")
    print(OUT / "index.html", f"{len(html) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
