"""THE DIAL'S PER-DETECTION OPTION (user decision 2026-10-14): the dial lam ("false alarms accepted to
avoid one miss") applied to each detection instead of through one confidence threshold.

A gradient-boosted model, trained on the BDD100K train detections (21,831 images, YOLOv8x @1280, boxes
>= 0.001, grouped matching), gives every detection its chance of being a real object from YOLOv8x's
own outputs only: score and class, box size / shape / position, rank in the image, overlap with
stronger boxes (likely duplicates), confidence-weighted support from overlapping boxes and how crowded
the image is. A detection is kept iff that chance >= 1 / (1 + lam). When the chances are calibrated
this rule minimises the expected lam * misses + false alarms detection by detection, so a dial setting
means the same in both options (wad/one_threshold.py is the other) and both are fitted on the same
train images; nothing is chosen on BDD100K val or DAWN.

Statistical backing (results/threshold/per_detection_*.csv, figures/fig21_per_detection):
  calibration   predicted chance vs share real on BDD100K val per condition and DAWN per weather
                (never seen in training), in bins and right around the cut 1/(1+lam) at each graded
                setting (lam = 1, 3, 10, 30, 100: the four named presets plus 3)
  stability     5-fold by near-duplicate group inside train: out-of-fold calibration and cost (vs the
                one threshold fitted on the other folds), and the five fold models' spread on val / DAWN
  grading       recall, precision, false alarms per image and cost vs the one threshold (train-chosen)
                and no tuning (0.25), with 95% image-bootstrap intervals
The model's early stopping uses a random 10% of the train detections (boxes of one image can fall on
both sides; train only). Its features and settings were picked in exploratory runs of
wad/detection_rescore.py that were scored on BDD100K val / DAWN (alternatives within ~0.005), so
"nothing chosen on val" holds for the fit, not for that earlier model choice.
Evaluation keeps each detection's real / false label from the matching of all boxes >= 0.001 (as
wad/detection_rescore.py): if the model drops a matched box but keeps a weaker duplicate of the same
object, the duplicate still counts as false, so the per-detection numbers are slightly pessimistic.

Two models (PREREG_threshold.md, "Per-detection option, model 2"): model 1 ("context") reads the 20 context
features; model 2 ("yolo") adds YOLOv8x's own per-box features (wad/yolo_box_features.py): the class
probabilities, the box sides' DFL uncertainty, the pyramid level and the neck feature vector at the box's
anchor (32 principal components fitted on train). ADOPTED says which one the dial uses.

Model: weights/per_detection.joblib (model 1) / per_detection_yolo.joblib (model 2): the sklearn model, the
feature list and metadata (model 2 also its PCA). To use it on new detections: score(det, sizes[, box])
-> chance per detection (model 2 needs the per-box features: wad.yolo_box_features.Detector);
keep = chance >= 1 / (1 + lam); wad/dial.py wraps both options.
Run: python -m wad.per_detection [--features context|yolo]   (model 2 outputs carry "_yolo" and add
     results/threshold/per_detection_yolo_{vs_model1,criteria,importance}.csv)
     -> weights/per_detection.joblib, cache/thresh/per_detection_pred_{bdd,dawn}.parquet,
     cache/thresh/per_detection_oof_train.parquet (5-fold out-of-fold chances on train),
     results/threshold/per_detection_{calibration,cuts,eval,folds}.csv, figures/fig21_per_detection.png
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from wad.config import REPO, THRESH_CACHE
from wad.config import THRESH_RESULTS as RESULTS
from wad.one_threshold import DEFAULT, DIAL, PRESETS, TRAIN, eval_configs, train_curve, train_threshold

FEATURE_SETS = ("context", "yolo")
ADOPTED = "yolo"  # the model the dial uses: model 2 met the PREREG_threshold.md criteria on 2026-10-15
TRAIN_DET = "cache/thresh/det/v8x_1280/detections.parquet"
EVAL_DET = {"BDD100K": "cache/bdd/det/v8x_1280/detections.parquet", "DAWN": "cache/det/v8x_1280/detections.parquet"}
CLS = {"car": 0, "truck": 1, "bus": 2, "bicycle": 3, "motorcycle": 4, "person": 5}
GRP = {"car": 0, "truck": 0, "bus": 0, "bicycle": 1, "motorcycle": 1, "person": 2}
FULL = ["logit", "cls", "log_h", "log_w", "aspect", "x_center", "y_bottom", "y_top", "log_rank", "log_rank_group",
        "max_iou_stronger_same_group", "n_stronger_overlapping", "max_iou_stronger_any", "img_log_n_025", "img_log_n_005",
        "img_log_n_001", "img_max_score", "img_group_log_n_025", "support_same_group", "support_any"]
PARAMS = dict(max_iter=400, learning_rate=0.1, max_leaf_nodes=63, early_stopping=True, validation_fraction=0.1, random_state=0)
BDD_GROUPS = {"day": ["clear", "cloudy", "rain", "snow"], "night": ["clear_night", "rain_night", "snow_night"]}
BINS = np.array([0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0 + 1e-9])
COCO = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
N_PC = 32
YOLO_COLS = ([f"p_{k}" for k in COCO] + ["p_other_max", "p_top_margin", "p_sum", "p_entropy"]
             + [f"dfl_ent_{side}" for side in ("left", "top", "right", "bottom")] + ["dfl_ent_mean", "level"]
             + [f"neck_pc{i:02d}" for i in range(N_PC)])
BOX_SET = {"train": "train", "BDD100K": "bdd", "DAWN": "dawn"}  # wad.yolo_box_features set names


def cols_of(features: str) -> list:
    return FULL if features == "context" else FULL + YOLO_COLS


def _tag(features: str | None) -> str:
    features = features or ADOPTED
    assert features in FEATURE_SETS, features
    return "" if features == "context" else "_yolo"


def model_path(features: str | None = None):
    return REPO / "weights" / f"per_detection{_tag(features)}.joblib"


def pred_path(key: str, features: str | None = None):
    """Per-detection chances on BDD100K val ('bdd') or DAWN ('dawn'), in wad.curves.Cache pred order."""
    return THRESH_CACHE / f"per_detection{_tag(features)}_pred_{key}.parquet"


def oof_path(features: str | None = None):
    """5-fold out-of-fold chances on the train detections."""
    return THRESH_CACHE / f"per_detection{_tag(features)}_oof_train.parquet"


MODEL = model_path()  # the adopted model


# ---- features from the detector's outputs only (no labels) -----------------------------------------

def iou_rows(b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of boxes (n, 4)."""
    x1 = np.maximum(b[:, None, 0], b[None, :, 0])
    y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2])
    y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-9)


def features_raw(det: pd.DataFrame, sizes: pd.DataFrame) -> pd.DataFrame:
    """One feature row per detection, in det's order. det: image_id, cls (name), score, x1, y1, x2, y2
    (pixels), with each image's detections in consecutive rows; sizes: width and height in pixels, indexed
    by image_id (or with an image_id column). Also computes the two box-combination scores (softmax share,
    noisy-OR) used by wad.detection_rescore."""
    sizes = sizes.set_index("image_id") if "image_id" in sizes.columns else sizes
    ids = det.image_id.to_numpy()
    if len(det) == 0:
        return pd.DataFrame(columns=FULL + ["softmax_share", "noisy_or", "score"], dtype=float)
    unknown = sorted(set(det.cls) - set(CLS))
    if unknown:
        raise ValueError(f"unknown classes {unknown}; expected class names {sorted(CLS)}")
    missing = sorted(set(ids) - set(sizes.index))
    if missing:
        raise ValueError(f"no width / height for {len(missing)} images, e.g. {missing[:3]}")
    H = sizes.height.loc[ids].to_numpy().astype(float)
    W = sizes.width.loc[ids].to_numpy().astype(float)
    b = det[["x1", "y1", "x2", "y2"]].to_numpy(float)
    s = det.score.to_numpy()
    cls = det.cls.map(CLS).to_numpy()
    grp = det.cls.map(GRP).to_numpy()
    h, w = (b[:, 3] - b[:, 1]) / H, (b[:, 2] - b[:, 0]) / W
    f = pd.DataFrame(dict(logit=np.log(s / (1 - s + 1e-9) + 1e-9), cls=cls, log_h=np.log(np.clip(h, 1e-4, None)),
                          log_w=np.log(np.clip(w, 1e-4, None)), aspect=np.log(np.clip(w * W, 1e-3, None) / np.clip(h * H, 1e-3, None)),
                          x_center=(b[:, 0] + b[:, 2]) / 2 / W, y_bottom=b[:, 3] / H, y_top=b[:, 1] / H))
    n = len(det)
    rank, rank_g = np.zeros(n), np.zeros(n)
    m_same, n_over, m_any = np.zeros(n), np.zeros(n), np.zeros(n)
    sup_same, sup_any = np.zeros(n), np.zeros(n)  # confidence-weighted overlapping area from all other boxes
    share, nor = np.zeros(n), np.zeros(n)
    img_cols = {k: np.zeros(n) for k in ("img_log_n_025", "img_log_n_005", "img_log_n_001", "img_max_score", "img_group_log_n_025")}
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    if len(starts) != len(set(ids)):
        raise ValueError("each image's detections must be in consecutive rows")
    ends = np.r_[starts[1:], n]
    for a, z in zip(starts, ends):
        sl = slice(a, z)
        ss, gg = s[sl], grp[sl]
        order = np.argsort(-ss, kind="stable")
        r = np.empty(len(ss))
        r[order] = np.arange(len(ss))
        rank[sl] = r
        rg = np.zeros(len(ss))
        for g in np.unique(gg):
            mm = np.flatnonzero(gg == g)
            o = mm[np.argsort(-ss[mm], kind="stable")]
            rg[o] = np.arange(len(o))
        rank_g[sl] = rg
        iou = iou_rows(b[sl])
        stronger = ss[None, :] > ss[:, None]  # column detection is stronger than row detection
        same = gg[:, None] == gg[None, :]
        m_same[sl] = np.where(stronger & same, iou, 0).max(1, initial=0)
        n_over[sl] = (stronger & same & (iou > 0.5)).sum(1)
        m_any[sl] = np.where(stronger, iou, 0).max(1, initial=0)
        wgt = np.where(same, iou, 0.0)  # overlap weights within the class group
        np.fill_diagonal(wgt, 1.0)  # a box counts fully for itself (zero-area boxes would get 0/0 otherwise)
        lg = np.log(np.clip(ss, 1e-6, 1 - 1e-6) / (1 - np.clip(ss, 1e-6, 1 - 1e-6)))
        ez = np.exp(lg - lg.max())
        share[sl] = ez / (wgt * ez[None, :]).sum(1)
        nor[sl] = 1 - np.exp((np.log1p(-np.clip(wgt * ss[None, :], 0, 1 - 1e-9))).sum(1))
        np.fill_diagonal(iou, 0)
        sup_same[sl] = (np.where(same, iou, 0) * ss[None, :]).sum(1)
        sup_any[sl] = (iou * ss[None, :]).sum(1)
        img_cols["img_log_n_025"][sl] = np.log1p((ss >= 0.25).sum())
        img_cols["img_log_n_005"][sl] = np.log1p((ss >= 0.05).sum())
        img_cols["img_log_n_001"][sl] = np.log1p((ss >= 0.01).sum())
        img_cols["img_max_score"][sl] = ss.max()
        img_cols["img_group_log_n_025"][sl] = np.log1p(np.array([(ss[gg == g] >= 0.25).sum() for g in gg]))
    f["log_rank"], f["log_rank_group"] = np.log1p(rank), np.log1p(rank_g)
    f["max_iou_stronger_same_group"], f["n_stronger_overlapping"], f["max_iou_stronger_any"] = m_same, n_over, m_any
    f["support_same_group"], f["support_any"] = sup_same, sup_any
    f["softmax_share"], f["noisy_or"] = share, nor
    for k, v in img_cols.items():
        f[k] = v
    f["score"] = s
    return f


def labelled_features(c, det_path: str) -> pd.DataFrame:
    """Features for the detections of a matched set (wad.curves.Cache), in c.pred order, plus their
    real / false label (tp) and image index (img)."""
    det = pd.read_parquet(REPO / det_path)
    det["row"] = np.arange(len(det))  # position in the detections file (the per-box features follow it)
    det = det[det.image_id.isin(c.idx)]
    pos = det.image_id.map(c.idx).to_numpy()
    det = det.iloc[np.argsort(pos, kind="stable")].reset_index(drop=True)  # images in Cache order, rows as matched
    p = c.pred.reset_index(drop=True)
    assert len(det) == len(p) and (det.image_id.to_numpy() == p.image_id.to_numpy()).all()
    assert np.allclose(det.score.to_numpy(), p.score.to_numpy()) and (det.cls.to_numpy() == p.cls.to_numpy()).all()
    f = features_raw(det, c.images.set_index("image_id")[["width", "height"]])
    f["tp"] = p.tp.to_numpy().astype(int)
    f["img"] = p.image_id.map(c.idx).to_numpy()
    f["row"] = det.row.to_numpy()
    return f


# ---- YOLOv8x's own per-box features (model 2) ----------------------------------------------------------

def box_columns(cls: np.ndarray, dfl: np.ndarray, neck: np.ndarray, level: np.ndarray, pca) -> pd.DataFrame:
    """Model 2's per-box columns from the head outputs at the box's anchor (wad.yolo_box_features)."""
    cls = np.asarray(cls, np.float32)
    d = np.asarray(dfl, np.float32).reshape(-1, 4, 16)
    own = list(COCO.values())
    other = np.setdiff1d(np.arange(cls.shape[1]), own)
    top2 = np.sort(cls, 1)[:, -2:]
    q = cls / np.clip(cls.sum(1, keepdims=True), 1e-9, None)
    ent = -(d * np.log(np.clip(d, 1e-9, None))).sum(-1)  # per box side (left, top, right, bottom)
    out = {f"p_{k}": cls[:, i] for k, i in COCO.items()}
    out.update(p_other_max=cls[:, other].max(1), p_top_margin=top2[:, 1] - top2[:, 0], p_sum=cls.sum(1),
               p_entropy=-(q * np.log(np.clip(q, 1e-9, None))).sum(1))
    for j, side in enumerate(("left", "top", "right", "bottom")):
        out[f"dfl_ent_{side}"] = ent[:, j]
    out["dfl_ent_mean"], out["level"] = ent.mean(1), np.asarray(level, np.float32)
    pcs = pca.transform(np.asarray(neck, np.float32))
    for i in range(N_PC):
        out[f"neck_pc{i:02d}"] = pcs[:, i]
    return pd.DataFrame(out)


def fit_pca(seed=0, n=200_000, det_id="v8x_1280", box_set="train"):
    """PCA of the neck vectors, fitted on a random 200,000 train detections."""
    from sklearn.decomposition import PCA

    from wad.yolo_box_features import load as load_box

    neck = load_box(box_set, det_id)["neck"]
    rows = np.sort(np.random.default_rng(seed).choice(len(neck), n, replace=False))
    return PCA(n_components=N_PC, random_state=seed).fit(np.asarray(neck[rows], np.float32))


def add_box_columns(f: pd.DataFrame, box_set: str, pca, chunk=400_000, det_id="v8x_1280") -> pd.DataFrame:
    """Join model 2's per-box columns to labelled features (by the detections-file row)."""
    from wad.yolo_box_features import load as load_box

    b = load_box(box_set, det_id)
    rows = f.row.to_numpy()
    parts = [box_columns(b["cls"][rows[i:i + chunk]], b["dfl"][rows[i:i + chunk]], b["neck"][rows[i:i + chunk]],
                         b["level"][rows[i:i + chunk]], pca) for i in range(0, len(rows), chunk)]
    extra = pd.concat(parts, ignore_index=True)
    assert len(extra) == len(f) and not extra.isna().any().any(), "per-box features missing for some detections"
    return pd.concat([f.reset_index(drop=True), extra], axis=1)


# ---- the model --------------------------------------------------------------------------------------

def train_model(f: pd.DataFrame, cols=FULL):
    from sklearn.ensemble import HistGradientBoostingClassifier

    m = HistGradientBoostingClassifier(**PARAMS, categorical_features=[cols.index("cls")])
    return m.fit(f[cols].to_numpy(), f.tp.to_numpy())


def load(path=None) -> dict:
    import joblib

    return joblib.load(path or model_path())


def score(det: pd.DataFrame, sizes: pd.DataFrame, bundle: dict | None = None, box: dict | None = None) -> np.ndarray:
    """The model's chance that each detection is real (det and sizes as for features_raw; model 2 also
    needs box: the per-box head outputs cls, dfl, neck, level for det's rows)."""
    bundle = bundle or load()
    if len(det) == 0:
        return np.zeros(0)
    f = features_raw(det, sizes)
    if bundle.get("features", "context") == "yolo":
        if box is None:
            raise ValueError("this per-detection model needs YOLOv8x's per-box features (box=; wad.yolo_box_features.Detector)")
        f = pd.concat([f.reset_index(drop=True), box_columns(box["cls"], box["dfl"], box["neck"], box["level"], bundle["pca"])], axis=1)
    return bundle["model"].predict_proba(f[bundle["cols"]].to_numpy())[:, 1]


def keep(chance: np.ndarray, lam: float) -> np.ndarray:
    """The dial's per-detection rule: keep iff the chance of being real >= 1 / (1 + lam)."""
    return np.asarray(chance) >= 1.0 / (1.0 + lam)


# ---- statistical backing -----------------------------------------------------------------------------

def calibration(p: np.ndarray, tp: np.ndarray) -> tuple[pd.DataFrame, float]:
    b = np.digitize(p, BINS) - 1
    rows = [dict(bin_lo=BINS[k], bin_hi=min(BINS[k + 1], 1.0), n=int((b == k).sum()), mean_chance=float(p[b == k].mean()),
                 share_real=float(tp[b == k].mean())) for k in range(len(BINS) - 1) if (b == k).any()]
    t = pd.DataFrame(rows)
    return t, float((t.n * (t.mean_chance - t.share_real).abs()).sum() / t.n.sum())


def near_cuts(p: np.ndarray, tp: np.ndarray, lams=PRESETS) -> list[dict]:
    """Detections whose chance is within a factor 1.5 of each cut 1/(1+lam): are they real that often?"""
    out = []
    for lam in lams:
        cut = 1 / (1 + lam)
        m = (p >= cut / 1.5) & (p <= min(cut * 1.5, 1.0))
        out.append(dict(lam=lam, cut=cut, n_near=int(m.sum()), mean_chance_near=float(p[m].mean()), share_real_near=float(tp[m].mean())))
    return out


def per_image(img: np.ndarray, kept: np.ndarray, tp: np.ndarray, n_img: int) -> tuple[np.ndarray, np.ndarray]:
    return (np.bincount(img, weights=kept & tp, minlength=n_img), np.bincount(img, weights=kept & ~tp, minlength=n_img))


def groups_of(ds: str, weathers) -> dict:
    w = sorted(set(weathers))
    g = {x: [x] for x in w}
    if ds == "BDD100K":
        g.update({k: [x for x in v if x in w] for k, v in BDD_GROUPS.items()})
    g["all"] = w
    return g


def grade(ds: str, c, f: pd.DataFrame, p: np.ndarray, tau: dict, lams, B=1000, seed=0) -> pd.DataFrame:
    """Per condition and lam: the three policies, and cost ratios with 95% image-bootstrap intervals."""
    from wad.curves import ALL

    tp, img, sc = f.tp.to_numpy().astype(bool), f.img.to_numpy(), f.score.to_numpy()
    n_gt = c.n_gt(ALL).astype(float)
    rng = np.random.default_rng(seed)
    sels = {g: np.flatnonzero(np.isin(c.weather, ws)) for g, ws in groups_of(ds, c.weather).items()}
    boot = {g: np.stack([np.bincount(rng.integers(0, len(sel), len(sel)), minlength=len(sel)) for _ in range(B)]).astype(float)
            for g, sel in sels.items()}
    rows = []
    for lam in lams:
        pol = {"per detection": (1 / (1 + lam), p >= 1 / (1 + lam)), "one threshold (train)": (tau[lam], sc >= tau[lam]),
               "no tuning (0.25)": (DEFAULT, sc >= DEFAULT)}
        counts = {name: per_image(img, kept, tp, c.n_img) for name, (thr, kept) in pol.items()}
        for g, sel in sels.items():
            Wb, cost = boot[g], {}
            for name, (thr, _) in pol.items():
                tk, fk, gt = counts[name][0][sel], counts[name][1][sel], n_gt[sel]
                cost[name] = lam * (gt - tk) + fk
                rows.append(dict(dataset=ds, group=g, lam=lam, policy=name, cut_or_threshold=thr, images=len(sel),
                                 recall=tk.sum() / gt.sum(), precision=tk.sum() / max(tk.sum() + fk.sum(), 1),
                                 fppi=fk.sum() / len(sel), cost_per_img=cost[name].mean()))
            ref = cost["one threshold (train)"]
            for r, name in zip(rows[-3:], pol):
                ratio = (Wb @ cost[name]) / (Wb @ ref)
                r.update(cost_vs_one=cost[name].sum() / ref.sum(), ci_lo=float(np.quantile(ratio, 0.025)),
                         ci_hi=float(np.quantile(ratio, 0.975)))
    return pd.DataFrame(rows)


def folds(tr: pd.DataFrame, im: pd.DataFrame, evals: dict, k=5, seed=0, cols=FULL) -> tuple[pd.DataFrame, np.ndarray]:
    """k-fold by near-duplicate group inside train: out-of-fold calibration and cost against the one
    threshold fitted on the other folds; each fold model's cost on BDD100K val / DAWN against the one
    threshold fitted on all of train."""
    grp = dict(zip(im.image_id, im.group))
    ug = np.array(sorted(set(grp.values())))
    fold_of_group = dict(zip(ug, np.random.default_rng(seed).permutation(len(ug)) % k))
    fold = tr.image_id.map(grp).map(fold_of_group).to_numpy()
    n_gt = dict(zip(im.image_id, im.n_gt))
    oof = np.full(len(tr), np.nan)
    rows = []
    for j in range(k):
        t0 = time.perf_counter()
        inn, out = fold != j, fold == j
        m = train_model(tr[inn], cols)
        oof[out] = m.predict_proba(tr.loc[out, cols].to_numpy())[:, 1]
        _, curve = train_curve(tr.loc[inn, ["score", "tp"]])
        ho = tr[out]
        held = [i for i in n_gt if fold_of_group[grp[i]] == j]  # every held-out image, with or without detections
        g_out = float(sum(n_gt[i] for i in held))
        y, s = ho.tp.to_numpy().astype(bool), ho.score.to_numpy()
        _, ece = calibration(oof[out], y)
        pe_cache = {}  # this fold model's chances on BDD100K val / DAWN
        for lam in PRESETS:
            t = train_threshold(curve, lam)
            kd, k1 = oof[out] >= 1 / (1 + lam), s >= t
            c_det = lam * (g_out - (kd & y).sum()) + (kd & ~y).sum()
            c_one = lam * (g_out - (k1 & y).sum()) + (k1 & ~y).sum()
            row = dict(fold=j, lam=lam, n_iter=int(m.n_iter_), heldout_images=len(held), heldout_ece=ece,
                       heldout_cost_vs_one=c_det / c_one)
            for ds, (f, ng, tau) in evals.items():
                pe = pe_cache.setdefault(ds, m.predict_proba(f[cols].to_numpy())[:, 1])
                yy, ss = f.tp.to_numpy().astype(bool), f.score.to_numpy()
                kd, k1 = pe >= 1 / (1 + lam), ss >= tau[lam]
                row[f"{ds}_cost_vs_one"] = (lam * (ng - (kd & yy).sum()) + (kd & ~yy).sum()) / (lam * (ng - (k1 & yy).sum()) + (k1 & ~yy).sum())
            rows.append(row)
        print(f"fold {j}: {m.n_iter_} iterations, {time.perf_counter() - t0:.0f} s", flush=True)
    return pd.DataFrame(rows), oof


def figure(cal: pd.DataFrame, ev_dial: pd.DataFrame, name="fig21_per_detection"):
    import matplotlib.pyplot as plt

    from wad.config import Cfg
    from wad.figures import INK, INK2, ORDINAL_BLUE, SERIES, save, style

    style()
    show = [("BDD100K", "day", ORDINAL_BLUE[1], "-", "BDD100K day"), ("BDD100K", "night", INK, "-", "BDD100K night"),
            ("BDD100K", "rain_night", SERIES[1], "-", "BDD100K rain night"), ("DAWN", "fog", SERIES[2], "--", "DAWN fog"),
            ("DAWN", "all", SERIES[3], "--", "DAWN all")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.plot([0.002, 1], [0.002, 1], color=INK2, lw=0.8, ls=":")
    for ds, g, col, ls, lab in show:
        t = cal[(cal.dataset == ds) & (cal.group == g) & (cal.n >= 50)]
        ax.plot(t.mean_chance, t.share_real, color=col, ls=ls, marker="o", ms=3.5, lw=1.4, label=lab)
    ax.set(xscale="log", yscale="log", xlim=(0.002, 1.05), ylim=(0.002, 1.05), xlabel="model's chance the detection is real",
           ylabel="share actually real", title="Calibration on images never seen in training")
    ax.legend(fontsize=7, loc="upper left")
    for ax, ds in ((axes[1], "BDD100K"), (axes[2], "DAWN")):
        e = ev_dial[(ev_dial.dataset == ds) & (ev_dial.group == "all")]
        for policy, col, ls in (("per detection", ORDINAL_BLUE[0], "-"), ("no tuning (0.25)", SERIES[1], "--")):
            d = e[e.policy == policy]
            ax.plot(d.lam, d.cost_vs_one, color=col, ls=ls, lw=1.8, label=policy)
            ax.fill_between(d.lam, d.ci_lo, d.ci_hi, color=col, alpha=0.18, lw=0)
        ax.axhline(1.0, color=INK2, lw=0.8)
        ax.text(290, 1.012, "one threshold (train) = 1", fontsize=7.5, color=INK2, va="bottom", ha="right")
        ax.set(xscale="log", ylim=(0.85, None), xlabel="dial λ: false alarms accepted per avoided miss",
               ylabel="cost relative to one threshold (λ·misses + false alarms)", title=f"{ds}: cost at each dial setting (95% band)")
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return save(fig, Cfg(dict(paths=Cfg(dict(results=RESULTS)))), name)


def versus_model1(ds: str, c, f: pd.DataFrame, p2: np.ndarray, p1: np.ndarray, B=1000, seed=0) -> pd.DataFrame:
    """Model 2 against model 1 per condition and preset: cost ratio with a 95% image-bootstrap interval."""
    from wad.curves import ALL

    tp, img = f.tp.to_numpy().astype(bool), f.img.to_numpy()
    n_gt = c.n_gt(ALL).astype(float)
    rng = np.random.default_rng(seed)
    rows = []
    for g, ws in groups_of(ds, c.weather).items():
        sel = np.flatnonzero(np.isin(c.weather, ws))
        Wb = np.stack([np.bincount(rng.integers(0, len(sel), len(sel)), minlength=len(sel)) for _ in range(B)]).astype(float)
        for lam in PRESETS:
            cost = []
            for p in (p2, p1):
                tk, fk = per_image(img, p >= 1 / (1 + lam), tp, c.n_img)
                cost.append(lam * (n_gt[sel] - tk[sel]) + fk[sel])
            ratio = (Wb @ cost[0]) / (Wb @ cost[1])
            rows.append(dict(dataset=ds, group=g, lam=lam, images=len(sel), cost_model2_vs_model1=cost[0].sum() / cost[1].sum(),
                             ci_lo=float(np.quantile(ratio, 0.025)), ci_hi=float(np.quantile(ratio, 0.975))))
    return pd.DataFrame(rows)


def group_importance(model, f: pd.DataFrame, cols: list, n=200_000, repeats=3, seed=0) -> pd.DataFrame:
    """Log-loss increase on BDD100K val detections scored 0.05-0.5 when a whole feature group is shuffled."""
    from sklearn.metrics import log_loss

    band = f[(f.score >= 0.05) & (f.score < 0.5)]
    band = band.sample(min(n, len(band)), random_state=seed)
    X, y = band[cols].to_numpy(), band.tp.to_numpy()
    base = log_loss(y, model.predict_proba(X)[:, 1])
    groups = {"context (20 features of model 1)": FULL, "class probabilities": [c for c in cols if c.startswith("p_")],
              "box-side uncertainty (DFL)": [c for c in cols if c.startswith("dfl_")], "pyramid level": ["level"],
              "neck feature components": [c for c in cols if c.startswith("neck_pc")]}
    rng = np.random.default_rng(seed)
    rows = []
    for name, g in groups.items():
        idx = [cols.index(c) for c in g]
        inc = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, idx] = Xp[rng.permutation(len(Xp))][:, idx]
            inc.append(log_loss(y, model.predict_proba(Xp)[:, 1]) - base)
        rows.append(dict(group=name, features=len(g), log_loss_increase=float(np.mean(inc))))
    return pd.DataFrame(rows).sort_values("log_loss_increase", ascending=False)


def main(features: str = "context"):
    import joblib
    import sklearn

    from wad.curves import ALL, Cache
    from wad.thresh_net import cfg as train_cfg

    t0 = time.perf_counter()
    cols, tag = cols_of(features), _tag(features)
    ct = Cache(train_cfg())
    tr = labelled_features(ct, TRAIN_DET)
    tr["image_id"] = ct.image_ids[tr.img.to_numpy()]
    pca = None
    if features == "yolo":
        pca = fit_pca()
        tr = add_box_columns(tr, "train", pca)
    im = pd.DataFrame(dict(image_id=ct.image_ids, n_gt=ct.n_gt(ALL)))  # objects per train image, as the cost counts them
    im["group"] = pd.read_parquet(THRESH_CACHE / "images.parquet", columns=["image_id", "group"]).set_index("image_id").group.loc[ct.image_ids].to_numpy()
    print(f"train features ({features}, {len(cols)} columns): {len(tr)} detections, {time.perf_counter() - t0:.0f} s", flush=True)
    model = train_model(tr, cols)
    joblib.dump(dict(model=model, cols=cols, features=features, pca=pca, cls=CLS, grp=GRP, params=PARAMS, n_iter=int(model.n_iter_),
                     sklearn=sklearn.__version__,
                     trained_on="BDD100K train, 21,831 images, YOLOv8x @1280 detections >= 0.001, grouped matching (IoU 0.5)",
                     rule="keep iff chance >= 1 / (1 + lam)"), model_path(features))
    print(f"model: {model.n_iter_} iterations, {time.perf_counter() - t0:.0f} s -> {model_path(features)}", flush=True)
    _, curve = train_curve(pd.read_parquet(TRAIN, columns=["score", "tp"]))
    lams = np.union1d(DIAL, PRESETS)
    tau = {lam: train_threshold(curve, lam) for lam in lams}
    cal_rows, cut_rows, ev, evals, vs, imp = [], [], [], {}, [], None
    for ds, v in eval_configs().items():
        c = Cache(v)
        f = labelled_features(c, EVAL_DET[ds])
        if features == "yolo":
            f = add_box_columns(f, BOX_SET[ds], pca)
        p = model.predict_proba(f[cols].to_numpy())[:, 1]
        key = "bdd" if ds == "BDD100K" else "dawn"
        pd.DataFrame(dict(image_id=c.image_ids[f.img.to_numpy()], score=f.score.to_numpy(), tp=f.tp.to_numpy(), p=p)).to_parquet(
            pred_path(key, features), index=False)
        evals[ds] = (f, float(c.n_gt(ALL).sum()), tau)
        w_det = c.weather[f.img.to_numpy()]
        y = f.tp.to_numpy().astype(bool)
        for g, ws in groups_of(ds, c.weather).items():
            m = np.isin(w_det, ws)
            t, ece = calibration(p[m], y[m])
            cal_rows.append(t.assign(dataset=ds, group=g, ece=ece))
            cut_rows += [dict(dataset=ds, group=g, **r) for r in near_cuts(p[m], y[m])]
        ev.append(grade(ds, c, f, p, tau, lams))
        if features == "yolo":
            p1 = pd.read_parquet(pred_path(key, "context"))
            assert (p1.image_id.to_numpy() == c.image_ids[f.img.to_numpy()]).all()
            vs.append(versus_model1(ds, c, f, p, p1.p.to_numpy()))
            if ds == "BDD100K":
                imp = group_importance(model, f, cols)
        print(f"{ds}: graded, {time.perf_counter() - t0:.0f} s", flush=True)
    cal = pd.concat(cal_rows, ignore_index=True)[["dataset", "group", "bin_lo", "bin_hi", "n", "mean_chance", "share_real", "ece"]]
    cuts, ev = pd.DataFrame(cut_rows), pd.concat(ev, ignore_index=True)
    fo, oof = folds(tr, im, evals, cols=cols)
    # out-of-fold chances on train (each from the fold model that did not see that image): the honest
    # per-detection operating curve on train, used to map F-beta to the dial (wad.fbeta_dial)
    pd.DataFrame(dict(image_id=tr.image_id.to_numpy(), score=tr.score.to_numpy(), tp=tr.tp.to_numpy(), p=oof)).to_parquet(
        oof_path(features), index=False)
    out = f"per_detection{tag}"
    cal.to_csv(RESULTS / f"{out}_calibration.csv", index=False, float_format="%.5g")
    cuts.to_csv(RESULTS / f"{out}_cuts.csv", index=False, float_format="%.5g")
    ev[ev.lam.isin(PRESETS)].to_csv(RESULTS / f"{out}_eval.csv", index=False, float_format="%.5g")
    fo.to_csv(RESULTS / f"{out}_folds.csv", index=False, float_format="%.5g")
    figure(cal, ev[ev.lam.isin(DIAL)], "fig21_per_detection" if features == "context" else "fig23_per_detection_yolo")
    pd.set_option("display.width", 220)
    print(cal.groupby(["dataset", "group"]).ece.first().round(4).to_string())
    print(cuts[cuts.group.isin(["all", "day", "night", "fog"])].round(3).to_string(index=False))
    e = ev[ev.lam.isin(PRESETS) & ev.group.isin(["all", "day", "night", "rain_night", "fog"])]
    print(e[["dataset", "group", "lam", "policy", "recall", "precision", "fppi", "cost_vs_one", "ci_lo", "ci_hi"]].round(3).to_string(index=False))
    print(fo.groupby("lam")[["heldout_ece", "heldout_cost_vs_one", "BDD100K_cost_vs_one", "DAWN_cost_vs_one"]].agg(["min", "max"]).round(4).to_string())
    if features == "yolo":
        vs = pd.concat(vs, ignore_index=True)
        vs.to_csv(RESULTS / f"{out}_vs_model1.csv", index=False, float_format="%.5g")
        imp.to_csv(RESULTS / f"{out}_importance.csv", index=False, float_format="%.5g")
        prim = vs[(vs.dataset == "BDD100K") & (vs.group == "all") & (vs.lam == 10)].iloc[0]
        dawn = vs[(vs.dataset == "DAWN") & (vs.group == "all") & (vs.lam == 10)].iloc[0]
        cb = cuts[(cuts.dataset == "BDD100K") & (cuts.group == "all")]
        rel = cb.share_real_near / cb.cut
        crit = pd.DataFrame([
            dict(criterion="BDD100K val all, lam 10: cost model 2 / model 1, 95% interval below 1",
                 value=f"{prim.cost_model2_vs_model1:.4f} ({prim.ci_lo:.4f}-{prim.ci_hi:.4f})", met=bool(prim.ci_hi < 1)),
            dict(criterion="DAWN all, lam 10: model 2 not worse (point estimate <= 1)",
                 value=f"{dawn.cost_model2_vs_model1:.4f} ({dawn.ci_lo:.4f}-{dawn.ci_hi:.4f})", met=bool(dawn.cost_model2_vs_model1 <= 1)),
            dict(criterion="BDD100K val all: share real near each preset cut within x1.25 of the cut",
                 value=", ".join(f"{x:.3f}" for x in rel), met=bool(((rel >= 0.8) & (rel <= 1.25)).all()))])
        crit.to_csv(RESULTS / f"{out}_criteria.csv", index=False)
        print(vs[vs.group.isin(["all", "day", "night", "rain_night", "fog"])].round(4).to_string(index=False))
        print(imp.round(5).to_string(index=False))
        print(crit.to_string(index=False))
    print(f"done, {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=FEATURE_SETS, default="context")
    main(ap.parse_args().features)
