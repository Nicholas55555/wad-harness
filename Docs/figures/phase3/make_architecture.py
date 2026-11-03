"""Architecture figure, v2 (2026-10-29): the v1 pipeline with real images and real numbers from the project.

Two stages:
  python make_architecture.py --prepare   needs the harness caches; writes architecture_assets.json
                                          (thumbnails, boxes, one box's YOLOv8x outputs, calibration, curves)
  python make_architecture.py             builds architecture.svg from that JSON (no caches needed)
Then render the PDF (Overleaf) and the PNG (slides fallback) from the same SVG:  .\render_architecture.ps1

Everything drawn comes from the cached YOLOv8x @1280 detections, the adopted per-detection model (model 2) and
the result tables. The example frame (BDD100K val, rain at night) was chosen to illustrate, because the dial
recovers several objects there; the typical effect sizes are in the tables (doc 08). Boxes are drawn as vector
rectangles over the embedded photos. Text widths are checked with Segoe UI metrics, so every line fits.
"""
from __future__ import annotations

import base64
import io
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[2]
ASSETS = HERE / "architecture_assets.json"
OUT = HERE / "architecture.svg"

MAIN = ("BDD100K", "rain_night/ca4269e1-a3a6e21a")
INPUTS = [("day", "BDD100K", "clear/c650b257-a545cebf"), ("night", "BDD100K", "clear_night/b2e2f4ed-6ba045d0"),
          ("rain at night", "BDD100K", MAIN[1]), ("fog (DAWN)", "DAWN", "fog/mist-034")]
TRAIN_WEATHERS = ["clear", "cloudy", "rain", "snow", "clear_night", "rain_night"]
LAM = 10
COCO80 = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
          "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie "
          "suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
          "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
          "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
          "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier "
          "toothbrush").split()


# =============================================================================================== stage 1
def jpeg(im, width, quality=82, crop169=False):
    if crop169:
        w, h = im.size
        th = round(w * 9 / 16)
        if th < h:
            top = (h - th) // 2
            im = im.crop((0, top, w, top + th))
        else:
            tw = round(h * 16 / 9)
            left = (w - tw) // 2
            im = im.crop((left, 0, left + tw, h))
    im = im.resize((width, round(im.size[1] * width / im.size[0])))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), im.size


def prepare():
    sys.path.insert(0, str(REPO))
    import numpy as np
    import pandas as pd

    from wad import yolo_box_features as ybf
    from wad.curves import Cache
    from wad.imageio import load_rgb
    from wad.one_threshold import eval_configs, train_curve
    from wad.per_detection import pred_path

    caches = {ds: Cache(v) for ds, v in eval_configs().items()}
    det_path = {"BDD100K": "cache/bdd/det/v8x_1280/detections.parquet", "DAWN": "cache/det/v8x_1280/detections.parquet"}
    gt_path = {"BDD100K": "cache/bdd/gt.parquet", "DAWN": "cache/gt.parquet"}
    out = {}

    def path_of(ds, iid):
        return REPO / caches[ds].images.set_index("image_id").loc[iid, "path"]

    # input thumbnails (16:9 crops, 2x the drawn size)
    out["inputs"] = []
    for label, ds, iid in INPUTS:
        src, _ = jpeg(load_rgb(path_of(ds, iid), "apply"), 220, crop169=True)
        out["inputs"].append(dict(label=label, id=iid, src=src))

    # the example frame: every candidate, the kept boxes at lam 10 (per detection), the objects
    ds, iid = MAIN
    c = caches[ds]
    sel = (c.pred.image_id == iid).to_numpy()
    pr = c.pred[sel].reset_index(drop=True)
    p = pd.read_parquet(pred_path("bdd")).p.to_numpy()[sel]
    det_all = pd.read_parquet(REPO / det_path[ds])
    rows = np.flatnonzero((det_all.image_id == iid).to_numpy())  # parquet order = cache order within an image
    d = det_all.iloc[rows].reset_index(drop=True)
    assert len(d) == len(pr) and np.allclose(d.score.to_numpy(), pr.score.to_numpy())
    tp = pr.tp.to_numpy().astype(bool)
    keep = p >= 1 / (1 + LAM)
    gtb = pd.read_parquet(REPO / gt_path[ds])
    gt = gtb[(gtb.image_id == iid) & gtb.gt_id.isin(c.gt[c.gt.image_id == iid].gt_id)]
    found = set(pr.gt_id[keep & tp].tolist())
    im = load_rgb(path_of(ds, iid), "apply")
    W0, H0 = im.size
    src, _ = jpeg(im, 640)
    s25 = d.score.to_numpy() >= 0.25
    cand = [[round(b.x1, 1), round(b.y1, 1), round(b.x2, 1), round(b.y2, 1), round(float(b.score), 4)] for b in d.itertuples()]
    kept = [[round(b.x1, 1), round(b.y1, 1), round(b.x2, 1), round(b.y2, 1), int(t)] for b, k, t in zip(d.itertuples(), keep, tp) if k]
    missed = [[round(g.x1, 1), round(g.y1, 1), round(g.x2, 1), round(g.y2, 1)] for g in gt.itertuples() if g.gt_id not in found]
    out["main"] = dict(id=iid, src=src, w=W0, h=H0, cand=cand, kept=kept, missed=missed, n_obj=len(gt),
                       found=int((keep & tp).sum()), false=int((keep & ~tp).sum()),
                       found25=int((s25 & tp).sum()), false25=int((s25 & ~tp).sum()),
                       est_fa=round(float((1 - p[keep]).sum()), 2),
                       smin=round(float(d.score.min()), 4), smax=round(float(d.score.max()), 3))

    # one recovered box (kept and real at lam 10, below the default 0.25): its YOLOv8x per-box outputs
    cand_idx = np.flatnonzero(keep & tp & ~s25)
    area = ((d.x2 - d.x1) * (d.y2 - d.y1)).to_numpy()
    k = int(cand_idx[np.argmax(area[cand_idx])])
    feats = ybf.load("bdd")
    r = int(rows[k])
    cls = np.asarray(feats["cls"][r], dtype=float)
    dfl = np.asarray(feats["dfl"][r], dtype=float).reshape(4, 16)
    top = np.argsort(-cls)[:4]
    b = d.iloc[k]
    bw, bh = b.x2 - b.x1, b.y2 - b.y1
    box = (max(0, b.x1 - 0.35 * bw), max(0, b.y1 - 0.35 * bh), min(W0, b.x2 + 0.35 * bw), min(H0, b.y2 + 0.35 * bh))
    crop = im.crop(tuple(round(v) for v in box))
    csrc, csize = jpeg(crop, 160, quality=88)
    out["hbox"] = dict(index=k, score=round(float(b.score), 3), p=round(float(p[k]), 3), cls=b.cls,
                       box=[round(float(v), 1) for v in (b.x1, b.y1, b.x2, b.y2)], crop_box=[round(v, 1) for v in box],
                       crop=csrc, top=[[COCO80[i].replace("_", " "), round(float(cls[i]), 3)] for i in top],
                       cls80=[float(f"{v:.3g}") for v in cls],
                       dfl=[[round(float(v), 4) for v in side] for side in dfl], level=int(feats["level"][r]))

    # offline: train mosaic and one labelled train frame
    tim = pd.read_parquet(REPO / "cache/thresh/images.parquet")
    out["train"] = []
    for w in TRAIN_WEATHERS:
        x = tim[(tim.weather == w) & tim.n_gt.between(4, 12) & tim.keep].sort_values("image_id").iloc[3]
        src, _ = jpeg(load_rgb(REPO / x.path, "apply"), 156, quality=80, crop169=True)
        out["train"].append(dict(weather=w, id=x.image_id, src=src))
    from wad.one_threshold import TRAIN

    tgt = pd.read_parquet(REPO / "cache/thresh/gt.parquet")
    pool = tim[(tim.weather == "clear") & tim.n_gt.between(6, 10) & tim.keep].sort_values("image_id").head(400)
    tm = pd.read_parquet(TRAIN, columns=["image_id", "score", "tp"])
    tm = tm[tm.image_id.isin(pool.image_id) & (tm.score >= 0.05)]
    tdet = pd.read_parquet(REPO / "cache/thresh/det/v8x_1280/detections.parquet")
    tdet = tdet[tdet.image_id.isin(pool.image_id) & (tdet.score >= 0.05)]
    tm_by, td_by = dict(tuple(tm.groupby("image_id"))), dict(tuple(tdet.groupby("image_id")))
    for x in pool.itertuples():
        m = tm_by.get(x.image_id, tm.iloc[:0])
        if m.score.duplicated().any() or not (5 <= m.tp.sum() <= 9 and 2 <= (~m.tp).sum() <= 4):
            continue
        dd = td_by.get(x.image_id, tdet.iloc[:0])
        lab = dict(zip(m.score.round(6), m.tp))
        if len(dd) != len(m) or not all(round(s, 6) in lab for s in dd.score):
            continue
        im2 = load_rgb(REPO / x.path, "apply")
        src, _ = jpeg(im2, 480)
        g = tgt[tgt.image_id == x.image_id]
        out["label"] = dict(id=x.image_id, src=src, w=im2.size[0], h=im2.size[1],
                            gt=[[round(v, 1) for v in (o.x1, o.y1, o.x2, o.y2)] for o in g.itertuples()],
                            det=[[round(o.x1, 1), round(o.y1, 1), round(o.x2, 1), round(o.y2, 1), int(lab[round(o.score, 6)])]
                                 for o in dd.itertuples()])
        break

    # numbers for the mini plots
    cal = pd.read_csv(HERE.parents[1] / "tables" / "per_detection_yolo_calibration.csv")
    cb = cal[(cal.dataset == "BDD100K") & (cal.group == "all")]
    out["calib"] = dict(points=[[round(a, 4), round(b, 4)] for a, b in zip(cb.mean_chance, cb.share_real)],
                        ece=round(float(cb.ece.iloc[0]) * 100, 2))
    _, (ss, ctp, cfp) = train_curve()
    n_obj = len(tgt)
    taus = np.geomspace(0.002, 0.95, 160)
    idx = np.searchsorted(-ss, -taus, side="right") - 1  # last rule with score >= tau
    curves = {}
    for lam in (1, 10, 30):
        cost = lam * (n_obj - ctp) + cfp
        rel = cost[idx] / cost.min()
        j = int(np.argmin(cost))
        curves[str(lam)] = dict(points=[[round(float(t), 5), round(float(v), 4)] for t, v in zip(taus, rel)],
                                tau=round(float(ss[j]), 4))
    out["cost_curves"] = curves
    ft = pd.read_csv(REPO / "weights" / "fbeta_table.csv")
    ft = ft[(ft.beta >= 0.25) & (ft.beta <= 4)]
    out["lam_eq"] = [[round(float(b), 3), round(float(v), 3)] for b, v in zip(ft.beta, ft.det_lam_equivalent)][::8]
    ev = pd.read_csv(HERE.parents[1] / "tables" / "per_detection_yolo_eval.csv")
    grade = []
    for dsn, grp, label in (("BDD100K", "all", "BDD100K val, all"), ("BDD100K", "day", "day"), ("BDD100K", "night", "night"),
                            ("BDD100K", "rain_night", "rain at night"), ("DAWN", "all", "DAWN, all"), ("DAWN", "fog", "DAWN fog")):
        x = ev[(ev.dataset == dsn) & (ev.group == grp) & (ev.lam == LAM)].set_index("policy")
        det_ = x.loc["per detection"]
        grade.append(dict(label=label, det=round(float(det_.cost_vs_one), 3), lo=round(float(det_.ci_lo), 3),
                          hi=round(float(det_.ci_hi), 3), none=round(float(x.loc["no tuning (0.25)"].cost_vs_one), 3)))
    out["grade"] = grade
    ASSETS.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {ASSETS} ({ASSETS.stat().st_size / 1e3:.0f} kB); main frame: {out['main']['found']}/{out['main']['n_obj']} "
          f"found, {out['main']['false']} false at lam {LAM}; default 0.25: {out['main']['found25']} found, "
          f"{out['main']['false25']} false; highlight box s={out['hbox']['score']} p={out['hbox']['p']} {out['hbox']['top']}")


# =============================================================================================== stage 2
FONT = "'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif"
FONTS = {400: "C:/Windows/Fonts/segoeui.ttf", 600: "C:/Windows/Fonts/seguisb.ttf", 700: "C:/Windows/Fonts/segoeuib.ttf"}
INK, SUB = "#1F2933", "#52606D"
NEUTRAL = dict(fill="#F4F6F8", stroke="#7B8794")
WHITE = dict(fill="#FFFFFF", stroke="#7B8794")
NEW = dict(fill="#FFF1DA", stroke="#C26A00")
NEW_GROUP = dict(fill="#FFF9F0", stroke="#D97706")
NEW_INK = "#7A3E00"
PLAN = dict(fill="#FFFFFF", stroke="#2F6FDB")
PLAN_INK = "#1E4FA3"
BASE = dict(fill="#F3F3F3", stroke="#9AA0A6")
BASE_INK = "#62676C"
OFF_BAND, OFF_INK = "#EEF6F0", "#2F6B45"
ARROW, PARAM = "#3E4C59", "#C26A00"
REAL, FALSE, MISS, CANDC = "#2BD46F", "#FF4D4D", "#FFB020", "#8FD3FF"

parts: list[str] = []
problems: list[str] = []


def width(s, size, weight=400):
    from PIL import ImageFont

    plain = re.sub(r"_\{([^}]*)\}", r"\1", s).replace("*", "")
    return ImageFont.truetype(FONTS.get(weight, FONTS[400]), size).getlength(plain)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rich(s, size):
    """'*x*' -> italic (variables); '_{eq}' -> subscript."""
    out = []
    for i, chunk in enumerate(re.split(r"\*", s)):
        if not chunk:
            continue
        pieces = []
        for j, sub in enumerate(re.split(r"_\{([^}]*)\}", chunk)):
            if j % 2:
                pieces.append(f'<tspan dy="{size * 0.28:.1f}" font-size="{size * 0.72:.1f}">{esc(sub)}</tspan>'
                              f'<tspan dy="{-size * 0.28:.1f}">\u200b</tspan>')
            elif sub:
                pieces.append(esc(sub))
        body = "".join(pieces)
        out.append(f'<tspan font-style="italic">{body}</tspan>' if i % 2 else body)
    return "".join(out)


def text(x, y, s, size=16, weight=400, fill=INK, anchor="start", maxw=None, spacing=None):
    if maxw is not None and width(s, size, weight) > maxw:
        problems.append(f"too wide ({width(s, size, weight):.0f} > {maxw:.0f}): {s!r}")
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    s = re.sub(r"^ +", lambda mm: " " * (2 * len(mm.group(0))), s)  # SVG collapses spaces; indent with NBSP
    parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill}" text-anchor="{anchor}"{ls}>{rich(s, size)}</text>')


def rect(x, y, w, h, fill, stroke, sw=1.6, rx=10, dash=None, opacity=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' stroke-opacity="{opacity:.2f}"' if opacity is not None else ""
    parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}{o}/>')


def line(x1, y1, x2, y2, color, sw=1.2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}"{d}/>')


def polyline(pts, color, sw=1.8, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{color}" '
                 f'stroke-width="{sw}" stroke-linejoin="round"{d}/>')


def arrow(pts, color=ARROW, sw=2.0, dash=None, head="a-dark"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#{head})"/>')


def card(x, y, w, h, title, style, ink=INK, tsize=20, dash=None, sw=1.6):
    rect(x, y, w, h, style["fill"], style["stroke"], sw=sw, dash=dash)
    text(x + w / 2, y + 30, title, tsize, 600, ink, "middle", w - 16)


def lines(x, y, rows, anchor="middle", maxw=None, lh=None):
    """rows: list of (text, size, weight, colour)."""
    for s, size, weight, col in rows:
        text(x, y, s, size, weight, col, anchor, maxw)
        y += lh or size + 5
    return y


_clip = [0]


def photo(src, x, y, w, h, frame=None, boxes=(), ring=None):
    """An embedded photo with vector box overlays. boxes: (x1, y1, x2, y2, colour, width, opacity, dash) in the
    photo's own pixel frame (scaled by frame=(W0, H0))."""
    _clip[0] += 1
    cid = f"clip{_clip[0]}"
    parts.append(f'<clipPath id="{cid}"><rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4"/></clipPath>')
    parts.append(f'<image x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" preserveAspectRatio="xMidYMid slice" '
                 f'clip-path="url(#{cid})" href="{src}" xlink:href="{src}"/>')
    if boxes:
        sx, sy = w / frame[0], h / frame[1]
        g = [f'<g clip-path="url(#{cid})" fill="none">']
        for x1, y1, x2, y2, col, sw, op, dash in boxes:
            d = f' stroke-dasharray="{dash}"' if dash else ""
            g.append(f'<rect x="{x + x1 * sx:.1f}" y="{y + y1 * sy:.1f}" width="{max(0.6, (x2 - x1) * sx):.1f}" '
                     f'height="{max(0.6, (y2 - y1) * sy):.1f}" stroke="{col}" stroke-width="{sw}" stroke-opacity="{op:.2f}"{d}/>')
        g.append("</g>")
        parts.append("".join(g))
    rect(x, y, w, h, "none", ring or "#3E4C59", sw=1.6 if ring else 0.8, rx=4)


def network_icon(cx, y, h, color="#7B8794", fill="#E4E7EB"):
    """YOLO-like backbone -> neck -> head as narrowing then widening blocks."""
    hs = [1.0, 0.8, 0.62, 0.46, 0.62, 0.8]
    bw, gap = 13, 5
    x = cx - (len(hs) * bw + (len(hs) - 1) * gap) / 2
    for f in hs:
        bh = h * f
        rect(x, y + (h - bh) / 2, bw, bh, fill, color, sw=1.2, rx=3)
        x += bw + gap


def mini_axes(x, y, w, h):
    line(x, y + h, x + w, y + h, "#9AA5B1", 1)
    line(x, y, x, y + h, "#9AA5B1", 1)


def build():
    A = json.loads(ASSETS.read_text(encoding="utf-8"))
    m, hb = A["main"], A["hbox"]
    W, H = 1740, 972
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')
    marks = "".join(
        f'<marker id="{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{c}"/></marker>'
        for i, c in (("a-dark", ARROW), ("a-param", PARAM), ("a-grey", BASE["stroke"]), ("a-plan", PLAN["stroke"])))
    parts.append(f"<defs>{marks}</defs>")

    CY0, CY1 = 96, 410
    MID = 236
    text(22, 34, "ONLINE · EVERY CAMERA FRAME", 15, 700, SUB, spacing="1.2")

    # -- c1 input ---------------------------------------------------------------------------------------------
    x0, w = 18, 232
    card(x0, CY0, w, CY1 - CY0, "Camera frame *x*", NEUTRAL)
    tw, th, gx = 100, 56, 8
    for i, it in enumerate(A["inputs"]):
        cx = x0 + 12 + (i % 2) * (tw + gx)
        cy = CY0 + 48 + (i // 2) * (th + 26)
        main = it["id"] == m["id"]
        photo(it["src"], cx, cy, tw, th, ring="#1F2933" if main else None)
        text(cx + tw / 2, cy + th + 17, it["label"], 14, 700 if main else 400, INK if main else SUB, "middle", tw + 6)
    lines(x0 + w / 2, CY0 + 236, [("one image, any weather", 16, 400, SUB),
                                    ("example used here:", 14, 400, SUB),
                                    ("BDD100K val, rain at night", 14, 600, INK)], maxw=w - 16)
    arrow([(x0 + w, MID), (272, MID)])

    # -- c2 detector ------------------------------------------------------------------------------------------
    x0, w = 274, 144
    card(x0, CY0, w, CY1 - CY0, "YOLOv8x", NEUTRAL)
    network_icon(x0 + w / 2, CY0 + 50, 92)
    lines(x0 + w / 2, CY0 + 182, [("frozen", 16, 600, INK), ("COCO weights", 16, 400, SUB), ("1280 px", 16, 400, SUB),
                                   ("floor 0.001", 16, 400, SUB), ("NMS 0.5, ≤ 300", 16, 400, SUB)], maxw=w - 14)
    arrow([(x0 + w, MID), (440, MID)])

    # -- c3 candidates ----------------------------------------------------------------------------------------
    x0, w = 442, 286
    card(x0, CY0, w, CY1 - CY0, "All candidates *D*(*x*)", NEUTRAL)
    pw, ph = 262, 147
    boxes = [(b[0], b[1], b[2], b[3], CANDC, 1.0, 0.18 + 0.82 * math.sqrt(b[4]), None) for b in sorted(m["cand"], key=lambda b: b[4])]
    photo(m["src"], x0 + 12, CY0 + 46, pw, ph, (m["w"], m["h"]), boxes)
    lines(x0 + w / 2, CY0 + 222, [(f"{len(m['cand'])} boxes with scores {m['smin']:.3f}–{m['smax']:.2f}", 16, 400, INK),
                                   ("each: box · class · score *s*", 16, 400, SUB),
                                   ("brighter = higher score", 14, 400, SUB)], maxw=w - 16)

    # -- NEW group ---------------------------------------------------------------------------------------------
    GX0, GX1 = 742, 1464
    rect(GX0, 58, GX1 - GX0, CY1 - 58 + 14, NEW_GROUP["fill"], NEW_GROUP["stroke"], sw=2.4, rx=14)
    text(GX0 + 14, 82, "NEW · PER-DETECTION TRADE-OFF DIAL", 15, 700, NEW_INK, spacing="1.0")
    arrow([(728, MID), (756, MID)])

    # -- c4 per-box outputs -----------------------------------------------------------------------------------
    x0, w = 758, 232
    card(x0, CY0, w, CY1 - CY0, "Per-box outputs *h*(*d*)", NEW, NEW_INK)
    cx0 = x0 + 12
    cb = hb["crop_box"]
    cw = 74
    chh = cw * (cb[3] - cb[1]) / (cb[2] - cb[0])
    chh = min(chh, 60)
    bx = hb["box"]
    inner = [(bx[0] - cb[0], bx[1] - cb[1], bx[2] - cb[0], bx[3] - cb[1], REAL, 2.0, 1.0, None)]
    photo(hb["crop"], cx0, CY0 + 44, cw, chh, (cb[2] - cb[0], cb[3] - cb[1]), inner)
    lines(cx0 + cw + 10, CY0 + 60, [("one recovered box", 14, 600, NEW_INK),
                                     (f"score *s* = {hb['score']:.2f}", 15, 400, INK),
                                     ("below default 0.25", 13, 400, SUB)], anchor="start", maxw=w - cw - 30, lh=19)
    # class probabilities: all 80 on a log scale (1e-4 .. 1), the six road classes highlighted
    by = CY0 + 124
    text(cx0, by, "80 class probabilities (log scale)", 14, 600, NEW_INK, maxw=w - 20)
    sw80, sh = (w - 24) / 80, 40
    base_y = by + 8 + sh
    road = {0, 1, 2, 3, 5, 7}
    for i, v in enumerate(hb["cls80"]):
        vh = sh * (max(math.log10(max(v, 1e-6)), -4) + 4) / 4
        col = "#C26A00" if i == 2 else ("#E9A23B" if i in road else "#C9CED6")
        if vh > 0:
            parts.append(f'<rect x="{cx0 + i * sw80:.1f}" y="{base_y - vh:.1f}" width="{sw80 - 0.5:.2f}" height="{vh:.1f}" fill="{col}"/>')
    line(cx0, base_y, cx0 + 80 * sw80, base_y, "#9AA5B1", 0.8)
    line(cx0, by + 8, cx0 + 80 * sw80, by + 8, "#EADBC4", 0.6, dash="2 2")
    text(cx0 + 80 * sw80, by + 18, "1", 11, 400, SUB, "end")
    tops = " · ".join(f"{n} {v:.3f}".replace("0.", ".") for n, v in hb["top"][:3])
    text(cx0, base_y + 16, tops, 13, 400, INK, maxw=w - 20)
    # DFL
    dy = by + 90
    text(cx0, dy, "box-edge certainty (DFL)", 14, 600, NEW_INK, maxw=w - 20)
    hw, hh = 46, 26
    for i, (side, vals) in enumerate(zip("LTRB", hb["dfl"])):
        hx = cx0 + i * (hw + 7)
        mx = max(vals)
        bw = hw / len(vals)
        for j, v in enumerate(vals):
            vh = hh * v / mx
            parts.append(f'<rect x="{hx + j * bw:.1f}" y="{dy + 8 + hh - vh:.1f}" width="{bw - 0.6:.1f}" height="{vh:.1f}" fill="#C26A00"/>')
        line(hx, dy + 8 + hh, hx + hw, dy + 8 + hh, "#9AA5B1", 0.8)
        text(hx + hw / 2, dy + 8 + hh + 14, side, 12, 600, SUB, "middle")
    text(cx0, CY1 - 14, "+ neck feature (320-d), level", 13, 400, NEW_INK, maxw=w - 20)
    arrow([(x0 + w, MID), (1012, MID)])

    # -- c5 chance real ---------------------------------------------------------------------------------------
    x0, w = 1014, 228
    card(x0, CY0, w, CY1 - CY0, "Chance real *p*(*d*)", NEW, NEW_INK)
    sx0, sw_ = x0 + 12, w - 24
    segs = [(20, "#9AA5B1"), (10, "#E9A23B"), (5, "#F3C77A"), (32, "#C26A00"), (1, "#7A3E00")]
    xx = sx0
    for n, col in segs:
        ww = sw_ * n / 68
        parts.append(f'<rect x="{xx:.1f}" y="{CY0 + 46}" width="{ww - 1:.1f}" height="14" fill="{col}"/>')
        xx += ww
    text(sx0, CY0 + 78, "φ(*d*): 20 context + 48 per-box", 14, 400, NEW_INK, maxw=sw_)
    # tree icon + label
    tx, ty = sx0 + 8, CY0 + 90
    for k in range(3):
        ox = tx + k * 16
        parts.append(f'<polygon points="{ox},{ty + 18} {ox + 6},{ty + 2} {ox + 12},{ty + 18}" fill="#7A3E00"/>')
    text(tx + 54, ty + 15, "boosted trees, log-loss", 14, 400, NEW_INK, maxw=sw_ - 54)
    # reliability diagram
    px0, py0, pw2, ph2 = x0 + 46, CY0 + 128, 120, 104
    rect(px0, py0, pw2, ph2, "#FFFFFF", "#E2D3BC", sw=1, rx=2)
    line(px0, py0 + ph2, px0 + pw2, py0, "#9AA5B1", 1, dash="3 3")
    pts = [(px0 + pw2 * a, py0 + ph2 * (1 - b)) for a, b in A["calib"]["points"]]
    polyline(pts, "#C26A00", 1.6)
    for x, y in pts:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#C26A00" stroke="#FFFFFF" stroke-width="1"/>')
    text(px0 - 6, py0 + 8, "1", 12, 400, SUB, "end")
    text(px0 - 6, py0 + ph2, "0", 12, 400, SUB, "end")
    text(px0 + pw2, py0 + ph2 + 14, "1", 12, 400, SUB, "end")
    text(px0 + pw2 / 2, py0 + ph2 + 14, "predicted chance", 12, 400, SUB, "middle")
    parts.append(f'<text x="{px0 - 16:.1f}" y="{py0 + ph2 / 2:.1f}" font-family="{FONT}" font-size="12" fill="{SUB}" '
                 f'text-anchor="middle" transform="rotate(-90 {px0 - 16:.1f} {py0 + ph2 / 2:.1f})">share real</text>')
    lines(x0 + w / 2, CY0 + 268, [(f"calibrated: ECE {A['calib']['ece']:.2f} pt", 14, 400, NEW_INK),
                                   (f"this box: *p* = {hb['p']:.2f}", 16, 700, NEW_INK)], maxw=w - 16, lh=21)
    arrow([(x0 + w, MID), (1264, MID)])

    # -- c6 dial ------------------------------------------------------------------------------------------------
    x0, w = 1266, 184
    card(x0, CY0, w, CY1 - CY0, "λ dial", NEW, NEW_INK)
    gcx, gcy, R = x0 + w / 2, CY0 + 132, 64
    lo, hi = math.log10(0.3), math.log10(300)

    def ang(lam):
        return math.pi * (1 - (math.log10(lam) - lo) / (hi - lo))

    pts = [(gcx + R * math.cos(a), gcy - R * math.sin(a)) for a in [math.pi * k / 60 for k in range(61)]]
    polyline(pts, "#E2C9A0", 9)
    a10 = ang(LAM)
    arc = [(gcx + R * math.cos(a), gcy - R * math.sin(a)) for a in [math.pi - (math.pi - a10) * k / 30 for k in range(31)]]
    polyline(arc, "#C26A00", 9)
    for lam in (1, 3, 10, 30, 100):
        a = ang(lam)
        line(gcx + (R + 6) * math.cos(a), gcy - (R + 6) * math.sin(a), gcx + (R + 12) * math.cos(a), gcy - (R + 12) * math.sin(a),
             NEW_INK, 1.4)
        text(gcx + (R + 22) * math.cos(a), gcy - (R + 22) * math.sin(a) + 4, f"{lam:g}", 12, 600 if lam == LAM else 400,
             NEW_INK, "middle")
    line(gcx, gcy, gcx + (R - 14) * math.cos(a10), gcy - (R - 14) * math.sin(a10), NEW_INK, 3)
    parts.append(f'<circle cx="{gcx:.1f}" cy="{gcy:.1f}" r="5" fill="{NEW_INK}"/>')
    text(gcx - R - 2, gcy + 18, "fewer false", 12, 400, SUB, "start")
    text(gcx + R + 2, gcy + 18, "fewer misses", 12, 400, SUB, "end")
    lines(gcx, CY0 + 186, [(f"λ = {LAM}: keep *d* if", 15, 400, INK), ("*p*(*d*) ≥ 1/(1+λ) = 9.1%", 16, 700, NEW_INK),
                           ("one-threshold option:", 13, 400, SUB), (f"*s* ≥ τ(λ) = {A['cost_curves'][str(LAM)]['tau']:.3f}", 14, 400, INK),
                           ("or set by Fβ: λ = λ_{eq}(β)", 13, 400, SUB)], maxw=w - 14, lh=21)
    arrow([(x0 + w, MID), (1476, MID)])
    # use-case input
    text(1346, 26, "set per use case or weather:", 15, 600, NEW_INK, "end", maxw=330)
    text(1346, 46, "λ = false alarms accepted per avoided miss", 15, 600, NEW_INK, "end", maxw=360)
    arrow([(1358, 18), (1358, CY0 - 2)], color=PARAM, head="a-param")

    # -- c7 output ----------------------------------------------------------------------------------------------
    x0, w = 1478, 244
    card(x0, CY0, w, CY1 - CY0, "Kept boxes → planner", NEUTRAL)
    pw, ph = 220, 124
    boxes = [(b[0], b[1], b[2], b[3], MISS, 2.0, 1.0, "4 3") for b in m["missed"]]
    boxes += [(b[0], b[1], b[2], b[3], REAL if b[4] else FALSE, 2.0, 1.0, None) for b in m["kept"]]
    photo(m["src"], x0 + 12, CY0 + 46, pw, ph, (m["w"], m["h"]), boxes)
    ly = CY0 + 186
    for i, (col, lab, dash) in enumerate(((REAL, "real", None), (FALSE, "false alarm", None), (MISS, "missed", "4 3"))):
        lx = x0 + 14 + [0, 64, 168][i]
        rect(lx, ly - 10, 16, 11, "none", col, sw=2, rx=1, dash=dash)
        text(lx + 21, ly, lab, 13, 400, INK)
    lines(x0 + w / 2, ly + 26, [(f"λ = {LAM}: {m['found']} of {m['n_obj']} objects found,", 15, 600, INK),
                                (f"{m['false']} false alarms", 15, 600, INK),
                                (f"default 0.25: {m['found25']} found, {m['false25']} false", 14, 400, SUB),
                                (f"expected false alarms Σ(1 − *p*) = {m['est_fa']:.1f}", 13, 400, SUB)],
          maxw=w - 14, lh=20)

    # -- baseline lane -------------------------------------------------------------------------------------------
    BX0, BX1, BY0, BY1 = 274, 990, 446, 552
    rect(BX0, BY0, BX1 - BX0, BY1 - BY0, BASE["fill"], BASE["stroke"], dash="5 4")
    arrow([(585, CY1), (585, BY0 - 2)], color=BASE["stroke"], dash="4 3", head="a-grey")
    lines(BX0 + 16, BY0 + 26, [("Phase-1 baseline (for comparison)", 16, 600, BASE_INK),
                                ("SeqCRC step 1 per weather bin:", 14, 400, BASE_INK),
                                ("keep *s* ≥ τ̂(*b*), one cut per bin", 14, 400, BASE_INK),
                                ("for a fixed miss budget α", 14, 400, BASE_INK)], anchor="start", maxw=300, lh=19)
    # tau-hat per DAWN bin (alpha 0.2)
    taus = [("fog", 0.0251), ("rain", 0.1515), ("sand", 0.1297), ("snow", 0.2524)]
    cx, cy, ch = BX0 + 330, BY0 + 20, 58
    text(cx, cy - 4, "τ̂(*b*), DAWN, α = 0.2", 13, 600, BASE_INK, maxw=150)
    for i, (n, v) in enumerate(taus):
        bh = ch * v / 0.26
        rect(cx + 6 + i * 34, cy + 8 + ch - bh, 22, bh, "#B9BEC4", "#B9BEC4", sw=0, rx=2)
        text(cx + 17 + i * 34, cy + 8 + ch + 14, n, 12, 400, BASE_INK, "middle")
    # price of the budget
    qx, qy = BX0 + 500, BY0 + 24
    text(qx, qy, "false alarms per image", 13, 600, BASE_INK, maxw=200)
    text(qx, qy + 15, "to hold α = 0.2:", 13, 600, BASE_INK, maxw=200)
    for i, (lab, v) in enumerate((("DAWN snow", 1.6), ("rain at night", 117.1))):
        yy = qy + 30 + i * 22
        rect(qx, yy, max(2, 96 * v / 117.1), 12, "#B9BEC4", "#B9BEC4", sw=0, rx=2)
        text(qx + max(2, 96 * v / 117.1) + 6, yy + 11, f"{v:g}", 13, 600, BASE_INK)
        text(qx + 134, yy + 11, lab, 12, 400, BASE_INK, maxw=80)

    # -- planned ------------------------------------------------------------------------------------------------
    PX0, PY0 = 1478, 446
    rect(PX0, PY0, 244, 128, PLAN["fill"], PLAN["stroke"], dash="6 4")
    arrow([(1600, CY1), (1600, PY0 - 2)], color=PLAN["stroke"], dash="5 4", head="a-plan")
    sx, sy = PX0 + 18, PY0 + 22  # shield
    parts.append(f'<path d="M{sx} {sy} l14 -6 l14 6 v12 c0 10 -7 16 -14 19 c-7 -3 -14 -9 -14 -19 z" fill="#E8F0FC" stroke="{PLAN["stroke"]}" stroke-width="1.8"/>')
    parts.append(f'<path d="M{sx + 7} {sy + 11} l5 5 l9 -10" fill="none" stroke="{PLAN["stroke"]}" stroke-width="2.2"/>')
    lines(PX0 + 56, PY0 + 30, [("Planned (Phase 4)", 17, 600, PLAN_INK)], anchor="start", maxw=180)
    lines(PX0 + 18, PY0 + 66, [("CRC over λ per condition:", 15, 400, PLAN_INK), ("certify miss ≤ α, then", 15, 400, PLAN_INK),
                               ("stop / slow-down tiers", 15, 400, PLAN_INK)], anchor="start", maxw=214, lh=20)

    # -- offline band ----------------------------------------------------------------------------------------------
    OY0, OY1 = 598, 936
    rect(10, OY0, W - 20, OY1 - OY0, OFF_BAND, OFF_BAND, sw=0, rx=16)
    text(26, OY0 + 28, "OFFLINE · FITTED ONCE ON BDD100K TRAIN (THE ONLY LABELS USED FOR FITTING)", 15, 700, OFF_INK,
         spacing="1.0", maxw=760)
    QY0, QY1 = OY0 + 46, OY1 - 14
    QM = (QY0 + QY1) / 2

    # o1 train mosaic
    x0, w = 18, 244
    card(x0, QY0, w, QY1 - QY0, "BDD100K train", WHITE)
    mw, mh, g = 70, 39, 6
    for i, it in enumerate(A["train"]):
        photo(it["src"], x0 + 15 + (i % 3) * (mw + g), QY0 + 48 + (i // 3) * (mh + g), mw, mh)
    lines(x0 + w / 2, QY0 + 162, [("21,831 labelled frames", 16, 600, INK), ("clear, cloudy, rain, snow,", 14, 400, SUB),
                                   ("fog, dusk and night", 14, 400, SUB)], maxw=w - 16)
    arrow([(x0 + w, QM), (284, QM)])

    # o2 detect once
    x0, w = 286, 168
    card(x0, QY0, w, QY1 - QY0, "Detect once", WHITE)
    network_icon(x0 + w / 2 - 26, QY0 + 52, 56)
    cyx, cyy = x0 + w / 2 + 42, QY0 + 56  # cache cylinder
    parts.append(f'<path d="M{cyx - 18} {cyy} v40 a18 6 0 0 0 36 0 v-40" fill="#E4E7EB" stroke="#7B8794" stroke-width="1.4"/>')
    parts.append(f'<ellipse cx="{cyx}" cy="{cyy}" rx="18" ry="6" fill="#F4F6F8" stroke="#7B8794" stroke-width="1.4"/>')
    lines(x0 + w / 2, QY0 + 146, [("same frozen YOLOv8x", 15, 400, SUB), ("3.34 M boxes ≥ 0.001", 15, 600, INK),
                                   ("+ per-box outputs *h*(*d*)", 15, 400, SUB), ("cached to disk", 15, 400, SUB)],
          maxw=w - 12, lh=20)
    arrow([(x0 + w, QM), (476, QM)])

    # o3 label each box
    x0, w = 478, 252
    card(x0, QY0, w, QY1 - QY0, "Label each box", WHITE)
    L = A["label"]
    boxes = [(b[0], b[1], b[2], b[3], "#FFFFFF", 1.6, 0.95, "4 3") for b in L["gt"]]
    boxes += [(b[0], b[1], b[2], b[3], REAL if b[4] else FALSE, 1.8, 1.0, None) for b in L["det"]]
    photo(L["src"], x0 + 13, QY0 + 46, 226, 127, (L["w"], L["h"]), boxes)
    lines(x0 + w / 2, QY0 + 196, [("greedy IoU ≥ 0.5, grouped classes", 14, 400, SUB),
                                   ("→ *y*(*d*) = real or false", 16, 600, INK)], maxw=w - 14, lh=22)
    arrow([(x0 + w, QM), (752, QM)])

    # o4 fit the dial
    x0, w = 754, 446
    card(x0, QY0, w, QY1 - QY0, "Fit the dial", NEW, NEW_INK)
    # (a) cost curves: the one threshold per lambda is the minimiser
    ax0, ay0, aw, ah = x0 + 44, QY0 + 62, 150, 82
    mini_axes(ax0, ay0, aw, ah)
    lt0, lt1 = math.log10(0.002), math.log10(0.95)
    for lam, col, sw in (("1", "#D9B98A", 1.6), ("30", "#D9B98A", 1.6), ("10", "#C26A00", 2.4)):
        cc = A["cost_curves"][lam]
        pts = []
        for t, v in cc["points"]:
            v = min(v, 1.6)
            pts.append((ax0 + aw * (math.log10(t) - lt0) / (lt1 - lt0), ay0 + ah * (1 - (v - 1) / 0.6)))
        polyline(pts, col, sw)
        tx = ax0 + aw * (math.log10(cc["tau"]) - lt0) / (lt1 - lt0)
        parts.append(f'<circle cx="{tx:.1f}" cy="{ay0 + ah:.1f}" r="3.5" fill="{col}"/>')
    text(ax0 + aw / 2, ay0 + ah + 15, "threshold τ (log)", 12, 400, SUB, "middle")
    text(ax0 + 4, ay0 - 6, "cost, λ = 1 · 10 · 30", 12, 400, SUB)
    lines(ax0 + aw / 2, ay0 + ah + 36, [("τ(λ): exact minimiser,", 14, 600, NEW_INK), ("258 switch points", 14, 400, NEW_INK)],
          maxw=196, lh=18)
    # (b) model, a row across the bottom of the card
    for k in range(3):
        ox = x0 + 28 + k * 20
        parts.append(f'<polygon points="{ox},{QY0 + 262} {ox + 8},{QY0 + 238} {ox + 16},{QY0 + 262}" fill="#C26A00"/>')
    lines(x0 + 100, QY0 + 246, [("*p*(*d*): boosted trees on 68 features of 3.34 M boxes,", 14, 600, NEW_INK),
                                ("5-fold out-of-fold by near-duplicate group", 14, 400, NEW_INK)],
          anchor="start", maxw=w - 110, lh=19)
    # (c) lam_eq(beta)
    cx0, cy0, cw2, ch2 = x0 + 296, QY0 + 62, 100, 82
    mini_axes(cx0, cy0, cw2, ch2)
    b0, b1, l0, l1 = math.log10(0.25), math.log10(4), math.log10(0.1), math.log10(30)
    pts = [(cx0 + cw2 * (math.log10(b) - b0) / (b1 - b0), cy0 + ch2 * (1 - (math.log10(v) - l0) / (l1 - l0))) for b, v in A["lam_eq"]]
    polyline(pts, "#C26A00", 2)
    for bb, vv, lab in ((1, 1.81, "F1"), (2, 6.18, "F2")):
        px = cx0 + cw2 * (math.log10(bb) - b0) / (b1 - b0)
        py = cy0 + ch2 * (1 - (math.log10(vv) - l0) / (l1 - l0))
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{NEW_INK}"/>')
        text(px - 6, py - 6, lab, 12, 600, NEW_INK, "end")
    text(cx0 + cw2 / 2, cy0 + ch2 + 15, "β (log)", 12, 400, SUB, "middle")
    text(cx0, cy0 - 6, "λ_{eq}", 12, 400, SUB)
    lines(cx0 + cw2 / 2, cy0 + ch2 + 36, [("λ_{eq}(β) = (1+β²)/F∗ − 1", 14, 600, NEW_INK), ("F-score units", 14, 400, NEW_INK)],
          maxw=176, lh=18)
    arrow([(x0 + w, QM), (1222, QM)])

    # o5 check, then grade
    x0, w = 1224, 498
    card(x0, QY0, w, QY1 - QY0, "Check, then grade on unseen data", WHITE)
    gx0, gy0, gw, gh = x0 + 118, QY0 + 62, 184, 6 * 24
    v0, v1 = 0.85, 1.35

    def gxv(v):
        return gx0 + gw * (v - v0) / (v1 - v0)

    line(gxv(1), gy0 - 8, gxv(1), gy0 + gh - 4, "#7B8794", 1.2, dash="3 3")
    text(gxv(1), gy0 - 12, "one threshold = 1", 12, 400, SUB, "middle")
    for i, r in enumerate(A["grade"]):
        yy = gy0 + 10 + i * 24
        text(gx0 - 8, yy + 4, r["label"], 13, 400, INK, "end", maxw=110)
        line(gxv(r["lo"]), yy, gxv(r["hi"]), yy, "#C26A00", 2.2)
        parts.append(f'<circle cx="{gxv(r["det"]):.1f}" cy="{yy}" r="4.5" fill="#C26A00"/>')
        parts.append(f'<circle cx="{gxv(r["none"]):.1f}" cy="{yy}" r="4.5" fill="#FFFFFF" stroke="#7B8794" stroke-width="1.6"/>')
    for v in (0.9, 1.1, 1.3):
        text(gxv(v), gy0 + gh + 10, f"{v:.1f}", 12, 400, SUB, "middle")
    text(gx0 + gw / 2, gy0 + gh + 27, f"cost at λ = {LAM} vs one threshold", 12, 400, SUB, "middle")
    parts.append(f'<circle cx="{gx0 - 96}" cy="{gy0 + gh + 46}" r="4.5" fill="#C26A00"/>')
    text(gx0 - 87, gy0 + gh + 50, "per detection (95% CI)", 12, 400, INK)
    parts.append(f'<circle cx="{gx0 + 54}" cy="{gy0 + gh + 46}" r="4.5" fill="#FFFFFF" stroke="#7B8794" stroke-width="1.6"/>')
    text(gx0 + 63, gy0 + gh + 50, "default 0.25", 12, 400, INK)
    lines(x0 + 328, QY0 + 72, [("• calibration per cut", 14, 400, INK), ("• 5-fold + bootstrap", 14, 400, INK),
                               ("   stability", 14, 400, INK), ("• pre-registered", 14, 400, INK),
                               ("   adoption test", 14, 400, INK), ("• graded on BDD100K", 14, 600, INK),
                               ("   val (8,090) + DAWN", 14, 600, INK), ("   (889), never fitted", 14, 400, INK)],
          anchor="start", maxw=w - 338, lh=20)
    for i, it in enumerate((A["inputs"][1], A["inputs"][3])):  # graded frames: BDD100K val night, DAWN fog
        photo(it["src"], x0 + 330 + i * 78, QY0 + 228, 70, 39)

    # fitted parameters flow up into the online lane
    arrow([(1100, QY0), (1100, CY1 + 2)], color=PARAM, dash="6 4", head="a-param")
    arrow([(1180, QY0), (1180, 584), (1340, 584), (1340, CY1 + 2)], color=PARAM, dash="6 4", head="a-param")
    text(1092, 470, "fitted *p*", 15, 600, PARAM, "end")
    text(1348, 470, "τ(λ), λ_{eq}(β)", 15, 600, PARAM, "start")

    # -- legend ----------------------------------------------------------------------------------------------
    LY = 958
    items = [(NEUTRAL, None, "unchanged (as in the Phase-1 baseline)"), (NEW, None, "new in our method"),
             (PLAN, "6 4", "planned for Phase 4"), (BASE, "5 4", "baseline path (comparison)"),
             (dict(fill=OFF_BAND, stroke=OFF_BAND), None, "offline fitting")]
    x = 20
    for st, dash, label in items:
        rect(x, LY - 13, 26, 17, st["fill"], st["stroke"], sw=1.6, rx=4, dash=dash)
        text(x + 34, LY + 1, label, 15, 400, SUB)
        x += 34 + width(label, 15) + 28
    text(W - 20, LY + 1, "photos: BDD100K val / train and DAWN; boxes: cached YOLOv8x @1280 detections", 13, 400, SUB, "end")

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" '
           f'width="{W}" height="{H}" role="img" aria-label="Architecture of the weather-adaptive trade-off dial">\n'
           "<title>Per-detection trade-off dial on a frozen YOLOv8x: online pipeline with real frames, offline fitting, "
           "Phase-1 baseline</title>\n" + "\n".join(parts) + "\n</svg>\n")
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT} ({len(svg) / 1e3:.0f} kB)")
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)


if __name__ == "__main__":
    if "--prepare" in sys.argv:
        prepare()
    else:
        build()
