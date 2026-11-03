"""Draw the Phase 3 architecture figure (architecture.svg) for the slides and the Overleaf methodology.

Run:  python make_architecture.py            -> architecture.svg next to this file
The PDF (Overleaf) and PNG (slides fallback) are rendered from the same SVG with Edge, see README in Docs.
Text widths are checked with Segoe UI metrics (Windows fonts) so every line fits its box.
"""
import re
import sys
from pathlib import Path

from PIL import ImageFont

OUT = Path(__file__).with_name("architecture_v1.svg")  # v1 (text-only), superseded 2026-10-29 by make_architecture.py
W, H = 1200, 690
FONT = "'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif"
FONTS = {400: "C:/Windows/Fonts/segoeui.ttf", 600: "C:/Windows/Fonts/seguisb.ttf", 700: "C:/Windows/Fonts/segoeuib.ttf"}

INK, SUB = "#1F2933", "#52606D"
NEUTRAL = dict(fill="#F4F6F8", stroke="#7B8794")
NEW = dict(fill="#FFE9C2", stroke="#C26A00")
NEW_GROUP = dict(fill="#FFF8EC", stroke="#D97706")
NEW_INK = "#7A3E00"
PLAN = dict(fill="#FFFFFF", stroke="#2F6FDB")
PLAN_INK = "#1E4FA3"
BASE = dict(fill="#F3F3F3", stroke="#9AA0A6")
BASE_INK = "#62676C"
OFF_BAND, OFF_INK = "#EEF6F0", "#2F6B45"
ARROW, PARAM = "#3E4C59", "#C26A00"

parts, problems = [], []


def width(s, size, weight=400):
    plain = s.replace("*", "")
    return ImageFont.truetype(FONTS.get(weight, FONTS[400]), size).getlength(plain)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rich(s, size):
    """'*x*' -> italic tspan (variables); '_{eq}' -> subscript tspan; leading spaces kept (indent)."""
    out = []
    for i, chunk in enumerate(re.split(r"\*", s)):
        if not chunk:
            continue
        pieces = []
        for j, sub in enumerate(re.split(r"_\{([^}]*)\}", chunk)):
            if j % 2:  # subscript: shift down, smaller, then shift back
                pieces.append(f'<tspan dy="{size * 0.28:.1f}" font-size="{size * 0.72:.1f}">{esc(sub)}</tspan>'
                              f'<tspan dy="{-size * 0.28:.1f}">​</tspan>')
            elif sub:
                pieces.append(esc(sub))
        body = "".join(pieces)
        out.append(f'<tspan font-style="italic">{body}</tspan>' if i % 2 else body)
    return "".join(out)


def text(x, y, s, size=15, weight=400, fill=INK, anchor="start", maxw=None, spacing=None):
    if maxw is not None and width(re.sub(r"_\{([^}]*)\}", r"\1", s), size, weight) > maxw:
        problems.append(f"too wide ({width(s, size, weight):.0f} > {maxw:.0f}): {s!r}")
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    s = re.sub(r"^ +", lambda m: " " * (2 * len(m.group(0))), s)  # SVG collapses spaces; indent with NBSP
    parts.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill}" text-anchor="{anchor}"{ls}>{rich(s, size)}</text>')


def rect(x, y, w, h, fill, stroke, sw=1.6, rx=10, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def arrow(pts, color=ARROW, sw=2.0, dash=None, head="a-dark"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    path = "M " + " L ".join(f"{x} {y}" for x, y in pts)
    parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#{head})"/>')


def box(x, y, w, h, title, lines, style, title_ink=INK, body_ink=SUB, tsize=18, bsize=15, lh=19.5, dash=None,
        title_lines=None, center=True, sw=1.6):
    rect(x, y, w, h, style["fill"], style["stroke"], sw=sw, dash=dash)
    cx = x + w / 2 if center else x + 12
    anchor = "middle" if center else "start"
    inner = w - 20
    ty = y + 27
    for t in (title_lines or [title]):
        text(cx, ty, t, tsize, 600, title_ink, anchor, inner)
        ty += tsize + 4
    by = ty + 2
    for ln in lines:
        if isinstance(ln, tuple):  # (text, size, weight, colour)
            s, sz, wt, col = ln
            text(cx, by, s, sz, wt, col, anchor, inner)
            by += sz + 5
        else:
            text(cx, by, ln, bsize, 400, body_ink, anchor, inner)
            by += lh
    if by - lh + 8 > y + h:
        problems.append(f"box overflow: {title!r} ({by - lh + 8:.0f} > {y + h})")


# ---------------------------------------------------------------- canvas and defs
parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')
parts.append(
    "<defs>"
    f'<marker id="a-dark" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{ARROW}"/></marker>'
    f'<marker id="a-param" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{PARAM}"/></marker>'
    f'<marker id="a-grey" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{BASE["stroke"]}"/></marker>'
    f'<marker id="a-plan" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{PLAN["stroke"]}"/></marker>'
    '<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#AFC2D6"/><stop offset="1" stop-color="#E3EAF1"/></linearGradient>'
    "</defs>")

# ---------------------------------------------------------------- online lane
text(20, 40, "ONLINE · EVERY CAMERA FRAME", 14, 700, SUB, spacing="1.2")
Y0, Y1 = 96, 236                       # top / bottom of the online boxes
YM = (Y0 + Y1) // 2

# A: camera frame with a small picture
rect(20, Y0, 120, Y1 - Y0, NEUTRAL["fill"], NEUTRAL["stroke"])
text(80, Y0 + 26, "Camera", 18, 600, INK, "middle", 100)
text(80, Y0 + 47, "frame *x*", 18, 600, INK, "middle", 100)
px, py, pw, ph = 32, Y0 + 60, 96, 52
parts.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="4" fill="url(#sky)" stroke="#7B8794" stroke-width="1"/>')
parts.append(f'<polygon points="{px + 34},{py + 22} {px + 62},{py + 22} {px + pw},{py + ph} {px},{py + ph}" fill="#5F6B78"/>')
parts.append(f'<rect x="{px}" y="{py + 18}" width="{pw}" height="5" fill="#FFFFFF" opacity="0.55"/>')   # fog band
for i in range(4):
    parts.append(f'<line x1="{px + 48}" y1="{py + 26 + 6 * i}" x2="{px + 48}" y2="{py + 29 + 6 * i}" stroke="#FFFFFF" stroke-width="2"/>')
for i in range(7):
    xx = px + 10 + 13 * i
    parts.append(f'<line x1="{xx}" y1="{py + 4}" x2="{xx - 4}" y2="{py + 14}" stroke="#FFFFFF" stroke-width="1.2" opacity="0.8"/>')
text(80, Y1 - 10, "any weather", 15, 400, SUB, "middle", 100)

arrow([(140, YM), (162, YM)])

# B: frozen detector
box(164, Y0, 160, Y1 - Y0, "YOLOv8x", ["frozen, COCO", "weights · 1280 px", "floor 0.001", "NMS 0.5 · ≤ 300"], NEUTRAL)

# chips: what the detector hands on
c1y, c2y = Y0, 174
rect(348, c1y, 140, 62, NEUTRAL["fill"], NEUTRAL["stroke"])
text(418, c1y + 25, "boxes *D*(*x*)", 15, 600, INK, "middle", 124)
text(418, c1y + 46, "class, score *s*", 15, 400, SUB, "middle", 124)
rect(348, c2y, 140, 62, NEW["fill"], NEW["stroke"])
text(418, c2y + 25, "per-box outputs", 15, 600, NEW_INK, "middle", 124)
text(418, c2y + 46, "*h*(*d*) · now kept", 15, 400, NEW_INK, "middle", 124)
arrow([(324, c1y + 31), (346, c1y + 31)])
arrow([(324, c2y + 31), (346, c2y + 31)])

# NEW group: features -> chance -> dial
GX0, GX1 = 510, 1020
rect(GX0, 60, GX1 - GX0, Y1 - 60 + 14, NEW_GROUP["fill"], NEW_GROUP["stroke"], sw=2.4, rx=14)
text(GX0 + 14, 82, "NEW · PER-DETECTION TRADE-OFF DIAL", 14, 700, NEW_INK, spacing="1.0")
arrow([(488, c1y + 31), (520, c1y + 31)])
arrow([(488, c2y + 31), (520, c2y + 31)])
box(522, Y0, 150, Y1 - Y0, "Features φ(*d*)", ["20 context:", "score, class, size,", "rank, overlaps", "+ 48 from *h*(*d*)"],
    NEW, NEW_INK, NEW_INK)
arrow([(672, YM), (688, YM)])
box(690, Y0, 150, Y1 - Y0, "Chance real", ["*p*(*d*) = P(real | φ)", "boosted trees,", "calibrated on", "unseen frames"],
    NEW, NEW_INK, NEW_INK)
arrow([(840, YM), (856, YM)])
box(858, Y0, 150, Y1 - Y0, "λ dial", ["keep *d* if", ("*p*(*d*) ≥ 1/(1+λ)", 16, 700, NEW_INK), "simple option:", "*s* ≥ τ(λ)"],
    NEW, NEW_INK, NEW_INK)
arrow([(1008, YM), (1042, YM)])

# knob: what the user sets
kx, ky = 806, 30
parts.append(f'<circle cx="{kx}" cy="{ky}" r="15" fill="#FFFFFF" stroke="{NEW_GROUP["stroke"]}" stroke-width="2.2"/>')
for ang in range(-120, 121, 40):
    import math
    a = math.radians(ang - 90)
    parts.append(f'<line x1="{kx + 18 * math.cos(a):.1f}" y1="{ky + 18 * math.sin(a):.1f}" x2="{kx + 22 * math.cos(a):.1f}" '
                 f'y2="{ky + 22 * math.sin(a):.1f}" stroke="{NEW_GROUP["stroke"]}" stroke-width="1.6"/>')
a = math.radians(40 - 90)
parts.append(f'<line x1="{kx}" y1="{ky}" x2="{kx + 11 * math.cos(a):.1f}" y2="{ky + 11 * math.sin(a):.1f}" '
             f'stroke="{NEW_INK}" stroke-width="2.6" stroke-linecap="round"/>')
text(834, 25, "set per use case or weather: λ = false", 15, 600, NEW_INK, maxw=350)
text(834, 45, "alarms accepted per avoided miss (or β)", 15, 600, NEW_INK, maxw=350)
arrow([(933, 52), (933, Y0 - 2)], color=PARAM, head="a-param")

# F: output
box(1044, Y0, 136, Y1 - Y0, "Kept boxes", ["→ planner", "+ expected false", "alarms Σ(1 − *p*),", "no labels needed"], NEUTRAL)

# ---------------------------------------------------------------- baseline lane (comparison) and planned step
BY0, BY1 = 278, 358
rect(164, BY0, 536, BY1 - BY0, BASE["fill"], BASE["stroke"], dash="5 4")
text(178, BY0 + 24, "Phase-1 baseline (comparison): SeqCRC step 1 per weather bin", 15, 600, BASE_INK, maxw=510)
text(178, BY0 + 46, "bin *b* → threshold τ̂(*b*) for a miss budget α → keep if *s* ≥ τ̂(*b*)", 15, 400, BASE_INK, maxw=510)
text(178, BY0 + 67, "one cut for every box; false alarms are an output, not a choice", 15, 400, BASE_INK, maxw=510)
arrow([(244, Y1), (244, BY0 - 2)], color=BASE["stroke"], dash="4 3", head="a-grey")

box(1044, BY0, 136, 128, "Planned (Ph. 4)", ["CRC over λ per", "condition: miss", "budget α; stop /", "slow tiers"],
    PLAN, PLAN_INK, PLAN_INK, tsize=16, dash="6 4")
arrow([(1112, Y1), (1112, BY0 - 2)], color=PLAN["stroke"], dash="5 4", head="a-plan")

# ---------------------------------------------------------------- offline band
OY0, OY1 = 420, 634
rect(10, OY0, 1180, OY1 - OY0, OFF_BAND, OFF_BAND, sw=0, rx=16)
text(24, OY0 + 26, "OFFLINE · FITTED ONCE ON BDD100K TRAIN (THE ONLY LABELS USED FOR FITTING)", 14, 700, OFF_INK, spacing="1.0", maxw=700)
BY = OY0 + 46
BH = 152
WHITE = dict(fill="#FFFFFF", stroke="#7B8794")
box(20, BY, 140, BH, "BDD100K", ["train frames", "21,831, labelled", "day · night ·", "rain · snow"], WHITE,
    title_lines=["BDD100K"])
arrow([(160, BY + BH / 2), (182, BY + BH / 2)])
box(184, BY, 160, BH, "Detect once", ["same frozen model", "3.34 M boxes", "≥ 0.001, + *h*(*d*)", "cached to disk"], WHITE)
arrow([(344, BY + BH / 2), (366, BY + BH / 2)])
box(368, BY, 160, BH, "Label each box", ["greedy IoU ≥ 0.5,", "grouped classes", "→ *y*(*d*) = real", "or false"], WHITE)
arrow([(528, BY + BH / 2), (550, BY + BH / 2)])
box(552, BY, 332, BH, "Fit the dial", ["• τ(λ): exact switch points of", "   λ·misses + false alarms (258)",
                                       "• *p*(*d*): boosted trees, 5-fold out-of-",
                                       "   fold by near-duplicate group", "• λ_{eq}(β) = (1 + β²)/F∗ − 1 on train"],
    NEW, NEW_INK, NEW_INK, center=False)
arrow([(884, BY + BH / 2), (906, BY + BH / 2)])
box(908, BY, 272, BH, "Check, then grade", ["• calibration at every cut", "• fold + bootstrap stability",
                                             "• pre-registered adoption test", "• graded on BDD100K val (8,090)",
                                             "   + DAWN (889), 95% intervals"], WHITE, center=False)

# fitted parameters flow up into the online lane
arrow([(765, BY), (765, Y1 + 2)], color=PARAM, dash="6 4", head="a-param")
arrow([(872, BY), (872, Y1 + 2)], color=PARAM, dash="6 4", head="a-param")
text(757, 392, "fitted *p*", 15, 600, PARAM, "end")
text(880, 392, "τ(λ), λ_{eq}(β)", 15, 600, PARAM, "start")

# ---------------------------------------------------------------- legend
LY = 664
items = [(NEUTRAL, None, "unchanged (as in the Phase-1 baseline)"), (NEW, None, "new in our method"),
         (PLAN, "6 4", "planned for Phase 4"), (BASE, "5 4", "baseline path (comparison)"),
         (dict(fill=OFF_BAND, stroke=OFF_BAND), None, "offline fitting")]
x = 20
for st, dash, label in items:
    rect(x, LY - 13, 26, 17, st["fill"], st["stroke"], sw=1.6, rx=4, dash=dash)
    text(x + 34, LY + 1, label, 14, 400, SUB)
    x += 34 + width(label, 14) + 26

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
       f'role="img" aria-label="Architecture of the weather-adaptive trade-off dial">\n'
       f"<title>Per-detection trade-off dial on a frozen YOLOv8x: online pipeline, offline fitting, Phase-1 baseline</title>\n"
       + "\n".join(parts) + "\n</svg>\n")
OUT.write_text(svg, encoding="utf-8")
print(f"wrote {OUT} ({len(svg):,} bytes); legend ends at x={x:.0f}")
if problems:
    print("PROBLEMS:\n  " + "\n  ".join(problems))
    sys.exit(1)
