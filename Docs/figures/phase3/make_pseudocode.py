"""Slide images of the pseudocode and key equations, line for line the same as Docs/latex/methodology.tex.

The report typesets the algorithms with algorithmicx; this script draws the same lines (same numbering, same
comments) in Computer Modern with matplotlib, so the slides need no PDF conversion. If methodology.tex changes,
change ALG1/ALG2/EQUATIONS below to match (or screenshot Docs/latex/algorithms_for_slides.pdf instead).

Run:  python make_pseudocode.py   -> pseudocode_alg1/alg2/equations .svg and .png next to this file
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.unicode_minus": False, "svg.fonttype": "path"})
PT = 1 / 72  # inch per point
SIZE, NUMSIZE, LINE, INDENT = 11, 8.5, 15.2, 16.5
WIDTH = 6.9  # inches, about the report's text width

# Each line: (number or None, indent level, text in mathtext, comment or None).  r"\mathbf{...}" = keyword.
ALG1 = ("Algorithm 1", "Fitting the dial (offline, BDD100K train only)", [
    (None, 0, r"$\mathbf{Require}$: labelled train frames $\mathcal{D}_{\mathrm{train}}$; frozen detector $f$", None),
    (1, 0, r"$\mathbf{for\ all}\ x \in \mathcal{D}_{\mathrm{train}}\ \mathbf{do}$", None),
    (2, 1, r"$(D(x), h) \leftarrow f(x)$", r"floor 0.001, NMS 0.5, $\leq 300$ boxes; keep per-box outputs"),
    (3, 1, r"$y(d) \leftarrow$ greedy IoU $\geq 0.5$ match of $d \in D(x)$ to labels", "grouped classes"),
    (4, 0, r"$\mathbf{Option\ A}$: sort scores; keep ends of equal-score blocks; cumulative real and false counts", None),
    (5, 1, r"$\tau(\cdot) \leftarrow$ corners of the lower convex hull", r"all $\lambda$ at once"),
    (6, 0, r"$\mathbf{Option\ B}$: $\varphi(d) \leftarrow [\mathrm{context}(d, D(x)),\ \mathrm{per\text{-}box}(h(d))]$", "PCA fitted on train"),
    (7, 1, r"$g \leftarrow$ gradient boosting on $\{(\varphi(d), y(d))\}$", "log-loss"),
    (8, 1, r"$p_{\mathrm{oof}} \leftarrow$ 5-fold out-of-fold probabilities, folds by near-duplicate group", None),
    (9, 0, r"$\mathbf{for\ all}\ \beta\ \mathbf{do}$", None),
    (10, 1, r"$\lambda_{\mathrm{eq}}(\beta) \leftarrow (1+\beta^2)/F^*_\beta - 1$", r"F-score map, $F^*_\beta$ on train"),
    (11, 0, r"check calibration near every cut $1/(1+\lambda)$ and fold stability; adopt only if the pre-registered", None),
    (None, 0, r"criteria hold", None),
    (12, 0, r"$\mathbf{return}\ \tau(\cdot),\ g$ (with PCA), $\lambda_{\mathrm{eq}}(\cdot)$", None),
])
ALG2 = ("Algorithm 2", "Applying the dial (online, every camera frame)", [
    (None, 0, r"$\mathbf{Require}$: frame $x$; setting $\lambda$ (or $\beta$); option $\in \{$A, B$\}$", None),
    (1, 0, r"$\mathbf{if}\ \beta$ is given $\mathbf{then}$", None),
    (2, 1, r"$\lambda \leftarrow \lambda_{\mathrm{eq}}(\beta)$", "F-score map"),
    (3, 0, r"$\mathbf{if}$ a budget $\alpha$ is certified for condition $c$ $\mathbf{then}$", None),
    (4, 1, r"$\lambda \leftarrow \max(\lambda, \hat{\lambda}_c)$", r"planned: CRC over $\lambda$"),
    (5, 0, r"$(D, h) \leftarrow f(x)$", "one detector pass"),
    (6, 0, r"$\mathbf{for\ all}\ d \in D\ \mathbf{do}$", None),
    (7, 1, r"$\mathbf{if}$ option A $\mathbf{then}$", None),
    (8, 2, r"keep $d$ iff $s(d) \geq \tau(\lambda)$", None),
    (9, 1, r"$\mathbf{else}$", None),
    (10, 2, r"$p(d) \leftarrow \sigma(g(\varphi(d)))$; keep $d$ iff $p(d) \geq 1/(1+\lambda)$", "cost-optimal cut"),
    (11, 0, r"$\widehat{\mathrm{FA}} \leftarrow \Sigma_{d\ \mathrm{kept}} (1 - p(d))$", "expected false alarms, no labels needed"),
    (12, 0, r"$\mathbf{return}$ kept detections, $\widehat{\mathrm{FA}}$", None),
])
EQUATIONS = [  # (label, mathtext) - the key equations of methodology.tex
    ("cost of a setting", r"$C_\lambda(k) = \sum_x\,[\,\lambda\, M_k(x) + F_k(x)\,]$"),
    ("one threshold (option A)", r"$\tau(\lambda) = \arg\min_\tau \sum_{x \in \mathcal{D}_{\mathrm{train}}} [\,\lambda\, M_\tau(x) + F_\tau(x)\,]$"),
    ("per detection (option B)", r"$\mathrm{keep}\ d \Leftrightarrow 1 - p(d) \leq \lambda\, p(d) \Leftrightarrow p(d) \geq \frac{1}{1+\lambda}$"),
    ("model loss (log-loss)", r"$\ell = -\sum_d\, [\,y_d \log p_d + (1-y_d) \log(1-p_d)\,]$"),
    ("F-score units", r"$\lambda_{\mathrm{eq}}(\beta) = \frac{1+\beta^2}{F^*_\beta} - 1$"),
    ("planned: CRC over $\\lambda$", r"$\hat{\lambda}_c = \inf\{\lambda : \frac{n_c}{n_c+1}\hat{R}_c(\lambda) + \frac{1}{n_c+1} \leq \alpha\}$"),
]


def draw_algorithm(name, title, lines, out):
    h = (44 + (len(lines) - 1) * LINE) * PT  # 4 pt margins, caption block 30 pt, last line + 6 pt rule
    fig = plt.figure(figsize=(WIDTH, h))
    to_in = fig.dpi_scale_trans
    top = h - 4 * PT
    fig.add_artist(plt.Line2D([0, WIDTH], [top, top], transform=to_in, color="black", lw=0.8))
    y = top - 12.5 * PT
    fig.text(0.02, y, r"$\mathbf{" + name.replace(" ", r"\ ") + "}$  " + title, transform=to_in, fontsize=SIZE, va="baseline")
    y -= 5 * PT
    fig.add_artist(plt.Line2D([0, WIDTH], [y, y], transform=to_in, color="black", lw=0.5))
    y -= 12.5 * PT
    for num, level, body, comment in lines:
        if num is not None:
            fig.text(0.26, y, f"{num}:", transform=to_in, fontsize=NUMSIZE, ha="right", va="baseline")
        x = 0.33 + level * INDENT * PT if num is not None or body.startswith("criteria") else 0.02
        fig.text(x, y, body, transform=to_in, fontsize=SIZE, va="baseline")
        if comment:
            fig.text(WIDTH - 0.02, y, r"$\triangleright$ " + comment, transform=to_in, fontsize=SIZE, ha="right", va="baseline")
        y -= LINE * PT
    y += LINE * PT - 6 * PT
    fig.add_artist(plt.Line2D([0, WIDTH], [y, y], transform=to_in, color="black", lw=0.8))
    for ext in ("svg", "png"):
        fig.savefig(HERE / f"{out}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def draw_equations(out):
    rowh = 0.52
    h = rowh * len(EQUATIONS) + 0.42
    fig = plt.figure(figsize=(WIDTH, h))
    to_in = fig.dpi_scale_trans
    for i, (label, eq) in enumerate(EQUATIONS):
        y = h - 0.46 - i * rowh
        fig.text(0.02, y, label, transform=to_in, fontsize=SIZE, va="center", color="#444444")
        fig.text(2.15, y, eq, transform=to_in, fontsize=SIZE + 3, va="center")
    for ext in ("svg", "png"):
        fig.savefig(HERE / f"{out}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


draw_algorithm(*ALG1, "pseudocode_alg1")
draw_algorithm(*ALG2, "pseudocode_alg2")
draw_equations("pseudocode_equations")
print("wrote pseudocode_alg1/alg2/equations .svg/.png")
