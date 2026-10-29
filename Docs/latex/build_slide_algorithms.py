"""Typeset the pseudocode for the slides from exactly the algorithms in methodology.tex.

Every algorithm environment is copied verbatim from methodology.tex onto its own small page
(algorithms_for_slides.tex), so the slides and Overleaf show the same pseudocode. Run from this folder:
    python build_slide_algorithms.py
    tectonic algorithms_for_slides.tex        (or compile it on Overleaf)
then screenshot or export each page (one algorithm per page) for the slide.
"""
import re
from pathlib import Path

here = Path(__file__).parent
src = (here / "methodology.tex").read_text(encoding="utf-8")
algs = re.findall(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", src, flags=re.S)
assert algs, "no algorithm environment found in methodology.tex"
algs = [a.replace("\\begin{algorithm}[t]", "\\begin{algorithm}[H]") for a in algs]

preamble = r"""\documentclass[11pt]{article}
\usepackage[paperwidth=6.9in,paperheight=3.9in,margin=0.12in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{algorithm}
\usepackage[noend]{algpseudocode}
\pagestyle{empty}
\begin{document}
"""
doc = preamble + "\n\\clearpage\n".join(algs) + "\n\\end{document}\n"
(here / "algorithms_for_slides.tex").write_text(doc, encoding="utf-8")
print(f"{len(algs)} algorithms copied from methodology.tex -> algorithms_for_slides.tex")
