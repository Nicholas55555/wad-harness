"""Slide 2 chart: what the Phase-1 baseline pays for a fixed miss budget (per-bin CRC, alpha = 0.2).

Numbers: held-out false alarms per image of the per-bin CRC threshold, 200 calibration/test splits,
YOLOv8x @1280, grouped loss; DAWN with clean labels (results/run2/dawn/v8x_1280__group__clean/seeds200) and
BDD100K val (results/run2/run2.md, policies at alpha = 0.2). Every bin holds the budget (held-out miss 0.188-0.198).
Run:  python make_baseline_price.py  -> baseline_price.svg / .png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
                     "svg.fonttype": "path"})
INK, MUTED, GRID, BAR = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"   # reference palette: text, muted, grid, series-1

rows = [  # (group, condition, false alarms per image at alpha = 0.2)
    ("DAWN (daytime)", "snow", 1.63), ("DAWN (daytime)", "sand", 1.88), ("DAWN (daytime)", "rain", 2.63),
    ("DAWN (daytime)", "fog", 6.16),
    ("BDD100K day", "cloudy", 8.3), ("BDD100K day", "snow", 10.1), ("BDD100K day", "clear", 11.4),
    ("BDD100K day", "rain", 14.2),
    ("BDD100K night", "snow", 42.7), ("BDD100K night", "clear", 58.9), ("BDD100K night", "rain", 117.1),
]

fig, ax = plt.subplots(figsize=(8.6, 4.9))
y, ticks, labels, group_y = 0.0, [], [], {}
for i, (group, cond, v) in enumerate(rows):
    if i and rows[i - 1][0] != group:
        y += 0.7                                   # gap between groups
    ax.barh(y, v, height=0.62, color=BAR, edgecolor="white", linewidth=2)
    ax.text(v + 1.5, y, f"{v:.1f}" if v < 10 else f"{v:.0f}", va="center", ha="left", fontsize=11, color=INK)
    ticks.append(y)
    labels.append(cond)
    group_y.setdefault(group, []).append(y)
    y += 1
ax.set_yticks(ticks, labels, fontsize=11, color=INK)
ax.invert_yaxis()
for group, ys in group_y.items():
    ax.text(-0.115, (min(ys) + max(ys)) / 2, group, transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=11, fontweight="semibold", color=MUTED)
ax.set_xlim(0, 130)
ax.set_xlabel("false alarms per image needed to hold a 20% miss budget", fontsize=11, color=MUTED)
ax.xaxis.grid(True, color=GRID, linewidth=1)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="x", colors=MUTED, labelsize=10)
ax.tick_params(axis="y", length=0, pad=6)
fig.suptitle("The baseline's price: one fixed miss budget, very different false-alarm bills",
             x=0.02, ha="left", fontsize=14, fontweight="semibold", color=INK)
fig.text(0.02, 0.905, "Per-bin CRC threshold at α = 0.2 (SeqCRC step 1), YOLOv8x, 200 calibration/test splits. "
         "Every bin holds the budget;\nthe false alarms are whatever results. No tuning (0.25) runs at 0.8–1.8 per image "
         "but misses 19–56% of objects.", ha="left", va="top", fontsize=9.5, color=MUTED)
fig.subplots_adjust(left=0.31, right=0.97, top=0.80, bottom=0.12)
for ext in ("svg", "png"):
    fig.savefig(HERE / f"baseline_price.{ext}", dpi=250, facecolor="white")
print("wrote baseline_price.svg/.png")
