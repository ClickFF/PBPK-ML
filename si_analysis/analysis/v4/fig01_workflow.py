#!/usr/bin/env python3
"""Figure 1 (v4): study workflow in three stages, no fills or nested cards. Sample counts, encoder reuse,
statistical tests and the scenario inventory are in the caption, Table 1, Table 3 and Tables S4/S5.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v4/fig01_workflow.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
import style_v4 as S                                              # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch   # noqa: E402

S.use()

STAGES = [
    ("a", "PK-parameter prediction",
     [("Curated human PK data", S.OBS), ("ATFP graph embeddings\n+ RDKit descriptors", S.DLML),
      ("SVR regressor per endpoint", S.DLML), ("Predicted Fu, CL, VDss", S.DLML)]),
    ("b", "Held-out ML benchmarking",
     [("Test Set #1", S.OBS), ("DL–ML model (ML Group A)", S.DLML),
      ("vs descriptor models and\nsingle-representation controls", S.OBS),
      ("Compound-paired comparison", S.OBS)]),
    ("c", "PBPK input substitution",
     [("41 intravenous drugs,\nSimcyp minimal PBPK", S.OBS),
      ("Inputs replaced one source at a time\n(S+ or DL–ML)", S.SPLUS),
      ("Simulated vs clinical profiles", S.OBS), ("C–T profile, AUC, Cmax error", S.OBS)]),
]


def main():
    fig = S.figure(S.DOUBLE, 2.5)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.axis("off")
    bw, bh, x0s = 0.78, 0.14, [0.5, 1.5, 2.5]
    ys = [0.76, 0.545, 0.33, 0.115]
    for (letter, head, boxes), xc in zip(STAGES, x0s):
        ax.text(xc - bw / 2, 0.925, "(%s)" % letter, fontsize=9, fontweight="bold", ha="left", va="center")
        ax.text(xc - bw / 2 + 0.11, 0.925, head, fontsize=8, ha="left", va="center")
        for k, ((txt, col), y) in enumerate(zip(boxes, ys)):
            ax.add_patch(FancyBboxPatch((xc - bw / 2, y - bh / 2), bw, bh, boxstyle="square,pad=0",
                                        fc="white", ec=col, lw=0.8))
            ax.text(xc, y, txt, fontsize=7, ha="center", va="center", color="black")
            if k < len(boxes) - 1:
                ax.add_patch(FancyArrowPatch((xc, y - bh / 2), (xc, ys[k + 1] + bh / 2), arrowstyle="-|>",
                                             mutation_scale=7, lw=0.7, color=S.OBS))
    # hand-over arrows between stages
    for xa, xb, ya, yb in ((x0s[0] + bw / 2, x0s[1] - bw / 2, ys[1], ys[1]),
                           (x0s[1] + bw / 2, x0s[2] - bw / 2, ys[1], ys[1])):
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=7, lw=0.7,
                                     color=S.OBS, connectionstyle="arc3,rad=0"))
    S.save(fig, "main", "Figure1_workflow")


if __name__ == "__main__":
    main()
