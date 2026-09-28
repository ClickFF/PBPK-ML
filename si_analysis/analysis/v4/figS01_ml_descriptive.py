#!/usr/bin/env python3
"""Figure S1 (v4): descriptive performance of ML Group A and the single-representation controls (ML Groups E,
F, G) on Test Set #1, as a dot plot with no best-value highlighting (decision D4). Values are those of Table S1
(outputs/tables/ml_table2_recomputed.csv); paired inference is in Figure 3 and Figures S2, S3.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v4/figS01_ml_descriptive.py
"""
import sys
from pathlib import Path

import pandas as pd

S21 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S21 / "lib"))
import style_v4 as S                                              # noqa: E402

S.use()
GROUPS = [("A", "A, merged embeddings + RDKit"), ("E", "E, ATFP embeddings"), ("F", "F, RDKit descriptors"),
          ("G", "G, merged embeddings")]
EP = [("Fu", "^", "f$_u$"), ("CL", "o", "CL"), ("VDss", "s", "VD$_{ss}$")]
MET = [("MAE", "MAE (log$_{10}$)"), ("RMSE", "RMSE (log$_{10}$)"), ("GMFE", "GMFE"), ("R2", "R$^2$"),
       ("FE2", "Within 2-fold (%)")]


def main():
    d = pd.read_csv(S21 / "outputs" / "tables" / "ml_table2_recomputed.csv").set_index(["group", "endpoint"])
    fig, axes = S.plt.subplots(1, 5, figsize=(S.DOUBLE, 2.9))
    for k, (ax, (m, xl)) in enumerate(zip(axes, MET)):
        y, ticks = 0, []
        for g, lab in GROUPS:
            for ep, mk, _ in EP:
                ax.plot(d.loc[(g, ep), m], y, mk, ms=3.8, mfc="white", mec=S.OBS, mew=0.8)
                y -= 1
            ticks.append(y + 2)
            y -= 0.7
        ax.set_yticks(ticks)
        ax.set_yticklabels([l for _, l in GROUPS] if k == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(y + 0.3, 0.7)
        ax.set_xlabel(xl)
        ax.margins(x=0.14)
        ax.grid(axis="x", color="#e5e5e5", lw=0.3)
        S.panel_label(ax, "abcde"[k], dx=-0.1 if k else -1.35, dy=1.01)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], ls="", marker=mk, mfc="white", mec=S.OBS, ms=3.8, label=l) for _, mk, l in EP],
               loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6)
    S.save(fig, "si", "FigureS1_ml_descriptive")


if __name__ == "__main__":
    main()
