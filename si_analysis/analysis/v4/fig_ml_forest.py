#!/usr/bin/env python3
"""Figure 3 and Figures S2, S3 (v4): compound-paired differences ML Group A - comparator on Test Set #1.

Source: outputs/tables/ml_benchmark_paired_stats_holm.csv (analysis/06_ml_benchmark_stats.py; 2000 bootstrap
resamples, seed 0; Wilcoxon on paired |error| for MAE, exact McNemar for within-2-fold; Holm within the
15-test family A vs C-G x 3 endpoints, separately per test).

Encoding (spec section 5): filled marker = Holm-adjusted p < 0.05; open = Holm p >= 0.05; the 95% CI is always
drawn. Delta RMSE and Delta R^2 have no paired test and are always open (descriptive).

  Figure 3  : (a) Delta MAE, (b) Delta within 2-fold, comparators C-G
  Figure S2 : Delta MAE, RMSE, R^2, within 2-fold, comparators C, D
  Figure S3 : same, comparators E, F, G

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v4/fig_ml_forest.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

S21 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S21 / "lib"))
import style_v4 as S                                              # noqa: E402

S.use()
D = pd.read_csv(S21 / "outputs" / "tables" / "ml_benchmark_paired_stats_holm.csv")
NAME = {"C": "C, published setting re-implemented", "D": "D, published protocol reproduced",
        "E": "E, ATFP embeddings only", "F": "F, RDKit descriptors only", "G": "G, merged embeddings only"}
EP = [("Fu", "^", "f$_u$"), ("CL", "o", "CL"), ("VDss", "s", "VD$_{ss}$")]
METRIC = {
    "MAE": ("delta_MAE", "delta_MAE_CI95_lo", "delta_MAE_CI95_hi", "p_holm_wilcoxon", 1,
            "ΔMAE (log$_{10}$ units)"),
    "RMSE": ("delta_RMSE", "delta_RMSE_CI95_lo", "delta_RMSE_CI95_hi", None, 1, "ΔRMSE (log$_{10}$ units)"),
    "R2": ("delta_R2", "delta_R2_CI95_lo", "delta_R2_CI95_hi", None, 1, "ΔR$^2$"),
    "FE2": ("delta_FE<2", "delta_FE<2_CI95_lo", "delta_FE<2_CI95_hi", "p_holm_mcnemar", 100,
            "Δ within 2-fold (% points)"),
}


def forest(ax, comps, metric, show_labels):
    col, lo, hi, ph, k, xl = METRIC[metric]
    y, ticks, labels = 0.0, [], []
    for c in comps:
        for ep, mk, _ in EP:
            r = D[(D.Comparator == c) & (D.Endpoint == ep)].iloc[0]
            fill = ph is not None and S.filled(r[ph])
            ax.plot([k * r[lo], k * r[hi]], [y, y], color=S.DLML, lw=0.9, solid_capstyle="butt")
            ax.plot(k * r[col], y, mk, ms=4.2, mfc=S.DLML if fill else "white", mec=S.DLML, mew=0.9)
            y -= 1
        ticks.append(y + 2)
        labels.append(NAME[c])
        y -= 0.8
    ax.axvline(0, color=S.OBS, lw=0.6, ls=(0, (3, 2)))
    ax.set_ylim(y + 0.3, 0.8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels if show_labels else [])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(xl)


def legend(fig, y, open_label="Holm-adjusted p ≥ 0.05"):
    h = [Line2D([], [], ls="", marker=m, mfc="white", mec=S.DLML, ms=4.2, label=l) for _, m, l in EP]
    h += [Line2D([], [], ls="", marker="o", mfc=S.DLML, mec=S.DLML, ms=4.2, label="Holm-adjusted p < 0.05"),
          Line2D([], [], ls="", marker="o", mfc="white", mec=S.DLML, ms=4.2, label=open_label)]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.55, y), handletextpad=0.3, columnspacing=1.1)


def main():
    comps = ["C", "D", "E", "F", "G"]
    fig, axes = S.plt.subplots(1, 2, figsize=(S.DOUBLE, 3.6))
    for k, (ax, m) in enumerate(zip(axes, ("MAE", "FE2"))):
        forest(ax, comps, m, k == 0)
        S.panel_label(ax, "ab"[k], dx=-0.05 if k else -0.62, dy=1.01)
    axes[0].set_title("Lower is better for ML Group A", fontsize=7.5, color=S.OBS)
    axes[1].set_title("Higher is better for ML Group A", fontsize=7.5, color=S.OBS)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.5)
    legend(fig, -0.005)
    S.save(fig, "main", "Figure3_ml_paired_forest")
    for name, cs, h in (("FigureS2_ml_paired_A_vs_CD", ["C", "D"], 2.6), ("FigureS3_ml_paired_A_vs_EFG", ["E", "F", "G"], 3.3)):
        fig, axes = S.plt.subplots(1, 4, figsize=(S.DOUBLE, h))
        for k, (ax, m) in enumerate(zip(axes, ("MAE", "RMSE", "R2", "FE2"))):
            forest(ax, cs, m, k == 0)
            S.panel_label(ax, "abcd"[k], dx=-0.1 if k else -1.3, dy=1.01)
        fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.8)
        legend(fig, -0.01, "Holm-adjusted p ≥ 0.05, or no test (ΔRMSE, ΔR$^2$)")
        S.save(fig, "si", name)


if __name__ == "__main__":
    main()
