#!/usr/bin/env python3
"""Figure S9 (v4): compound-level effect of predicting clearance, C-T log-NRMSE (n = 40).
(a) per compound: log-NRMSE with observed CL on the S+ background (s1_run3) and the median over the four
predicted-CL scenarios (h1_run0, h1_run0_noCLr, h2_run4, h2_run0); (b) the two against each other, identity
and 2-fold band. Corrects the v3 axis label (v3 said "RMSE log10, time-weighted" but plotted log-NRMSE) and
removes the in-image title. Source: outputs/tables/per_compound_metrics.csv; statistics written to
outputs/tables/v4_figureS9_stats.csv.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v4/figS09_compound_heterogeneity.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from matplotlib.lines import Line2D

S21 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S21 / "lib"))
import style_v4 as S                                              # noqa: E402

S.use()
T = S21 / "outputs" / "tables"
PRED = ["h1_run0", "h1_run0_noCLr", "h2_run4", "h2_run0"]


def main():
    d = pd.read_csv(T / "per_compound_metrics.csv")
    w = d.pivot(index="compound_id", columns="scenario", values="log_NRMSE")
    nm = d.drop_duplicates("compound_id").set_index("compound_id").compound
    x = w["s1_run3"]
    y = w[PRED].median(axis=1)
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    diff = y - x
    rho, p = spearmanr(x, y)
    st = dict(n=len(x), n_worse=int((diff > 0).sum()), median_diff=float(diff.median()),
              wilcoxon_p=float(wilcoxon(y, x).pvalue), spearman_rho=rho, spearman_p=p)
    pd.DataFrame([st]).to_csv(T / "v4_figureS9_stats.csv", index=False, float_format="%.6g")

    order = sorted(x.index, key=lambda c: str(nm[c]).lower())
    fig = S.figure(S.DOUBLE, 6.4)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    for i, c in enumerate(order):
        yy = -i
        ax.plot([x[c], y[c]], [yy, yy], color=S.GREY, lw=0.7, zorder=1)
        ax.plot(x[c], yy, "o", ms=3.2, mfc=S.OBS, mec=S.OBS, zorder=2)
        ax.plot(y[c], yy, "o", ms=3.2, mfc="white", mec=S.OBS, mew=0.8, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks([-i for i in range(len(order))])
    ax.set_yticklabels([str(nm[c]).capitalize() for c in order], fontsize=6.5)
    ax.set_ylim(-len(order) + 0.3, 0.7)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("C–T log-NRMSE")
    ax.legend(handles=[Line2D([], [], ls="", marker="o", mfc=S.OBS, mec=S.OBS, ms=3.2, label="Observed CL"),
                       Line2D([], [], ls="", marker="o", mfc="white", mec=S.OBS, ms=3.2,
                              label="Predicted CL (median of 4)")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.2)
    S.panel_label(ax, "a", dx=-0.42, dy=1.04)

    ax = fig.add_subplot(gs[1].subgridspec(2, 1, height_ratios=[1, 0.9])[0])
    lo, hi = min(x.min(), y.min()) / 1.5, max(x.max(), y.max()) * 1.5
    g = np.array([lo, hi])
    ax.fill_between(g, g / 2, g * 2, color=S.LIGHT, lw=0, zorder=0)
    ax.plot(g, g, color=S.OBS, lw=0.7)
    ax.plot(x, y, "o", ms=3.4, mfc=S.GREY, mec="white", mew=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("C–T log-NRMSE, observed CL")
    ax.set_ylabel("C–T log-NRMSE, predicted CL (median of 4)")
    ax.text(0.05, 0.95, "Spearman ρ = %.2f\n%s; N = %d" % (rho, S.pfmt(p), len(x)), transform=ax.transAxes,
            ha="left", va="top", fontsize=7)
    S.panel_label(ax, "b", dx=-0.42, dy=1.02)
    S.save(fig, "si", "FigureS9_compound_heterogeneity")
    print(st)


if __name__ == "__main__":
    main()
