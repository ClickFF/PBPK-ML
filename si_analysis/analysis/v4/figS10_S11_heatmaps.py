#!/usr/bin/env python3
"""Figures S10 and S11 (v4): compound x scenario heatmaps of C-T profile error, drawn by one function so that
compound order (alphabetical), scenario order, figure size, group separators and reader-facing labels are
identical. S10 = log-NRMSE (range-normalised, primary; phenobarbital missing by the 0.2 log10 range rule);
S11 = RMSE of log10 concentrations (non-normalised). Each has its own colour bar (cividis, capped at the 97th
percentile of that metric; missing = grey). Lower values = better profile agreement.
Source: outputs/tables/per_compound_metrics.csv.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v4/figS10_S11_heatmaps.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

S21 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S21 / "lib"))
import style_v4 as S                                              # noqa: E402

S.use()
GROUPS = [["v0_run0", "v1_run0", "s1_run1", "s1_run2", "s1_run3"],
          ["h1_run0", "h1_run0_noCLr", "h2_run4", "h2_run0"], S.FULL]
GAP = 0.6


def heatmap(d, col, cbar_label, name):
    w = d.pivot(index="compound_id", columns="scenario", values=col)
    nm = d.drop_duplicates("compound_id").set_index("compound_id").compound
    rows = sorted(w.index, key=lambda c: str(nm[c]).lower())
    vmax = np.nanpercentile(w.to_numpy(), 97)
    cmap = matplotlib.colormaps[S.HEATMAP].copy()
    cmap.set_bad(S.MISSING)
    fig = S.figure(S.DOUBLE, 8.8)
    ax = fig.add_axes([0.2, 0.2, 0.68, 0.77])
    x0, xt, xl = 0.0, [], []
    for g in GROUPS:
        m = np.ma.masked_invalid(w.loc[rows, g].to_numpy(float))
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest",
                       extent=(x0, x0 + len(g), len(rows), 0))
        xt += [x0 + i + 0.5 for i in range(len(g))]
        xl += [S.LABEL[r].replace("Full PBPK, ", "") for r in g]
        x0 += len(g) + GAP
    ax.set_xlim(0, x0 - GAP)
    ax.set_ylim(len(rows), 0)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, rotation=60, ha="right", rotation_mode="anchor", fontsize=6.5)
    ax.set_yticks([i + 0.5 for i in range(len(rows))])
    ax.set_yticklabels(["%s (%d)" % (str(nm[c]).capitalize(), c) for c in rows], fontsize=6.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # group headings above the columns (plain text, no borders)
    x0 = 0.0
    for g, lab in zip(GROUPS, ["PBPK Group A", "PBPK Group B", "Full PBPK (sensitivity)"]):
        ax.text(x0 + len(g) / 2, -0.6, lab, ha="center", va="bottom", fontsize=7)
        x0 += len(g) + GAP
    cax = fig.add_axes([0.9, 0.55, 0.018, 0.3])
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label(cbar_label + "\n(lower = better agreement)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_linewidth(0.5)
    S.save(fig, "si", name)
    return vmax


def main():
    d = pd.read_csv(S21 / "outputs" / "tables" / "per_compound_metrics.csv")
    v1 = heatmap(d, "log_NRMSE", "C–T log-NRMSE", "FigureS10_heatmap_logNRMSE")
    v2 = heatmap(d, "rmse_log10", "RMSE of log$_{10}$ concentrations", "FigureS11_heatmap_RMSE_log10")
    print("97th-percentile caps: log-NRMSE %.3f, RMSE %.3f" % (v1, v2))


if __name__ == "__main__":
    main()
