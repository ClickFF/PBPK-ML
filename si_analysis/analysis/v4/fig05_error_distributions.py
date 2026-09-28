#!/usr/bin/env python3
"""Figure 5 (v4): per-compound error of the nine minimal-PBPK scenarios (full-PBPK scenarios are summarised in
Table S9, lower block, and appear in the Figure S10/S11 heatmaps). Panels: (a) C-T log-NRMSE (n = 40),
(b) C-T RMSE of log10 concentrations (n = 41), (c) AUC0-t and (d) Cmax fold error (predicted/observed, log2
axis, 2-fold band). Medians are written to outputs/tables/v4_figure5_medians.csv for the caption.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v4/fig05_error_distributions.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S21 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S21 / "lib"))
import style_v4 as S                                              # noqa: E402

S.use()
T = S21 / "outputs" / "tables"
COLOR = {"v0_run0": S.OBS, "v1_run0": S.OBS, "s1_run1": S.SPLUS, "s1_run2": S.SPLUS, "s1_run3": S.SPLUS,
         "h1_run0": S.BU, "h1_run0_noCLr": S.BU, "h2_run4": S.TD, "h2_run0": S.DLML}
PANELS = [("log_NRMSE", "C–T log-NRMSE\n", False), ("rmse_log10", "C–T RMSE\n(log$_{10}$ concentration)", False),
          ("fe_auc_rel", "AUC$_{0–t}$ fold error\n(predicted/observed)", True), ("fe_cmax_rel", "C$_{max}$ fold error\n(predicted/observed)", True)]


def main():
    d = pd.read_csv(T / "per_compound_metrics.csv")
    runs = S.MINIMAL
    fig, axes = S.plt.subplots(1, 4, figsize=(S.DOUBLE, 3.5), sharey=True)
    rng = np.random.default_rng(3)
    med = []
    for k, (ax, (col, xl, fold)) in enumerate(zip(axes, PANELS)):
        for i, run in enumerate(runs):
            v = d[d.scenario == run][col].dropna().to_numpy(float)
            y = -i
            ax.boxplot([v], positions=[y], orientation="horizontal", widths=0.55, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor="white", edgecolor=COLOR[run], lw=0.7),
                       medianprops=dict(color=COLOR[run], lw=1.2), whiskerprops=dict(color=COLOR[run], lw=0.6),
                       capprops=dict(color=COLOR[run], lw=0.6))
            ax.plot(v, y + rng.uniform(-0.18, 0.18, len(v)), "o", ms=1.6, color=COLOR[run], alpha=0.55, zorder=3)
            med.append(dict(scenario=run, label=S.LABEL[run], endpoint=col, n=len(v), median=float(np.median(v)),
                            median_fold=float(2 ** np.median(v)) if fold else np.nan))
        if fold:
            ax.axvspan(-1, 1, color=S.LIGHT, lw=0, zorder=0)
            ax.axvline(0, color=S.OBS, lw=0.6)
            ax.set_xlim(-4.6, 4.6)
            ax.set_xticks([-4, -2, 0, 2, 4])
            ax.set_xticklabels(["1/16", "1/4", "1", "4", "16"])
        ax.set_xlabel(xl)
        S.panel_label(ax, "abcd"[k], dx=-0.08 if k else -1.28, dy=1.01)
    axes[0].set_yticks([-i for i in range(len(runs))])
    axes[0].set_yticklabels([S.LABEL[r] for r in runs])
    axes[0].set_ylim(-len(runs) + 0.4, 0.6)
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=0.6)
    pd.DataFrame(med).to_csv(T / "v4_figure5_medians.csv", index=False, float_format="%.4f")
    S.save(fig, "main", "Figure5_error_distributions")


if __name__ == "__main__":
    main()
