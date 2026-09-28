#!/usr/bin/env python3
"""Figure S15 (v4; v3 Figure S14). Redraw SI Figure S14 (embedding-distance applicability domain) in the v3 figure style from the saved
outputs of run_embedding_ad_analysis.py; no statistic is recomputed.

v3 style: bold panel titles with (a)-(c) labels, "Spearman ρ" and "p < 0.001" notation, one shared
legend, 7.2-in print width.

    .venv_pbpk/bin/python repo/claude_new_Sep21/response_letter/minor_points/applicability_domain/scripts/replot_figure_s14.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt                                   # noqa: E402

S21 = Path(__file__).resolve().parents[2]
OUT = S21 / "response_letter" / "minor_points" / "applicability_domain" / "outputs"
sys.path.insert(0, str(S21 / "lib"))
import style_v4 as S                                              # noqa: E402
ENDP = [("Fu", "f$_u$", S.DLML), ("CL", "CL", S.DLML), ("VDss", "VD$_{ss}$", S.DLML)]


def main():
    d = pd.read_csv(OUT / "embedding_ad_compound_level.csv")
    s = pd.read_csv(OUT / "embedding_ad_summary.csv").set_index("Endpoint")
    S.use()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    for k, (ax, (ep, lab, col)) in enumerate(zip(axes, ENDP)):
        g = d[d.Endpoint == ep]
        out = g.Outside_AD.astype(bool)
        ax.scatter(g.Mean_5NN_distance[~out], g.Abs_error_log10[~out], s=7, alpha=0.6, color=col,
                   edgecolor="none", label="Inside domain")
        ax.scatter(g.Mean_5NN_distance[out], g.Abs_error_log10[out], s=14, alpha=0.9, color=S.OBS,
                   edgecolor="white", linewidth=0.3, label="Outside domain")
        c = np.polyfit(g.Mean_5NN_distance, g.Abs_error_log10, 1)
        xl = np.linspace(g.Mean_5NN_distance.min(), g.Mean_5NN_distance.max(), 50)
        ax.plot(xl, c[0] * xl + c[1], color="#222222", lw=1.0)
        r = s.loc[ep]
        ax.axvline(r.AD_threshold_train_LOO_p95, color="#555555", ls="--", lw=0.9)
        p = r.Spearman_permutation_p
        pt = "p < 0.001" if p < 0.001 else "p = %.3f" % p
        ax.text(0.03, 0.97, "Spearman ρ = %.2f\n%s\noutside: %d of %d" % (r.Spearman_rho, pt, r.N_outside_AD, r.N_test),
                transform=ax.transAxes, ha="left", va="top", fontsize=7,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        ax.set_title(lab, fontsize=8)
        ax.text(-0.2, 1.07, "(%s)" % "abc"[k], transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean 5-NN embedding distance", fontsize=8)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("Absolute error (log$_{10}$)", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.0)
    S.save(fig, "si", "FigureS15_applicability_domain")
    print("wrote Figure S15")


if __name__ == "__main__":
    main()
