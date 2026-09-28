#!/usr/bin/env python3
"""Figure S14 (v4; v3 Figure S13): sensitivity of the Cmax bias to the read-out definition, redrawn from the
saved outputs of response_letter/P6/scripts/cmax_readout_sensitivity.py (no statistic recomputed).
(a) mean signed log2 Cmax FE per minimal scenario under three read-outs; (b) bias with all matched points vs
bias restricted to observed times inside the simulated time range; (c) per-compound signed Cmax FE vs time of
the observed peak in the observed-input control (Spearman from cmax_bias_vs_peak_time.csv).
Note: P6 read its per-scenario errors from data/eval_notebook_rerun_2026_09_20 (recorded in
figure_layout_qc_v4.md); values differ from per_compound_metrics.csv in the third decimal.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v4/figS14_cmax_readout.py
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
P6 = S21 / "response_letter" / "P6" / "outputs"


def main():
    rd = pd.read_csv(P6 / "cmax_readout_definition_summary.csv").set_index("run_id").loc[S.MINIMAL]
    rs = pd.read_csv(P6 / "cmax_interpolation_support_sensitivity.csv").set_index("run_id").loc[S.MINIMAL]
    bc = pd.read_csv(P6 / "cmax_readout_by_compound.csv")
    pk = pd.read_csv(P6 / "cmax_bias_vs_peak_time.csv").set_index("run_id").loc["v1_run0"]
    fig = S.figure(S.DOUBLE, 3.1)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    defs = [("bias_published", "o", S.OBS, "PCHIP at observed times (used)"),
            ("bias_sim_window", "s", S.GREY, "Simulated peak, observed window"),
            ("bias_sim_full", "^", "#bdbdbd", "Simulated peak, full grid")]
    for i, run in enumerate(S.MINIMAL):
        for (c, mk, col, _), dy in zip(defs, (0.22, 0.0, -0.22)):
            ax.plot(rd.loc[run, c], -i + dy, mk, ms=3.6, mfc="white" if col == "#bdbdbd" else col, mec=col if col != "#bdbdbd" else S.GREY, mew=0.8)
    ax.axvline(0, color=S.OBS, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks([-i for i in range(len(S.MINIMAL))])
    ax.set_yticklabels([S.LABEL[r] for r in S.MINIMAL], fontsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-1.0, 0.1)
    ax.set_xlabel("Mean signed log$_2$ C$_{max}$ FE")
    ax.legend(handles=[Line2D([], [], ls="", marker=mk, ms=3.6, mfc="white" if col == "#bdbdbd" else col,
                              mec=col if col != "#bdbdbd" else S.GREY, label=l) for _, mk, col, l in defs],
              loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=1, handletextpad=0.2)
    S.panel_label(ax, "a", dx=-1.05, dy=1.02)

    ax = fig.add_subplot(gs[1])
    g = np.array([-1.0, 0.0])
    ax.plot(g, g, color=S.OBS, lw=0.7)
    ax.plot(rs.bias_published, rs.bias_restricted, "o", ms=3.4, mfc=S.GREY, mec="white", mew=0.3)
    ax.set_xlim(-1.0, 0.0)
    ax.set_ylim(-1.0, 0.0)
    ax.set_aspect("equal")
    ax.set_xlabel("Bias, all matched points")
    ax.set_ylabel("Bias, points inside\nsimulated time range")
    S.panel_label(ax, "b", dx=-0.45, dy=1.02)

    ax = fig.add_subplot(gs[2])
    v = bc[bc.run_id == "v1_run0"]
    ax.axhline(0, color=S.OBS, lw=0.6)
    ax.plot(v.t_obs_peak, v.bias_published, "o", ms=3, mfc=S.OBS, mec="white", mew=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Time of observed peak (h)")
    ax.set_ylabel("Signed log$_2$ C$_{max}$ FE")
    ax.text(0.97, 0.04, "ρ = %.2f\n%s; N = %d" % (pk.spearman_rho, S.pfmt(pk.p), pk.n), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
    S.panel_label(ax, "c", dx=-0.42, dy=1.02)
    S.save(fig, "si", "FigureS14_cmax_readout")


if __name__ == "__main__":
    main()
