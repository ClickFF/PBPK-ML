#!/usr/bin/env python3
"""Figure 6 (v4): paired contrasts against the reference of each PBPK comparison group.
(a) PBPK Group A, reference = observed-input control (v1_run0); (b) PBPK Group B, reference = observed CL on
S+ background (s1_run3). Hodges-Lehmann estimate with 95% bootstrap CI (10,000 resamples); filled marker =
Holm-adjusted p < 0.05 (Holm within group x endpoint), open otherwise; CI always drawn.
Source: outputs/tables/paired_contrasts.csv.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v4/fig06_paired_contrasts.py
"""
import sys
from pathlib import Path

import pandas as pd
from matplotlib.lines import Line2D

S21 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S21 / "lib"))
import style_v4 as S                                              # noqa: E402

S.use()
COLOR = {"v0_run0": S.OBS, "s1_run1": S.SPLUS, "s1_run2": S.SPLUS, "s1_run3": S.SPLUS,
         "h1_run0": S.BU, "h1_run0_noCLr": S.BU, "h2_run4": S.TD, "h2_run0": S.DLML}
ENDP = ["log_NRMSE", "rmse_log10", "fe_auc_abs", "fe_cmax_abs"]
HEAD = {"log_NRMSE": "Δ C–T log-NRMSE", "rmse_log10": "Δ C–T RMSE (log$_{10}$)",
        "fe_auc_abs": "Δ AUC$_{0–t}$ |log$_2$ FE|", "fe_cmax_abs": "Δ C$_{max}$ |log$_2$ FE|"}
GROUPS = [("A", ["v0_run0", "s1_run1", "s1_run2", "s1_run3"], "Reference: observed-input control"),
          ("B", ["h1_run0", "h1_run0_noCLr", "h2_run4", "h2_run0"], "Reference: observed CL on S+ background")]


def main():
    pc = pd.read_csv(S21 / "outputs" / "tables" / "paired_contrasts.csv")
    fig, axes = S.plt.subplots(2, 4, figsize=(S.DOUBLE, 4.1), sharex="col")
    for r, (g, runs, ref) in enumerate(GROUPS):
        for c, ep in enumerate(ENDP):
            ax = axes[r, c]
            for i, run in enumerate(runs):
                x = pc[(pc.group == g) & (pc.run_id == run) & (pc.endpoint == ep)].iloc[0]
                col = COLOR[run]
                ax.plot([x.ci_low, x.ci_high], [-i, -i], color=col, lw=0.9, solid_capstyle="butt")
                ax.plot(x.estimate, -i, S.MARK[ep], ms=4.2, mec=col, mew=0.9,
                        mfc=col if S.filled(x.p_holm) else "white")
            ax.axvline(0, color=S.OBS, lw=0.6, ls=(0, (3, 2)))
            ax.set_ylim(-len(runs) + 0.4, 0.6)
            ax.set_yticks([-i for i in range(len(runs))])
            ax.set_yticklabels([S.LABEL[x] for x in runs] if c == 0 else [])
            ax.tick_params(axis="y", length=0)
            if r == 0:
                ax.set_title(HEAD[ep])
            if r == 1:
                ax.set_xlabel("test − reference")
        dy = 1.2 if r == 0 else 1.06
        axes[r, 0].text(-1.55, dy, "(%s)" % "ab"[r], transform=axes[r, 0].transAxes, fontsize=9,
                        fontweight="bold", va="bottom")
        axes[r, 0].text(-1.33, dy, ref, transform=axes[r, 0].transAxes, fontsize=7.5, va="bottom",
                        color=S.OBS)
    h = [Line2D([], [], ls="", marker="o", mfc=S.OBS, mec=S.OBS, ms=4.2, label="Holm-adjusted p < 0.05"),
         Line2D([], [], ls="", marker="o", mfc="white", mec=S.OBS, ms=4.2, label="Holm-adjusted p ≥ 0.05")]
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), h_pad=1.8, w_pad=0.6)
    fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.6, -0.01))
    S.save(fig, "main", "Figure6_paired_contrasts")


if __name__ == "__main__":
    main()
