#!/usr/bin/env python3
"""Figure S16 (v4; v3 Figure S15): information-matched clearance-strategy comparison and renal-clearance
sensitivity (decision D5). Supplementary, audit-level figure; neutral styling.

(a) Information-matched primary comparison, S+ CLint (renal CL = 0) - DL-ML total CL (h1_run0_noCLr - h2_run4),
    four absolute-error endpoints; Hodges-Lehmann estimate and 95% bootstrap CI; filled = Holm-adjusted
    p < 0.05 (Holm within the contrast's absolute-error family), open otherwise.
(b) Retained renal-clearance sensitivity: signed AUC0-t log2 FE per compound without (renal CL = 0) and with
    the template renal CL (h1_run0_noCLr -> h1_run0).
(c) Exploratory: change in signed AUC FE caused by the template renal CL vs template CLRbase as a fraction of
    the observed total CL (neutral grey).
The secondary pragmatic comparison (h1_run0 - h2_run4) is reported in Table S11 only.
Sources: outputs/tables/clearance_hierarchy_{contrasts,clr_mechanistic,spearman}.csv.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v4/figS16_clearance_hierarchy.py
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
ENDP = ["log_NRMSE", "rmse_log10", "fe_auc_abs", "fe_cmax_abs"]
LAB = {"log_NRMSE": "C–T log-NRMSE", "rmse_log10": "C–T RMSE (log$_{10}$)", "fe_auc_abs": "AUC$_{0–t}$ |log$_2$ FE|",
       "fe_cmax_abs": "C$_{max}$ |log$_2$ FE|"}


def main():
    c = pd.read_csv(T / "clearance_hierarchy_contrasts.csv")
    mech = pd.read_csv(T / "clearance_hierarchy_clr_mechanistic.csv")
    sp = pd.read_csv(T / "clearance_hierarchy_spearman.csv")
    fig = S.figure(S.DOUBLE, 2.8)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.8, 1], wspace=0.6)

    ax = fig.add_subplot(gs[0])
    a = c[c.contrast == "A"].set_index("endpoint")
    for i, ep in enumerate(ENDP):
        r = a.loc[ep]
        ax.plot([r.ci_low, r.ci_high], [-i, -i], color=S.OBS, lw=0.9)
        ax.plot(r.estimate, -i, S.MARK[ep], ms=4.2, mec=S.OBS, mew=0.9, mfc=S.OBS if S.filled(r.p_holm) else "white")
    ax.axvline(0, color=S.OBS, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks([-i for i in range(len(ENDP))])
    ax.set_yticklabels([LAB[e] for e in ENDP])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-len(ENDP) + 0.4, 0.6)
    ax.set_xlabel("S+ CLint (renal CL = 0) − DL–ML total CL")
    S.panel_label(ax, "a", dx=-0.85, dy=1.03)

    ax = fig.add_subplot(gs[1])
    for _, r in mech.iterrows():
        ax.plot([0, 1], [r.auc_rel_h1_run0_noCLr, r.auc_rel_h1_run0], color="#bdbdbd", lw=0.6, zorder=1)
    ax.plot(np.zeros(len(mech)), mech.auc_rel_h1_run0_noCLr, "o", ms=2.4, color=S.BU, zorder=2)
    ax.plot(np.ones(len(mech)), mech.auc_rel_h1_run0, "o", ms=2.4, color=S.BU, zorder=2)
    for x, col in ((0, "auc_rel_h1_run0_noCLr"), (1, "auc_rel_h1_run0")):
        ax.plot([x - 0.18, x + 0.18], [mech[col].median()] * 2, color=S.OBS, lw=1.4, zorder=3)
    ax.axhspan(-1, 1, color=S.LIGHT, lw=0, zorder=0)
    ax.axhline(0, color=S.OBS, lw=0.6)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Renal CL = 0", "Template\nrenal CL"])
    ax.set_ylabel("Signed AUC$_{0–t}$ log$_2$ FE")
    ax.set_title("S+ CLint scenario", fontsize=7.5)
    S.panel_label(ax, "b", dx=-0.55, dy=1.03)

    ax = fig.add_subplot(gs[2])
    nz = mech[mech.CLRbase_template_Lh > 0]
    z = mech[mech.CLRbase_template_Lh == 0]
    ax.plot(nz.CLR_frac_of_obsCL, nz.d_fe_auc_rel, "o", ms=3, mfc=S.GREY, mec="white", mew=0.3)
    ax.plot(np.full(len(z), 3e-5), z.d_fe_auc_rel, "x", ms=3.2, color=S.GREY)
    ax.set_xscale("log")
    ax.axhline(0, color=S.OBS, lw=0.6)
    ax.set_xlabel("Template CLRbase / observed total CL")
    ax.set_ylabel("Δ signed AUC$_{0–t}$ log$_2$ FE")
    r = sp[(sp.x == "CLR_frac_of_obsCL") & (sp.y == "d_fe_auc_rel") & (sp.subset == "all 41")].iloc[0]
    ax.text(0.04, 0.05, "Spearman ρ = %.2f; N = %d" % (r.spearman_rho, r.n), transform=ax.transAxes, fontsize=6.5)
    S.panel_label(ax, "c", dx=-0.42, dy=1.03)
    S.save(fig, "si", "FigureS16_clearance_hierarchy")


if __name__ == "__main__":
    main()
