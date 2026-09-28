#!/usr/bin/env python3
"""Figure S13 (v4; v3 Figure S12 + former main Figure 7c + full-PBPK scenarios): systematic Cmax
underprediction and model structure.

(a) mean signed log2 Cmax fold error per scenario with 95% bootstrap CI, nine minimal and nine full-PBPK
    scenarios (outputs/tables/cmax_bias_by_arm.csv);
(b) Simcyp library model relative to the observed-input control: change in signed Cmax fold error
    (v0_run0 - v1_run0) by template structure, full PBPK (n = 20) vs minimal (n = 21)
    (outputs/tables/v0_structure_contrast_by_compound.csv; statistics in v0_structure_contrast.csv);
(c), (d) signed Cmax fold error vs observed VDss in the observed-input control (v1_run0) and in the scenario
    with DL-ML Fu, VDss and CL (h2_run0), n = 40 with an observed VDss in the clinical data file; Spearman rho recomputed (outputs/tables/v4_figureS13_spearman.csv).

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v4/figS13_cmax_structure.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

S21 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S21 / "lib"))
import style_v4 as S                                              # noqa: E402

S.use()
T = S21 / "outputs" / "tables"


def main():
    cb = pd.read_csv(T / "cmax_bias_by_arm.csv").set_index("run_id")
    fig = S.figure(S.DOUBLE, 7.0)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], hspace=0.75, wspace=0.55)
    ax = fig.add_subplot(gs[:, 0])
    runs = S.MINIMAL + S.FULL
    ys = [-i for i in range(len(S.MINIMAL))] + [-(i + len(S.MINIMAL) + 0.8) for i in range(len(S.FULL))]
    for run, y in zip(runs, ys):
        r = cb.loc[run]
        col = S.OBS if run in S.MINIMAL else S.GREY
        ax.plot([r.ci_low, r.ci_high], [y, y], color=col, lw=0.9)
        ax.plot(r["mean"], y, "^", ms=4.2, mfc=col, mec=col)
    ax.axvline(0, color=S.OBS, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(ys)
    ax.set_yticklabels([S.LABEL[r].replace("Full PBPK, ", "") for r in runs], fontsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(min(ys) - 0.6, 0.6)
    ax.axhline(-(len(S.MINIMAL) - 0.1), color="#dddddd", lw=0.5)
    ax.text(0.02, -(len(S.MINIMAL) + 0.05), "Full PBPK (structural sensitivity)", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=6.5, color=S.GREY)
    ax.set_xlabel("Mean signed log$_2$ C$_{max}$ fold error (95% CI)")
    S.panel_label(ax, "a", dx=-0.95, dy=1.01)

    v0 = pd.read_csv(T / "v0_structure_contrast_by_compound.csv")
    ax = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(5)
    for i, (t, lab) in enumerate((("full", "Full PBPK\ntemplate"), ("minimal", "Minimal\ntemplate"))):
        v = v0[v0.tmpl == t].d_cmax.to_numpy()
        ax.plot(i + rng.uniform(-0.12, 0.12, len(v)), v, "o", ms=3, mfc=S.GREY, mec="white", mew=0.3)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=S.OBS, lw=1.2)
    ax.axhline(0, color=S.OBS, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Full PBPK\ntemplate (n = 20)", "Minimal\ntemplate (n = 21)"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Δ signed log$_2$ C$_{max}$ FE,\nlibrary − observed-input")
    S.panel_label(ax, "b", dx=-0.38, dy=1.02)

    per = pd.read_csv(T / "per_compound_metrics.csv")
    m = pd.read_csv(S21 / "data" / "compounds" / "pbpk_mechanistic_master_v3.csv").set_index("compound_id")
    out = []
    for k, (run, lab) in enumerate((("v1_run0", "Observed-input control"), ("h2_run0", "DL–ML Fu, VDss, CL"))):
        ax = fig.add_subplot(gs[k + 1, 1])
        # observed VDss from the clinical data file, as in the v3 analysis (P6); identical to the observed-input
        # values where present, one compound has no value there (n = 40)
        vd = pd.read_csv(S21 / "data" / "observed" / "observed_data_cleaned_deduplicated.csv"
                         ).groupby("compound_id")["vd(l/kg)"].first().astype(float)
        y = per[per.scenario == run].set_index("compound_id").fe_cmax_rel
        j = pd.concat([y.rename("y"), vd.rename("x")], axis=1).dropna()
        x, y = j.x, j.y
        rho, p = spearmanr(x, y)
        out.append(dict(scenario=run, n=len(y), spearman_rho=rho, p=p))
        ax.axhline(0, color=S.OBS, lw=0.6)
        ax.plot(x, y, "o", ms=2.8, mfc=S.OBS if k == 0 else S.DLML, mec="white", mew=0.3)
        ax.set_xscale("log")
        ax.set_ylim(-4.6, 1.8)
        ax.set_title(lab, fontsize=7)
        ax.text(0.04, 0.04, "ρ = %.2f\n%s" % (rho, S.pfmt(p)), transform=ax.transAxes, fontsize=6.5, va="bottom",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
        ax.set_xlabel("Observed VD$_{ss}$ (L kg$^{-1}$)")
        ax.set_ylabel("Signed log$_2$ C$_{max}$ FE")
        S.panel_label(ax, "cd"[k], dx=-0.38, dy=1.04)
    pd.DataFrame(out).to_csv(T / "v4_figureS13_spearman.csv", index=False, float_format="%.4f")
    S.save(fig, "si", "FigureS13_cmax_structure")
    print(pd.DataFrame(out).round(3))


if __name__ == "__main__":
    main()
