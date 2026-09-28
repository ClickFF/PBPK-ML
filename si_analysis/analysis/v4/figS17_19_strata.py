#!/usr/bin/env python3
"""Figures S17-S19 (v4; v3 Figures S16-S18): exploratory stratification (SI Section S7), neutral styling, no
p values in panels (multiplicity and exploratory status are stated in the captions and Table S13).

  S17  ECCS class x information-matched delta (h1_run0_noCLr - h2_run4): log-NRMSE, AUC, Cmax
  S18  secondary strata x matched delta, AUC |log2 FE|: clearance group, ionization, logP bin, VDss bin
  S19  ECCS class x cost of predicting CL (median of four predicted-CL scenarios - observed CL)
Source: data/compounds/pbpk_mechanistic_master_v3.csv (built by analysis/12_mechanistic_strata.py).

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v4/figS17_19_strata.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S21 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S21 / "lib"))
import style_v4 as S                                              # noqa: E402

S.use()
M = pd.read_csv(S21 / "data" / "compounds" / "pbpk_mechanistic_master_v3.csv")
ECCS = ["Class 1", "Class 2", "Class 3", "Class 4"]
STRATA = [("clearance_group", ["Metabolism-leaning", "Renal-leaning", "Uptake/Biliary-influenced", "Mixed/Other"],
           "Clearance group (template)"),
          ("ionization_pH74", ["Acid (ionized)", "Base (ionized)", "Ampholyte", "Neutral at pH 7.4"], "Ionization at pH 7.4"),
          ("logP_bin", ["<2", "2–4", ">4"], "logP (template)"),
          ("VDss_bin", ["Low (<0.7)", "Mid (0.7–2)", "High (>2)"], "Observed VD$_{ss}$ (L kg$^{-1}$)")]


def boxes(ax, y, groups, cats, rng):
    data = [y[groups == c].dropna().to_numpy() for c in cats]
    ax.boxplot(data, positions=range(len(cats)), widths=0.5, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="white", edgecolor=S.GREY, lw=0.7), medianprops=dict(color=S.OBS, lw=1.1),
               whiskerprops=dict(color=S.GREY, lw=0.6), capprops=dict(color=S.GREY, lw=0.6))
    for i, g in enumerate(data):
        ax.plot(i + rng.uniform(-0.13, 0.13, len(g)), g, "o", ms=2.4, mfc=S.OBS, mec="none", alpha=0.7)
    ax.axhline(0, color=S.OBS, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(range(len(cats)))
    lab = [c.replace("/", "/\n").replace("-leaning", "-\nleaning").replace(" at pH 7.4", "\nat pH 7.4")
           .replace(" (ionized)", "\n(ionized)") for c in cats]
    ax.set_xticklabels(["%s\nn = %d" % (l, len(g)) for l, g in zip(lab, data)], fontsize=6.5)
    return data


def eccs_figure(prefix, suffix, ylab, name):
    rng = np.random.default_rng(1)
    fig, axes = S.plt.subplots(1, 3, figsize=(S.DOUBLE, 2.6))
    for k, (ax, (col, head)) in enumerate(zip(axes, (("log_NRMSE", "C–T log-NRMSE"), ("fe_auc_abs", "AUC$_{0–t}$ |log$_2$ FE|"),
                                                      ("fe_cmax_abs", "C$_{max}$ |log$_2$ FE|")))):
        boxes(ax, M["%s%s%s" % (prefix, col, suffix)], M.ECCS_class, ECCS, rng)
        ax.set_title(head)
        if k == 0:
            ax.set_ylabel(ylab)
        S.panel_label(ax, "abc"[k], dx=-0.4 if k == 0 else -0.22, dy=1.04)
    fig.tight_layout(w_pad=1.2)
    S.save(fig, "si", name)


def main():
    eccs_figure("delta_", "__BU0_minus_TD", "Δ error\n(CLint, renal CL 0 − DL–ML CL)",
                "FigureS17_ECCS_matched_delta")
    rng = np.random.default_rng(2)
    fig, axes = S.plt.subplots(2, 2, figsize=(S.DOUBLE, 5.0), sharey=True)
    y = M["delta_fe_auc_abs__BU0_minus_TD"]
    for k, (ax, (var, cats, head)) in enumerate(zip(axes.flat, STRATA)):
        boxes(ax, y, M[var], cats, rng)
        ax.set_title(head)
        if k % 2 == 0:
            ax.set_ylabel("Δ AUC$_{0–t}$ |log$_2$ FE|\n(CLint, renal CL 0 − DL–ML CL)")
        S.panel_label(ax, "abcd"[k], dx=-0.2 if k % 2 == 0 else -0.08, dy=1.04)
    fig.tight_layout(h_pad=1.4, w_pad=1.0)
    S.save(fig, "si", "FigureS18_strata_matched_delta_AUC")
    eccs_figure("cost_", "__predCL_minus_obsCL", "Δ error, predicted CL\n− observed CL",
                "FigureS19_ECCS_cost_of_predicted_CL")


if __name__ == "__main__":
    main()
