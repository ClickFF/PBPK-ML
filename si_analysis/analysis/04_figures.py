#!/usr/bin/env python3
"""Main-text Figures 5, 6 and 7 -> outputs/figures/main/, from outputs/tables/ only.

Conventions (Manuscript/v1/CONVENTIONS.md, carried into Sep21):
  * drawn at print width (7.2 in), shared style lib/style.py, 400 dpi raster + vector copy
  * panel titles describe axes; interpretation lives in the caption
  * endpoints distinguishable without colour: circle = AUC, triangle = Cmax,
    diamond = C-T log-NRMSE, pentagon = C-T RMSE; a non-detectable contrast is a grey square
  * minimal-PBPK and full-PBPK arms never pooled
  * profile error is log-NRMSE / RMSE; log2 fold error is used for AUC and Cmax only

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/04_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import paths as P                                                 # noqa: E402
import arms as A                                                  # noqa: E402
import matplotlib                                                 # noqa: E402

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
import style                                                      # noqa: E402

style.use()
matplotlib.rcParams.update({"font.family": "sans-serif",
                            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                            "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                            "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                            "mathtext.default": "regular"})

W = 7.2
C_ENDP = {"log_NRMSE": ("#4a3aa7", "D"), "rmse_log10": ("#1baf7a", "p"),
          "fe_auc_abs": ("#2a78d6", "o"), "fe_cmax_abs": ("#eb6834", "^")}
GREY, INK, INK2, BAND = "#8b8a85", "#0b0b0b", "#52514e", "#efeeea"
MINIMAL = ["v0_run0", "v1_run0", "s1_run1", "s1_run2", "s1_run3",
           "h1_run0", "h1_run0_noCLr", "h2_run4", "h2_run0"]
FULL = ["h1_run1", "h1_run2", "h1_run3", "h1_run1_noCLr", "h1_run2_noCLr", "h1_run3_noCLr",
        "h2_run1", "h2_run2", "h2_run3"]


def per():
    return pd.read_csv(P.TABLES / "per_compound_metrics.csv")


# --------------------------------------------------------------------------------- Figure 5
def figure5():
    """v3: AUC and Cmax on the fold-error scale (predicted/observed, log axis) with the 2-fold band
    labelled; the median of every scenario is printed at the right of each panel (profile panels:
    median score; exposure panels: median fold error)."""
    d = per()
    order = MINIMAL + [None] + FULL
    ypos, y = {}, 0.0
    for r in order:
        if r is None:
            y += 0.9
            continue
        ypos[r] = y
        y += 1
    specs = [("log_NRMSE", "C–T profile\nlog-NRMSE", "score"),
             ("rmse_log10", "C–T profile\nRMSE (log$_{10}$ conc.)", "score"),
             ("fe_auc_rel", "AUC$_{0–t}$\nfold error (pred/obs)", "fold"),
             ("fe_cmax_rel", "C$_{max}$\nfold error (pred/obs)", "fold")]
    colours = {"log_NRMSE": C_ENDP["log_NRMSE"][0], "rmse_log10": C_ENDP["rmse_log10"][0],
               "fe_auc_rel": C_ENDP["fe_auc_abs"][0], "fe_cmax_rel": C_ENDP["fe_cmax_abs"][0]}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 4, figsize=(W, 6.6), sharey=True)
    for k, (ax, (col, xlab, kind)) in enumerate(zip(axes, specs)):
        c = colours[col]
        if kind == "fold":
            ax.set_xscale("log", base=2)
            ax.axvspan(0.5, 2, color=BAND, zorder=0)
            ax.axvline(1, color=INK2, lw=0.8, zorder=1)
            for f in (1 / 3, 3):
                ax.axvline(f, color=GREY, lw=0.6, ls=(0, (1, 1.6)), zorder=1)
        for r in MINIMAL + FULL:
            v = d.loc[d.scenario == r, col].dropna().to_numpy()
            if kind == "fold":
                v = np.clip(2.0 ** v, 1 / 16, 16)
            yy = ypos[r]
            ax.boxplot([v], positions=[yy], widths=0.6, orientation="horizontal",
                       showfliers=False, patch_artist=True, zorder=3,
                       medianprops=dict(color=INK, lw=1.4),
                       boxprops=dict(facecolor="white", edgecolor=c, lw=1.0),
                       whiskerprops=dict(color=c, lw=0.9), capprops=dict(color=c, lw=0.9))
            ax.plot(v, yy + rng.uniform(-0.2, 0.2, v.size), ls="none", marker="o", ms=2.2,
                    mfc=c, mec="none", alpha=0.45, zorder=4)
            med = np.median(v)
            ax.text(1.02, yy, ("%.2f" % med) if kind == "score" else ("%.2f×" % med),
                    transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6.6, color=INK2)
        ax.text(1.02, -1.15, "median", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=6.6, color=INK2, fontweight="bold")
        ax.set_xlabel(xlab, fontsize=8.8)
        ax.grid(True, axis="x", color="#e4e3df", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=7.5)
        if kind == "fold":
            ax.set_xlim(1 / 16, 16)
            ax.set_xticks([1 / 8, 1 / 2, 2, 8])
            ax.set_xticklabels(["1/8", "1/2", "2", "8"])
            ax.minorticks_off()
            ax.text(1.0, -1.15, "2-fold", ha="center", va="center", fontsize=6.8,
                    color=INK2, fontweight="bold", zorder=5)
        else:
            ax.set_xlim(left=0)
    axes[0].set_yticks([ypos[r] for r in MINIMAL + FULL])
    axes[0].set_yticklabels(MINIMAL + FULL, fontsize=8, family="monospace")
    axes[0].set_ylim(y - 0.4, -1.6)
    axes[0].tick_params(axis="y", length=0)
    for ax in axes:
        ax.axhline(ypos["h2_run0"] + 0.95, color=INK2, lw=0.7, ls=(0, (3, 2)))
    axes[0].text(-0.02, ypos["v0_run0"] - 1.15, "minimal PBPK", transform=axes[0].get_yaxis_transform(),
                 ha="right", fontsize=8.5, fontweight="bold")
    axes[0].text(-0.02, ypos["h1_run1"] - 0.75, "full PBPK", transform=axes[0].get_yaxis_transform(),
                 ha="right", fontsize=8.5, fontweight="bold")
    for ax, t_ in zip(axes, "abcd"):
        ax.text(0.0, 1.015, "(%s)" % t_, transform=ax.transAxes, fontsize=10.5, fontweight="bold")
    fig.tight_layout(w_pad=2.6)
    style.save(fig, P.FIG_MAIN, "Figure5_arm_error_distributions")
    plt.close(fig)
    print("  Figure 5")


# --------------------------------------------------------------------------------- Figure 6
def figure6():
    c = pd.read_csv(P.TABLES / "paired_contrasts.csv")
    fig, axes = plt.subplots(2, 4, figsize=(W, 4.4), sharex="col",
                             gridspec_kw=dict(height_ratios=[4, 4], hspace=0.45, wspace=0.12))
    titles = {"log_NRMSE": "C–T profile\nΔ log-NRMSE", "rmse_log10": "C–T profile\nΔ RMSE (log$_{10}$)",
              "fe_auc_abs": "AUC\nΔ |log$_2$ FE|", "fe_cmax_abs": "C$_{max}$\nΔ |log$_2$ FE|"}
    for r, g in enumerate(("A", "B")):
        sub = c[c.group == g]
        runs = list(dict.fromkeys(sub.run_id))
        for k, (col, *_rest) in enumerate(A.ENDPOINTS):
            ax = axes[r][k]
            colour, mk = C_ENDP[col]
            ax.axvline(0, color=INK2, lw=0.8)
            for y, run in enumerate(runs):
                row = sub[(sub.run_id == run) & (sub.endpoint == col)].iloc[0]
                det = bool(row.detectable)
                ax.plot([row.ci_low, row.ci_high], [y, y], color=colour if det else GREY, lw=1.6,
                        solid_capstyle="round", zorder=2)
                ax.plot([row.estimate], [y], ls="none", marker=mk if det else "s",
                        ms=6 if det else 5, mfc=colour if det else "white",
                        mec=colour if det else GREY, mew=1.1, zorder=3)
            ax.set_yticks(range(len(runs)))
            ax.set_yticklabels(runs if k == 0 else [], fontsize=8.5, family="monospace")
            ax.set_ylim(len(runs) - 0.5, -0.5)
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(True, axis="x", color="#e4e3df", lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(titles[col], fontsize=9.5)
        ref = A.run_of(A.REFERENCE[g])
        axes[r][0].text(-0.72, 1.50 if r == 0 else 1.07, "(%s) Group %s, reference %s" % ("ab"[r], g, ref),
                        transform=axes[r][0].transAxes, fontsize=10, fontweight="bold")
    fig.text(0.56, 0.005, "Hodges–Lehmann paired difference, test − reference "
             "(positive = larger error than the reference)", ha="center", fontsize=9)
    handles = [Line2D([], [], ls="none", marker=C_ENDP[e][1], mfc=C_ENDP[e][0], mec=C_ENDP[e][0],
                      ms=6, label=l) for e, l in (("log_NRMSE", "log-NRMSE"), ("rmse_log10", "RMSE"),
                                                ("fe_auc_abs", "AUC"), ("fe_cmax_abs", "C$_{max}$"))]
    handles.append(Line2D([], [], ls="none", marker="s", mfc="white", mec=GREY, ms=5,
                          label="interval covers zero"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 1.07), ncol=5,
               fontsize=8.5, frameon=False, handletextpad=0.3, columnspacing=1.0)
    fig.subplots_adjust(left=0.17, right=0.99, top=0.80, bottom=0.12)
    style.save(fig, P.FIG_MAIN, "Figure6_paired_contrasts")
    plt.close(fig)
    print("  Figure 6")


# --------------------------------------------------------------------------------- Figure 7
def figure7():
    lad = pd.read_csv(P.TABLES / "decomposition_ladder.csv")
    rout = pd.read_csv(P.TABLES / "decomposition_routing.csv")
    bias = pd.read_csv(P.TABLES / "cmax_bias_by_arm.csv").set_index("run_id")
    sc = pd.read_csv(P.TABLES / "v0_structure_contrast.csv")

    fig = plt.figure(figsize=(W, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1],
                          hspace=0.55, wspace=0.55)

    # (a) ladder
    ax = fig.add_subplot(gs[:, 0])
    steps = list(dict.fromkeys(lad.step))
    for off, col in ((-0.14, "fe_auc_abs"), (0.14, "fe_cmax_abs")):
        colour, mk = C_ENDP[col]
        s = lad[lad.endpoint == col].set_index("step")
        for y, st in enumerate(steps):
            r = s.loc[st]
            det = bool(r.detectable)
            ax.plot([r.ci_low, r.ci_high], [y + off] * 2, color=colour if det else GREY, lw=1.6)
            ax.plot([r.estimate], [y + off], ls="none", marker=mk, ms=6.5,
                    mfc=colour if det else "white", mec=colour if det else GREY, mew=1.1)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.axhspan(2.5, 4.5, color=BAND, zorder=0)
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels(steps, fontsize=8.8)
    ax.set_ylim(len(steps) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("change in |log$_2$ fold error| vs observed\nFu, V$_{Dss}$ and CL (v1_run0)",
                  fontsize=9.5)
    ax.grid(True, axis="x", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", mfc=C_ENDP["fe_auc_abs"][0],
                              mec=C_ENDP["fe_auc_abs"][0], ms=6, label="AUC"),
                       Line2D([], [], ls="none", marker="^", mfc=C_ENDP["fe_cmax_abs"][0],
                              mec=C_ENDP["fe_cmax_abs"][0], ms=6, label="C$_{max}$"),
                       Line2D([], [], ls="none", marker="o", mfc="white", mec=GREY, ms=6,
                              label="covers zero")],
              loc="upper right", fontsize=8.5, frameon=False)
    ax.text(-0.05, 1.03, "(a)", transform=ax.transAxes, fontsize=11, fontweight="bold",
            ha="right")

    # (b) routing
    ax = fig.add_subplot(gs[0, 1])
    ex = ["AUC, signed log2 FE", "Cmax, signed log2 FE", "C–T, log-NRMSE"]
    mat = rout.pivot(index="parameter", columns="exposure", values="abs_rho").loc[
        ["Fu", "VDss", "CLsys"], ex]
    pv = rout.pivot(index="parameter", columns="exposure", values="p").loc[["Fu", "VDss", "CLsys"], ex]
    im = ax.imshow(mat.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, "%.2f%s" % (v, "" if pv.values[i, j] < 0.05 else "\nn.s."),
                    ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.6 else INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["AUC", "C$_{max}$", "C–T\nlog-NRMSE"], fontsize=8.8)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["f$_u$", "V$_{Dss}$", "CL"], fontsize=9)
    ax.set_title("|Spearman ρ|, parameter vs exposure error", fontsize=9.2)
    ax.text(-0.3, 1.1, "(b)", transform=ax.transAxes, fontsize=11, fontweight="bold")

    # (c) Cmax bias per arm
    ax = fig.add_subplot(gs[1, 1])
    runs = ["v0_run0", "v1_run0", "s1_run3", "h1_run0_noCLr", "h2_run4", "h2_run0"]
    for y, r in enumerate(runs):
        b = bias.loc[r]
        ax.plot([b.ci_low, b.ci_high], [y, y], color=C_ENDP["fe_cmax_abs"][0], lw=1.6)
        ax.plot([b["mean"]], [y], ls="none", marker="^", ms=6, color=C_ENDP["fe_cmax_abs"][0])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels(runs, fontsize=8.3, family="monospace")
    ax.set_ylim(len(runs) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("mean signed log$_2$ C$_{max}$ fold error", fontsize=9.2)
    ax.grid(True, axis="x", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    # the full- vs minimal-template contrast is reported in the caption (v3), not in the panel
    ax.text(-0.3, 1.05, "(c)", transform=ax.transAxes, fontsize=11, fontweight="bold")
    style.save(fig, P.FIG_MAIN, "Figure7_parameter_vs_structure")
    plt.close(fig)
    print("  Figure 7")


if __name__ == "__main__":
    figure5()
    figure6()
    figure7()
