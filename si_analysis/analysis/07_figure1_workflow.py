#!/usr/bin/env python3
"""Figure 1 — executed study workflow (replaces the v7.2 conceptual schematic) and a simplified
TOC graphic that keeps the original two-stage concept.

Every count on the figure is taken from the manuscript tables (Table 1, Sep21 outputs):
Training #1 / Test #1: Fu 4042 / 633, CL 1287 / 177, VDss 1287 / 177; Training #2: Fu 4585,
CL 1354, VDss 1354; Test #2: 110 (Fu measured for 90); 41 Simcyp-ready compounds.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/07_figure1_workflow.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch     # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "manuscript" / "v3" / "figures"
matplotlib.rcParams.update({"font.family": "sans-serif",
                            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                            "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                            "mathtext.default": "regular", "pdf.fonttype": 42,
                            "svg.fonttype": "none"})
INK, INK2, GREY = "#1a1a1a", "#4d4d4d", "#8b8a85"
C_DATA, C_ML, C_PBPK, C_WARN = "#e8f0fa", "#eef5ee", "#fbf1e6", "#b3261e"
E_DATA, E_ML, E_PBPK = "#2a78d6", "#2e8b57", "#c8741f"


def box(ax, x, y, w, h, text, fc="white", ec=INK2, fs=6.6, bold=False, lw=0.8, ls="-", color=INK,
        ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    tx = x + w / 2 if ha == "center" else x + 0.012
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", zorder=3, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=INK2, ls="-", lw=0.9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 color=color, lw=lw, ls=ls, zorder=1, shrinkA=0, shrinkB=0))


def panel(ax, x, y, w, h, letter, title, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.015",
                                fc=fc, ec=ec, lw=1.0, zorder=0))
    ax.text(x + 0.012, y + h - 0.022, letter, fontsize=10, fontweight="bold", va="top", color=INK)
    ax.text(x + 0.045, y + h - 0.024, title, fontsize=7.6, fontweight="bold", va="top", color=ec)


def figure1():
    fig = plt.figure(figsize=(7.2, 6.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------------- A: data and endpoint-specific model development ------------------------
    panel(ax, 0.01, 0.53, 0.485, 0.46, "A", "Data and endpoint-specific model development", C_DATA, E_DATA)
    box(ax, 0.03, 0.855, 0.205, 0.085,
        "Curated human PK data (Jia et al.)\nFu · CL (L/h/kg) · VDss (L/kg)\nlog$_{10}$-transformed", fs=6.3)
    box(ax, 0.26, 0.895, 0.215, 0.045, "Training Set #1\nFu 4042 · CL 1287 · VDss 1287", fs=6.1)
    box(ax, 0.26, 0.845, 0.215, 0.045, "Test Set #1 (held out)\nFu 633 · CL 177 · VDss 177", fs=6.1,
        ec=E_DATA, lw=1.1)
    arrow(ax, 0.235, 0.905, 0.26, 0.915)
    arrow(ax, 0.235, 0.885, 0.26, 0.866)
    box(ax, 0.03, 0.715, 0.205, 0.105,
        "Endpoint-specific ATFP encoders\n(Fu, CL, VDss; graph attention,\nBayesian hyperparameter search)\ntrained once on Training #1",
        fs=6.1)
    box(ax, 0.26, 0.755, 0.215, 0.065, "RDKit descriptors\n(217 physicochemical /\ntopological features)", fs=6.1)
    arrow(ax, 0.30, 0.895, 0.13, 0.822)
    arrow(ax, 0.37, 0.845, 0.37, 0.822)
    box(ax, 0.03, 0.60, 0.205, 0.08,
        "Embeddings from the final\nrepresentation layer, concatenated\nacross the three encoders", fs=6.1)
    arrow(ax, 0.1325, 0.715, 0.1325, 0.682)
    box(ax, 0.26, 0.60, 0.215, 0.12,
        "Merged feature vector\n(embeddings + RDKit)\n↓\nFinal ML regressors per endpoint\n(SVR selected; 10-fold CV × 3,\nGridSearchCV)", fs=6.1)
    arrow(ax, 0.235, 0.64, 0.26, 0.64)
    arrow(ax, 0.37, 0.755, 0.37, 0.722)
    ax.text(0.03, 0.548, "Test Set #1 is used once, for benchmarking;\nits labels never enter regressor fitting.",
            fontsize=5.8, color=INK2, style="italic")

    # ---------------- B: ML benchmarking ------------------------------------------------------
    panel(ax, 0.505, 0.53, 0.485, 0.46, "B", "ML benchmarking on Test Set #1 (Table 2)", C_ML, E_ML)
    rows = [("A", "ATFP emb. + RDKit, SVR (this work)", True),
            ("B", "published models (reported)", False),
            ("C", "published setting, re-implemented", False),
            ("D", "reproduced, same protocol", False),
            ("E", "ATFP embeddings only", False),
            ("F", "RDKit descriptors only", False),
            ("G", "merged embeddings only", False)]
    y0 = 0.90
    for k, (g, t, a) in enumerate(rows):
        yy = y0 - k * 0.037
        box(ax, 0.525, yy, 0.30, 0.027, "ML Group %s — %s" % (g, t), fs=5.8, ha="left",
            ec=E_ML if a else INK2, lw=1.2 if a else 0.7, fc="white", bold=a)
    box(ax, 0.84, 0.855, 0.135, 0.076, "Descriptive only\nML Group B:\npublished scores\n(no per-compound\npredictions)", fs=5.8,
        ls=(0, (3, 2)))
    box(ax, 0.84, 0.66, 0.135, 0.17,
        "Matched-compound\ninference\nA vs C, D, E, F, G\npaired by PubChem CID\nΔMAE, ΔRMSE, ΔR²,\nΔ within 2-fold\nbootstrap 95% CI\nWilcoxon / McNemar\nHolm-adjusted", fs=5.8)
    arrow(ax, 0.825, 0.866, 0.84, 0.89, color=GREY, ls=(0, (3, 2)))
    arrow(ax, 0.825, 0.75, 0.84, 0.75)
    box(ax, 0.525, 0.575, 0.45, 0.06,
        "Outcome: merged representation comparable to descriptor models; clearest paired gains\n"
        "vs ATFP-only for CL and vs embeddings-only for VDss — complementary, not superior",
        fs=6.0, fc="white", ec=E_ML)

    # ---------------- C: PBPK-oriented refitting ---------------------------------------------
    panel(ax, 0.01, 0.01, 0.37, 0.505, "C", "PBPK-oriented refitting", C_DATA, E_DATA)
    box(ax, 0.03, 0.375, 0.155, 0.08, "Test Set #2: 110 compounds\n(Fu measured: 90)\nexcluded from\nregressor fitting", fs=5.9,
        ec=E_DATA, lw=1.1)
    box(ax, 0.205, 0.375, 0.155, 0.08, "Training Set #2\nall remaining data\nFu 4585 · CL 1354\nVDss 1354",
        fs=6.0)
    box(ax, 0.205, 0.255, 0.155, 0.09,
        "Regressors refitted\n(architecture, features and\nhyperparameters frozen\nfrom panel A)", fs=5.9)
    arrow(ax, 0.2825, 0.375, 0.2825, 0.347)
    box(ax, 0.03, 0.235, 0.155, 0.11,
        "Encoders reused from\nTraining #1 — NOT retrained\n81 of 110 Test #2\ncompounds (32 of 41\nPBPK) seen by encoders",
        fs=5.8, ec=C_WARN, lw=1.2, color=C_WARN)
    arrow(ax, 0.185, 0.29, 0.205, 0.29, color=C_WARN)
    box(ax, 0.03, 0.09, 0.33, 0.11,
        "Predicted Fu, CL, VDss for Test #2\n→ 41 compounds with a Simcyp v24 library model\nand curated IV clinical C–T data (healthy adults)\n→ secondary held-out assessment,\n    not an external validation", fs=5.8)
    arrow(ax, 0.2825, 0.255, 0.2825, 0.202)
    arrow(ax, 0.1075, 0.375, 0.1075, 0.347)
    ax.text(0.03, 0.028, "S+ (ADMET Predictor v11): physicochemical properties,\nFu, VDss and HLM CLint predictions.",
            fontsize=5.7, color=INK2, style="italic")

    # ---------------- D: controlled PBPK evaluation ------------------------------------------
    panel(ax, 0.39, 0.01, 0.60, 0.505, "D", "Controlled PBPK evaluation (Simcyp v24, R interface; n = 41)", C_PBPK, E_PBPK)
    box(ax, 0.41, 0.33, 0.27, 0.13,
        "PBPK Group A — clearance fixed at observed\nv0 Simcyp library model (reference)\nv1 observed Fu, VDss, CL (reference)\n"
        "+ S+ physicochemical → + S+ Fu → + S+ VDss", fs=5.9, ha="left")
    box(ax, 0.70, 0.33, 0.275, 0.13,
        "PBPK Group B — only the clearance source varies\nobserved CL (reference, S+ Fu/VDss)\nbottom-up: S+ HLM CLint (IVIVE), ± renal CL\n"
        "top-down: DL–ML CL\nall predicted: DL–ML Fu, VDss, CL", fs=5.9, ha="left")
    box(ax, 0.41, 0.225, 0.27, 0.075,
        "Primary analysis: minimal PBPK, identical\nstructure across arms (no SAC); paired\nby compound", fs=5.9, ec=E_PBPK, lw=1.1)
    box(ax, 0.70, 0.225, 0.275, 0.075,
        "Structural sensitivity: full PBPK (Kp methods\n1–3), never pooled with minimal arms", fs=5.9,
        ls=(0, (3, 2)))
    arrow(ax, 0.545, 0.33, 0.545, 0.302)
    arrow(ax, 0.8375, 0.33, 0.8375, 0.302)
    box(ax, 0.41, 0.045, 0.565, 0.145,
        "Outputs\n• C–T profile: log-NRMSE (primary; n = 40, range rule) and RMSE of log$_{10}$ conc. (n = 41);\n"
        "   time-weighted RMSE as sensitivity\n• AUC$_{0–t}$ and C$_{max}$: log$_{2}$ fold error, within 2-fold\n"
        "• Paired Hodges–Lehmann, bootstrap 95% CI, Wilcoxon, Holm\n"
        "• Parameter-to-exposure analysis (upstream fold error vs exposure error)", fs=5.9, ha="left")
    arrow(ax, 0.545, 0.225, 0.545, 0.192)
    arrow(ax, 0.8375, 0.225, 0.8375, 0.192, ls=(0, (3, 2)))

    # between panels
    arrow(ax, 0.475, 0.66, 0.525, 0.66, color=E_DATA, lw=1.2)
    arrow(ax, 0.36, 0.60, 0.36, 0.518, color=E_DATA, lw=1.2)
    arrow(ax, 0.36, 0.157, 0.41, 0.157, color=E_PBPK, lw=1.2)
    for ext, kw in (("png", {"dpi": 450}), ("pdf", {}), ("svg", {})):
        fig.savefig(OUT / ("Figure1_executed_workflow.%s" % ext), facecolor="white", **kw)
    plt.close(fig)


def toc():
    """Graphical abstract (ACS TOC is 3.25 x 1.75 in): the original two-stage concept, reduced."""
    fig = plt.figure(figsize=(3.25, 1.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.02, 0.35, 0.22, 0.4, "Molecular\ngraph\n+ RDKit", fc=C_DATA, ec=E_DATA, fs=6.5, bold=True)
    box(ax, 0.30, 0.35, 0.22, 0.4, "DL–ML\npredicted\nFu · CL · VDss", fc=C_ML, ec=E_ML, fs=6.5, bold=True)
    box(ax, 0.58, 0.35, 0.18, 0.4, "Hybrid\nPBPK\n(Simcyp)", fc=C_PBPK, ec=E_PBPK, fs=6.5, bold=True)
    arrow(ax, 0.24, 0.55, 0.30, 0.55, lw=1.1)
    arrow(ax, 0.52, 0.55, 0.58, 0.55, lw=1.1)
    arrow(ax, 0.76, 0.55, 0.80, 0.55, lw=1.1)
    import numpy as np
    t = np.linspace(0, 1, 60)
    x0, y0, w, h = 0.81, 0.33, 0.17, 0.44
    ax.plot(x0 + w * t, y0 + h * (0.9 * np.exp(-3.2 * t) + 0.05), color=E_DATA, lw=1.2)
    ax.plot(x0 + w * t[::7], y0 + h * (0.9 * np.exp(-2.8 * t[::7]) + 0.05), "o", ms=2.2, color=E_PBPK)
    ax.plot([x0, x0, x0 + w], [y0 + h, y0, y0], color=INK2, lw=0.6)
    ax.text(0.5, 0.14, "Clearance accuracy, not its formulation, limits exposure prediction",
            ha="center", fontsize=6.1, color=INK2, style="italic")
    for ext, kw in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(OUT / ("TOC_graphic.%s" % ext), facecolor="white", **kw)
    plt.close(fig)


if __name__ == "__main__":
    figure1()
    toc()
    print("wrote Figure1_executed_workflow.* and TOC_graphic.* to", OUT)
