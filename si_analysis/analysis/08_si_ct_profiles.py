#!/usr/bin/env python3
"""SI C-T grids (Figures S4-S8): Sep21 copy of Sep20 make_internal_review_ct_figures.py,
repointed to the sealed corrected batch in data/; titles and captions live in the SI document.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/08_si_ct_profiles.py [arm ...]
"""
from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
S21 = HERE.parent
SIM = S21 / "data" / "simulation_output_batch"          # sealed corrected batch
FIG = S21 / "outputs" / "figures" / "si"
OBS = S21 / "data" / "observed" / "observed_data_cleaned_deduplicated.csv"
MASTER = S21 / "data" / "compounds" / "pbpk_physchem_mechanistic_master.csv"
ROOT = S21.parent.parent

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BLUE, BAND_B, ORANGE = "#2a78d6", "#cfe1f7", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
W2, XMAX = 7.0, 48.0

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 1.8, "ytick.major.size": 1.8,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
    "grid.color": GRID, "grid.linewidth": 0.4, "legend.frameon": False,
    "mathtext.fontset": "custom",               # keep subscripts in Arial, not DejaVu
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "mathtext.default": "regular",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

CAPTION = (
    "One panel per compound, titled with its Drug_Name and ordered alphabetically; all 41 compounds "
    "of the PBPK evaluation set are shown. The line is the simulated population mean and the band "
    "its 5th–95th percentile; points are the digitised clinical observations. Concentration is on a "
    "log scale and time is truncated at 48 h, the simulation window. Panel axes are scaled to the "
    "mean profile and the observations, and the percentile band is drawn but clipped: for several "
    "compounds the 5th percentile falls below 1e-15 within the window, which would otherwise "
    "flatten every panel. Simulations are population means for a healthy-volunteer population at "
    "the dose of the matching clinical study, so a vertical offset is a systematic exposure error "
    "and a difference in slope is a clearance error."
)

ARM_TITLE = {
    "v0_run0": "PBPK_v0 — Simcyp compound files as shipped",
    "v1_run0": "PBPK_v1 — observed Fu, V$_{ss}$ and CL$_{sys}$",
    "h1_run0": "PBPK_v2 — bottom-up, S+ HLM CL$_{int}$ (template CL$_R$ retained)",
    "h1_run0_noCLr": "PBPK_v2 — bottom-up, S+ HLM CL$_{int}$ (CL$_R$ = 0)",
    "h2_run4": "PBPK_v3 — top-down, DL–ML CL$_{sys}$",
    "h2_run0": "PBPK_v4 — all inputs predicted (DL–ML Fu, V$_{ss}$, CL$_{sys}$)",
}


def figure_for(arm, names, obs):
    ct = SIM / arm / "ct_wide_all.xlsx"
    if not ct.exists():
        print("  %s: no ct_wide_all.xlsx" % arm)
        return
    sim = pd.read_excel(ct)
    sim = sim[sim["mean"] > 0].sort_values(["compound_id", "Time_hr"])
    ids = sorted(sim["compound_id"].unique(), key=lambda i: names.get(int(i), str(i)).lower())

    ncol, nrow = 5, int(np.ceil(len(ids) / 5))
    fig, axes = plt.subplots(nrow, ncol, figsize=(W2, 0.94 * nrow + 0.55), squeeze=False)
    for k, cid in enumerate(ids):
        ax = axes[k // ncol][k % ncol]
        sg = sim[sim["compound_id"] == cid]
        og = obs[obs["compound_id"] == cid]
        b = sg[sg["lower"].notna() & sg["upper"].notna() & (sg["lower"] > 0)]
        if len(b):
            ax.fill_between(b["Time_hr"], b["lower"], b["upper"], color=BAND_B,
                            linewidth=0, zorder=1)
        ax.plot(sg["Time_hr"], sg["mean"], color=BLUE, linewidth=0.9, zorder=3)
        if len(og):
            ax.plot(og["Time_hr"], og["Conc_mgl"], linestyle="none", marker="o", markersize=1.9,
                    markerfacecolor=ORANGE, markeredgecolor="white", markeredgewidth=0.25,
                    zorder=4)
        ax.set_yscale("log")
        ax.set_xlim(0, XMAX)
        # scale to the mean profile and the observations; the 5th percentile runs to ~1e-20 for
        # several compounds and would otherwise flatten every panel
        v = [sg["mean"][sg["Time_hr"] <= XMAX].to_numpy()]
        if len(og):
            v.append(og["Conc_mgl"].to_numpy())
        v = np.concatenate(v)
        v = v[np.isfinite(v) & (v > 0)]
        if v.size:
            lo, hi = v.min() / 3.0, v.max() * 3.0
            if hi / lo < 30:
                mid = np.sqrt(hi * lo)
                lo, hi = mid / 5.5, mid * 5.5
            ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=3))
        ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs="all", numticks=10))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks([0, 24, 48])
        ax.tick_params(labelsize=5, pad=1.2)
        ax.grid(True, color=GRID, linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(names.get(int(cid), str(cid)), fontsize=5.6, color=INK, pad=1.6, loc="left")
    for k in range(len(ids), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.supxlabel("Time (h)", fontsize=7, color=INK, y=0.008)
    fig.supylabel("Plasma concentration (mg L$^{-1}$, log scale)", fontsize=7, color=INK, x=0.004)
    handles = [Line2D([], [], color=BLUE, linewidth=0.9, label="simulated population mean"),
               Patch(facecolor=BAND_B, label="simulated 5th–95th percentile"),
               Line2D([], [], linestyle="none", marker="o", markersize=2.6,
                      markerfacecolor=ORANGE, markeredgecolor="white", label="observed")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.008), ncol=3,
               fontsize=5.8, labelcolor=INK2, handletextpad=0.5, columnspacing=1.2)
    fig.tight_layout(rect=(0.012, 0.016, 1.0, 0.972))
    fig.subplots_adjust(hspace=0.78, wspace=0.42)
    # title and caption live in the SI document (Sep21)

    FIG.mkdir(parents=True, exist_ok=True)
    stem = "FigS_ct_profiles_%s" % arm
    fig.savefig(FIG / (stem + ".pdf"))
    fig.savefig(FIG / (stem + ".png"), dpi=600)
    plt.close(fig)
    print("  %s.pdf / .png  (%d compounds)" % (stem, len(ids)))


def main():
    names = dict(zip(pd.read_csv(MASTER)["ID_trend_pbpk"].astype(int),
                     pd.read_csv(MASTER)["Drug_Name"].astype(str)))
    obs = pd.read_csv(OBS)
    obs["Conc_mgl"] = obs["Conc_ng_ml"] / 1000.0
    obs = obs[obs["Conc_mgl"] > 0].sort_values(["compound_id", "Time_hr"])
    for arm in (sys.argv[1:] or list(ARM_TITLE)):
        figure_for(arm, names, obs)
    print("\nfigures in %s" % FIG)


if __name__ == "__main__":
    main()
