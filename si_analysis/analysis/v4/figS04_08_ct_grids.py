#!/usr/bin/env python3
"""Figures S4-S8 (v4): simulated vs observed C-T profiles, 41 compounds per scenario, split over two pages
(compounds 1-21 and 22-41 in alphabetical order), 4 columns, legend repeated on each page, compound titles
>= 7 pt. Data and axis-scaling rules as in analysis/08_si_ct_profiles.py (v3): population mean and
5th-95th percentile band (clipped), digitised observations, log concentration, 0-48 h.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v4/figS04_08_ct_grids.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

S21 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S21 / "lib"))
import style_v4 as S                                              # noqa: E402

S.use()
SIM = S21 / "data" / "simulation_output_batch"
OBS = S21 / "data" / "observed" / "observed_data_cleaned_deduplicated.csv"
MASTER = S21 / "data" / "compounds" / "pbpk_physchem_mechanistic_master.csv"
FIGS = [("S4", "v0_run0"), ("S5", "v1_run0"), ("S6", "h1_run0"), ("S7", "h2_run4"), ("S8", "h2_run0")]
BAND = "#d6e2ee"
XMAX = 48.0


def panel(ax, sg, og, title):
    b = sg[sg["lower"].notna() & sg["upper"].notna() & (sg["lower"] > 0)]
    if len(b):
        ax.fill_between(b.Time_hr, b.lower, b.upper, color=BAND, lw=0, zorder=1)
    ax.plot(sg.Time_hr, sg["mean"], color=S.DLML, lw=0.9, zorder=3)
    if len(og):
        ax.plot(og.Time_hr, og.Conc_mgl, "o", ms=2.2, mfc=S.OBS, mec="white", mew=0.25, zorder=4)
    ax.set_yscale("log")
    ax.set_xlim(0, XMAX)
    v = np.r_[sg["mean"][sg.Time_hr <= XMAX].to_numpy(), og.Conc_mgl.to_numpy()]
    v = v[np.isfinite(v) & (v > 0)]
    lo, hi = v.min() / 3.0, v.max() * 3.0
    if hi / lo < 30:
        mid = np.sqrt(hi * lo)
        lo, hi = mid / 5.5, mid * 5.5
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=3))
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs="all", numticks=10))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([0, 24, 48])
    ax.tick_params(labelsize=6.5, pad=1.5)
    ax.set_title(title, fontsize=7, loc="left", pad=2)


def main():
    m = pd.read_csv(MASTER)
    names = dict(zip(m.ID_trend_pbpk.astype(int), m.Drug_Name.astype(str)))
    obs = pd.read_csv(OBS)
    obs["Conc_mgl"] = obs.Conc_ng_ml / 1000.0
    obs = obs[obs.Conc_mgl > 0].sort_values(["compound_id", "Time_hr"])
    handles = [Line2D([], [], color=S.DLML, lw=0.9, label="Simulated population mean"),
               Patch(facecolor=BAND, label="Simulated 5th–95th percentile"),
               Line2D([], [], ls="none", marker="o", ms=3, mfc=S.OBS, mec="white", label="Observed")]
    for fig_no, arm in FIGS:
        sim = pd.read_excel(SIM / arm / "ct_wide_all.xlsx")
        sim = sim[sim["mean"] > 0].sort_values(["compound_id", "Time_hr"])
        ids = sorted(sim.compound_id.unique(), key=lambda i: names.get(int(i), str(i)).lower())
        for page, chunk in enumerate((ids[:21], ids[21:]), start=1):
            ncol, nrow = 4, 6
            fig, axes = S.plt.subplots(nrow, ncol, figsize=(S.DOUBLE, 8.6), squeeze=False)
            for k, cid in enumerate(chunk):
                panel(axes[k // ncol][k % ncol], sim[sim.compound_id == cid], obs[obs.compound_id == cid],
                      names.get(int(cid), str(cid)))
            for k in range(len(chunk), nrow * ncol):
                axes[k // ncol][k % ncol].axis("off")
            fig.supxlabel("Time (h)", fontsize=8, y=0.03)
            fig.supylabel("Plasma concentration (mg L$^{-1}$)", fontsize=8, x=0.01)
            fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005))
            fig.tight_layout(rect=(0.02, 0.045, 1, 1), h_pad=1.0, w_pad=0.8)
            S.save(fig, "si", "Figure%s_ct_%s_part%d" % (fig_no, arm, page))
        print(fig_no, arm, len(ids))


if __name__ == "__main__":
    main()
