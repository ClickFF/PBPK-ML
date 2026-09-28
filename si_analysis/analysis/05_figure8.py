#!/usr/bin/env python3
"""Figure 8 — performance of the all-predicted arm (PBPK_v4_ML, h2_run0).

Layout after Geci et al. 2024 (Arch Toxicol 98:2659-2676, Fig. 6), chosen by the author to
replace the v1 compound-heterogeneity figure: goodness of fit on top, per-compound error in the
middle, three representative concentration-time profiles at the bottom, the three compounds
marked in the middle strip.

Brought to the Sep21 rules:
  * AUC and Cmax values are the evaluation's own (per_compound_metrics.csv); nothing recomputed
  * observed-vs-predicted panels carry identity, 2-fold and 3-fold references and no R^2
  * the C-T strip shows per-compound log-NRMSE; signed log2 fold error only for AUC and Cmax
  * compounds ranked by AUC fold error — the author's explicit choice for this figure, an
    exception to the alphabetical-order convention, recorded in README.md

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/05_figure8.py [run_id]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import paths as P                                                 # noqa: E402
import matplotlib                                                 # noqa: E402

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from matplotlib.patches import Patch                              # noqa: E402
from scipy.interpolate import PchipInterpolator                   # noqa: E402
import style                                                      # noqa: E402

style.use()
matplotlib.rcParams.update({"font.family": "sans-serif",
                            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                            "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                            "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                            "mathtext.default": "regular"})

W = 7.2
BLUE, ORANGE, RED, PURPLE = "#2a78d6", "#eb6834", "#e34948", "#4a3aa7"
BAND_B, INK, INK2, INK3, BAND = "#cfe1f7", "#0b0b0b", "#52514e", "#8b8a85", "#efeeea"


def gof(ax, o, p, xlab, ylab, dense=False):
    ok = np.isfinite(o) & np.isfinite(p) & (o > 0) & (p > 0)
    o, p = o[ok], p[ok]
    lo, hi = min(o.min(), p.min()) * 0.45, max(o.max(), p.max()) * 2.2
    x = np.array([lo, hi])
    ax.fill_between(x, x / 2, x * 2, color=BAND, lw=0, zorder=0)
    ax.plot(x, x, color=INK2, lw=0.9, zorder=2)
    for k, ls in ((2, (0, (4, 2))), (3, (0, (1, 1.6)))):
        ax.plot(x, x * k, color=INK3, lw=0.7, ls=ls, zorder=2)
        ax.plot(x, x / k, color=INK3, lw=0.7, ls=ls, zorder=2)
    if dense:
        ax.plot(o, p, ls="none", marker="o", ms=2.3, mfc=BLUE, mec="none", alpha=0.28, zorder=3,
                rasterized=True)
    else:
        ax.plot(o, p, ls="none", marker="o", ms=3.8, mfc=BLUE, mec="white", mew=0.4, alpha=0.9,
                zorder=3)
    lo_, lp_ = np.log10(o), np.log10(p)
    r2 = 1 - np.sum((lp_ - lo_) ** 2) / np.sum((lo_ - lo_.mean()) ** 2)
    e = np.abs(lp_ - lo_)
    ax.text(0.04, 0.96, "n = %d\nR$^2$ = %.2f\nwithin 2-fold = %.0f%%\nwithin 3-fold = %.0f%%"
            % (o.size, r2, 100 * np.mean(e <= np.log10(2)), 100 * np.mean(e <= np.log10(3))),
            transform=ax.transAxes, fontsize=6.8, color=INK2, va="top", linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8), zorder=6)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(xlab, fontsize=7.8)
    ax.set_ylabel(ylab, fontsize=7.8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))
    return r2


def fold_strip(ax, order, vals, picks, ylab, show_x=False):
    """signed log2 FE -> fold error (pred/obs) on a log2 axis, 2-fold band, 3-fold lines."""
    y = 2.0 ** np.array([vals.get(c, np.nan) for c in order], dtype=float)
    x = np.arange(len(order))
    ax.set_yscale("log", base=2)
    ax.axhspan(0.5, 2, color=BAND, zorder=0)
    ax.axhline(1, color=INK2, lw=0.7, zorder=2)
    for f, ls in ((2, (0, (4, 2))), (0.5, (0, (4, 2))), (3, (0, (1, 1.6))), (1 / 3, (0, (1, 1.6)))):
        ax.axhline(f, color=INK3, lw=0.7, ls=ls, zorder=2)
    yc = np.clip(y, 1 / 24, 12)
    inside = (y >= 0.5) & (y <= 2)
    ax.plot(x[inside], yc[inside], ls="none", marker="o", ms=3.4, mfc=INK, mec="none", zorder=3)
    ax.plot(x[~inside], yc[~inside], ls="none", marker="o", ms=3.4, mfc=RED, mec="none", zorder=3)
    for c in picks:
        i = order.index(c)
        ax.axvline(i, color=BLUE, lw=0.8, zorder=1)
        ax.plot([i], [yc[i]], marker="o", ms=5, mfc=BLUE, mec="white", mew=0.6, zorder=5)
    ax.set_ylim(1 / 24, 12)
    ax.set_yticks([1 / 16, 1 / 3, 1 / 2, 1, 2, 3, 8])
    ax.set_yticklabels(["1/16", "1/3", "1/2", "1", "2", "3", "8"], fontsize=6.5)
    ax.minorticks_off()
    ax.set_ylabel(ylab, fontsize=7.6)
    ax.set_xlim(-0.8, len(order) - 0.2)
    ax.tick_params(labelsize=6.8)
    if show_x:
        ax.set_xlabel("compound, ranked by AUC fold error", fontsize=7.8)
        ax.set_xticks([0, len(order) // 2, len(order) - 1])
        ax.set_xticklabels(["1", str(len(order) // 2 + 1), str(len(order))])
    else:
        ax.set_xticks([])


def strip(ax, order, vals, picks, ylab, signed, show_x=False):
    y = np.array([vals.get(c, np.nan) for c in order], dtype=float)
    x = np.arange(len(order))
    if signed:
        ax.axhspan(-1, 1, color=BAND, zorder=0)
        ax.axhline(0, color=INK2, lw=0.7, zorder=2)
        inside = np.abs(y) <= 1
        ax.plot(x[inside], y[inside], ls="none", marker="o", ms=3.4, mfc=INK, mec="none", zorder=3)
        ax.plot(x[~inside], y[~inside], ls="none", marker="o", ms=3.4, mfc=RED, mec="none", zorder=3)
    else:
        ax.axhline(np.nanmedian(y), color=INK3, lw=0.7, ls=(0, (4, 2)), zorder=2)
        ax.plot(x, y, ls="none", marker="D", ms=3.2, mfc=PURPLE, mec="none", zorder=3)
        miss = ~np.isfinite(y)
        for i in np.where(miss)[0]:
            ax.text(i, 0.0, "×", ha="center", va="bottom", fontsize=8, color=INK3)
        ax.set_ylim(bottom=0)
    for c in picks:
        i = order.index(c)
        ax.axvline(i, color=BLUE, lw=0.9, zorder=1)
        if np.isfinite(y[i]):
            ax.plot([i], [y[i]], marker="o", ms=4.8, mfc=BLUE, mec="white", mew=0.6, zorder=5)
    ax.set_ylabel(ylab, fontsize=8.2)
    ax.set_xlim(-0.8, len(order) - 0.2)
    ax.grid(True, axis="y", color="#e4e3df", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.8)
    if show_x:
        ax.set_xlabel("compound, ranked by signed AUC fold error", fontsize=8.8)
        ax.set_xticks([0, len(order) // 2, len(order) - 1])
        ax.set_xticklabels(["1", str(len(order) // 2 + 1), str(len(order))])
    else:
        ax.set_xticks([])


def profile(ax, sg, og, title):
    b = sg[sg["lower"].notna() & sg["upper"].notna() & (sg["lower"] > 0)]
    if len(b):
        ax.fill_between(b["Time_hr"], b["lower"], b["upper"], color=BAND_B, lw=0, zorder=1)
    ax.plot(sg["Time_hr"], sg["mean"], color=BLUE, lw=1.4, zorder=3)
    ax.plot(og["Time_hr"], og["Conc_mgl"], ls="none", marker="o", ms=3.6, mfc=ORANGE,
            mec="white", mew=0.4, zorder=4)
    ax.set_yscale("log")
    ax.set_xlim(0, 48)
    v = np.concatenate([sg["mean"][sg["Time_hr"] <= 48].to_numpy(), og["Conc_mgl"].to_numpy()])
    v = v[np.isfinite(v) & (v > 0)]
    lo, hi = v.min() / 3.0, v.max() * 3.0
    if hi / lo < 30:
        mid = np.sqrt(hi * lo)
        lo, hi = mid / 5.5, mid * 5.5
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=4))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([0, 12, 24, 36, 48])
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=9, color=BLUE, pad=3)
    ax.set_xlabel("time (h)", fontsize=8.8)


def tag(ax, letter, dx=-0.26, dy=1.1):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "h2_run0"
    d = pd.read_csv(P.TABLES / "per_compound_metrics.csv")
    d = d[d.scenario == run].set_index("compound_id")
    obs = pd.read_csv(P.OBS)
    obs["Conc_mgl"] = obs["Conc_ng_ml"] / 1000
    obs = obs[obs.Conc_mgl > 0].sort_values(["compound_id", "Time_hr"])
    sim = pd.read_excel(P.SIM / run / "ct_wide_all.xlsx")
    sim = sim[sim["mean"] > 0].sort_values(["compound_id", "Time_hr"])
    obs = obs[obs.compound_id.isin(sim.compound_id.unique())]

    order = list(d["fe_auc_rel"].sort_values().index)
    picks = [order[0], order[len(order) // 2], order[-1]]
    labs = ["most underpredicted AUC", "median AUC error", "most overpredicted AUC"]

    # matched concentrations for panel A (dose-normalised so compounds share one axis)
    O, Pp = [], []
    for cid, g in obs.groupby("compound_id"):
        sg = sim[sim.compound_id == cid]
        pr = PchipInterpolator(sg.Time_hr, sg["mean"])(g.Time_hr.values)
        dose = g.Dose_mg_kg.to_numpy(float)
        k = (pr > 0) & (dose > 0)
        O.append(g.Conc_mgl.to_numpy()[k] / dose[k]); Pp.append(pr[k] / dose[k])
    O, Pp = np.concatenate(O), np.concatenate(Pp)

    fig = plt.figure(figsize=(W, 7.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 0.95, 0.95], hspace=0.42, wspace=0.42,
                          left=0.09, right=0.985, top=0.975, bottom=0.075)
    a = fig.add_subplot(gs[0, 0])
    r2s = [gof(a, O, Pp, "observed conc./dose\n(mg L$^{-1}$ per mg kg$^{-1}$)",
               "predicted conc./dose\n(mg L$^{-1}$ per mg kg$^{-1}$)", dense=True)]
    tag(a, "(a)", dx=-0.36, dy=1.08)
    b = fig.add_subplot(gs[0, 1])
    r2s.append(gof(b, d.auc_obs.to_numpy(), d.auc_pred.to_numpy(), "observed AUC$_{0-t}$ (mg h L$^{-1}$)",
                   "predicted AUC$_{0-t}$ (mg h L$^{-1}$)"))
    tag(b, "(b)", dx=-0.36, dy=1.08)
    c = fig.add_subplot(gs[0, 2])
    r2s.append(gof(c, d.cmax_obs.to_numpy(), d.cmax_pred.to_numpy(), "observed C$_{max}$ (mg L$^{-1}$)",
                   "predicted C$_{max}$ (mg L$^{-1}$)"))
    tag(c, "(c)", dx=-0.36, dy=1.08)

    m = pd.read_csv(P.MASTER)
    nm = dict(zip(m.ID_trend_pbpk.astype(int), m.Name_final.str.strip()))
    inner = gs[1, :].subgridspec(2, 1, hspace=0.12)
    for k, (col, ylab) in enumerate([("fe_auc_rel", "AUC$_{0-t}$\nfold error"),
                                     ("fe_cmax_rel", "C$_{max}$\nfold error")]):
        ax = fig.add_subplot(inner[k])
        fold_strip(ax, order, d[col].to_dict(), picks, ylab, show_x=(k == 1))
        if k == 0:
            tag(ax, "(d)", dx=-0.075, dy=1.32)
            for cpd in picks:
                ax.annotate(nm.get(int(cpd), cpd), xy=(order.index(cpd), 12), xytext=(0, 1.5),
                            textcoords="offset points", fontsize=7, color=BLUE, ha="center",
                            va="bottom", fontweight="bold")
    for k, (cpd, lab) in enumerate(zip(picks, labs)):
        ax = fig.add_subplot(gs[2, k])
        profile(ax, sim[sim.compound_id == cpd], obs[obs.compound_id == cpd],
                "%s (%s)" % (nm.get(int(cpd), cpd), lab))
        ax.title.set_fontsize(7.8)
        ax.title.set_fontweight("bold")
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_fontsize(7.8)
        if k == 0:
            ax.set_ylabel("concentration (mg L$^{-1}$)", fontsize=7.8)
        tag(ax, "(%s)" % "efg"[k], dx=-0.3, dy=1.16)
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=1.4, label="simulated population mean"),
                        Patch(facecolor=BAND_B, edgecolor="none", label="5th–95th percentile"),
                        Line2D([], [], ls="none", marker="o", ms=4, mfc=ORANGE, mec="white",
                               label="observed"),
                        Line2D([], [], ls="none", marker="o", ms=4, mfc=RED, mec="none",
                               label="outside 2-fold")],
               loc="lower center", bbox_to_anchor=(0.54, -0.005), ncol=4, fontsize=7.2,
               frameon=False, columnspacing=1.2, handletextpad=0.4)
    style.save(fig, P.FIG_MAIN, "Figure8_v4ML_overview_AUCranked")
    plt.close(fig)
    print("Figure 8: picks =", [(nm.get(int(c)), round(d.loc[c, "fe_auc_rel"], 2)) for c in picks],
          "R2 conc/AUC/Cmax =", [round(x, 3) for x in r2s])


if __name__ == "__main__":
    main()
