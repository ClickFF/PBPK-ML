#!/usr/bin/env python3
"""Figure 4 (v3) — accuracy of the predicted ADME inputs for the 41 PBPK compounds, with R².

Observed = values entered in v1_run0; predicted = S+ (s1_run3) and DL-ML (h2_run0), from
outputs/tables/upstream_parameter_fold_errors.csv. Per panel: n, R² (coefficient of
determination on log10 values), % within 2-fold, GMFE. Identity, 2-fold band, 3-fold lines.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/11_figure4.py
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
import style                                                      # noqa: E402

style.use()
matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                            "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                            "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                            "mathtext.default": "regular"})
BAND, INK2, INK3 = "#efeeea", "#52514e", "#8b8a85"
COL = {"S+": "#eb6834", "DL-ML": "#2a78d6"}
PAR = [("Fu", "f$_u$ (unitless)"), ("CLsys", "CL (L h$^{-1}$ kg$^{-1}$)"), ("VDss", "V$_{Dss}$ (L kg$^{-1}$)")]


def stats(o, p):
    lo, lp = np.log10(o), np.log10(p)
    e = lp - lo
    r2 = 1 - np.sum(e ** 2) / np.sum((lo - lo.mean()) ** 2)
    return len(o), r2, 100 * np.mean(np.abs(e) <= np.log10(2)), 10 ** np.mean(np.abs(e))


def main():
    fe = pd.read_csv(P.TABLES / "upstream_parameter_fold_errors.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    rows = []
    for r, pred in enumerate(["S+", "DL-ML"]):
        for c, (par, lab) in enumerate(PAR):
            ax = axes[r, c]
            s = fe[(fe.predictor == pred) & (fe.parameter == par)]
            if r == 0:
                ax.set_title(lab.split(" (")[0], fontsize=10, fontweight="bold", pad=8)
            if s.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, "S+ provides no\nsystemic clearance", ha="center", va="center",
                        fontsize=8.5, color=INK3, transform=ax.transAxes)
                continue
            ax.text(-0.3, 1.06, "(%s)" % "abcde"[len(rows)], transform=ax.transAxes,
                    fontsize=10, fontweight="bold", va="bottom")
            o, p = s.obs.to_numpy(float), s.pred.to_numpy(float)
            lo = min(o.min(), p.min()) / 2.2
            hi = max(o.max(), p.max()) * 2.2
            x = np.array([lo, hi])
            ax.fill_between(x, x / 2, x * 2, color=BAND, lw=0, zorder=0)
            ax.plot(x, x, color=INK2, lw=0.9)
            for k, ls in ((2, (0, (4, 2))), (3, (0, (1, 1.6)))):
                ax.plot(x, x * k, color=INK3, lw=0.7, ls=ls)
                ax.plot(x, x / k, color=INK3, lw=0.7, ls=ls)
            ax.plot(o, p, ls="none", marker="o", ms=3.6, mfc=COL[pred], mec="white", mew=0.4, alpha=0.9)
            ax.set_xscale("log"), ax.set_yscale("log")
            ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            n, r2, w2, g = stats(o, p)
            rows.append(dict(predictor=pred, parameter=par, n=n, R2=r2, within2=w2, GMFE=g))
            ax.text(0.04, 0.96, "n = %d\nR$^2$ = %.2f\nwithin 2-fold = %.1f%%\nGMFE = %.2f" % (n, r2, w2, g),
                    transform=ax.transAxes, va="top", fontsize=7, color=INK2, linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))
            ax.tick_params(labelsize=7)
            ax.set_xlabel("observed " + lab, fontsize=7.8)
            ax.set_ylabel("%s predicted" % ("S+" if pred == "S+" else "DL–ML"), fontsize=7.8)
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    style.save(fig, P.FIG_MAIN, "Figure4_upstream_adme_gof")
    plt.close(fig)
    out = pd.DataFrame(rows)
    out.to_csv(P.TABLES / "upstream_parameter_r2.csv", index=False, float_format="%.4f")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
