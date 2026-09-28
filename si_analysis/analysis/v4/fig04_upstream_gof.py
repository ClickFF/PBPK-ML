#!/usr/bin/env python3
"""Figure 4 (v4): predicted vs observed Fu, CL and VDss for the 41 PBPK compounds, five balanced panels.
Top row: S+ Fu, S+ VDss (S+ provides no CL); bottom row: DL-ML Fu, CL, VDss. Identical axis limits per
parameter; N and R^2 (log10) in the panel; within-2-fold and GMFE in the caption
(outputs/tables/v4_figure4_metrics.csv). Source: outputs/tables/upstream_parameter_fold_errors.csv.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v4/fig04_upstream_gof.py
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
UNIT = {"Fu": ("f$_u$", ""), "CLsys": ("CL", " (L h$^{-1}$ kg$^{-1}$)"), "VDss": ("VD$_{ss}$", " (L kg$^{-1}$)")}
PANELS = [("S+", "Fu", (0, slice(1, 3))), ("S+", "VDss", (0, slice(3, 5))),
          ("DL-ML", "Fu", (1, slice(0, 2))), ("DL-ML", "CLsys", (1, slice(2, 4))), ("DL-ML", "VDss", (1, slice(4, 6)))]


def main():
    fe = pd.read_csv(T / "upstream_parameter_fold_errors.csv")
    lim = {}
    for par in UNIT:
        s = fe[fe.parameter == par]
        v = np.r_[s.obs, s.pred]
        lim[par] = (v.min() / 2.2, v.max() * 2.2)
    fig = S.figure(S.DOUBLE, 4.9)
    gs = fig.add_gridspec(2, 6, hspace=0.55, wspace=1.2)
    rows = []
    for k, (pred, par, (r, c)) in enumerate(PANELS):
        ax = fig.add_subplot(gs[r, c])
        s = fe[(fe.predictor == pred) & (fe.parameter == par)]
        o, p = s.obs.to_numpy(float), s.pred.to_numpy(float)
        lo, hi = lim[par]
        x = np.array([lo, hi])
        ax.fill_between(x, x / 2, x * 2, color=S.LIGHT, lw=0, zorder=0)
        ax.plot(x, x, color=S.OBS, lw=0.7)
        for f in (3, 1 / 3):
            ax.plot(x, x * f, color=S.GREY, lw=0.5, ls=":")
        col = S.SPLUS if pred == "S+" else S.DLML
        ax.plot(o, p, "o", ms=3.2, mfc=col, mec="white", mew=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        e = np.log10(p) - np.log10(o)
        r2 = 1 - np.sum(e ** 2) / np.sum((np.log10(o) - np.log10(o).mean()) ** 2)
        rows.append(dict(predictor=pred, parameter=par, n=len(o), R2=r2,
                         within2=100 * np.mean(np.abs(e) <= np.log10(2)), GMFE=10 ** np.mean(np.abs(e))))
        ax.text(0.05, 0.95, "N = %d\nR$^2$ = %.2f" % (len(o), r2), transform=ax.transAxes, ha="left", va="top",
                fontsize=7)
        sym, unit = UNIT[par]
        ax.set_xlabel("Observed %s%s" % (sym, unit))
        ax.set_ylabel("%s predicted %s" % ("S+" if pred == "S+" else "DL–ML", sym))
        ax.set_title("%s %s" % ("S+" if pred == "S+" else "DL–ML", sym))
        S.panel_label(ax, "abcde"[k], dx=-0.42, dy=1.03)
    pd.DataFrame(rows).to_csv(T / "v4_figure4_metrics.csv", index=False, float_format="%.4f")
    S.save(fig, "main", "Figure4_upstream_gof")


if __name__ == "__main__":
    main()
