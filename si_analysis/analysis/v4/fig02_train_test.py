#!/usr/bin/env python3
"""Figure 2 (v4): DL-ML (ML Group A) predicted vs observed, training and held-out test set separately.
Only N and R^2 are printed in the panels; RMSE, GMFE and within-2-fold coverage go to the caption
(written to outputs/tables/v4_figure2_metrics.csv, identical definitions to the v3 figure).

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v4/fig02_train_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S21 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S21 / "lib"))
import style_v4 as S                                              # noqa: E402
import paths as P                                                 # noqa: E402

S.use()
ENDPOINTS = [
    ("Fu", "ML+training-raw/DL+ML_trainV4/pred_res/res_v1/res_Fu.csv", "lgFu_train", "lgFu_test",
     "Plasma unbound fraction", "log$_{10}$ f$_u$"),
    ("CL", "ML+training-raw/DL+ML_trainV4/pred_res/res_v1/res_CL.csv", "lgCL_train", "lgCL_test",
     "Systemic clearance", "log$_{10}$ CL"),
    ("VDss", "ML+training-raw/DL+ML_trainV4/pred_res/res_v1/res_VDss.csv", "lgVD_train", "lgVD_test",
     "Volume of distribution", "log$_{10}$ VD$_{ss}$"),
]
SETS = [("Training set", "train", S.GREY), ("Held-out test set", "test", S.DLML)]
L2, L3 = np.log10(2), np.log10(3)


def load(path, tr, te):
    df = pd.read_csv(P.ROOT / path)
    return {k: df[df.Dataset.astype(str).str.contains(tok, case=False, na=False)][["Actual", "Predicted"]].dropna()
            for k, tok in (("train", tr), ("test", te))}


def metrics(y, p):
    e = p - y
    return dict(N=len(y), R2=1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2), RMSE=float(np.sqrt(np.mean(e ** 2))),
                GMFE=float(10 ** np.mean(np.abs(e))), within2=float(np.mean(np.abs(e) <= L2)))


def main():
    fig, axes = S.plt.subplots(3, 2, figsize=(5.6, 8.1))
    rows = []
    for ri, (ep, path, tr, te, name, sym) in enumerate(ENDPOINTS):
        d = load(path, tr, te)
        v = np.concatenate([np.r_[f.Actual, f.Predicted] for f in d.values()])
        lo, hi = v.min(), v.max()
        lo, hi = lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)
        g = np.array([lo, hi])
        for ci, (lab, key, col) in enumerate(SETS):
            ax = axes[ri, ci]
            y, p = d[key].Actual.to_numpy(float), d[key].Predicted.to_numpy(float)
            m = metrics(y, p)
            rows.append(dict(endpoint=ep, set=lab, **m))
            ax.fill_between(g, g - L2, g + L2, color=S.LIGHT, lw=0, zorder=0)
            ax.plot(g, g, color=S.OBS, lw=0.7, zorder=2)
            for s in (L3, -L3):
                ax.plot(g, g + s, color=S.GREY, lw=0.5, ls=":", zorder=2)
            big = len(y) > 1000
            ax.scatter(y, p, s=3 if big else 6, color=col, alpha=0.35 if big else 0.7, lw=0, zorder=3,
                       rasterized=big)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.text(0.04, 0.96, "N = %d\nR$^2$ = %.2f" % (m["N"], m["R2"]), transform=ax.transAxes,
                    ha="left", va="top", fontsize=7)
            ax.set_xlabel("Observed %s" % sym)
            ax.set_ylabel("Predicted %s" % sym)
            if ri == 0:
                ax.set_title(lab, fontsize=8)
            if ci == 0:
                ax.annotate(name, xy=(-0.42, 0.5), xycoords="axes fraction", rotation=90, ha="center",
                            va="center", fontsize=8, fontweight="bold")
            S.panel_label(ax, "abcdef"[ri * 2 + ci], dx=-0.28, dy=1.02)
    fig.tight_layout(h_pad=1.2, w_pad=2.0)
    pd.DataFrame(rows).to_csv(P.TABLES / "v4_figure2_metrics.csv", index=False, float_format="%.4f")
    S.save(fig, "main", "Figure2_train_test")


if __name__ == "__main__":
    main()
