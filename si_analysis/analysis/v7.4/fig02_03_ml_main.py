#!/usr/bin/env python3
"""v7.4 Figures 2 and 3 (Round 4): the original v7.2 ML displays, with terminology and numerical corrections only.

Figure 2  training vs held-out observed-versus-predicted panels, the v7.2 Figure 2 layout. Corrections: CLsys
          symbol; the curated training count (1287 for CLsys and VDss) is shown next to the number of compounds
          with archived predictions (1284).
Figure 3  head-to-head bar chart of ML Groups A-D over five metrics and three endpoints, the v7.2 Figure 3
          layout. Corrections: CLsys column title, and every ML Group A, C and D value recomputed from the
          archived per-compound prediction files (outputs/tables/ml_table2_recomputed.csv, written by
          analysis/10_ml_tables_figures.py). ML Group B remains the literature-reported score.
          Best-value emphasis is off by default (HIGHLIGHT_BEST = False): the compound-paired analysis in the SI
          detects no difference between ML Group A and the descriptor models, so marking a winner per metric
          would assert more than the statistics support. Set HIGHLIGHT_BEST = True to restore the v7.2 styling.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v7.4/fig02_03_ml_main.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

S21 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S21 / "lib"))
import style_v74 as S                                             # noqa: E402
import paths as P                                                 # noqa: E402

S.use()
HIGHLIGHT_BEST = False

RES = "ML+training-raw/DL+ML_trainV4/pred_res/res_v1/res_%s.csv"
ENDPOINTS = [("Fu", "lgFu_train", "lgFu_test", "Plasma unbound fraction (f$_u$)", "log$_{10}$ f$_u$", 4042),
             ("CL", "lgCL_train", "lgCL_test", "Systemic clearance (CL$_{sys}$)", "log$_{10}$ CL$_{sys}$", 1287),
             ("VDss", "lgVD_train", "lgVD_test", "Volume of distribution (VD$_{ss}$)", "log$_{10}$ VD$_{ss}$", 1287)]
L2, L3 = np.log10(2), np.log10(3)

EPS = ["Fu", "CL", "VDss"]
EP_TITLE = {"Fu": "f$_u$", "CL": "CL$_{sys}$", "VDss": "VD$_{ss}$"}
METRICS = ["MAE", "RMSE", "GMFE", "R2", "FE2"]
MLAB = {"MAE": "MAE", "RMSE": "RMSE", "GMFE": "GMFE", "R2": "R²", "FE2": "Within 2-fold (%)"}
HIGHER = {"R2", "FE2"}
GROUPS = list("ABCD")
GCOL = {"A": S.DLML, "B": S.SPLUS, "C": S.BU, "D": S.GREY}


def load(stem, tr, te):
    df = pd.read_csv(P.ROOT / (RES % stem))
    return {k: df[df.Dataset.astype(str) == tok][["Actual", "Predicted"]].dropna() for k, tok in (("train", tr), ("test", te))}


def r2(y, p):
    return 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)


def figure2():
    fig, axes = S.plt.subplots(3, 2, figsize=(5.8, 8.2))
    rows = []
    for ri, (key, tr, te, name, sym, n_cur) in enumerate(ENDPOINTS):
        d = load("VDss" if key == "VDss" else key, tr, te)
        v = np.concatenate([np.r_[f.Actual, f.Predicted] for f in d.values()])
        lo, hi = v.min(), v.max()
        lo, hi = lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)
        g = np.array([lo, hi])
        for ci, (lab, part, col) in enumerate((("Training set", "train", S.GREY),
                                               ("Held-out test set", "test", S.DLML))):
            ax = axes[ri, ci]
            y, p = d[part].Actual.to_numpy(float), d[part].Predicted.to_numpy(float)
            e = p - y
            m = dict(N=len(y), R2=r2(y, p), RMSE=float(np.sqrt(np.mean(e ** 2))),
                     GMFE=float(10 ** np.mean(np.abs(e))), FE2=100 * float(np.mean(np.abs(e) <= L2)))
            ax.fill_between(g, g - L2, g + L2, color=S.LIGHT, lw=0, zorder=0)
            ax.plot(g, g, color=S.OBS, lw=0.7, zorder=2)
            for s in (L3, -L3):
                ax.plot(g, g + s, color=S.GREY, lw=0.5, ls=":", zorder=2)
            big = len(y) > 1000
            ax.scatter(y, p, s=3 if big else 6, color=col, alpha=0.35 if big else 0.7, lw=0, zorder=3, rasterized=big)
            n_txt = "N = %d" % len(y) if (part == "test" or n_cur == len(y)) else "N = %d (%d plotted)" % (n_cur, len(y))
            ax.text(0.04, 0.96, "%s\nR$^2$ = %.2f\nRMSE = %.2f\nWithin 2-fold = %.0f%%" % (n_txt, m["R2"], m["RMSE"], m["FE2"]),
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.6)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Observed %s" % sym)
            ax.set_ylabel("Predicted %s" % sym)
            if ri == 0:
                ax.set_title(lab, fontsize=8.5, fontweight="bold")
            if ci == 0:
                ax.annotate(name, xy=(-0.40, 0.5), xycoords="axes fraction", rotation=90, ha="center",
                            va="center", fontsize=8, fontweight="bold")
            S.panel_label(ax, "abcdef"[ri * 2 + ci], dx=-0.26, dy=1.02)
            rows.append(dict(endpoint=key, set=lab, N_curated=n_cur if part == "train" else len(y),
                             N_plotted=len(y), **{k: v for k, v in m.items() if k != "N"}))
    fig.tight_layout(h_pad=1.2, w_pad=2.0)
    S.save(fig, "main", "Figure2_ml_train_test")
    return pd.DataFrame(rows)


def disp(m, v):
    return "%.0f" % v if m == "FE2" else "%.2f" % v


def figure3():
    d = pd.read_csv(S.TABLES / "ml_table2_recomputed.csv")
    fig, axes = S.plt.subplots(len(METRICS), len(EPS), figsize=(7.2, 7.6))
    S.plt.subplots_adjust(left=0.11, right=0.90, top=0.94, bottom=0.10, hspace=0.36, wspace=0.12)
    for c, ep in enumerate(EPS):
        sub = d[(d.endpoint == ep) & d.group.isin(GROUPS)].set_index("group").loc[GROUPS].reset_index()
        for r, m in enumerate(METRICS):
            ax = axes[r, c]
            vals = sub[m].to_numpy(float)
            best = []
            if HIGHLIGHT_BEST:
                shown = [float(disp(m, v)) for v in vals]
                b = max(shown) if m in HIGHER else min(shown)
                tied = [g for g, v in zip(GROUPS, shown) if v == b]
                best = ["A"] if "A" in tied else tied
            bars = ax.bar(range(len(GROUPS)), vals, width=0.66, color=[GCOL[g] for g in GROUPS],
                          edgecolor="#222222", linewidth=0.6)
            for b_, g in zip(bars, GROUPS):
                b_.set_alpha(0.95 if (not HIGHLIGHT_BEST or g in best) else 0.30)
            rowmax = d[d.group.isin(GROUPS)][m].max()
            top = max(0.9, rowmax * 1.25) if m == "R2" else (max(80, rowmax * 1.25) if m == "FE2" else rowmax * 1.28)
            ax.set_ylim(0, top)
            for i, (v, g) in enumerate(zip(vals, GROUPS)):
                hl = HIGHLIGHT_BEST and g in best
                ax.text(i, v + top * 0.02, disp(m, v), ha="center", va="bottom", fontsize=6.8,
                        color="#111111" if (hl or not HIGHLIGHT_BEST) else "#9a9a9a",
                        fontweight="bold" if hl else "normal")
            ax.set_xticks(range(len(GROUPS)))
            ax.set_xticklabels(GROUPS if r == len(METRICS) - 1 else [], fontsize=7.5)
            ax.tick_params(axis="x", length=0)
            ax.tick_params(axis="y", labelsize=6.8)
            if c:
                ax.set_yticklabels([])
            ax.grid(axis="y", alpha=0.25, lw=0.6)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(EP_TITLE[ep], fontsize=10, fontweight="bold", pad=6)
            if c == len(EPS) - 1:
                ax.text(1.04, 0.5, MLAB[m], transform=ax.transAxes, rotation=270, va="center", fontsize=8.5,
                        fontweight="bold")
    fig.text(0.02, 0.52, "Performance metric value", rotation=90, va="center", fontsize=9.5, fontweight="bold")
    fig.legend(handles=[Patch(facecolor=GCOL[g], edgecolor="#222222", lw=0.6, label="ML Group %s" % g) for g in GROUPS],
               ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=8)
    S.save(fig, "main", "Figure3_head_to_head")
    return d[d.group.isin(GROUPS)]


def main():
    out = S.OUT_TABLES
    out.mkdir(parents=True, exist_ok=True)
    f2 = figure2()
    f2.to_csv(out / "figure2_train_test_metrics.csv", index=False, float_format="%.4f")
    f3 = figure3()
    f3.to_csv(out / "figure3_head_to_head_values.csv", index=False, float_format="%.4f")
    print(f2.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
