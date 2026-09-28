#!/usr/bin/env python3
"""Figure 7 (v4): (a) cumulative input substitution relative to the observed-input control; (b) association
between DL-ML parameter fold error and exposure error in the scenario with DL-ML Fu, VDss and CL (h2_run0).
The residual Cmax structural analysis (v3 Figure 7c) is in Figure S13.

(a) Hodges-Lehmann paired change in |log2 FE| (AUC0-t circles, Cmax triangles) vs v1_run0 with 95% bootstrap
    CI (outputs/tables/decomposition_ladder.csv). Filled = Holm-adjusted p < 0.05, Holm across the six steps
    within each endpoint (computed here from the Wilcoxon p in the ladder table); CI always drawn.
(b) |Spearman rho| (outputs/tables/decomposition_routing.csv); cell text = rho with sign; n.s. = p >= 0.05.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v4/fig07_substitution_association.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

S21 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S21 / "lib"))
import style_v4 as S                                              # noqa: E402

S.use()
T = S21 / "outputs" / "tables"
STEPS = [("s1_run1", "S+ physicochemical inputs", S.SPLUS), ("s1_run2", "+ S+ Fu", S.SPLUS),
         ("s1_run3", "+ S+ VDss", S.SPLUS), ("h2_run4", "+ DL–ML total CL", S.TD),
         ("h1_run0_noCLr", "+ S+ CLint, renal CL = 0", S.BU), ("h2_run0", "+ DL–ML Fu, VDss", S.DLML)]


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    lad = pd.read_csv(T / "decomposition_ladder.csv")
    rout = pd.read_csv(T / "decomposition_routing.csv")
    fig = S.figure(6.3, 3.0)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    out = []
    for ep, mk, dy in (("fe_auc_abs", "o", 0.15), ("fe_cmax_abs", "^", -0.15)):
        s = lad[lad.endpoint == ep].set_index("run_id").loc[[r for r, _, _ in STEPS]]
        ph = holm(s.p.to_numpy())
        for i, ((run, lab, col), (_, r), p) in enumerate(zip(STEPS, s.iterrows(), ph)):
            y = -i + dy
            ax.plot([r.ci_low, r.ci_high], [y, y], color=col, lw=0.9, solid_capstyle="butt")
            ax.plot(r.estimate, y, mk, ms=4.2, mec=col, mew=0.9, mfc=col if S.filled(p) else "white")
            out.append(dict(endpoint=ep, run_id=run, step=lab, estimate=r.estimate, ci_low=r.ci_low,
                            ci_high=r.ci_high, p=r.p, p_holm=p))
    ax.axvline(0, color=S.OBS, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks([-i for i in range(len(STEPS))])
    ax.set_yticklabels([l for _, l, _ in STEPS])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-len(STEPS) + 0.4, 0.6)
    ax.set_xlabel("Change in |log$_2$ fold error| vs observed-input control")
    ax.legend(handles=[Line2D([], [], ls="", marker="o", mfc=S.OBS, mec=S.OBS, ms=4, label="AUC$_{0–t}$"),
                       Line2D([], [], ls="", marker="^", mfc=S.OBS, mec=S.OBS, ms=4, label="C$_{max}$"),
                       Line2D([], [], ls="", marker="o", mfc="white", mec=S.OBS, ms=4, label="Holm p ≥ 0.05")],
              loc="upper right", handletextpad=0.2)
    S.panel_label(ax, "a", dx=-0.75, dy=1.02)
    pd.DataFrame(out).to_csv(T / "v4_figure7a_holm.csv", index=False, float_format="%.5f")

    ax = fig.add_subplot(gs[1])
    ex = ["AUC, signed log2 FE", "Cmax, signed log2 FE", "C–T, RMSE log10", "C–T, log-NRMSE"]
    pars = ["Fu", "VDss", "CLsys"]
    rho = rout.pivot(index="parameter", columns="exposure", values="rho").loc[pars, ex]
    pv = rout.pivot(index="parameter", columns="exposure", values="p").loc[pars, ex]
    im = ax.imshow(rho.abs().to_numpy(), cmap=S.HEATMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(len(pars)):
        for j in range(len(ex)):
            v, p = rho.iloc[i, j], pv.iloc[i, j]
            ax.text(j, i, "%+.2f" % v + ("" if p < 0.05 else "\nn.s."), ha="center", va="center", fontsize=7,
                    color="white" if abs(v) < 0.55 else "black")
    ax.set_xticks(range(len(ex)))
    ax.set_xticklabels(["AUC$_{0–t}$", "C$_{max}$", "C–T RMSE", "C–T\nlog-NRMSE"])
    ax.set_yticks(range(len(pars)))
    ax.set_yticklabels(["f$_u$", "VD$_{ss}$", "CL"])
    ax.set_ylabel("DL–ML parameter fold error")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("|Spearman ρ|")
    cb.outline.set_linewidth(0.5)
    S.panel_label(ax, "b", dx=-0.3, dy=1.02)
    S.save(fig, "main", "Figure7_substitution_association")


if __name__ == "__main__":
    main()
