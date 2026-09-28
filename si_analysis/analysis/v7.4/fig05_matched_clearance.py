#!/usr/bin/env python3
"""v7.4 Figure 5 (Round 4): information-matched predicted CLint vs predicted CLsys (redesign of v4 Figure S16a + the
S16b paired-strip layout applied to the matched pair).

(a) Direct matched contrast h1_run0_noCLr - h2_run4 (contrast A of outputs/tables/clearance_hierarchy_contrasts.csv,
    = Table S11 / v4 Figure S16a): four absolute-error endpoints (one Holm family), then two signed rows (bias
    direction). Filled = Holm-adjusted p < 0.05; open otherwise; CI always drawn.
(b) Per-compound signed AUC0-t log2 FE under the two routes, connected by compound (per_compound_metrics.csv);
    median bars and over/under-2-fold counts cross-checked against clearance_hierarchy_signed_summary.csv.
Nothing is recomputed.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v7.4/fig05_matched_clearance.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

S21 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S21 / "lib"))
import style_v74 as S                                             # noqa: E402

S.use()
BU, TD = "h1_run0_noCLr", "h2_run4"
ABS = ["log_NRMSE", "rmse_log10", "fe_auc_abs", "fe_cmax_abs"]
SIGNED = ["fe_auc_rel", "fe_cmax_rel"]
ROWLAB = {"log_NRMSE": "C–T log-NRMSE (n = 40)", "rmse_log10": "C–T RMSE (log$_{10}$ conc.)",
          "fe_auc_abs": "AUC$_{0–t}$ |log$_2$ FE|", "fe_cmax_abs": "C$_{max}$ |log$_2$ FE|",
          "fe_auc_rel": "AUC$_{0–t}$ signed log$_2$ FE", "fe_cmax_rel": "C$_{max}$ signed log$_2$ FE"}
MK = dict(S.MARK, fe_auc_rel="o", fe_cmax_rel="^")


def main():
    c = pd.read_csv(S.TABLES / "clearance_hierarchy_contrasts.csv")
    a = c[c.contrast == "A"].set_index("endpoint")
    assert a.loc["fe_auc_abs", "test"] == BU and a.loc["fe_auc_abs", "reference"] == TD
    per = pd.read_csv(S.TABLES / "per_compound_metrics.csv")
    w = per[per.scenario.isin([BU, TD])].pivot(index="compound_id", columns="scenario", values="fe_auc_rel")
    summ = pd.read_csv(S.TABLES / "clearance_hierarchy_signed_summary.csv")
    summ = summ[summ.endpoint == "fe_auc_rel"].set_index("scenario")
    for run in (BU, TD):  # cross-check the figure against the archived summary
        v = w[run]
        assert abs(v.median() - summ.loc[run, "median"]) < 1e-3
        assert int((v > 1).sum()) == summ.loc[run, "n_over_2fold"]
        assert int((v < -1).sum()) == summ.loc[run, "n_under_2fold"]

    fig = S.figure(S.DOUBLE, 3.1)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], left=0.25, right=0.985, top=0.88, bottom=0.20, wspace=0.42)
    ax = fig.add_subplot(gs[0])
    ys, rows = [], []
    y = 0.0
    for ep in ABS + SIGNED:
        if ep == SIGNED[0]:
            y -= 0.6
            ax.axhline(y + 0.5, color=S.GREY, lw=0.5, ls=":")
        r = a.loc[ep]
        signed = ep in SIGNED
        col = S.GREY if signed else S.OBS
        ax.plot([r.ci_low, r.ci_high], [y, y], color=col, lw=0.9, solid_capstyle="butt")
        ax.plot(r.estimate, y, MK[ep], ms=4.2, mec=col, mew=0.9, mfc=col if S.filled(r.p_holm) else "white")
        ys.append(y)
        rows.append(dict(endpoint=ep, n=int(r.n), estimate=r.estimate, ci_low=r.ci_low, ci_high=r.ci_high,
                         p=r.p, p_holm=r.p_holm))
        y -= 1
    ax.axvline(0, color=S.OBS, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(ys)
    ax.set_yticklabels([ROWLAB[e] for e in ABS + SIGNED])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y + 0.4, 0.6)
    ax.set_xlabel("S+ CL$_{int}$ (renal CL = 0) − DL–ML CL$_{sys}$")
    ax.text(1.0, ys[1] + 0.05, "accuracy\n(one Holm family)", transform=ax.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=6.3, color=S.OBS)
    ax.text(1.0, ys[-1] - 0.35, "bias direction", transform=ax.get_yaxis_transform(), ha="right", va="top",
            fontsize=6.3, color=S.GREY)
    S.panel_label(ax, "a", dx=-0.62, dy=1.05)

    ax = fig.add_subplot(gs[1])
    for _, r in w.iterrows():
        ax.plot([0, 1], [r[BU], r[TD]], color="#cfcfcf", lw=0.6, zorder=1)
    rng = np.random.default_rng(4)
    for x, run, col in ((0, BU, S.BU), (1, TD, S.TD)):
        v = w[run].to_numpy(float)
        ax.plot(np.full(len(v), x), v, "o", ms=2.6, color=col, zorder=2)
        ax.plot([x - 0.2, x + 0.2], [np.median(v)] * 2, color=S.OBS, lw=1.5, zorder=3)
        s = summ.loc[run]
        ax.text(x, 5.75, "median %+.2f\n%d over 2-fold\n%d under 2-fold" % (s["median"], s.n_over_2fold,
                s.n_under_2fold), ha="center", va="bottom", fontsize=6.3, color=S.OBS)
    ax.axhspan(-1, 1, color=S.LIGHT, lw=0, zorder=0)
    ax.axhline(0, color=S.OBS, lw=0.6)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-3.6, 8.2)
    ax.set_yticks([-2, 0, 2, 4])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["S+ CL$_{int}$,\nrenal CL = 0", "DL–ML\nCL$_{sys}$"])
    ax.set_ylabel("Signed AUC$_{0–t}$ log$_2$ FE")
    S.panel_label(ax, "b", dx=-0.32, dy=1.05)

    fig.legend(handles=[Line2D([], [], ls="", marker="o", mfc=S.OBS, mec=S.OBS, ms=4, label="Holm-adjusted p < 0.05"),
                        Line2D([], [], ls="", marker="o", mfc="white", mec=S.OBS, ms=4, label="Holm-adjusted p ≥ 0.05")],
               loc="lower left", ncol=2, bbox_to_anchor=(0.12, -0.01))
    out = S.OUT_TABLES
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "figure5a_values.csv", index=False, float_format="%.5f")
    S.save(fig, "main", "Figure5_matched_clearance")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
