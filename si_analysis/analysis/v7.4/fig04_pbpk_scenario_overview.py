#!/usr/bin/env python3
"""v7.4 Figure 4: global PBPK scenario overview (Round 4 compaction; source design = v4 Figure 5).

Seven minimal-PBPK scenarios in two blocks, so that the step at which clearance becomes predicted is visible:
  clearance observed  : v1_run0, s1_run1, s1_run2, s1_run3
  clearance predicted : h1_run0_noCLr, h2_run4, h2_run0
Panels: (a) C-T log-NRMSE (n = 40), (b) AUC0-t fold error, (c) Cmax fold error (n = 41, log2 axis, 2-fold band).
The C-T RMSE panel of v4 Figure 5 and the nine full-PBPK scenarios stay in the SI.

Descriptive only; no test is shown. Medians and 2-fold coverage are computed from the archived per-compound
table, exactly as v4 Figure 5 computed its medians, and are written to outputs/tables/v7.4/.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v7.4/fig04_pbpk_scenario_overview.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S21 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S21 / "lib"))
import style_v74 as S                                             # noqa: E402

S.use()
BLOCKS = [("Clearance observed", ["v1_run0", "s1_run1", "s1_run2", "s1_run3"]),
          ("Clearance predicted", ["h1_run0_noCLr", "h2_run4", "h2_run0"])]
ROWS = [r for _, runs in BLOCKS for r in runs]
PANELS = [("log_NRMSE", "C–T log-NRMSE", False), ("fe_auc_rel", "AUC$_{0–t}$ fold error\n(predicted/observed)", True),
          ("fe_cmax_rel", "C$_{max}$ fold error\n(predicted/observed)", True)]
SHORTLAB = {"v1_run0": "Observed-input control", "s1_run1": "S+ physicochemical inputs",
            "s1_run2": "S+ physicochemical inputs + f$_u$",
            "s1_run3": "S+ physicochemical inputs + f$_u$ + VD$_{ss}$",
            "h1_run0_noCLr": "S+ CL$_{int}$, renal CL = 0", "h2_run4": "DL–ML CL$_{sys}$",
            "h2_run0": "DL–ML f$_u$, VD$_{ss}$ and CL$_{sys}$"}


def main():
    d = pd.read_csv(S.TABLES / "per_compound_metrics.csv")
    ypos, summary = {}, []
    y = 0.0
    for b, (_, runs) in enumerate(BLOCKS):
        if b:
            y -= 0.9
        for r in runs:
            ypos[r] = y
            y -= 1.0
    split = (ypos["s1_run3"] + ypos["h1_run0_noCLr"]) / 2

    fig = S.figure(S.DOUBLE, 3.5)
    gs = fig.add_gridspec(1, 3, left=0.325, right=0.99, top=0.82, bottom=0.17, wspace=0.16)
    rng = np.random.default_rng(3)
    axes = []
    for k, (col, xl, fold) in enumerate(PANELS):
        ax = fig.add_subplot(gs[k])
        axes.append(ax)
        ax.axhspan(min(ypos.values()) - 0.6, split, color="#F4F4F4", lw=0, zorder=0)
        for run in ROWS:
            v = d[d.scenario == run][col].dropna().to_numpy(float)
            c = S.COLOR[run]
            ax.boxplot([v], positions=[ypos[run]], orientation="horizontal", widths=0.6, showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor="white", edgecolor=c, lw=0.7),
                       medianprops=dict(color=c, lw=1.3), whiskerprops=dict(color=c, lw=0.6),
                       capprops=dict(color=c, lw=0.6), zorder=2)
            ax.plot(v, ypos[run] + rng.uniform(-0.19, 0.19, len(v)), "o", ms=1.8, color=c, alpha=0.6, zorder=3)
            if k == 0:
                summary.append(dict(scenario=run, label=SHORTLAB[run], n_profile=len(v),
                                    median_log_NRMSE=float(np.median(v))))
        if fold:
            ax.axvspan(-1, 1, color=S.LIGHT, lw=0, zorder=1)
            ax.axvline(0, color=S.OBS, lw=0.6, zorder=1)
            ax.set_xlim(-4.6, 4.6)
            ax.set_xticks([-2, -1, 0, 1, 2])
            ax.set_xticklabels(["1/4", "1/2", "1", "2", "4"])
        else:
            ax.set_xlim(0, 1.05)
        ax.set_xlabel(xl)
        ax.set_ylim(min(ypos.values()) - 0.6, 0.65)
        ax.set_yticks([ypos[r] for r in ROWS])
        ax.set_yticklabels([SHORTLAB[r] for r in ROWS] if k == 0 else [])
        ax.tick_params(axis="y", length=0)
        S.panel_label(ax, "abc"[k], dx=-0.06 if k else -0.90, dy=1.10)

    # 2-fold coverage of AUC and Cmax, printed beside the panel they belong to
    for k, (col, _, fold) in enumerate(PANELS):
        if not fold:
            continue
        for run in ROWS:
            v = d[d.scenario == run][col].dropna().to_numpy(float)
            axes[k].text(4.5, ypos[run] + 0.34, "%.0f%%" % (100 * np.mean(np.abs(v) <= 1)), ha="right",
                         va="center", fontsize=6.2, color=S.OBS)
        axes[k].text(4.5, 0.62, "within\n2-fold", ha="right", va="bottom", fontsize=6.2, color=S.OBS)

    for (name, runs), va in zip(BLOCKS, ("bottom", "bottom")):
        yy = ypos[runs[0]] + 0.62
        axes[0].text(-0.88, yy, name, transform=axes[0].get_yaxis_transform(), fontsize=7.4, style="italic",
                     color=S.OBS, ha="left", va=va)
    for ax in axes:
        ax.axhline(split, color=S.GREY, lw=0.6, ls=":", zorder=4)

    out = S.OUT_TABLES
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in ROWS:
        r = dict(scenario=run, label=SHORTLAB[run])
        for col, _, _ in PANELS:
            v = d[d.scenario == run][col].dropna().to_numpy(float)
            r["n_" + col] = len(v)
            r["median_" + col] = float(np.median(v))
            if col != "log_NRMSE":
                r["within2_" + col] = float(100 * np.mean(np.abs(v) <= 1))
                r["median_fold_" + col] = float(2 ** np.median(v))
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(out / "figure4_scenario_summary.csv", index=False, float_format="%.4f")
    S.save(fig, "main", "Figure4_pbpk_scenario_overview")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
