#!/usr/bin/env python3
"""New main-text Figure 6: upstream input error and its propagation to exposure error.

Merges the two displays that were Figures S15 and S16 of the v7.4 SI into one main-text figure:

  (a) |log2 CLsys fold error| vs C-T RMSE of log10 concentrations   (was S15a)
  (b) signed log2 CLsys fold error vs signed log2 AUC0-t fold error (was S15b)
  (c) signed log2 VDss fold error vs signed log2 Cmax fold error    (was S15c)
  (d) |Spearman rho| for every DL-ML parameter x exposure endpoint  (was S16)

All four panels describe the same scenario, the all-predicted workflow (DL-ML Fu, VDss and CLsys with S+
physicochemical inputs; h2_run0), which is why they belong in one figure. Statistics are recomputed here from
the same archived CSVs the SI figures used, with the same estimator, bootstrap and seed, so the values are
identical to the frozen captions.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v7.4_final01/fig06_parameter_to_exposure.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

S21 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S21 / "lib"))
import style_draft02 as S                                         # noqa: E402

S.use()
T = S21 / "outputs" / "tables"
N_BOOT = 10000
SEED = 20260920


def boot(x, y):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    b = np.array([spearmanr(x[i], y[i]).statistic for i in idx])
    b = b[np.isfinite(b)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    fe = pd.read_csv(T / "upstream_parameter_fold_errors.csv")
    fe = fe[fe.predictor == "DL-ML"].pivot(index="compound_id", columns="parameter", values="log2_fe")
    per = pd.read_csv(T / "per_compound_metrics.csv")
    per = per[per.scenario == "h2_run0"].set_index("compound_id")
    d = fe.join(per[["rmse_log10", "log_NRMSE", "fe_auc_rel", "fe_cmax_rel"]])
    rout = pd.read_csv(T / "decomposition_routing.csv")

    spec = [("a", "CLsys", "rmse_log10", True,
             "|log$_2$ CL$_{sys}$ fold error|", "C–T RMSE (log$_{10}$ concentration)", "|CLsys FE| vs C–T RMSE"),
            ("b", "CLsys", "fe_auc_rel", False,
             "log$_2$ CL$_{sys}$ fold error", "log$_2$ AUC$_{0–t}$ fold error", "CLsys FE vs AUC FE"),
            ("c", "VDss", "fe_cmax_rel", False,
             "log$_2$ VD$_{ss}$ fold error", "log$_2$ C$_{max}$ fold error", "VDss FE vs Cmax FE"),
            (None, "CLsys", "log_NRMSE", True, None, None, "|CLsys FE| vs C–T log-NRMSE (sensitivity)")]

    fig = S.figure(S.DOUBLE, 5.0)
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.34,
                          left=0.085, right=0.965, top=0.93, bottom=0.085)
    axes = {"a": fig.add_subplot(gs[0, 0]), "b": fig.add_subplot(gs[0, 1]),
            "c": fig.add_subplot(gs[1, 0]), "d": fig.add_subplot(gs[1, 1])}
    rows = []
    for letter, par, ex, ab, xl, yl, lab in spec:
        j = d[[par, ex]].dropna()
        x = j[par].abs().to_numpy() if ab else j[par].to_numpy()
        y = j[ex].to_numpy()
        rho, p = spearmanr(x, y)
        lo, hi = boot(x, y)
        rows.append(dict(association=lab, n=len(x), spearman_rho=rho, ci_low=lo, ci_high=hi, p=p))
        if letter is None:
            continue
        ax = axes[letter]
        if not ab:
            ax.axhline(0, color=S.OBS, lw=0.6)
            ax.axvline(0, color=S.OBS, lw=0.6)
        ax.plot(x, y, "o", ms=3.4, mfc=S.DLML, mec="white", mew=0.35, alpha=0.95)
        lab_txt = ("ρ = %.2f [%.2f, %.2f]\n%s; N = %d" % (rho, lo, hi, S.pfmt(p), len(x))
                   ).replace("-", "\u2212")          # true minus, matching the tick labels
        ax.text(0.03, 0.97 if ab else 0.03, lab_txt,
                transform=ax.transAxes, ha="left", va="top" if ab else "bottom", fontsize=7,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.4))
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        S.panel_label(ax, letter, dx=-0.20, dy=1.02)

    # (d) association summary across every parameter x exposure pair
    ax = axes["d"]
    ex_cols = ["AUC, signed log2 FE", "Cmax, signed log2 FE", "C–T, RMSE log10", "C–T, log-NRMSE"]
    pars = ["Fu", "VDss", "CLsys"]
    rho = rout.pivot(index="parameter", columns="exposure", values="rho").loc[pars, ex_cols]
    pv = rout.pivot(index="parameter", columns="exposure", values="p").loc[pars, ex_cols]
    im = ax.imshow(rho.abs().to_numpy(), cmap=S.HEATMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(len(pars)):
        for j2 in range(len(ex_cols)):
            v, p = rho.iloc[i, j2], pv.iloc[i, j2]
            cell = ("%+.2f" % v).replace("-", "\u2212") + ("" if p < 0.05 else "\nn.s.")
            ax.text(j2, i, cell, ha="center", va="center",
                    fontsize=7, color="white" if abs(v) < 0.55 else "black")
    ax.set_xticks(range(len(ex_cols)))
    ax.set_xticklabels(["AUC$_{0–t}$", "C$_{max}$", "C–T\nRMSE", "C–T\nlog-NRMSE"])
    ax.set_yticks(range(len(pars)))
    ax.set_yticklabels(["f$_u$", "VD$_{ss}$", "CL$_{sys}$"])
    ax.set_ylabel("DL–ML input fold error")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cb.set_label("|Spearman ρ|")
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=2)
    S.panel_label(ax, "d", dx=-0.20, dy=1.02)

    S.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(S.OUT_TABLES / "figure6_parameter_to_exposure_spearman.csv", index=False, float_format="%.4f")
    S.save(fig, "main", "Figure6_parameter_to_exposure")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
