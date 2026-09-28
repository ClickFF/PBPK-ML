#!/usr/bin/env python3
"""Figure S12 (v4; v3 Figure S11): upstream parameter error vs exposure error in the scenario with DL-ML Fu,
VDss and CL (h2_run0).

(a) |log2 CL fold error| vs C-T RMSE of log10 concentrations (principal profile panel, n = 41);
(b) signed log2 CL fold error vs signed log2 AUC0-t fold error (n = 41);
(c) signed log2 VDss fold error vs signed log2 Cmax fold error (n = 41).
Spearman rho, 95% percentile bootstrap CI over compounds (10,000 resamples, seed 20260920, same implementation
as response_letter/P9/scripts/parameter_to_exposure.py) and p are recomputed here from
outputs/tables/upstream_parameter_fold_errors.csv and per_compound_metrics.csv. The log-NRMSE association
(n = 40) is computed the same way and reported as a sensitivity analysis in the caption only.
Output: outputs/tables/v4_figureS12_spearman.csv.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v4/figS12_parameter_exposure.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

S21 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S21 / "lib"))
import style_v4 as S                                              # noqa: E402

S.use()
T = S21 / "outputs" / "tables"
N_BOOT = 10000


def boot(x, y):
    rng = np.random.default_rng(20260920)
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
    spec = [("a", "CLsys", "rmse_log10", True, "|log$_2$ CL fold error|", "C–T RMSE (log$_{10}$ concentration)", "|CL FE| vs C–T RMSE"),
            ("b", "CLsys", "fe_auc_rel", False, "log$_2$ CL fold error", "log$_2$ AUC$_{0–t}$ fold error", "CL FE vs AUC FE"),
            ("c", "VDss", "fe_cmax_rel", False, "log$_2$ VD$_{ss}$ fold error", "log$_2$ C$_{max}$ fold error", "VDss FE vs Cmax FE"),
            (None, "CLsys", "log_NRMSE", True, None, None, "|CL FE| vs C–T log-NRMSE (sensitivity)")]
    rows = []
    fig, axes = S.plt.subplots(1, 3, figsize=(S.DOUBLE, 2.5))
    for letter, par, ex, ab, xl, yl, lab in spec:
        j = d[[par, ex]].dropna()
        x = j[par].abs().to_numpy() if ab else j[par].to_numpy()
        y = j[ex].to_numpy()
        rho, p = spearmanr(x, y)
        lo, hi = boot(x, y)
        rows.append(dict(association=lab, n=len(x), spearman_rho=rho, ci_low=lo, ci_high=hi, p=p))
        if letter is None:
            continue
        ax = axes["abc".index(letter)]
        if not ab:
            ax.axhline(0, color=S.OBS, lw=0.6)
            ax.axvline(0, color=S.OBS, lw=0.6)
        ax.plot(x, y, "o", ms=3.2, mfc=S.DLML, mec="white", mew=0.3)
        ax.text(0.03, 0.97 if ab else 0.03, "ρ = %.2f [%.2f, %.2f]\n%s; N = %d" % (rho, lo, hi, S.pfmt(p), len(x)),
                transform=ax.transAxes, ha="left", va="top" if ab else "bottom", fontsize=7,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        S.panel_label(ax, letter, dx=-0.3, dy=1.02)
    fig.tight_layout(w_pad=1.2)
    pd.DataFrame(rows).to_csv(T / "v4_figureS12_spearman.csv", index=False, float_format="%.4f")
    S.save(fig, "si", "FigureS12_parameter_exposure")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
