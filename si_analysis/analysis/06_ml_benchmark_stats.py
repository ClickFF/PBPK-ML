#!/usr/bin/env python3
"""ML benchmarking statistics for Table 2 -> outputs/tables/ml_*.csv, outputs/figures/ml/.

Sources (read only):
  repo/benchmarking_statistics_table2/outputs/summary/table2_groupA_vs_controls_stats_testset.csv
      ML Group A vs C, D, E, F, G on Test Set #1, matched by PUBCHEM_CID (2000 bootstrap
      resamples, seed 0; Wilcoxon on paired |error|; McNemar on paired within-2-fold)
  (ML Group B is not analysed: the exact published compound-level predictions are not
   available, so A vs B is descriptive only; group_b() below is retained for audit, not called)

Added here, nothing else recomputed for C-G:
  * Holm adjustment within the prespecified family A vs {C, D, E, F, G} x {Fu, CL, VDss} (15
    tests), separately for the Wilcoxon |error| test and the McNemar within-2-fold test
  * a reviewer-facing forest plot of delta MAE and delta within-2-fold (A - comparator)

Delta = Group A - comparator: negative delta MAE and positive delta FE<2 favour Group A.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/06_ml_benchmark_stats.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEP21 = HERE.parent
REPO = SEP21.parent
ROOT = REPO.parent
BENCH = REPO / "benchmarking_statistics_table2"
sys.path.insert(0, str(BENCH / "scripts"))
sys.path.insert(0, str(SEP21 / "lib"))
import run_pairwise_stats as R                                     # noqa: E402
import matplotlib                                                  # noqa: E402

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt                                    # noqa: E402
import style                                                       # noqa: E402

OUT_T = SEP21 / "outputs" / "tables"
OUT_F = SEP21 / "outputs" / "figures" / "ml"
OUT_F.mkdir(parents=True, exist_ok=True)
LABEL = {"B": "B, published (106-CID subset)", "C": "C, optimal ML (reported settings)",
         "D": "D, reproduced ML", "E": "E, ATFP embeddings only", "F": "F, RDKit only",
         "G": "G, merged embeddings only"}


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def group_b():
    raw = pd.read_csv(ROOT / "Table2_data" / "GroupB" / "RawData.csv")
    spec = {"Fu": ("fu", "pred_fu"), "CL": ("CL_final(L/hour/kg)", "pred_CL_(L/hour/kg)"),
            "VDss": ("VD_final(L/kg)", "pred_VD_(L/kg)")}
    rows = []
    for ep, (ac, pc) in spec.items():
        b = pd.DataFrame({"PUBCHEM_CID": raw["PubChem_CID"].astype(int),
                          "Actual": raw[ac].astype(float).map(math.log10),
                          "Predicted": raw[pc].astype(float).map(math.log10)})
        a = R.load_endpoint("A", ep)
        m = a.merge(b, on="PUBCHEM_CID", suffixes=("_A", "_B"))
        ta, pa = m.Actual_A.to_numpy(float), m.Predicted_A.to_numpy(float)
        tb, pb = m.Actual_B.to_numpy(float), m.Predicted_B.to_numpy(float)
        ea, eb = np.abs(pa - ta), np.abs(pb - tb)
        fa, fb = R.fold_error_log10(ta, pa), R.fold_error_log10(tb, pb)
        ma, mb = R.metric_block(ta, pa), R.metric_block(tb, pb)
        d_mae, lo_mae, hi_mae = R.paired_bootstrap_delta(ea, eb)
        d_rmse, lo_rmse, hi_rmse = R.bootstrap_delta_rmse(ea ** 2, eb ** 2)
        d_fe, lo_fe, hi_fe = R.paired_bootstrap_delta((fa <= 2).astype(float), (fb <= 2).astype(float))
        d_r2, lo_r2, hi_r2 = R.bootstrap_delta_r2(ta, pa, pb)
        rows.append({"Comparison": "A vs B", "Comparator": "B", "Endpoint": ep, "N_paired": len(m),
                     "max_abs_true_diff_log10": float(np.max(np.abs(ta - tb))),
                     "A_MAE": ma["MAE"], "Comparator_MAE": mb["MAE"], "delta_MAE": d_mae,
                     "delta_MAE_CI95_lo": lo_mae, "delta_MAE_CI95_hi": hi_mae,
                     "wilcoxon_p_abs_err": R.wilcoxon_p(ea, eb),
                     "A_RMSE": ma["RMSE"], "Comparator_RMSE": mb["RMSE"], "delta_RMSE": d_rmse,
                     "delta_RMSE_CI95_lo": float(lo_rmse), "delta_RMSE_CI95_hi": float(hi_rmse),
                     "A_GMFE": ma["GMFE"], "Comparator_GMFE": mb["GMFE"],
                     "A_FE<2": ma["FE<2"], "Comparator_FE<2": mb["FE<2"], "delta_FE<2": d_fe,
                     "delta_FE<2_CI95_lo": lo_fe, "delta_FE<2_CI95_hi": hi_fe,
                     "mcnemar_p_FE<2": R.mcnemar_p(fa <= 2, fb <= 2),
                     "A_R2": ma["R2"], "Comparator_R2": mb["R2"], "delta_R2": d_r2,
                     "delta_R2_CI95_lo": lo_r2, "delta_R2_CI95_hi": hi_r2})
    return pd.DataFrame(rows)


def verdict(r):
    mae_det = r.delta_MAE_CI95_hi < 0 or r.delta_MAE_CI95_lo > 0
    fe_det = r["delta_FE<2_CI95_lo"] > 0 or r["delta_FE<2_CI95_hi"] < 0
    parts = []
    if r.p_holm_wilcoxon < 0.05:
        parts.append("MAE %s after Holm" % ("favours A" if r.delta_MAE < 0 else "favours comparator"))
    elif mae_det:
        parts.append("MAE CI excludes 0, not after Holm")
    if r["p_holm_mcnemar"] < 0.05:
        parts.append("FE<2 %s after Holm" % ("favours A" if r["delta_FE<2"] > 0 else "favours comparator"))
    elif fe_det:
        parts.append("FE<2 CI excludes 0, not after Holm")
    return "; ".join(parts) if parts else "no detectable difference"


def main():
    s = pd.read_csv(BENCH / "outputs" / "summary" / "table2_groupA_vs_controls_stats_testset.csv")
    s["family"] = "primary: A vs C-G (15 tests)"
    s["p_holm_wilcoxon"] = holm(s["wilcoxon_p_abs_err"])
    s["p_holm_mcnemar"] = holm(s["mcnemar_p_FE<2"])
    # ML Group B: not reported (author decision 2026-09-21) - the exact published compound-level
    # predictions are not available; Table2_data/GroupB/RawData.csv is not treated as that source.
    # A vs B stays descriptive (Table 2). group_b() is kept for audit only and is not called.
    allr = s.copy()
    allr["verdict"] = allr.apply(verdict, axis=1)
    order = {"B": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5}
    eo = {"Fu": 0, "CL": 1, "VDss": 2}
    allr = allr.sort_values(by=["Comparator", "Endpoint"],
                            key=lambda c: c.map(order) if c.name == "Comparator" else c.map(eo))
    allr.to_csv(OUT_T / "ml_benchmark_paired_stats_holm.csv", index=False, float_format="%.6g")
    show = allr[["Comparator", "Endpoint", "N_paired", "delta_MAE", "delta_MAE_CI95_lo",
                 "delta_MAE_CI95_hi", "wilcoxon_p_abs_err", "p_holm_wilcoxon", "delta_FE<2",
                 "delta_FE<2_CI95_lo", "delta_FE<2_CI95_hi", "mcnemar_p_FE<2", "p_holm_mcnemar",
                 "delta_R2", "verdict"]]
    pd.set_option("display.width", 250)
    print(show.round(4).to_string(index=False))
    forest(allr)


def forest(d):
    style.use()
    matplotlib.rcParams.update({"font.family": "sans-serif",
                                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
    comps = ["C", "D", "E", "F", "G"]
    eps = ["Fu", "CL", "VDss"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=True)
    ylab, ypos = [], []
    y = 0
    for c in comps:
        for e in eps:
            ypos.append(y); ylab.append("A vs %s  %s" % (c, e)); y += 1
        y += 0.6
    for ax, (col, lo, hi, padj, fav_neg, xl) in zip(axes, [
            ("delta_MAE", "delta_MAE_CI95_lo", "delta_MAE_CI95_hi", "p_holm_wilcoxon", True,
             "Δ MAE (log$_{10}$ units), A − comparator"),
            ("delta_FE<2", "delta_FE<2_CI95_lo", "delta_FE<2_CI95_hi", "p_holm_mcnemar", False,
             "Δ fraction within 2-fold, A − comparator")]):
        ax.axvline(0, color="#52514e", lw=0.8)
        k = 0
        for c in comps:
            for e in eps:
                r = d[(d.Comparator == c) & (d.Endpoint == e)].iloc[0]
                det = r[lo] > 0 or r[hi] < 0
                sig = r[padj] < 0.05
                col_ = "#2a78d6" if sig else ("#8fb8e8" if det else "#8b8a85")
                ax.plot([r[lo], r[hi]], [ypos[k]] * 2, color=col_, lw=1.6)
                ax.plot([r[col]], [ypos[k]], marker="o" if c != "B" else "s", ms=5,
                        mfc=col_ if (sig or det) else "white", mec=col_, mew=1.0)
                k += 1
        ax.set_xlabel(xl, fontsize=8.8)
        ax.grid(True, axis="x", color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(ylab, fontsize=8)
    axes[0].set_ylim(max(ypos) + 0.7, -0.7)
    axes[0].tick_params(axis="y", length=0)
    axes[0].text(0.02, 1.01, "(a)  favours A ←", transform=axes[0].transAxes, fontsize=9, fontweight="bold")
    axes[1].text(0.02, 1.01, "(b)  → favours A", transform=axes[1].transAxes, fontsize=9, fontweight="bold")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="#2a78d6", marker="o", lw=1.6, label="Holm-adjusted p < 0.05"),
         Line2D([], [], color="#8fb8e8", marker="o", lw=1.6, label="95% CI excludes 0, Holm p ≥ 0.05"),
         Line2D([], [], color="#8b8a85", marker="o", mfc="white", lw=1.6, label="95% CI includes 0")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=7.8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    for ext, kw in (("png", {"dpi": 400}), ("pdf", {}), ("svg", {})):
        fig.savefig(OUT_F / ("FigureR1_ml_paired_forest.%s" % ext), **kw)
    plt.close(fig)
    print("wrote", OUT_F / "FigureR1_ml_paired_forest.png")


if __name__ == "__main__":
    main()
