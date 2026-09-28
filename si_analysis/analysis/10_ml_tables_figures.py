#!/usr/bin/env python3
"""Table 2 / Table S1 values recomputed from the per-compound prediction files, and Figures 3 / S1.

Resolves conflicts C1/C2 (author decision 2026-09-21): every ML Group A and control value is
recomputed from `repo/benchmarking_statistics_table2/inputs/Group*/res_*.csv` on the Test Set #1
compounds (ML Group A restricted to the CIDs of the benchmark test sets: Fu 633, CL 177, VDss 177;
ML Groups C and D from the Sep 14 CID-paired files). ML Group B remains the literature-reported
score (Table2_data/Table2.xlsx), descriptive only.

Highlight rule (tables and figures identical): per endpoint and metric, the best displayed value
(2 decimals; %<2FE as integer percent) is bold + underlined; lower is better for MAE, RMSE, GMFE,
higher for R² and within 2-fold. If ML Group A ties for best, only ML Group A is highlighted.

Outputs: outputs/tables/ml_table2_recomputed.csv, manuscript/v3/tables/Table2.md, TableS1.md,
manuscript/v3/figures/Figure3_head_to_head.{png,pdf}, manuscript/v3/SI/figures/FigureS1_control_groups.{png,pdf}

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/10_ml_tables_figures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import Patch                               # noqa: E402

HERE = Path(__file__).resolve().parent
S21 = HERE.parent
REPO = S21.parent
ROOT = REPO.parent
INP = REPO / "benchmarking_statistics_table2" / "inputs"
V3 = S21 / "manuscript" / "v3"
TOK = {"Fu": "lgFu_test", "CL": "lgCL_test", "VDss": "lgVD_test"}
EPS = ["Fu", "CL", "VDss"]
METRICS = ["MAE", "RMSE", "GMFE", "R2", "FE2"]
HIGHER = {"R2", "FE2"}
CONFIG = {  # method, algorithm, descriptor — v7.2 Table 2 / Table S1, unchanged
    ("A", "Fu"): ("This work (DL–ML)", "SVM", "Merged GNN embeddings + RDKit"),
    ("A", "CL"): ("This work (DL–ML)", "SVM", "Merged GNN embeddings + RDKit"),
    ("A", "VDss"): ("This work (DL–ML)", "SVM", "Merged GNN embeddings + RDKit"),
    ("B", "Fu"): ("Literature reported score (ML)", "SVM", "Merged"),
    ("B", "CL"): ("Literature reported score (ML)", "Consensus", "Consensus"),
    ("B", "VDss"): ("Literature reported score (ML)", "Consensus", "Consensus"),
    ("C", "Fu"): ("Optimal ML", "SVM", "Merged"), ("C", "CL"): ("Optimal ML", "SVM", "Merged"),
    ("C", "VDss"): ("Optimal ML", "RF", "Mordred"),
    ("D", "Fu"): ("Reproduced ML", "SVM", "Merged"), ("D", "CL"): ("Reproduced ML", "SVM", "Merged"),
    ("D", "VDss"): ("Reproduced ML", "SVM", "RDKit"),
    ("E", "Fu"): ("Attentive FP", "GNN", "2D graph"), ("E", "CL"): ("Attentive FP", "GNN", "2D graph"),
    ("E", "VDss"): ("Attentive FP", "GNN", "2D graph"),
    ("F", "Fu"): ("RDKit + ML", "SVM", "RDKit"), ("F", "CL"): ("RDKit + ML", "RF", "RDKit"),
    ("F", "VDss"): ("RDKit + ML", "SVM", "RDKit"),
    ("G", "Fu"): ("Embeddings + ML", "Stacking", "Merged GNN embeddings"),
    ("G", "CL"): ("Embeddings + ML", "SVM", "Merged GNN embeddings"),
    ("G", "VDss"): ("Embeddings + ML", "Grad Boost", "Merged GNN embeddings"),
}
COLORS = {"A": "#32c4a6", "B": "#4C78A8", "C": "#72B7B2", "D": "#9D755D",
          "E": "#f6b7a1", "F": "#b9c7df", "G": "#f0c36b"}


def load(g, ep):
    d = pd.read_csv(INP / ("Group%s" % g) / ("res_%s.csv" % ep))
    return d[d.Dataset.astype(str).str.contains(TOK[ep], case=False)]


def metrics(d):
    err = d.Predicted - d.Actual
    ae = err.abs()
    return dict(n=len(d), MAE=ae.mean(), RMSE=np.sqrt((err ** 2).mean()), GMFE=10 ** ae.mean(),
                R2=1 - (err ** 2).sum() / ((d.Actual - d.Actual.mean()) ** 2).sum(),
                FE2=100 * (ae <= np.log10(2)).mean())


def compute():
    rows = []
    for ep in EPS:
        ref = set(load("E", ep).PUBCHEM_CID)                 # the benchmark Test Set #1 compounds
        for g in "ACDEFG":
            d = load(g, ep)
            if g == "A":
                d = d[d.PUBCHEM_CID.isin(ref)]
            assert set(d.PUBCHEM_CID) == ref, (g, ep)
            rows.append(dict(group=g, endpoint=ep, source="prediction file", **metrics(d)))
    t2 = pd.read_excel(ROOT / "Table2_data" / "Table2.xlsx", sheet_name="final_benchmarking")
    for _, r in t2[t2.Group == "B"].iterrows():
        rows.append(dict(group="B", endpoint=r.Endpoint.strip(), source="literature (reported)", n=np.nan,
                         MAE=r.MAE, RMSE=r.RMSE, GMFE=r.GMFE, R2=r.R2, FE2=r["<2-Fold %"]))
    d = pd.DataFrame(rows)
    d.to_csv(S21 / "outputs" / "tables" / "ml_table2_recomputed.csv", index=False, float_format="%.4f")
    return d


def disp(m, v):
    return "%d" % round(v) if m == "FE2" else "%.2f" % v


def best_groups(sub, m):
    vals = {g: float(disp(m, v)) for g, v in zip(sub.group, sub[m])}
    b = max(vals.values()) if m in HIGHER else min(vals.values())
    tied = [g for g, v in vals.items() if v == b]
    return ["A"] if "A" in tied else tied


def md_table(d, groups):
    head = "| ML Group | Method | Endpoint | Algorithm | Descriptor | MAE ↓ | RMSE ↓ | GMFE ↓ | R² ↑ | Within 2-fold ↑ |"
    lines = [head, "|---|---|---|---|---|---|---|---|---|---|"]
    best = {}
    for ep in EPS:
        sub = d[(d.endpoint == ep) & d.group.isin(groups)]
        for m in METRICS:
            best[(ep, m)] = best_groups(sub, m)
    for g in groups:
        for ep in EPS:
            r = d[(d.group == g) & (d.endpoint == ep)].iloc[0]
            cells = []
            for m in METRICS:
                s = disp(m, r[m]) + ("%" if m == "FE2" else "")
                cells.append("**<u>%s</u>**" % s if g in best[(ep, m)] else s)
            meth, alg, desc = CONFIG[(g, ep)]
            lines.append("| %s | %s | %s | %s | %s | %s |" % (g, meth, ep, alg, desc, " | ".join(cells)))
    return "\n".join(lines) + "\n"


def figure(d, groups, out):
    labels = {"MAE": "MAE", "RMSE": "RMSE", "GMFE": "GMFE", "R2": "R²", "FE2": "Within 2-fold (%)"}
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                                "pdf.fonttype": 42})
    fig, axes = plt.subplots(len(METRICS), len(EPS), figsize=(7.2, 7.6))
    plt.subplots_adjust(left=0.11, right=0.90, top=0.95, bottom=0.10, hspace=0.36, wspace=0.12)
    for c, ep in enumerate(EPS):
        sub = d[(d.endpoint == ep) & d.group.isin(groups)].set_index("group").loc[groups].reset_index()
        for r, m in enumerate(METRICS):
            ax = axes[r, c]
            best = best_groups(sub, m)
            vals = sub[m].to_numpy(float)
            bars = ax.bar(range(len(groups)), vals, width=0.66, color=[COLORS[g] for g in groups],
                          edgecolor="#222222", linewidth=0.6)
            for b, g in zip(bars, groups):
                b.set_alpha(0.95 if g in best else 0.30)
            rowmax = d[d.group.isin(groups)][m].max()          # one y-scale per metric row
            top = max(0.9, rowmax * 1.25) if m == "R2" else (max(80, rowmax * 1.25) if m == "FE2" else rowmax * 1.28)
            ax.set_ylim(0, top)
            for i, (v, g) in enumerate(zip(vals, groups)):
                ax.text(i, v + top * 0.02, disp(m, v), ha="center", va="bottom", fontsize=6.8,
                        color="#111111" if g in best else "#9a9a9a",
                        fontweight="bold" if g in best else "normal")
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups if r == len(METRICS) - 1 else [], fontsize=7.5)
            ax.tick_params(axis="x", length=0)
            ax.tick_params(axis="y", labelsize=6.8)
            if c:
                ax.set_yticklabels([])
            ax.grid(axis="y", alpha=0.25, lw=0.6)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(ep, fontsize=10, fontweight="bold", pad=6)
            if c == len(EPS) - 1:
                ax.text(1.04, 0.5, labels[m], transform=ax.transAxes, rotation=270, va="center",
                        fontsize=8.5, fontweight="bold")
    fig.text(0.02, 0.52, "Performance metric value", rotation=90, va="center", fontsize=9.5, fontweight="bold")
    fig.legend(handles=[Patch(facecolor=COLORS[g], edgecolor="#222222", lw=0.6, label="ML Group %s" % g) for g in groups],
               ncol=len(groups), loc="lower center", bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=400)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def main():
    d = compute()
    (V3 / "tables").mkdir(parents=True, exist_ok=True)
    (V3 / "tables" / "Table2.md").write_text(md_table(d, list("ABCD")))
    (V3 / "tables" / "TableS1.md").write_text(md_table(d, list("AEFG")))
    figure(d, list("ABCD"), V3 / "figures" / "Figure3_head_to_head.png")
    figure(d, list("AEFG"), V3 / "SI" / "figures" / "FigureS1_control_groups.png")
    print(d.round(3).to_string())


if __name__ == "__main__":
    main()
