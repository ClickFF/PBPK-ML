# Table 2 Benchmarking Statistical Analysis

This repository folder contains the matched compound-level statistical analysis used to support the revised Table 2 model comparison.

## Purpose

The analysis compares the primary DL-ML model, Group A, against available control groups using matched test-set compounds. It was prepared to address the reviewer request for statistical confidence intervals and paired testing of the reported benchmarking differences.

## Contents

| Folder | Description |
|---|---|
| `inputs/` | Normalized prediction files used for the paired comparisons. Each file contains `PUBCHEM_CID`, `Actual`, `Predicted`, and `Dataset`. Values are on the log10 scale used for scoring. |
| `scripts/` | Reproducible Python scripts for pairwise statistics and bootstrap confidence interval plots. |
| `outputs/pairwise/` | Matched compound-level tables and per-comparison statistical summaries. |
| `outputs/summary/` | Combined summary tables across all included controls. |
| `outputs/figures/` | Bootstrap confidence interval figures for CL and VDss. |

## Included Comparisons

Group A is compared with Groups C, D, E, F, and G where matched prediction results are available.

- Group A: DL-ML model using graph-derived embeddings plus descriptor information.
- Group C/D: descriptor-based control comparisons with available CL and VDss outputs.
- Group E: DL-only control.
- Group F: RDKit-only control.
- Group G: embeddings-only control.

Group B is not included because fully matched compound-level prediction outputs were not available from the original benchmarking source.

## Statistical Analysis

Compounds are matched by `PUBCHEM_CID` within each endpoint-specific test set. The scripts report MAE, RMSE, MSE, GMFE, fraction within 2-fold error, R2, bootstrap 95% confidence intervals, Wilcoxon signed-rank tests for paired absolute errors, and McNemar/binomial exact tests for paired within-2-fold classification.

Delta values are reported as `Group A - comparator`. For error metrics, negative values favor Group A. For R2 and FE<2, positive values favor Group A.

## Reproduce

Run from this folder:

```bash
python scripts/run_pairwise_stats.py
python scripts/build_summary_and_plots.py
```

The scripts regenerate the files under `outputs/`.

Python requirements: `numpy`, `pandas`, and `matplotlib`. `scipy` is optional; fallback implementations are included for the paired statistical tests if SciPy is unavailable.

## Key Outputs

- `outputs/summary/table2_groupA_vs_controls_stats_testset.csv`
- `outputs/summary/table2_groupA_vs_controls_stats_cl_vdss.csv`
- `outputs/figures/figure_bootstrap_ci_cl_vdss_combined_metrics.png`
- `outputs/figures/figure_bootstrap_ci_cl_vdss_combined_metrics.pdf`
