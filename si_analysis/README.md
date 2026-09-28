# Analysis pipeline

Produces the Supporting Information tables and figures and the main-text figures. Pure Python —
`pandas`, `numpy`, `matplotlib`, `scipy` — pinned in `environment/requirements-analysis.txt`.

The numbered scripts are the pipeline and run in order; `analysis/v4/`, `analysis/v7.4/` and
`analysis/v7.4_final01/` hold the figure builders.

```bash
pip install -r environment/requirements-analysis.txt
python si_analysis/analysis/01_evaluate.py      # then 02 .. 05
python si_analysis/analysis/06_ml_benchmark_stats.py
python si_analysis/analysis/09_si_tables.py
```

`lib/` holds the shared pieces: `metrics.py` (including log-NRMSE and the prespecified 0.2 log10
range rule that excludes phenobarbital), `arms.py` (scenario names and groupings), `paths.py`, and
`style_v74b.py` (the figure style used for the submitted set).

## Main-text figures

| Figure | Built by |
|---|---|
| 1 workflow | `analysis/v4/fig01_workflow.py` |
| 2 train / held-out performance | `analysis/v7.4/fig02_03_ml_main.py`, `analysis/v7.4_final01/fig02_train_test_3x3.py` |
| 3 predicted PBPK inputs | `analysis/v4/fig04_upstream_gof.py` |
| 4 error distributions across minimal scenarios | `analysis/v7.4/fig04_pbpk_scenario_overview.py` |
| 5 matched clearance contrast | `analysis/v7.4/fig05_matched_clearance.py` |
| 6 parameter → exposure | `analysis/v7.4_final01/fig06_parameter_to_exposure.py` |
| **7 all-predicted end-to-end** | `analysis/v7.4/fig06_allpredicted_overview.py` |

Figure 7 is the one to start from if you are following the ML–PBPK pipeline end to end; the
simulations behind it are in `pbpk/`. It was numbered Figure 8 in earlier drafts.

## Supporting Information figures

| SI | Built by |
|---|---|
| S1 | `analysis/v4/figS01_ml_descriptive.py` |
| S2, S3 | `analysis/10_ml_tables_figures.py` |
| S4 | `analysis/06_ml_benchmark_stats.py` |
| S5–S9 | `analysis/08_si_ct_profiles.py` with `analysis/v4/figS04_08_ct_grids.py` (one arm each: `v0_run0`, `v1_run0`, `h1_run0`, `h2_run4`, `h2_run0`) |
| S10 | `analysis/v4/figS09_compound_heterogeneity.py` |
| S11, S12 | `analysis/v4/figS10_S11_heatmaps.py` |
| S13 | `analysis/v4/fig05_error_distributions.py` |
| S14 | `analysis/v4/fig07_substitution_association.py` |
| S15 | `analysis/v4/figS13_cmax_structure.py` |
| S16 | `analysis/v4/figS14_cmax_readout.py` |
| S17 | `analysis/v4/figS15_applicability_domain.py` |
| S18, S19, S20 | `analysis/12_mechanistic_strata.py`, `analysis/v4/figS17_19_strata.py` |

## Supporting Information tables

Nine of the thirteen are generated; the rest are authored in the SI source.

| SI | Source |
|---|---|
| S1, S3, S5, S6 | authored in the SI document |
| S2 | `analysis/09_si_tables.py` from `outputs/tables/ml_benchmark_paired_stats_holm.csv` |
| S4 encoder exposure | `analysis/09_si_tables.py` — the audit behind the qualification in the root README |
| S7 41-compound master | `analysis/09_si_tables.py` |
| S8 paired PBPK contrasts | `analysis/09_si_tables.py` from `analysis/02_paired.py` |
| S9 C–T statistic sensitivity | `analysis/09_si_tables.py` |
| S10 applicability domain | `analysis/09_si_tables.py` |
| S11 clearance hierarchy | `analysis/13_clearance_hierarchy.py` |
| S12, S13 | `analysis/12_mechanistic_strata.py` |

`outputs/tables/` holds the generated tables; `outputs/draft02_MANIFEST.csv` records the sha256 and
build provenance of every display item in the submitted set.
