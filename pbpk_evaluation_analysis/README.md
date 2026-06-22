# PBPK Evaluation and Mechanistic Analysis

This repository folder contains the PBPK simulation outputs, evaluation notebooks, analysis-ready tables, and figures used for the revised PBPK comparison and mechanistic interpretation.

## Purpose

The folder provides a reproducible record for the Table 3, Table S2, Figure 6, and response-letter PBPK analyses. It focuses on evaluation of different PBPK parameterization strategies and descriptive mechanistic stratification of compound-level outcomes.

## Contents

| Path | Description |
|---|---|
| `eval key_compare.ipynb` | Main PBPK evaluation notebook for the key Table 3 comparison and mechanistic response-letter figures. |
| `eval all_pbpk_supplemental.ipynb` | Supplemental PBPK evaluation notebook for the broader Table S2 scenario comparison. |
| `simulation/Table3/` | Simulation outputs for the main PBPK comparison scenarios. |
| `simulation/TableS2/` | Simulation outputs for supplemental PBPK scenarios. |
| `observed/` | Observed concentration-time and cleaned observed data used for evaluation. |
| `evaluation/Table3/` | Main PBPK evaluation outputs, summary tables, compound-level comparisons, and plots. |
| `evaluation/TableS2/` | Supplemental scenario-level evaluation outputs. |
| `pbpk_physchem_mechanistic_master.csv` | Analysis-ready master table for compound-level mechanistic stratification. |
| `README_master_data.md` | Source and column-provenance notes for the mechanistic master table. |

## Main Evaluation Workflow

Run the notebooks from this folder so that relative paths resolve correctly.

```bash
jupyter notebook "eval key_compare.ipynb"
jupyter notebook "eval all_pbpk_supplemental.ipynb"
```

The notebooks read simulation files from `simulation/`, observed data from `observed/`, and write evaluation outputs to `evaluation/`.

## Key Outputs

Main Table 3 evaluation:

- `evaluation/Table3/overall/evaluation_all_raw.csv`
- `evaluation/Table3/overall/evaluation_summary_by_scenario.csv`
- `evaluation/Table3/overall/scenario_summary_key_ranking.csv`
- `evaluation/Table3/overall/scenario_summary_key_rel.csv`

Supplemental Table S2 evaluation:

- `evaluation/TableS2/overall/evaluation_all_raw.csv`
- `evaluation/TableS2/overall/evaluation_summary_by_scenario.csv`
- `evaluation/TableS2/overall/scenario_summary_key_ranking.csv`
- `evaluation/TableS2/overall/scenario_summary_key_rel.csv`

Mechanistic analysis and response-letter figures:

- `evaluation/Table3/plot/mechanistic/`
- `evaluation/Table3/mechanistic_summary/`

## Master Data

`pbpk_physchem_mechanistic_master.csv` is provided as Data S3 / analysis-ready data for the mechanistic stratification. It combines compound identifiers, empirical physicochemical and ADME descriptors, ECCS class, clearance annotations, and PBPK error metrics. See `README_master_data.md` for the column-source summary.

## Notes

The simulation folders include exported concentration-time profiles and selected ADME input summaries required to audit the PBPK evaluation. Proprietary Simcyp project files are not included, but the provided exported simulation results and evaluation notebooks reproduce the reported numerical analyses.
