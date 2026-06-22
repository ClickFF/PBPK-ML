# PBPK Mechanistic Master Table

`pbpk_physchem_mechanistic_master.csv` is provided as an analysis-ready table for the compound-level mechanistic stratification and response-letter figures. It is intended to be used as Data S3 in the revision package.

The table contains one row per PBPK evaluation compound. It combines compound identifiers, empirical physicochemical and ADME descriptors, mechanistic annotations, and PBPK evaluation metrics derived from the simulation outputs in this repository.

## Column Source Summary

| Column group | Representative columns | Source and processing |
|---|---|---|
| Compound identifiers | `PUBCHEM_CID`, `ID_trend_pbpk`, `Name_final`, `Drug_Name` | Harmonized compound identifiers used to link PBPK simulation results, observed data, and external compound-level annotations. `PUBCHEM_CID` is the primary compound identifier used for downstream analysis. |
| PBPK outcome labels | `class`, `class_control`, `scenario_A`, `scenario_B`, `scenario_control_A`, `scenario_control_B` | Derived from the PBPK evaluation workflow in `eval key_compare.ipynb` using the simulation results under `simulation/Table3/` and observed concentration-time/PK data under `observed/`. |
| ECCS annotation | `ECCS_Class`, `ECCS_new` | ECCS class was obtained from the available S+ prediction output and simplified into `ECCS_new` as `Class1`, `Class2`, `Class3`, or `Class4`. The simplified class is used only for descriptive mechanistic stratification. |
| Mechanistic strata | `clearance_group`, `ionization_group`, `logP_bin`, `Vd_bin` | Descriptive groupings used for mechanism-oriented interpretation. `clearance_group` summarizes the dominant clearance mechanism where available. `ionization_group` is based on compound ionization type. `logP_bin` and `Vd_bin` are binned physicochemical/distribution descriptors used for visualization and descriptive summaries. |
| Empirical physicochemical and ADME inputs | `MW_s2`, `LogP_s2`, `pKa1_key`, `pKa2_key`, `Fu_obs_s2`, `VD_obs_s2`, `CLsys_obs_s2`, `CLRbase_s2`, `CL_MetBas_s2` | Curated from the mapped Supporting Information/S2-style compound data and Simcyp/S+ ADME input sources available during model construction. Observed values are preferred where available for mechanism-oriented interpretation. |
| Predicted ADME fields | `Fu_pred_s2`, `VD_pred_s2`, `CLsys_pred_s2` | S+/ADME predicted values retained for traceability. These fields were not used as the primary basis for the mechanistic grouping when observed/empirical inputs were available. |
| Simcyp-related input flags | `Pcnt3AMetCL`, `ActiveUptakeHep`, `BiliaryClearanceType`, `EnteredPeff`, `MDCKValue`, `PAMPAValue`, `KpPredictionMethod` | Model-input descriptors or flags exported from the available Simcyp/S+ ADME input tables. These are retained for context and auditability. |
| Literature clearance annotation | `clearance_ref_*` | Literature-derived or curated clearance mechanism fields used to support clearance-mechanism interpretation. These fields are descriptive and are not used to train or tune ML models. |
| PBPK error metrics | `abs_log2_A`, `abs_log2_B`, `fe_cmax_abs_A`, `fe_cmax_abs_B`, `fe_auc_abs_A`, `fe_auc_abs_B`, `time_within_2fold_A`, `time_within_2fold_B` | Derived from `evaluation/Table3/overall/evaluation_all_raw.csv`. Scenario A corresponds to the v2_CLint strategy and Scenario B corresponds to the v3_CLsys strategy in the revised comparison. |
| Delta/benefit metrics | `delta_score_conc_v3_minus_v2`, `delta_score_auc_v3_minus_v2`, `delta_score_cmax_v3_minus_v2`, `delta_time_within_2fold_v3_minus_v2` | Scenario B minus Scenario A differences. Because lower error is better for concentration, AUC, and Cmax error metrics, the notebook defines benefit scores as the negative of these delta-score columns when plotting v3_CLsys benefit. |

## Intended Use

This master table supports the mechanistic figures and descriptive stratified summaries in the revised manuscript and response letter. It should be treated as an analysis-ready, compound-level table rather than a raw-data assembly script.

The reproducible PBPK evaluation itself is provided in:

- `eval key_compare.ipynb` for the main Table 3/Figure 6 comparisons.
- `eval all_pbpk_supplemental.ipynb` for the supplemental Table S2 scenario evaluation.

The mechanistic plots generated from this master table are saved under:

- `evaluation/Table3/plot/mechanistic/`

The corresponding descriptive summary tables are saved under:

- `evaluation/Table3/mechanistic_summary/`
