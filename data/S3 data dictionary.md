# `Data S3 compound-level analysis dataset.csv` — data dictionary and provenance

Replaces `pbpk_physchem_mechanistic_master.csv` (v7.2 Data S3; archived unchanged in the code repository under
`superseded_v7.2/pbpk_evaluation_analysis/`). One row per PBPK compound (41), ordered by name, built by
`si_analysis/analysis/12_mechanistic_strata.py` in the code repository. Every column has exactly one source; no column is copied
from a merged `_s2` field.

## Problems found in the v7.2 table (why it was rebuilt)

| Issue | Evidence | Consequence in v7.2 |
|---|---|---|
| `CLRbase_s2` row-misaligned | 40 of 41 values differ from the template renal CL (e.g. bufuralol 27.2 = pravastatin's value) | column unusable; the clearance groups were derived from the correct `CLRbase`, so they reproduce exactly |
| `LogP_s2`, `MW_s2`, `Fu_s2`, `VD_s2` misaligned for dabigatran and imipramine | dabigatran MW 627.8 (true 471.5), imipramine MW 358.8 (= etoricoxib) | two rows of those fields wrong |
| unsuffixed `Fu`, `VD` are template values, not observed | differ from the observed inputs entered in v1_run0 in 30 / 38 compounds (e.g. docetaxel Fu 1.0 vs 0.04) | `Vd_bin` binned the template VD: 7 compounds in the wrong bin (e.g. lansoprazole template 3.0 vs observed 0.28 L/kg) |
| `ionization_group` rule ignored acid/base type | weak bases labelled "Acidic" (alprazolam, voriconazole, itraconazole), acids labelled "Basic" (phenobarbital, phenytoin, zidovudine), neutrals labelled "Acidic" (digoxin, docetaxel, nifedipine) | ionization strata not interpretable |
| PBPK error columns from the published arms | `abs_log2_*`, `delta_score_*`, `class` computed for the mis-specified v2_CLint arm and with the withdrawn mean \|log₂\| profile statistic | all outcome columns stale |
| redundant columns | `LogP` / `LogP_key` / `LogP_s2`, `VD` / `VD_key` / `VD_s2` / `VD_obs_s2`, ... without source labels | ambiguous provenance |

## Columns

| Group | Columns | Source |
|---|---|---|
| Identifiers | `compound_id`, `PUBCHEM_CID`, `compound` | v7.2 master (verified identifiers) |
| Observed inputs | `Fu_obs`, `VDss_obs_L_per_kg`, `CL_obs_L_per_h_per_kg` | values entered in v1_run0 (`pbpk/applied/v1_run0/adme_key_inputs.csv`; CL = CL_MetBas / 70) |
| Simcyp template | `logP_template`, `BP_template`, `CLRbase_template_Lh`, `KpScalar_template`, `template_distribution`, `template_elimination`, `CLint_H_extra`, `Pcnt3AMetCL`, `ActiveUptakeHep`, `BiliaryClearanceType` | template read-back (Table R7b; `design_from_template.csv`); the four flags from the v7.2 master, whose row alignment was verified through `CLRbase` (41/41 match) |
| S+ (ADMET Predictor v11) | `MW_Splus`, `logP_Splus`, `pKa1_Splus`, `pKa2_Splus`, `compound_type_code_Splus`, `ECCS_class_raw_Splus`, `ECCS_class` | values entered in s1_run1 (`adme_key_inputs.csv`); ECCS from the S+ output, subclasses collapsed to Class 1–4 (docetaxel, phenytoin: not assigned) |
| Literature clearance annotation | `clearance_ref_*` | unchanged from v7.2 (descriptive) |
| Strata | `clearance_group` | v7.2 Methods rule on template inputs: Uptake/Biliary if ActiveUptakeHep > 1 or BiliaryClearanceType = 1; else Metabolism-leaning if CLint_H_extra > 0 or Pcnt3AMetCL ≥ 50; else Renal-leaning if CLRbase ≥ 1.0 L/h with weak hepatic input; else Mixed/Other |
| | `ionization_pH74` | S+ compound-type code (mapping inferred from unambiguous reference drugs: 2 monoprotic acid, 3 monoprotic base, 1 diprotic base, 5 ampholyte, 0/4 neutral or other) with S+ pKa: acid ionized if pKa ≤ 7.4, base ionized if highest pKa ≥ 7.4, otherwise "Neutral at pH 7.4" |
| | `logP_bin` (<2, 2–4, >4) | template logP |
| | `VDss_bin` (<0.7, 0.7–2, >2 L/kg) | observed VDss |
| PBPK errors | `<metric>__<scenario>` for log_NRMSE, rmse_log10, fe_auc_abs, fe_cmax_abs, fe_auc_signed, fe_cmax_signed; scenarios v0_run0, v1_run0, s1_run3, h1_run0, h1_run0_noCLr, h2_run4, h2_run0 | `si_analysis/outputs/tables/per_compound_metrics.csv` (corrected batch) |
| Strategy contrast | `delta_<metric>__BU0_minus_TD` (h1_run0_noCLr − h2_run4; information-matched, primary for SI Section S7), `delta_<metric>__BU_minus_TD` (h1_run0 − h2_run4; template-assisted sensitivity) | test − reference: negative = predicted-CLint workflow smaller error, positive = predicted-total-CL workflow smaller error |
| Cost of predicted CL | `cost_<metric>__predCL_minus_obsCL` | median over h1_run0, h1_run0_noCLr, h2_run4, h2_run0 minus s1_run3 |
| Outcome category | `outcome_BU0_vs_TD` | descriptive, h1_run0_noCLr vs h2_run4; acceptable = AUC and Cmax both within 2-fold (v7.2 definition) |

Known limitation: the S+ compound-type code book was not available; the mapping above is inferred
and three multiprotic acids coded 0 (methotrexate, tenofovir, valsartan) fall into
"Neutral at pH 7.4". The ionization strata are therefore descriptive only.

## Complete column list

All 92 columns, expanded from the patterns above so every literal name appears.

| # | Column | Group |
|---|---|---|
| 1 | `compound_id` | Identifiers, inputs, template, S+ and strata |
| 2 | `PUBCHEM_CID` | Identifiers, inputs, template, S+ and strata |
| 3 | `compound` | Identifiers, inputs, template, S+ and strata |
| 4 | `Fu_obs` | Identifiers, inputs, template, S+ and strata |
| 5 | `VDss_obs_L_per_kg` | Identifiers, inputs, template, S+ and strata |
| 6 | `CL_obs_L_per_h_per_kg` | Identifiers, inputs, template, S+ and strata |
| 7 | `logP_template` | Identifiers, inputs, template, S+ and strata |
| 8 | `BP_template` | Identifiers, inputs, template, S+ and strata |
| 9 | `CLRbase_template_Lh` | Identifiers, inputs, template, S+ and strata |
| 10 | `KpScalar_template` | Identifiers, inputs, template, S+ and strata |
| 11 | `template_distribution` | Identifiers, inputs, template, S+ and strata |
| 12 | `template_elimination` | Identifiers, inputs, template, S+ and strata |
| 13 | `CLint_H_extra` | Identifiers, inputs, template, S+ and strata |
| 14 | `Pcnt3AMetCL` | Identifiers, inputs, template, S+ and strata |
| 15 | `ActiveUptakeHep` | Identifiers, inputs, template, S+ and strata |
| 16 | `BiliaryClearanceType` | Identifiers, inputs, template, S+ and strata |
| 17 | `MW_Splus` | Identifiers, inputs, template, S+ and strata |
| 18 | `logP_Splus` | Identifiers, inputs, template, S+ and strata |
| 19 | `pKa1_Splus` | Identifiers, inputs, template, S+ and strata |
| 20 | `pKa2_Splus` | Identifiers, inputs, template, S+ and strata |
| 21 | `compound_type_code_Splus` | Identifiers, inputs, template, S+ and strata |
| 22 | `ECCS_class_raw_Splus` | Identifiers, inputs, template, S+ and strata |
| 23 | `ECCS_class` | Identifiers, inputs, template, S+ and strata |
| 24 | `clearance_ref_cas` | Literature clearance annotation |
| 25 | `clearance_ref_in_vivo_clintu_ml_min_kg` | Literature clearance annotation |
| 26 | `clearance_ref_hepatocyte_clintu_ml_min_kg` | Literature clearance annotation |
| 27 | `clearance_ref_hlm_clintu_ml_min_kg` | Literature clearance annotation |
| 28 | `clearance_ref_human_cl_ml_min_kg` | Literature clearance annotation |
| 29 | `clearance_ref_primary_mechanism` | Literature clearance annotation |
| 30 | `clearance_ref_mechanism_code` | Literature clearance annotation |
| 31 | `clearance_ref_mechanism_pct_ref` | Literature clearance annotation |
| 32 | `clearance_ref_mechanism_comments` | Literature clearance annotation |
| 33 | `clearance_ref_mechanism_notes` | Literature clearance annotation |
| 34 | `clearance_group` | Identifiers, inputs, template, S+ and strata |
| 35 | `ionization_pH74` | Identifiers, inputs, template, S+ and strata |
| 36 | `logP_bin` | Identifiers, inputs, template, S+ and strata |
| 37 | `VDss_bin` | Identifiers, inputs, template, S+ and strata |
| 38 | `log_NRMSE__v0_run0` | PBPK errors |
| 39 | `rmse_log10__v0_run0` | PBPK errors |
| 40 | `fe_auc_abs__v0_run0` | PBPK errors |
| 41 | `fe_cmax_abs__v0_run0` | PBPK errors |
| 42 | `fe_auc_signed__v0_run0` | PBPK errors |
| 43 | `fe_cmax_signed__v0_run0` | PBPK errors |
| 44 | `log_NRMSE__v1_run0` | PBPK errors |
| 45 | `rmse_log10__v1_run0` | PBPK errors |
| 46 | `fe_auc_abs__v1_run0` | PBPK errors |
| 47 | `fe_cmax_abs__v1_run0` | PBPK errors |
| 48 | `fe_auc_signed__v1_run0` | PBPK errors |
| 49 | `fe_cmax_signed__v1_run0` | PBPK errors |
| 50 | `log_NRMSE__s1_run3` | PBPK errors |
| 51 | `rmse_log10__s1_run3` | PBPK errors |
| 52 | `fe_auc_abs__s1_run3` | PBPK errors |
| 53 | `fe_cmax_abs__s1_run3` | PBPK errors |
| 54 | `fe_auc_signed__s1_run3` | PBPK errors |
| 55 | `fe_cmax_signed__s1_run3` | PBPK errors |
| 56 | `log_NRMSE__h1_run0` | PBPK errors |
| 57 | `rmse_log10__h1_run0` | PBPK errors |
| 58 | `fe_auc_abs__h1_run0` | PBPK errors |
| 59 | `fe_cmax_abs__h1_run0` | PBPK errors |
| 60 | `fe_auc_signed__h1_run0` | PBPK errors |
| 61 | `fe_cmax_signed__h1_run0` | PBPK errors |
| 62 | `log_NRMSE__h1_run0_noCLr` | PBPK errors |
| 63 | `rmse_log10__h1_run0_noCLr` | PBPK errors |
| 64 | `fe_auc_abs__h1_run0_noCLr` | PBPK errors |
| 65 | `fe_cmax_abs__h1_run0_noCLr` | PBPK errors |
| 66 | `fe_auc_signed__h1_run0_noCLr` | PBPK errors |
| 67 | `fe_cmax_signed__h1_run0_noCLr` | PBPK errors |
| 68 | `log_NRMSE__h2_run4` | PBPK errors |
| 69 | `rmse_log10__h2_run4` | PBPK errors |
| 70 | `fe_auc_abs__h2_run4` | PBPK errors |
| 71 | `fe_cmax_abs__h2_run4` | PBPK errors |
| 72 | `fe_auc_signed__h2_run4` | PBPK errors |
| 73 | `fe_cmax_signed__h2_run4` | PBPK errors |
| 74 | `log_NRMSE__h2_run0` | PBPK errors |
| 75 | `rmse_log10__h2_run0` | PBPK errors |
| 76 | `fe_auc_abs__h2_run0` | PBPK errors |
| 77 | `fe_cmax_abs__h2_run0` | PBPK errors |
| 78 | `fe_auc_signed__h2_run0` | PBPK errors |
| 79 | `fe_cmax_signed__h2_run0` | PBPK errors |
| 80 | `delta_log_NRMSE__BU_minus_TD` | Strategy contrast |
| 81 | `delta_log_NRMSE__BU0_minus_TD` | Strategy contrast |
| 82 | `cost_log_NRMSE__predCL_minus_obsCL` | Cost of predicted CL |
| 83 | `delta_rmse_log10__BU_minus_TD` | Strategy contrast |
| 84 | `delta_rmse_log10__BU0_minus_TD` | Strategy contrast |
| 85 | `cost_rmse_log10__predCL_minus_obsCL` | Cost of predicted CL |
| 86 | `delta_fe_auc_abs__BU_minus_TD` | Strategy contrast |
| 87 | `delta_fe_auc_abs__BU0_minus_TD` | Strategy contrast |
| 88 | `cost_fe_auc_abs__predCL_minus_obsCL` | Cost of predicted CL |
| 89 | `delta_fe_cmax_abs__BU_minus_TD` | Strategy contrast |
| 90 | `delta_fe_cmax_abs__BU0_minus_TD` | Strategy contrast |
| 91 | `cost_fe_cmax_abs__predCL_minus_obsCL` | Cost of predicted CL |
| 92 | `outcome_BU0_vs_TD` | Identifiers, inputs, template, S+ and strata |
