# -*- coding: utf-8 -*-
"""Round 5 draft_02: the SI figure remap, shared by every draft_02 builder.

Changes requested by the author:
  * SI Figures S15 + S16 are merged into a new main-text Figure 6; the old main-text Figure 6 becomes Figure 7
  * SI Figure S24 moves into Section S4 (PBPK compound-level results); Section S8 is dropped
  * SI Figures S2, S20 and S25 are deleted as duplicates of main-text displays
  * the remaining 20 SI figures are renumbered S1-S20 in document order
"""
from __future__ import annotations

# old Round 4C stem -> (new SI number, new stem) in the new document order.
# None means the figure leaves the SI; the note says where it went.
SI_ORDER = [
    # Section S1, ML benchmarking
    ("FigureS01_ml_descriptive", 1, "FigureS01_ml_descriptive"),
    ("FigureS03_ml_paired_A_vs_CD", 2, "FigureS02_ml_paired_A_vs_CD"),
    ("FigureS04_ml_paired_A_vs_EFG", 3, "FigureS03_ml_paired_A_vs_EFG"),
    ("FigureS05_ml_paired_forest", 4, "FigureS04_ml_paired_forest"),
    # Section S4, PBPK compound-level results
    ("FigureS06_ct_library_model", 5, "FigureS05_ct_library_model"),
    ("FigureS07_ct_observed_input", 6, "FigureS06_ct_observed_input"),
    ("FigureS08_ct_clint_template_renal", 7, "FigureS07_ct_clint_template_renal"),
    ("FigureS09_ct_dlml_clsys", 8, "FigureS08_ct_dlml_clsys"),
    ("FigureS10_ct_all_predicted", 9, "FigureS09_ct_all_predicted"),
    ("FigureS11_compound_heterogeneity", 10, "FigureS10_compound_heterogeneity"),
    ("FigureS12_heatmap_logNRMSE", 11, "FigureS11_heatmap_logNRMSE"),
    ("FigureS13_heatmap_RMSE_log10", 12, "FigureS12_heatmap_RMSE_log10"),
    ("FigureS14_error_distributions_all_scenarios", 13, "FigureS13_error_distributions_all_scenarios"),
    ("FigureS24_alt_input_substitution_branches", 14, "FigureS14_alt_input_substitution_branches"),
    ("FigureS17_cmax_structure", 15, "FigureS15_cmax_structure"),
    ("FigureS18_cmax_readout", 16, "FigureS16_cmax_readout"),
    # Section S5, applicability domain
    ("FigureS19_applicability_domain", 17, "FigureS17_applicability_domain"),
    # Section S6 keeps Table S11 and no figure
    # Section S7, exploratory mechanistic stratification
    ("FigureS21_ECCS_matched_delta", 18, "FigureS18_ECCS_matched_delta"),
    ("FigureS22_strata_matched_delta_AUC", 19, "FigureS19_strata_matched_delta_AUC"),
    ("FigureS23_ECCS_cost_of_predicted_CLsys", 20, "FigureS20_ECCS_cost_of_predicted_CLsys"),
]

REMOVED = {
    2: ("FigureS02_ml_head_to_head", "deleted; duplicates Table 2 of the main text"),
    15: ("FigureS15_parameter_to_exposure_scatter", "merged into main-text Figure 6"),
    16: ("FigureS16_parameter_to_exposure_heatmap", "merged into main-text Figure 6"),
    20: ("FigureS20_clearance_hierarchy", "deleted; duplicates main-text Figure 5"),
    25: ("FigureS25_allpredicted_extended", "deleted; duplicates main-text Figure 7"),
}

OLD_NUM = {stem: int(stem[7:9]) for stem, _, _ in SI_ORDER}
NEW_NUM = {old: new for old, (_, new) in zip(
    [int(s[7:9]) for s, _, _ in SI_ORDER], [(s, n) for s, n, _ in SI_ORDER])}
NEW_NUM = {int(stem[7:9]): new for stem, new, _ in SI_ORDER}
NEW_STEM = {stem: new_stem for stem, _, new_stem in SI_ORDER}

# Main-text figures: the new Figure 6 is inserted, so the old Figure 6 becomes Figure 7.
MAIN_RENAME = {"Figure6_allpredicted_overview": ("Figure7_allpredicted_overview", 7)}
MAIN_KEEP = ["Figure1_workflow", "Figure2_ml_train_test", "Figure3_upstream_inputs",
             "Figure4_pbpk_scenario_overview", "Figure5_matched_clearance"]
NEW_MAIN = ("Figure6_parameter_to_exposure", 6)
