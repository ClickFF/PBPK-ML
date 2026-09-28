# -*- coding: utf-8 -*-
"""Arm registry for claude_new_Sep21 — design carried over from response_letter/lib/arms.py.

Two comparison groups carry the primary analysis:

  A (control series)   clearance held at its observed value; the model structure and the
                       non-clearance inputs are varied. Reference A2 = v1_run0.
  B (clearance source) the other inputs held at the S+ setting of s1_run3; only the source of
                       clearance varies. Reference B1 = s1_run3 (observed clearance).
                       h2_run0 additionally replaces Fu and VDss with DL-ML predictions and is
                       the all-predicted application case.

Changes against the Sep20 registry:
  * INVALID is empty. v0_run0 x phenobarbital was excluded because the library workspace was
    dosed 100 mg oral; the re-simulated v0 doses 1.857 mg/kg IV with CL_IV = 0.31 L/h from the
    file's own oral clearance, so the cell is valid.
  * The primary concentration-time statistic is log-NRMSE, with RMSE (log10) as its companion.
    log2 fold errors are used for AUC and Cmax only.
"""

# (run_id, new_id, label, group)
ARMS = [
    ("v0_run0",       "A1", "PBPK_v0 Simcyp baseline",   "A"),
    ("v1_run0",       "A2", "PBPK_v1 observed inputs",   "A"),
    ("s1_run1",       "A4", "S+ physchem",               "A"),
    ("s1_run2",       "A5", "+ S+ Fu",                   "A"),
    ("s1_run3",       "A6", "+ S+ VDss",                 "A"),

    ("s1_run3",       "B1", "Observed CLsys (S+ Fu, VDss)", "B"),
    ("h1_run0",       "B2", "PBPK_v2 CLint",             "B"),
    ("h1_run0_noCLr", "B3", "PBPK_v2 CLint, CLR = 0",    "B"),
    ("h2_run4",       "B4", "PBPK_v3 CLsys",             "B"),
    ("h2_run0",       "B5", "PBPK_v4 ML",                "B"),

    ("h1_run1",       "S1", "Bottom-up (full) Kp 0",     "SI"),
    ("h1_run2",       "S2", "Bottom-up (full) Kp 1",     "SI"),
    ("h1_run3",       "S3", "Bottom-up (full) Kp 2",     "SI"),
    ("h1_run1_noCLr", "S4", "Bottom-up (full) Kp 0, CLR = 0", "SI"),
    ("h1_run2_noCLr", "S5", "Bottom-up (full) Kp 1, CLR = 0", "SI"),
    ("h1_run3_noCLr", "S6", "Bottom-up (full) Kp 2, CLR = 0", "SI"),
    ("h2_run1",       "S7", "Top-down (full) Kp 0",      "SI"),
    ("h2_run2",       "S8", "Top-down (full) Kp 1",      "SI"),
    ("h2_run3",       "S9", "Top-down (full) Kp 2",      "SI"),
]

INVALID = {}                         # none in the corrected batch
REFERENCE = {"A": "A2", "B": "B1"}

# endpoint registry: column, display label, family, whether lower is better
ENDPOINTS = [
    ("log_NRMSE",   "C–T profile, log-NRMSE",       "profile", True),
    ("rmse_log10",  "C–T profile, RMSE (log$_{10}$)", "profile", True),
    ("fe_auc_abs",  "AUC, |log$_2$ FE|",             "exposure", True),
    ("fe_cmax_abs", "C$_{max}$, |log$_2$ FE|",       "exposure", True),
]


def group(name):
    return [(r, n, l) for r, n, l, g in ARMS if g == name]


def run_of(new_id):
    return next((r for r, n, _, _ in ARMS if n == new_id), None)


def label_of(new_id):
    return next((l for _, n, l, _ in ARMS if n == new_id), new_id)


def runs():
    return list(dict.fromkeys(r for r, _, _, _ in ARMS))
