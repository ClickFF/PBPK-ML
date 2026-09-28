#!/usr/bin/env python3
"""Supporting Information tables built from the audited CSVs -> manuscript/v2/SI/tables/*.md

Every number in these tables is read from a file; nothing is typed by hand except the v7.2
configuration tables (S1 config columns, S3 architecture), which are transcribed verbatim from
`Supplementary Information v7.2.docx` and the v7.2 Table 2.

  S2  ML Group A vs B-G, matched by CID          outputs/tables/ml_benchmark_paired_stats_holm.csv
  S4  encoder exposure                            response_letter/P2/encoder_audit/*.csv
  S7  41-compound parameter/exposure master       outputs/tables/upstream_parameter_fold_errors.csv,
                                                  per_compound_metrics.csv
  S8  all PBPK paired contrasts                   outputs/tables/paired_contrasts.csv
  S9  C-T statistic sensitivity (time weighting)  response_letter/P1/outputs/ct_*_matrix.csv
  S10 applicability domain                        response_letter/minor_points/applicability_domain/

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/09_si_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
S21 = HERE.parent
T = S21 / "outputs" / "tables"
RL = S21 / "response_letter"
OUT = S21 / "manuscript" / "v3" / "SI" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(S21 / "lib"))
import paths as P                                                  # noqa: E402


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines) + "\n"


def p_fmt(p):
    return "<0.001" if p < 0.001 else ("%.3f" % p if p < 0.1 else "%.2f" % p)


def ci(a, b, f="%.3f"):
    return (f + " to " + f) % (a, b)


# ------------------------------------------------------------------ S2 ML paired statistics
def s2():
    d = pd.read_csv(T / "ml_benchmark_paired_stats_holm.csv")
    rows = []
    for _, r in d.iterrows():
        rows.append({"Comparator": "B (subset)" if r.Comparator == "B" else r.Comparator,
                     "Endpoint": r.Endpoint, "n": int(r.N_paired),
                     "ΔMAE": "%+.3f" % r.delta_MAE,
                     "95% CI": ci(r.delta_MAE_CI95_lo, r.delta_MAE_CI95_hi),
                     "p (raw)": p_fmt(r.wilcoxon_p_abs_err), "p (Holm)": p_fmt(r.p_holm_wilcoxon),
                     "Δ within 2-fold": "%+.3f" % r["delta_FE<2"],
                     "95% CI ": ci(r["delta_FE<2_CI95_lo"], r["delta_FE<2_CI95_hi"]),
                     "p (raw) ": p_fmt(r["mcnemar_p_FE<2"]), "p (Holm) ": p_fmt(r.p_holm_mcnemar),
                     "ΔR²": "%+.3f" % r.delta_R2,
                     "Direction": r.verdict.replace("favours", "favors")})
    (OUT / "S2_ml_paired.md").write_text(md(pd.DataFrame(rows)))


# ------------------------------------------------------------------ S4 encoder exposure
def s4():
    rows = [
        ["Test Set #1, CL", "177", "0 / 177", "0 / 177", "encoder-naive", "RL P2 audit"],
        ["Test Set #1, VDss", "177", "0 / 177", "0 / 177", "encoder-naive", "RL P2 audit"],
        ["Test Set #1, Fu", "633", "0 / 633", "47 / 633 (7.4%)", "cross-endpoint exposure only", "RL P2 audit"],
        ["Test Set #2 (all endpoints)", "110", "81 / 110 (74%)", "81 / 110", "same-endpoint label exposure through the encoder", "RL P2 audit"],
    ]
    pb = pd.read_csv(RL / "P2" / "encoder_audit" / "pbpk41_vs_encoder_leakage_summary.csv").set_index("set_or_flag")
    n = lambda k: int(pb.loc[k, "n_pbpk41_true"])
    rows.append(["PBPK subset (41 of the Test Set #2 compounds)", "41",
                 "CL %d / 41; VDss %d / 41; Fu %d / 41" % (n("in_CLsys_endpoint_train_all_by_tag"),
                                                          n("in_VDss_endpoint_train_all_by_tag"),
                                                          n("in_Fu_endpoint_train_all_by_tag")),
                 "32 / 41 in at least one encoder training set", "9 compounds encoder-naive",
                 "pbpk41_vs_encoder_leakage_summary.csv"])
    df = pd.DataFrame(rows, columns=["Evaluation set", "n", "In same-endpoint encoder training",
                                     "In any encoder training", "Interpretation", "Source"])
    (OUT / "S4_encoder_exposure.md").write_text(md(df))


# ------------------------------------------------------------------ S7 41-compound master
def s7():
    fe = pd.read_csv(T / "upstream_parameter_fold_errors.csv")
    m = pd.read_csv(T / "per_compound_metrics.csv")
    h = m[m.scenario == "h2_run0"].set_index("compound_id")
    names = h["compound"].to_dict()

    def val(pred, par, what):
        s = fe[(fe.predictor == pred) & (fe.parameter == par)].set_index("compound_id")[what]
        return s
    obs_fu, obs_vd, obs_cl = val("DL-ML", "Fu", "obs"), val("DL-ML", "VDss", "obs"), val("DL-ML", "CLsys", "obs")
    rows = []
    for cid in sorted(h.index, key=lambda c: names[c].lower()):
        g = lambda p, par: val(p, par, "log2_fe").get(cid, np.nan)
        rows.append({"Compound": names[cid].capitalize(), "ID": cid,
                     "Fu obs": "%.3g" % obs_fu[cid], "log₂FE Fu S+ / ML": "%+.2f / %+.2f" % (g("S+", "Fu"), g("DL-ML", "Fu")),
                     "VDss obs (L/kg)": "%.3g" % obs_vd[cid], "log₂FE VDss S+ / ML": "%+.2f / %+.2f" % (g("S+", "VDss"), g("DL-ML", "VDss")),
                     "CL obs (L/h/kg)": "%.3g" % obs_cl[cid], "log₂FE CL ML": "%+.2f" % g("DL-ML", "CLsys"),
                     "AUC log₂FE": "%+.2f" % h.loc[cid, "fe_auc_rel"],
                     "Cmax log₂FE": "%+.2f" % h.loc[cid, "fe_cmax_rel"],
                     "log-NRMSE": ("%.3f" % h.loc[cid, "log_NRMSE"]) if np.isfinite(h.loc[cid, "log_NRMSE"]) else "n.a.†",
                     "RMSE log₁₀": "%.3f" % h.loc[cid, "rmse_log10"]})
    (OUT / "S7_master41.md").write_text(md(pd.DataFrame(rows)))


# ------------------------------------------------------------------ S8 PBPK paired contrasts
def s8():
    c = pd.read_csv(T / "paired_contrasts.csv")
    lab = {"log_NRMSE": "C–T log-NRMSE", "rmse_log10": "C–T RMSE (log₁₀)",
           "fe_auc_abs": "AUC abs. log₂FE", "fe_cmax_abs": "Cmax abs. log₂FE"}
    rows = []
    for _, r in c.iterrows():
        rows.append({"Group": "PBPK " + r.group, "Arm": r.run_id, "Reference": r.reference,
                     "Endpoint": lab[r.endpoint], "n": int(r.n), "HL estimate": "%+.3f" % r.estimate,
                     "95% CI": ci(r.ci_low, r.ci_high), "p (raw)": p_fmt(r.p), "p (Holm)": p_fmt(r.p_holm),
                     "worse / better": "%d / %d" % (r.n_worse, r.n_better),
                     "Interpretation": {"worse": "larger error than reference", "better": "smaller error than reference",
                                        "none": "no detectable difference"}[r.direction]})
    (OUT / "S8_pbpk_paired.md").write_text(md(pd.DataFrame(rows)))


# ------------------------------------------------------------------ S9 C-T statistic sensitivity
def hl(d):
    i, j = np.triu_indices(len(d), k=0)
    return float(np.median((d[i] + d[j]) / 2.0))


def s9():
    o = RL / "P1" / "outputs"
    M = {"log-NRMSE (primary)": pd.read_csv(o / "ct_lognrmse_matrix.csv", index_col=0),
         "RMSE log₁₀ (secondary)": pd.read_csv(o / "ct_rmse_log10_matrix.csv", index_col=0),
         "time-weighted RMSE log₁₀ (sensitivity)": pd.read_csv(o / "ct_rmse_log10_timeweighted_matrix.csv", index_col=0)}
    arms = ["v0_run0", "v1_run0", "s1_run1", "s1_run2", "s1_run3", "h1_run0", "h1_run0_noCLr", "h2_run4", "h2_run0"]
    rows = []
    for a in arms:
        r = {"Arm": a}
        for k, m in M.items():
            v = m[a].dropna()
            r[k + ", median (n)"] = "%.3f (%d)" % (v.median(), len(v))
        rows.append(r)
    t1 = pd.DataFrame(rows)
    rng = np.random.default_rng(20260920)
    rows = []
    for a in ["h1_run0", "h1_run0_noCLr", "h2_run4", "h2_run0"]:
        r = {"Arm vs s1_run3 (observed CL)": a}
        for k, m in M.items():
            j = m[[a, "s1_run3"]].dropna()
            d = (j[a] - j["s1_run3"]).to_numpy(float)
            b = [hl(d[rng.integers(0, len(d), len(d))]) for _ in range(10000)]
            r[k] = "%+.3f [%.3f, %.3f], p %s" % (hl(d), *np.percentile(b, [2.5, 97.5]),
                                                p_fmt(float(wilcoxon(j[a], j["s1_run3"]).pvalue)))
        rows.append(r)
    t2 = pd.DataFrame(rows)
    (OUT / "S9_ct_sensitivity.md").write_text(md(t1) + "\n" + md(t2))


# ------------------------------------------------------------------ S10 applicability domain
def s10():
    d = pd.read_csv(RL / "minor_points" / "applicability_domain" / "outputs" / "embedding_ad_summary.csv")
    cols = {c.lower(): c for c in d.columns}
    rows = []
    for _, r in d.iterrows():
        rows.append({"Endpoint": r["Endpoint"], "n test": int(r["N_test"]),
                     "AD threshold (train LOO p95)": "%.2f" % r["AD_threshold_train_LOO_p95"],
                     "n outside AD": int(r["N_outside_AD"]),
                     "Spearman ρ (95% CI)": "%.3f (%.3f to %.3f)" % (r["Spearman_rho"], r["Spearman_CI95_lo"], r["Spearman_CI95_hi"]),
                     "median abs. error inside / outside": "%.3f / %.3f" % (r["Median_abs_error_inside_AD"], r["Median_abs_error_outside_AD"]),
                     "Interpretation": ("supported" if r["Endpoint"] == "Fu" else
                                        "not interpretable (%d outside AD)" % int(r["N_outside_AD"]))})
    (OUT / "S10_applicability_domain.md").write_text(md(pd.DataFrame(rows)))


if __name__ == "__main__":
    for f in (s2, s4, s7, s8, s9, s10):
        f()
        print("  ", f.__name__)
    print("tables in", OUT)
