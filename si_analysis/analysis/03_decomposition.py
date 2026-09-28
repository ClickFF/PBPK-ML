#!/usr/bin/env python3
"""Upstream parameter accuracy and the parameter-vs-structure decomposition.

Writes to outputs/tables/:
  upstream_parameter_accuracy.csv   Figure 4 numbers: predicted vs observed Fu, CLsys, VDss
  upstream_parameter_fold_errors.csv  per compound, per predictor
  decomposition_ladder.csv          Fig 7a: cumulative paired change vs the observed-input arm
  decomposition_routing.csv         Fig 7b: Spearman rho, parameter FE vs exposure error (h2_run0)
  cmax_bias_by_arm.csv              Fig 7c: mean signed log2 Cmax FE per arm, bootstrap CI
  v0_structure_contrast.csv         Fig 7c: full- vs minimal-template compounds within v0

Observed parameters are the values entered in v1_run0; predicted are those entered in s1_run3
(S+ Fu, VDss) and h2_run0 (DL-ML Fu, VDss, CLsys). CL_MetBas is L/h at 70 kg -> divided by 70.
Profile specificity in the structure contrast uses RMSE (log10) and log-NRMSE, not a log2 error.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/03_decomposition.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import paths as P                                                 # noqa: E402

N_BOOT = 10000
RNG = np.random.default_rng(20260921)


def hl(d):
    i, j = np.triu_indices(len(d), k=0)
    return float(np.median((d[i] + d[j]) / 2.0))


def hl_ci(diff):
    n = len(diff)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    boot = np.array([hl(diff[k]) for k in idx])
    return hl(diff), *np.percentile(boot, [2.5, 97.5])


def applied(run):
    d = pd.read_csv(P.SIM / run / "adme_key_inputs.csv").set_index("Drug")
    d.index = d.index.astype(int)
    return pd.DataFrame({"Fu": d["Fu"], "VDss": d["VD"], "CLsys": d["CL_MetBas"] / 70.0})


def upstream():
    obs = applied("v1_run0")
    rows, per = [], []
    for pred_name, run, params in (("S+", "s1_run3", ["Fu", "VDss"]),
                                   ("DL-ML", "h2_run0", ["Fu", "VDss", "CLsys"])):
        pr = applied(run)
        for p in params:
            j = pd.concat([obs[p].rename("obs"), pr[p].rename("pred")], axis=1).dropna()
            j = j[(j.obs > 0) & (j.pred > 0)]
            fe = np.log10(j.pred / j.obs)
            afe = np.abs(fe)
            rows.append(dict(predictor=pred_name, parameter=p, n=len(j),
                             within_2fold_pct=100 * np.mean(afe <= np.log10(2)),
                             within_3fold_pct=100 * np.mean(afe <= np.log10(3)),
                             gmfe=float(10 ** afe.mean()), afe_bias=float(10 ** fe.mean())))
            for cid, v in (np.log2(j.pred / j.obs)).items():
                per.append(dict(predictor=pred_name, parameter=p, compound_id=cid,
                                obs=j.loc[cid, "obs"], pred=j.loc[cid, "pred"], log2_fe=v))
    u = pd.DataFrame(rows)
    u.to_csv(P.TABLES / "upstream_parameter_accuracy.csv", index=False, float_format="%.3f")
    pd.DataFrame(per).to_csv(P.TABLES / "upstream_parameter_fold_errors.csv", index=False)
    return u, pd.DataFrame(per)


def main():
    d = pd.read_csv(P.TABLES / "per_compound_metrics.csv")
    val = lambda run, col: d[d.scenario == run].set_index("compound_id")[col]

    u, per = upstream()
    print("upstream parameter accuracy (n = 41):")
    print(u.round(2).to_string(index=False))

    # ---- 7a ladder: cumulative change vs the observed-input arm ------------------------------
    steps = [("s1_run1", "S+ physchem"), ("s1_run2", "+ S+ Fu"), ("s1_run3", "+ S+ VDss"),
             ("h2_run4", "+ DL–ML CL"), ("h1_run0_noCLr", "+ S+ CLint (bottom-up)"),
             ("h2_run0", "+ DL–ML Fu, VDss (all predicted)")]
    lad = []
    for col in ("fe_auc_abs", "fe_cmax_abs", "log_NRMSE", "rmse_log10"):
        for run, lab in steps:
            j = pd.concat([val(run, col).rename("t"), val("v1_run0", col).rename("r")],
                          axis=1).dropna()
            diff = (j.t - j.r).to_numpy(float)
            est, lo, hi = hl_ci(diff)
            p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(j.t, j.r).pvalue)
            lad.append(dict(endpoint=col, run_id=run, step=lab, n=len(j), estimate=est,
                            ci_low=lo, ci_high=hi, p=p, detectable=bool(lo > 0 or hi < 0)))
    pd.DataFrame(lad).to_csv(P.TABLES / "decomposition_ladder.csv", index=False,
                             float_format="%.5f")

    # ---- 7b routing: which parameter error reaches which exposure error (h2_run0) -------------
    ml = per[per.predictor == "DL-ML"].pivot(index="compound_id", columns="parameter",
                                             values="log2_fe")
    x = d[d.scenario == "h2_run0"].set_index("compound_id")
    rout = []
    for p in ("Fu", "CLsys", "VDss"):
        for col, lab in (("fe_auc_rel", "AUC, signed log2 FE"), ("fe_cmax_rel", "Cmax, signed log2 FE"),
                         ("log_NRMSE", "C–T, log-NRMSE"), ("rmse_log10", "C–T, RMSE log10")):
            j = pd.concat([ml[p].rename("a"), x[col].rename("b")], axis=1).dropna()
            a = j.a if col.startswith("fe_") else j.a.abs()      # profile scores are unsigned
            rho, pv = spearmanr(a, j.b)
            rout.append(dict(parameter=p, exposure=lab, n=len(j), rho=rho, abs_rho=abs(rho), p=pv))
    r = pd.DataFrame(rout)
    r.to_csv(P.TABLES / "decomposition_routing.csv", index=False, float_format="%.4f")
    print("\nrouting, h2_run0 (Spearman):")
    print(r.pivot(index="parameter", columns="exposure", values="rho").round(2).to_string())

    # ---- 7c Cmax bias per arm ---------------------------------------------------------------
    bias = []
    for run in d.scenario.unique():
        v = val(run, "fe_cmax_rel").dropna().to_numpy()
        boot = v[RNG.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
        bias.append(dict(run_id=run, n=len(v), mean=v.mean(), median=np.median(v),
                         ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5),
                         pct_under=100 * np.mean(v < 0)))
    pd.DataFrame(bias).to_csv(P.TABLES / "cmax_bias_by_arm.csv", index=False, float_format="%.4f")

    # ---- 7c structure contrast inside v0 -----------------------------------------------------
    dsg = pd.read_csv(P.SIM / "v0_run0" / "design_from_template.csv")
    dsg["tmpl"] = np.where(dsg.Template_distribution.str.contains("Full"), "full", "minimal")
    tm = dict(zip(dsg.Drug.astype(int), dsg.tmpl))
    j = pd.DataFrame({"cmax_v0": val("v0_run0", "fe_cmax_rel"), "cmax_v1": val("v1_run0", "fe_cmax_rel"),
                      "rmse_v0": val("v0_run0", "rmse_log10"), "rmse_v1": val("v1_run0", "rmse_log10"),
                      "nrmse_v0": val("v0_run0", "log_NRMSE"), "nrmse_v1": val("v1_run0", "log_NRMSE")})
    j["tmpl"] = j.index.map(tm)
    j["d_cmax"] = j.cmax_v0 - j.cmax_v1            # > 0: less Cmax under-prediction in v0
    j["d_rmse"] = j.rmse_v0 - j.rmse_v1
    j["d_nrmse"] = j.nrmse_v0 - j.nrmse_v1
    sc = []
    for k in ("full", "minimal"):
        g = j[j.tmpl == k]
        sc.append(dict(subset=k, n=len(g), median_d_cmax=g.d_cmax.median(),
                       n_less_under=int((g.d_cmax > 0).sum()),
                       wilcoxon_p=float(wilcoxon(g.cmax_v0, g.cmax_v1).pvalue),
                       median_d_rmse=g.d_rmse.median(), median_d_nrmse=g.d_nrmse.median()))
    f, m_ = j[j.tmpl == "full"], j[j.tmpl == "minimal"]
    u_c, p_c = mannwhitneyu(f.d_cmax, m_.d_cmax, alternative="greater")
    u_r, p_r = mannwhitneyu(f.d_rmse, m_.d_rmse, alternative="two-sided")
    fn, mn = f.d_nrmse.dropna(), m_.d_nrmse.dropna()
    u_n, p_n = mannwhitneyu(fn, mn, alternative="two-sided")
    s = pd.DataFrame(sc)
    s["mwu_p_cmax_full_gt_minimal"] = p_c
    s["mwu_p_rmse_specificity"] = p_r
    s["mwu_p_nrmse_specificity"] = p_n
    s.to_csv(P.TABLES / "v0_structure_contrast.csv", index=False, float_format="%.4f")
    j.to_csv(P.TABLES / "v0_structure_contrast_by_compound.csv", float_format="%.5f")
    print("\nv0 structure contrast:")
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
