#!/usr/bin/env python3
"""Mechanistic (ECCS and empirical) stratification, recomputed on the corrected Sep21 results.

Replaces the v7.2 Figure 6 / Data S3 analysis (github_publish/PBPK-ML/pbpk_evaluation_analysis,
archived unchanged under legacy/pbpk_evaluation_analysis_v72/). That analysis compared the
published v2_CLint arm (which ran template enzyme kinetics with observed VDss, not the S+ CLint)
with v3_CLsys, scored profiles by the withdrawn mean |log2| statistic, and read some inputs from
misaligned columns (see data/compounds/MASTER_V3_README.md).

1. Master table v3 (data/compounds/pbpk_mechanistic_master_v3.csv): one row per compound, every
   column from one named, verified source; strata re-derived by explicit rules; PBPK errors from
   outputs/tables/per_compound_metrics.csv.
2. Strategy contrast per compound (2026-09-21, goal_cl.md hierarchy): primary, information-matched
   delta = error(h1_run0_noCLr) - error(h2_run4), predicted CLint vs predicted total CL with no
   separate renal-clearance input in either; positive = the predicted-total-CL workflow has the
   smaller error, negative = the predicted-CLint workflow has the smaller error (test - reference,
   as in the main text). Sensitivity: template-assisted bottom-up h1_run0 - h2_run4.
   Endpoints: C-T log-NRMSE (n = 40) and RMSE log10; AUC and Cmax absolute log2 fold error.
3. Secondary: cost of predicting clearance = median error over the four predicted-CL scenarios
   (h1_run0, h1_run0_noCLr, h2_run4, h2_run0) - error with observed CL (s1_run3).
4. Per stratum: Kruskal-Wallis across classes (classes with n >= 3); per class the median,
   Hodges-Lehmann estimate with a 95% bootstrap CI (10,000 resamples) and a Wilcoxon signed-rank
   test against 0, Holm-adjusted across classes within each stratum x endpoint.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/12_mechanistic_strata.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, wilcoxon

HERE = Path(__file__).resolve().parent
S21 = HERE.parent
ROOT = S21.parents[1]
sys.path.insert(0, str(S21 / "lib"))
import paths as P                                                 # noqa: E402
import matplotlib                                                 # noqa: E402

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt                                   # noqa: E402
import style                                                      # noqa: E402

LEGACY_SRC = ROOT / "github_publish" / "PBPK-ML" / "pbpk_evaluation_analysis"
LEGACY = S21 / "legacy" / "pbpk_evaluation_analysis_v72"
OLD = P.DATA / "compounds" / "pbpk_physchem_mechanistic_master.csv"
NEW = P.DATA / "compounds" / "pbpk_mechanistic_master_v3.csv"
T = P.TABLES
FIG = S21 / "outputs" / "figures" / "si"
RNG = np.random.default_rng(20260921)
N_BOOT = 10000

BU, TD, BU0 = "h1_run0", "h2_run4", "h1_run0_noCLr"
PRED_CL = ["h1_run0", "h1_run0_noCLr", "h2_run4", "h2_run0"]
ENDP = [("log_NRMSE", "C–T log-NRMSE"), ("rmse_log10", "C–T RMSE (log₁₀)"),
        ("fe_auc_abs", "AUC |log₂ FE|"), ("fe_cmax_abs", "Cmax |log₂ FE|")]
STRATA = [("ECCS_class", ["Class 1", "Class 2", "Class 3", "Class 4"], "ECCS class (S+)"),
          ("clearance_group", ["Metabolism-leaning", "Renal-leaning", "Uptake/Biliary-influenced", "Mixed/Other"],
           "Clearance group"),
          ("ionization_pH74", ["Acid (ionized)", "Base (ionized)", "Ampholyte", "Neutral at pH 7.4"],
           "Ionization at pH 7.4"),
          ("logP_bin", ["<2", "2–4", ">4"], "logP bin"),
          ("VDss_bin", ["Low (<0.7)", "Mid (0.7–2)", "High (>2)"], "VDss bin")]


# ------------------------------------------------------------------ master table v3
def ionization(code, pka1, pka2):
    """S+ compound-type code as entered in the s1 scenarios. The code mapping was inferred from
    unambiguous reference drugs (code 2: warfarin, phenytoin, tolbutamide, probenecid, statins ->
    monoprotic acid; code 3: desipramine, verapamil, metoprolol, clozapine -> monoprotic base;
    code 1: quinidine, crizotinib, posaconazole -> diprotic base; code 5: buprenorphine,
    lorazepam, dabigatran, rosiglitazone -> ampholyte; codes 0 and 4 -> neutral/other).
    Ionized at pH 7.4: acid with pKa <= 7.4, base with its highest pKa >= 7.4."""
    pk = [x for x in (pka1, pka2) if pd.notna(x) and x > 0]
    if code == 5:
        return "Ampholyte"
    if code == 2:
        return "Acid (ionized)" if pk and min(pk) <= 7.4 else "Neutral at pH 7.4"
    if code in (1, 3):
        return "Base (ionized)" if pk and max(pk) >= 7.4 else "Neutral at pH 7.4"
    return "Neutral at pH 7.4"


def clearance_group(r):
    """v7.2 Methods rule, applied to the template-derived inputs (CLRbase in L/h)."""
    if (pd.notna(r.ActiveUptakeHep) and r.ActiveUptakeHep > 1) or r.BiliaryClearanceType == 1:
        return "Uptake/Biliary-influenced"
    if (pd.notna(r.CLint_H_extra) and r.CLint_H_extra > 0) or (pd.notna(r.Pcnt3AMetCL) and r.Pcnt3AMetCL >= 50):
        return "Metabolism-leaning"
    weak = (pd.isna(r.CLint_H_extra) or r.CLint_H_extra == 0) and (pd.isna(r.Pcnt3AMetCL) or r.Pcnt3AMetCL <= 5)
    if r.CLRbase_template_Lh >= 1.0 and weak:
        return "Renal-leaning"
    return "Mixed/Other"


def build_master():
    old = pd.read_csv(OLD).set_index("ID_trend_pbpk")
    v1 = pd.read_csv(P.SIM / "v1_run0" / "adme_key_inputs.csv").set_index("Drug")
    s1 = pd.read_csv(P.SIM / "s1_run1" / "adme_key_inputs.csv").set_index("Drug")
    r7 = pd.read_csv(S21 / "response_letter" / "shared" / "reviewer_tables" / "outputs" /
                     "Table_R7b_inherited_template_values_per_compound.csv").set_index("Trend_ID")
    dsg = pd.read_csv(P.SIM / "v0_run0" / "design_from_template.csv").set_index("Drug")
    per = pd.read_csv(T / "per_compound_metrics.csv")
    m = pd.DataFrame(index=old.index.rename("compound_id"))
    # identifiers
    m["PUBCHEM_CID"] = old.PUBCHEM_CID.astype("Int64")
    m["compound"] = old.Name_final.str.strip()
    # observed inputs, as entered in the observed-input control (v1_run0)
    m["Fu_obs"] = v1.Fu
    m["VDss_obs_L_per_kg"] = v1.VD
    m["CL_obs_L_per_h_per_kg"] = v1.CL_MetBas / 70.0
    # Simcyp template (library compound file)
    m["logP_template"] = r7.template_logP
    m["BP_template"] = r7.template_BP
    m["CLRbase_template_Lh"] = r7.CLRbase_Lh
    m["KpScalar_template"] = r7.KpScalar
    m["template_distribution"] = dsg.Template_distribution
    m["template_elimination"] = dsg.Template_elimination
    for c in ("CLint_H_extra", "Pcnt3AMetCL", "ActiveUptakeHep", "BiliaryClearanceType"):
        m[c] = old[c]                                        # template-derived flags (verified alignment via CLRbase)
    # S+ (ADMET Predictor) as entered in the s1 scenarios
    m["MW_Splus"] = s1.Mole_Wh
    m["logP_Splus"] = s1.LogP
    m["pKa1_Splus"], m["pKa2_Splus"] = s1.pKa1, s1.pKa2
    m["compound_type_code_Splus"] = s1.CompoundType
    m["ECCS_class_raw_Splus"] = old.ECCS_Class
    m["ECCS_class"] = old.ECCS_new.str.replace("Class", "Class ", regex=False)
    # literature clearance annotation (descriptive, unchanged)
    for c in [c for c in old.columns if c.startswith("clearance_ref_")]:
        m[c] = old[c]
    # strata (re-derived)
    m["clearance_group"] = m.apply(clearance_group, axis=1)
    m["ionization_pH74"] = [ionization(c, a, b) for c, a, b in zip(m.compound_type_code_Splus, m.pKa1_Splus, m.pKa2_Splus)]
    m["logP_bin"] = pd.cut(m.logP_template, [-np.inf, 2, 4, np.inf], labels=["<2", "2–4", ">4"], right=False).astype(str)
    m["VDss_bin"] = pd.cut(m.VDss_obs_L_per_kg, [-np.inf, 0.7, 2.0, np.inf],
                           labels=["Low (<0.7)", "Mid (0.7–2)", "High (>2)"], right=False).astype(str)
    # PBPK errors (Sep21 corrected batch)
    for run in ["v0_run0", "v1_run0", "s1_run3", "h1_run0", "h1_run0_noCLr", "h2_run4", "h2_run0"]:
        x = per[per.scenario == run].set_index("compound_id")
        for col, _ in ENDP:
            m["%s__%s" % (col, run)] = x[col]
        m["fe_auc_signed__%s" % run] = x.fe_auc_rel
        m["fe_cmax_signed__%s" % run] = x.fe_cmax_rel
    for col, _ in ENDP:
        m["delta_%s__BU_minus_TD" % col] = m["%s__%s" % (col, BU)] - m["%s__%s" % (col, TD)]
        m["delta_%s__BU0_minus_TD" % col] = m["%s__%s" % (col, BU0)] - m["%s__%s" % (col, TD)]
        m["cost_%s__predCL_minus_obsCL" % col] = (m[["%s__%s" % (col, r) for r in PRED_CL]].median(axis=1)
                                                  - m["%s__s1_run3" % col])
    # descriptive outcome category (v7.2 definition: acceptable = AUC and Cmax both within 2-fold)
    ok = lambda run: (m["fe_auc_abs__%s" % run] <= 1) & (m["fe_cmax_abs__%s" % run] <= 1)
    a, b = ok(BU0), ok(TD)
    m["outcome_BU0_vs_TD"] = np.select([a & b, a & ~b, ~a & b], ["Both acceptable", "Bottom-up only", "Top-down only"],
                                      "Neither (refinement)")
    m = m.sort_values("compound")
    m.to_csv(NEW, float_format="%.6g")
    # comparison with the legacy strata
    cmp = pd.DataFrame({"compound": m.compound,
                        "clearance_group_v72": old.clearance_group, "clearance_group_v3": m.clearance_group,
                        "ionization_v72": old.ionization_group, "ionization_v3": m.ionization_pH74,
                        "Vd_bin_v72": old.Vd_bin, "VDss_bin_v3": m.VDss_bin,
                        "logP_bin_v72": old.logP_bin, "logP_bin_v3": m.logP_bin})
    cmp.to_csv(T / "mechanistic_strata_v72_vs_v3.csv")
    return m, old, cmp


# ------------------------------------------------------------------ statistics
def hl(d):
    i, j = np.triu_indices(len(d), k=0)
    return float(np.median((d[i] + d[j]) / 2.0))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def stratify(m, prefix, label):
    rows = []
    for var, order, vlab in STRATA:
        for col, elab in ENDP:
            y = m["%s%s%s" % (prefix, col, label)]
            groups = [(c, y[m[var] == c].dropna().to_numpy()) for c in order]
            usable = [g for c, g in groups if len(g) >= 3]
            kw = kruskal(*usable).pvalue if len(usable) >= 2 else np.nan
            block = []
            for c, g in groups:
                if len(g) == 0:
                    continue
                est = hl(g)
                if len(g) >= 3:
                    boot = [hl(g[RNG.integers(0, len(g), len(g))]) for _ in range(N_BOOT)]
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    p = 1.0 if np.allclose(g, 0) else float(wilcoxon(g).pvalue)
                else:
                    lo = hi = p = np.nan
                block.append(dict(stratum=vlab, variable=var, cls=c, endpoint=elab, endpoint_col=col, n=len(g),
                                  median=float(np.median(g)), hl=est, ci_low=lo, ci_high=hi, p_wilcoxon=p,
                                  n_positive=int((g > 0).sum()), n_negative=int((g < 0).sum()), p_kruskal=kw))
            ps = [b["p_wilcoxon"] for b in block]
            ok = [i for i, v in enumerate(ps) if np.isfinite(v)]
            adj = holm([ps[i] for i in ok]) if ok else []
            for b in block:
                b["p_holm"] = np.nan
            for i, a in zip(ok, adj):
                block[i]["p_holm"] = a
            rows += block
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def fig_eccs(m, stat, out_stem, prefix, label, ylab_note, title_note):
    style.use()
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                                "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.default": "regular"})
    order = STRATA[0][1]
    panels = [("log_NRMSE", "C–T profile, log-NRMSE"), ("fe_auc_abs", "AUC$_{0–t}$, |log$_2$ FE|"),
              ("fe_cmax_abs", "C$_{max}$, |log$_2$ FE|")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharey=False)
    rng = np.random.default_rng(1)
    for k, (ax, (col, ttl)) in enumerate(zip(axes, panels)):
        y = m["%s%s%s" % (prefix, col, label)]
        data = [y[m.ECCS_class == c].dropna().to_numpy() for c in order]
        ax.axhline(0, color="#52514e", lw=0.8, ls=(0, (3, 2)))
        ax.boxplot(data, positions=range(4), widths=0.55, showfliers=False, patch_artist=True,
                   medianprops=dict(color="#b3261e", lw=1.4),
                   boxprops=dict(facecolor="#dbe7f5", edgecolor="#2a78d6", lw=0.9),
                   whiskerprops=dict(color="#2a78d6", lw=0.8), capprops=dict(color="#2a78d6", lw=0.8))
        for i, g in enumerate(data):
            ax.plot(i + rng.uniform(-0.15, 0.15, len(g)), g, "o", ms=3.2, mfc="#1a1a1a", mec="none", alpha=0.75)
        s = stat[(stat.variable == "ECCS_class") & (stat.endpoint_col == col)]
        ax.set_xticks(range(4))
        ax.set_xticklabels(["%s\nn = %d" % (c.replace("Class ", "Class "), len(g)) for c, g in zip(order, data)],
                           fontsize=7.2)
        ax.set_title(ttl, fontsize=8.6, fontweight="bold")
        kw = s.p_kruskal.iloc[0]
        ax.text(0.03, 0.97, "Kruskal–Wallis p = %.2f" % kw, transform=ax.transAxes, va="top", fontsize=7, color="#52514e")
        ax.tick_params(axis="y", labelsize=7)
        ax.text(-0.18, 1.07, "(%s)" % "abc"[k], transform=ax.transAxes, fontsize=10, fontweight="bold")
        if k == 0:
            ax.set_ylabel(ylab_note, fontsize=7.8)
        lim = np.nanmax(np.abs(np.concatenate([g for g in data if len(g)]))) * 1.15
        ax.set_ylim(-lim, lim)
    fig.tight_layout(w_pad=1.4)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext, kw_ in (("png", {"dpi": 400}), ("pdf", {})):
        fig.savefig(FIG / ("%s.%s" % (out_stem, ext)), **kw_)
    plt.close(fig)


def fig_strata(m, stat, col="log_NRMSE", label="__BU0_minus_TD", stem="FigS_mechanistic_strata_matched_delta",
               ylab="Δ C–T log-NRMSE,\nCLint (CLR = 0) − total CL"):
    style.use()
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey=True)
    rng = np.random.default_rng(2)
    y = m["delta_%s%s" % (col, label)]
    for k, (ax, (var, order, vlab)) in enumerate(zip(axes.flat, STRATA[1:])):
        data = [y[m[var] == c].dropna().to_numpy() for c in order]
        ax.axhline(0, color="#52514e", lw=0.8, ls=(0, (3, 2)))
        ax.boxplot(data, positions=range(len(order)), widths=0.55, showfliers=False, patch_artist=True,
                   medianprops=dict(color="#b3261e", lw=1.4),
                   boxprops=dict(facecolor="#dbe7f5", edgecolor="#2a78d6", lw=0.9),
                   whiskerprops=dict(color="#2a78d6", lw=0.8), capprops=dict(color="#2a78d6", lw=0.8))
        for i, g in enumerate(data):
            ax.plot(i + rng.uniform(-0.15, 0.15, len(g)), g, "o", ms=3, mfc="#1a1a1a", mec="none", alpha=0.75)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["%s\nn = %d" % (c.replace("/", "/\n").replace("-leaning", "-\nleaning") if len(c) > 12 else c, len(g))
                            for c, g in zip(order, data)], fontsize=6.6)
        kw = stat[(stat.variable == var) & (stat.endpoint_col == col)].p_kruskal.iloc[0]
        ax.set_title("%s (Kruskal–Wallis p = %.2f)" % (vlab, kw), fontsize=8.4, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
        ax.text(-0.12, 1.07, "(%s)" % "abcd"[k], transform=ax.transAxes, fontsize=10, fontweight="bold")
        if k % 2 == 0:
            ax.set_ylabel(ylab, fontsize=7.8)
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    for ext, kw_ in (("png", {"dpi": 400}), ("pdf", {})):
        fig.savefig(FIG / ("%s.%s" % (stem, ext)), **kw_)
    plt.close(fig)


# ------------------------------------------------------------------ SI (section S7)
def _pf(p):
    if not np.isfinite(p):
        return "–"
    return "<0.001" if p < 0.001 else ("%.3f" % p if p < 0.1 else "%.2f" % p)


def _f(x):
    s = "%+.3f" % x
    return "0.000" if s in ("+0.000", "-0.000") else s


def si_outputs(st0, st, sc):
    """Table S12 (ECCS per class, matched contrast), Table S13 (Kruskal-Wallis p for every stratum,
    endpoint and contrast) and the S7 figure copies into manuscript/v3/SI."""
    si = S21 / "manuscript" / "v3" / "SI"
    safe = {"log_NRMSE": "C–T log-NRMSE", "rmse_log10": "C–T RMSE (log₁₀)", "fe_auc_abs": "AUC₀₋ₜ, abs. log₂ FE",
            "fe_cmax_abs": "Cmax, abs. log₂ FE"}
    e = st0[st0.variable == "ECCS_class"]
    L = ["| Endpoint | ECCS class | n | Median Δ | HL Δ (95% CI) | p (raw) | p (Holm) | −/+ | KW p |",
         "|---|---|---|---|---|---|---|---|---|"]
    for col, _ in ENDP:
        elab = safe[col]
        blk = e[e.endpoint_col == col]
        for k, (_, r) in enumerate(blk.iterrows()):
            L.append("| %s | %s | %d | %s | %s (%s to %s) | %s | %s | %d/%d | %s |" % (
                elab if k == 0 else "", r.cls, r.n, _f(r["median"]), _f(r.hl), "%.3f" % r.ci_low, "%.3f" % r.ci_high,
                _pf(r.p_wilcoxon), _pf(r.p_holm), r.n_negative, r.n_positive, _pf(r.p_kruskal) if k == 0 else ""))
    (si / "tables" / "S12_ECCS_matched.md").write_text("\n".join(L) + "\n")
    ends = [c for c, _ in ENDP]
    L = ["| Contrast | Stratum | " + " | ".join(safe[c] for c, _ in ENDP) + " |", "|---|---|" + "---|" * len(ENDP)]
    for name, d in (("Matched: CLint (CLR = 0) − total CL", st0), ("Template-assisted CLint − total CL", st),
                    ("Predicted − observed CL (cost)", sc)):
        for k, (var, _, vlab) in enumerate(STRATA):
            ps = [d[(d.variable == var) & (d.endpoint_col == c)].p_kruskal.iloc[0] for c in ends]
            L.append("| %s | %s | %s |" % (name if k == 0 else "", vlab, " | ".join(_pf(x) for x in ps)))
    (si / "tables" / "S13_strata_kruskal.md").write_text("\n".join(L) + "\n")
    for src, dst in (("FigS_ECCS_matched_delta", "FigureS16_ECCS_matched_delta"),
                     ("FigS_mechanistic_strata_matched_delta_AUC", "FigureS17_strata_matched_delta_AUC"),
                     ("FigS_ECCS_cost_of_predicted_CL", "FigureS18_ECCS_cost_of_predicted_CL")):
        for ext in ("png", "pdf"):
            shutil.copy2(FIG / ("%s.%s" % (src, ext)), si / "figures" / ("%s.%s" % (dst, ext)))


def main():
    if not LEGACY.exists():
        shutil.copytree(LEGACY_SRC, LEGACY)
        print("archived", LEGACY_SRC, "->", LEGACY)
    m, old, cmp = build_master()
    st0 = stratify(m, "delta_", "__BU0_minus_TD")          # primary, information-matched
    st = stratify(m, "delta_", "__BU_minus_TD")            # sensitivity, template-assisted bottom-up
    sc = stratify(m, "cost_", "__predCL_minus_obsCL")
    st0.to_csv(T / "mechanistic_strata_matched_delta.csv", index=False, float_format="%.4f")
    st.to_csv(T / "mechanistic_strata_template_assisted_delta.csv", index=False, float_format="%.4f")
    sc.to_csv(T / "mechanistic_strata_cost_of_predicted_CL.csv", index=False, float_format="%.4f")
    for old_name in ("mechanistic_strata_strategy_delta.csv", "mechanistic_strata_strategy_delta_noCLr.csv"):
        (T / old_name).unlink(missing_ok=True)
    out = pd.crosstab(m.ECCS_class, m.outcome_BU0_vs_TD)
    out.to_csv(T / "mechanistic_outcome_by_ECCS.csv")
    fig_eccs(m, st0, "FigS_ECCS_matched_delta", "delta_", "__BU0_minus_TD",
             "Δ error, CLint (CLR = 0) − total CL\n(< 0: CLint smaller; > 0: total CL smaller)", "")
    fig_eccs(m, sc, "FigS_ECCS_cost_of_predicted_CL", "cost_", "__predCL_minus_obsCL",
             "Δ error, predicted − observed CL\n(> 0: prediction costs accuracy)", "")
    fig_strata(m, st0)
    si_outputs(st0, st, sc)
    fig_strata(m, st0, col="fe_auc_abs", stem="FigS_mechanistic_strata_matched_delta_AUC",
               ylab="Δ AUC |log$_2$ FE|,\nCLint (CLR = 0) − total CL")
    for f in FIG.glob("FigS_ECCS_strategy_delta.*"):
        f.unlink()
    for f in FIG.glob("FigS_mechanistic_strata_strategy_delta.*"):
        f.unlink()
    pd.set_option("display.width", 220)
    show = ["stratum", "cls", "endpoint", "n", "median", "hl", "ci_low", "ci_high", "p_wilcoxon", "p_holm", "p_kruskal"]
    print("\nPRIMARY matched delta (h1_run0_noCLr - h2_run4), ECCS:")
    print(st0[st0.variable == "ECCS_class"][show].round(3).to_string(index=False))
    print("\nKruskal-Wallis p, matched delta:")
    print(st0.groupby(["stratum", "endpoint"]).p_kruskal.first().unstack().round(3).to_string())
    print("\nKruskal-Wallis p, template-assisted sensitivity:")
    print(st.groupby(["stratum", "endpoint"]).p_kruskal.first().unstack().round(3).to_string())
    print("\nsensitivity ECCS:")
    print(st[st.variable == "ECCS_class"][show].round(3).to_string(index=False))
    print("\nKruskal-Wallis p, cost of predicted CL:")
    print(sc.groupby(["stratum", "endpoint"]).p_kruskal.first().unstack().round(3).to_string())
    print("\ncost of predicted CL, ECCS:")
    print(sc[sc.variable == "ECCS_class"][show].round(3).to_string(index=False))
    print("\nmatched delta, all strata, per class with p_holm < 0.05 or CI excluding 0:")
    sig = st0[(st0.ci_low > 0) | (st0.ci_high < 0)]
    print(sig[show].round(3).to_string(index=False))
    print("\noutcome by ECCS:\n", out)


if __name__ == "__main__":
    main()
