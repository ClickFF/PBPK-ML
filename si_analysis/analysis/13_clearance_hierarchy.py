#!/usr/bin/env python3
"""Clearance-strategy comparison reorganised into a prespecified hierarchy (goal_cl.md, 2026-09-21).

    A  information-matched primary   h1_run0_noCLr  vs  h2_run4        (neither receives a CLR input)
    B  renal-clearance sensitivity   h1_run0        vs  h1_run0_noCLr  (effect of the retained template CLR)
    C  secondary pragmatic workflow  h1_run0        vs  h2_run4        (template-assisted BU vs total-CL TD)

Every contrast is test - reference, compound-paired, on six endpoints: C-T log-NRMSE (n = 40), C-T RMSE
log10 (n = 41), |log2 FE| AUC and Cmax, and signed log2 FE AUC and Cmax. Absolute endpoints:
positive = the test scenario has the larger error. Signed endpoints: positive = the test scenario
predicts higher exposure; this is a shift in bias, not necessarily a change in accuracy.

Statistics are the v3 paired implementation of analysis/02_paired.py, imported rather than copied:
Hodges-Lehmann estimate, 95% percentile bootstrap over compounds (10 000 resamples, seed 20260920),
two-sided Wilcoxon signed-rank p. So that the numbers already quoted in the manuscript are carried over
unchanged, the random stream of 02_paired.py is replayed first (its A, B and C blocks in the same
order); the cells that exist in paired_contrasts.csv are asserted identical and the new cells continue
the same stream.

Holm families (prespecified): within each contrast, the four absolute-error endpoints form one family
and the two signed endpoints another. A conservative global Holm over the 12 absolute-error tests
(3 contrasts x 4 endpoints) is added as a column.

Exploratory mechanistic support (contrast B only): Spearman correlation of the CLR-induced change in
signed AUC error with the retained template CLRbase (L/h, Simcyp template read-back, Table R7b) and
with CLRbase / observed total CL (both L/h; observed CL = CL_MetBas entered in v1_run0). The template
CLR values are clinically informed library inputs, not prospective predictions.

Outputs
  outputs/tables/clearance_hierarchy_contrasts.csv
  outputs/tables/clearance_hierarchy_clr_mechanistic.csv   (per compound)
  outputs/tables/clearance_hierarchy_spearman.csv
  outputs/tables/clearance_hierarchy_signed_summary.csv    (median signed error, over/under counts)
  outputs/figures/si/FigS_clearance_hierarchy.{png,pdf}
  manuscript/v3/SI/tables/S11_clearance_hierarchy.md, manuscript/v3/SI/figures/FigureS15_clearance_hierarchy.png

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/13_clearance_hierarchy.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import paths as P                                                 # noqa: E402
import arms as A                                                  # noqa: E402
import matplotlib                                                 # noqa: E402
import matplotlib.ticker                                          # noqa: E402

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
import style                                                      # noqa: E402

spec = importlib.util.spec_from_file_location("paired02", HERE / "02_paired.py")
P02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P02)

CONTRASTS = [
    ("A", "h1_run0_noCLr", "h2_run4", "Information-matched primary",
     "Predicted CLint (CLR = 0) vs predicted total CL (CLR = 0)"),
    ("B", "h1_run0", "h1_run0_noCLr", "Renal-clearance sensitivity",
     "Predicted CLint + template CLR vs predicted CLint (CLR = 0)"),
    ("C", "h1_run0", "h2_run4", "Secondary pragmatic workflow",
     "Template-assisted bottom-up vs total-CL top-down"),
]
ENDPOINTS = [
    ("log_NRMSE", "C–T log-NRMSE", "absolute"),
    ("rmse_log10", "C–T RMSE (log10)", "absolute"),
    ("fe_auc_abs", "AUC |log2 FE|", "absolute"),
    ("fe_cmax_abs", "Cmax |log2 FE|", "absolute"),
    ("fe_auc_rel", "AUC signed log2 FE", "signed"),
    ("fe_cmax_rel", "Cmax signed log2 FE", "signed"),
]
ZERO = 1e-12


def replay_02(val):
    """Advance P02.RNG exactly as 02_paired.main() does and return its group C cells."""
    P02.RNG = np.random.default_rng(20260920)
    for g in ("A", "B"):
        ref = A.run_of(A.REFERENCE[g])
        tests = [(r, n, l) for r, n, l in A.group(g) if n != A.REFERENCE[g]]
        for col, *_ in A.ENDPOINTS:
            for run, _, _ in tests:
                P02.contrast(val(run, col), val(ref, col))
    cells = {}
    for col, *_ in A.ENDPOINTS:
        for run in ("h1_run0_noCLr", "h1_run0", "h2_run0"):
            cells[(run, "h2_run4", col)] = P02.contrast(val(run, col), val("h2_run4", col))
    return cells


def direction(r, kind):
    """Detectable = bootstrap CI excludes 0 AND Holm-adjusted p < 0.05; CI alone = nominal only."""
    if not (r["ci_low"] > 0 or r["ci_high"] < 0):
        return "not detectable"
    up = r["ci_low"] > 0
    if kind == "absolute":
        s = "test larger error" if up else "test smaller error"
    else:
        s = "test higher exposure (bias shift)" if up else "test lower exposure (bias shift)"
    return s if r["p_holm"] < 0.05 else "nominal only, not after Holm (%s)" % s.split(" (")[0]


def interpret(cid, col, r, kind):
    d = direction(r, kind)
    if d.startswith("nominal"):
        return ("CI excluded zero but not detectable after Holm adjustment; magnitude %+.4f log units."
                % r["estimate"])
    if kind == "absolute":
        if d == "not detectable":
            return "No detectable difference in error magnitude (not evidence of equivalence)."
        return "Detectable difference in error magnitude (%s)." % d
    if d == "not detectable":
        return "No detectable shift in exposure bias."
    return "Systematic shift in exposure bias (%s); not by itself a change in accuracy." % d.split(" (")[0]


def main():
    style.use()
    per = pd.read_csv(P.TABLES / "per_compound_metrics.csv")
    val = lambda run, col: per[per.scenario == run].set_index("compound_id")[col]

    cells = replay_02(val)
    old = pd.read_csv(P.TABLES / "paired_contrasts.csv")
    for _, o in old[old.group == "C"].iterrows():
        c = cells[(o.run_id, o.reference, o.endpoint)]
        for k in ("estimate", "ci_low", "ci_high", "p"):
            assert abs(c[k] - o[k]) < 6e-6, (o.run_id, o.endpoint, k, c[k], o[k])

    rows = []
    for cid, test, ref, level, lab in CONTRASTS:
        for col, elab, kind in ENDPOINTS:
            key = (test, ref, col)
            c = cells[key] if key in cells else P02.contrast(val(test, col), val(ref, col))
            j = pd.concat([val(test, col).rename("t"), val(ref, col).rename("r")], axis=1).dropna()
            dd = (j.t - j.r).to_numpy()
            rows.append(dict(contrast=cid, interpretation_level=level, comparison=lab, test=test,
                             reference=ref, endpoint=col, endpoint_label=elab, endpoint_type=kind,
                             n=c["n"], estimate=c["estimate"], ci_low=c["ci_low"], ci_high=c["ci_high"],
                             p=c["p"], n_positive=int((dd > ZERO).sum()), n_negative=int((dd < -ZERO).sum()),
                             n_zero=int((np.abs(dd) <= ZERO).sum()),
                             median_test=float(j.t.median()), median_reference=float(j.r.median())))
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for (cid, kind), g in out.groupby(["contrast", "endpoint_type"]):
        out.loc[g.index, "p_holm"] = P02.holm(g.p.to_numpy())
    ab = out.endpoint_type == "absolute"
    out["p_holm_global12_absolute"] = np.nan
    out.loc[ab, "p_holm_global12_absolute"] = P02.holm(out.loc[ab, "p"].to_numpy())
    out["holm_family"] = out.contrast + " x " + out.endpoint_type + " endpoints"
    out["direction"] = [direction(r, r.endpoint_type) for _, r in out.iterrows()]
    out["interpretation"] = [interpret(r.contrast, r.endpoint, r, r.endpoint_type) for _, r in out.iterrows()]
    out.to_csv(P.TABLES / "clearance_hierarchy_contrasts.csv", index=False, float_format="%.5f")

    # ---------------------------------------------------------------- signed-error summary
    summ = []
    for run in ("h1_run0", "h1_run0_noCLr", "h2_run4"):
        for col in ("fe_auc_rel", "fe_cmax_rel"):
            v = val(run, col)
            summ.append(dict(scenario=run, endpoint=col, n=len(v), median=v.median(),
                             n_over_2fold=int((v > 1).sum()), n_under_2fold=int((v < -1).sum()),
                             n_within_2fold=int((v.abs() <= 1).sum())))
    pd.DataFrame(summ).to_csv(P.TABLES / "clearance_hierarchy_signed_summary.csv", index=False,
                              float_format="%.4f")

    # ---------------------------------------------------------------- exploratory CLR mechanism
    m = pd.read_csv(P.DATA / "compounds" / "pbpk_mechanistic_master_v3.csv").set_index("compound_id")
    v1 = pd.read_csv(P.SIM / "v1_run0" / "adme_key_inputs.csv").set_index("Drug")
    sen = {r: pd.read_csv(P.SIM / r / "clearance_sentinel.csv").set_index("Drug").CL_sim_L_h_kg
           for r in ("h1_run0", "h1_run0_noCLr")}
    mech = pd.DataFrame(index=m.index)
    mech["compound"] = m.compound
    mech["CLRbase_template_Lh"] = m.CLRbase_template_Lh
    mech["CL_obs_Lh"] = v1.CL_MetBas
    mech["CLR_frac_of_obsCL"] = mech.CLRbase_template_Lh / mech.CL_obs_Lh
    mech["clearance_group"] = m.clearance_group
    mech["CL_sim_Lhkg_h1_run0"] = sen["h1_run0"]
    mech["CL_sim_Lhkg_h1_run0_noCLr"] = sen["h1_run0_noCLr"]
    for col in ("fe_auc_rel", "fe_auc_abs", "fe_cmax_rel", "log_NRMSE"):
        mech["d_" + col] = val("h1_run0", col) - val("h1_run0_noCLr", col)
    mech["auc_rel_h1_run0"] = val("h1_run0", "fe_auc_rel")
    mech["auc_rel_h1_run0_noCLr"] = val("h1_run0_noCLr", "fe_auc_rel")
    mech.to_csv(P.TABLES / "clearance_hierarchy_clr_mechanistic.csv", float_format="%.5f")
    # a compound whose template carries no CLR must be unchanged between the two scenarios
    z = mech[mech.CLRbase_template_Lh == 0]
    assert (z.d_fe_auc_rel.abs() < 1e-9).all(), z

    sp = []
    for x in ("CLRbase_template_Lh", "CLR_frac_of_obsCL"):
        for y in ("d_fe_auc_rel", "d_fe_auc_abs"):
            rho, p = spearmanr(mech[x], mech[y])
            sp.append(dict(x=x, y=y, n=len(mech), spearman_rho=rho, p=p, subset="all 41"))
            nz = mech[mech.CLRbase_template_Lh > 0]
            rho, p = spearmanr(nz[x], nz[y])
            sp.append(dict(x=x, y=y, n=len(nz), spearman_rho=rho, p=p, subset="CLRbase > 0"))
    rl = mech[mech.clearance_group == "Renal-leaning"].d_fe_auc_rel
    ml = mech[mech.clearance_group == "Metabolism-leaning"].d_fe_auc_rel
    sp.append(dict(x="Renal-leaning vs Metabolism-leaning", y="d_fe_auc_rel",
                   n=len(rl) + len(ml), spearman_rho=np.nan,
                   p=mannwhitneyu(rl, ml).pvalue,
                   subset="median %.3f (n=%d) vs %.3f (n=%d); Mann-Whitney" % (rl.median(), len(rl),
                                                                             ml.median(), len(ml))))
    spd = pd.DataFrame(sp)
    spd.to_csv(P.TABLES / "clearance_hierarchy_spearman.csv", index=False, float_format="%.4f")

    figure(out, mech, spd)
    si_table(out)
    pd.set_option("display.width", 220)
    print(out[["contrast", "endpoint", "n", "estimate", "ci_low", "ci_high", "p", "p_holm",
               "p_holm_global12_absolute", "n_positive", "n_negative", "n_zero",
               "direction"]].round(4).to_string(index=False))
    print(pd.DataFrame(summ).round(3).to_string(index=False))
    print(spd.round(4).to_string(index=False))


# ------------------------------------------------------------------------------ SI Table S11
def pf(p):
    return "<0.001" if p < 0.001 else ("%.3f" % p if p < 0.1 else "%.2f" % p)


def f3(x):
    s = "%+.3f" % x
    return "0.000" if s in ("+0.000", "-0.000") else s


def ci3(lo, hi):
    g = lambda x: ("0.000" if x == 0 else "%.5f" % x) if abs(x) < 5e-4 else "%.3f" % x
    return "%s to %s" % (g(lo), g(hi))


def si_table(out):
    lvl = {"A": "A, primary (matched)", "B": "B, renal sensitivity", "C": "C, secondary (pragmatic)"}
    elab = {"log_NRMSE": "C–T log-NRMSE", "rmse_log10": "C–T RMSE (log₁₀)",
            "fe_auc_abs": "AUC₀₋ₜ, abs. log₂ FE", "fe_cmax_abs": "Cmax, abs. log₂ FE",
            "fe_auc_rel": "AUC₀₋ₜ, signed log₂ FE", "fe_cmax_rel": "Cmax, signed log₂ FE"}
    short = {"not detectable": "no detectable difference",
             "test lower exposure (bias shift)": "bias shifted toward lower exposure",
             "test higher exposure (bias shift)": "bias shifted toward higher exposure",
             "test larger error": "larger error", "test smaller error": "smaller error"}
    L = ["| Comparison | Endpoint | N | HL estimate | 95% CI | p (raw) | p (Holm) | + / − / 0 | Interpretation |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        d = r.direction
        it = short.get(d, "CI excludes 0, not after Holm" if d.startswith("nominal") else d)
        L.append("| %s | %s | %d | %s | %s | %s | %s | %d / %d / %d | %s |" % (
            lvl[r.contrast], elab[r.endpoint], r.n, f3(r.estimate), ci3(r.ci_low, r.ci_high), pf(r.p),
            pf(r.p_holm), r.n_positive, r.n_negative, r.n_zero, it))
    si = P.S21 / "manuscript" / "v3" / "SI"
    (si / "tables" / "S11_clearance_hierarchy.md").write_text("\n".join(L) + "\n")
    import shutil
    shutil.copy2(P.FIG_SI / "FigS_clearance_hierarchy.png", si / "figures" / "FigureS15_clearance_hierarchy.png")
    shutil.copy2(P.FIG_SI / "FigS_clearance_hierarchy.pdf", si / "figures" / "FigureS15_clearance_hierarchy.pdf")


# ------------------------------------------------------------------------------------ figure
C_CON = {"A": "#1f5fa8", "B": "#b8860b", "C": "#7a7a7a"}
MK_CON = {"A": "o", "B": "s", "C": "D"}
LAB_CON = {"A": "A  Information-matched primary\n    BU CLint (CLR = 0) − TD CL",
           "B": "B  Renal-CL sensitivity\n    BU CLint + template CLR − BU CLint (CLR = 0)",
           "C": "C  Secondary pragmatic\n    BU CLint + template CLR − TD CL"}


def figure(out, mech, spd):
    matplotlib.rcParams.update({"font.family": "sans-serif",
                                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                                "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                                "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                                "mathtext.default": "regular"})
    fig = plt.figure(figsize=(7.2, 7.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 0.10, 1.25], hspace=0.42, wspace=0.55)
    gb = gs[2, :].subgridspec(1, 2, wspace=0.42)
    ends = [("log_NRMSE", "C–T log-NRMSE"), ("rmse_log10", "C–T RMSE (log$_{10}$)"),
            ("fe_auc_abs", "AUC |log$_2$ FE|"), ("fe_cmax_abs", "C$_{max}$ |log$_2$ FE|")]
    for k, (col, lab) in enumerate(ends):
        ax = fig.add_subplot(gs[0, k])
        sub = out[out.endpoint == col].set_index("contrast")
        for i, cid in enumerate("ABC"):
            r = sub.loc[cid]
            y = 2 - i
            det = (r.ci_low > 0 or r.ci_high < 0) and r.p_holm < 0.05
            ax.plot([r.ci_low, r.ci_high], [y, y], color=C_CON[cid], lw=1.8)
            ax.plot(r.estimate, y, MK_CON[cid], ms=6.5, mfc=C_CON[cid] if det else "white",
                    mec=C_CON[cid], mew=1.5)
        ax.axvline(0, color="#333333", lw=0.8, ls="--")
        ax.set_ylim(-0.6, 2.6)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(["A", "B", "C"] if k == 0 else [])
        ax.set_title(lab, fontsize=10.5, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8.5)
        lim = max(abs(sub.ci_low.min()), abs(sub.ci_high.max())) * 1.15
        ax.set_xlim(-lim, lim)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, symmetric=True))
        ax.set_xlabel("Paired Δ (HL, 95% CI)", fontsize=8.5)
        if k == 0:
            ax.text(-0.55, 1.10, "(a)", transform=ax.transAxes, fontsize=13, fontweight="bold")
    hand = [Line2D([], [], color=C_CON[c], marker=MK_CON[c], lw=1.8, mfc=C_CON[c], mec=C_CON[c],
                   label=LAB_CON[c]) for c in "ABC"]
    axl = fig.add_subplot(gs[1, :])
    axl.axis("off")
    axl.legend(handles=hand, loc="center", ncol=3, fontsize=7.4, frameon=False, handlelength=1.6,
               columnspacing=1.0)

    # (b) paired signed AUC error, CLR = 0 -> + template CLR
    axb = fig.add_subplot(gb[0])
    d = mech.sort_values("auc_rel_h1_run0_noCLr")
    for _, r in d.iterrows():
        dv = r.auc_rel_h1_run0 - r.auc_rel_h1_run0_noCLr
        col = "#c0392b" if dv < -1e-9 else ("#2a78d6" if dv > 1e-9 else "#9a9a9a")
        axb.plot([0, 1], [r.auc_rel_h1_run0_noCLr, r.auc_rel_h1_run0], color=col, lw=0.9, alpha=0.75)
    axb.scatter(np.zeros(len(d)), d.auc_rel_h1_run0_noCLr, s=12, color="#333333", zorder=3)
    axb.scatter(np.ones(len(d)), d.auc_rel_h1_run0, s=12, color="#333333", zorder=3)
    for x, run in ((0, "auc_rel_h1_run0_noCLr"), (1, "auc_rel_h1_run0")):
        med = d[run].median()
        axb.plot([x - 0.14, x + 0.14], [med, med], color="black", lw=2.6, zorder=4)
        axb.text(x + (0.30 if x == 1 else -0.30), med, "median\n%+.2f" % med, va="center",
                 ha="left" if x == 1 else "right", fontsize=8.5)
    axb.axhspan(-1, 1, color="#efeeea", zorder=0)
    axb.axhline(0, color="#333333", lw=0.7)
    axb.set_xlim(-0.85, 1.85)
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["BU CLint\nCLR = 0", "BU CLint\n+ template CLR"], fontsize=9.5)
    axb.set_ylabel("Signed AUC log$_2$ FE (pred/obs)")
    nneg = int((mech.d_fe_auc_rel < -1e-9).sum())
    nzero = int((mech.d_fe_auc_rel.abs() <= 1e-9).sum())
    axb.set_title("Signed AUC error, paired", fontsize=11, fontweight="bold")
    axb.text(-0.22, 1.04, "(b)", transform=axb.transAxes, fontsize=13, fontweight="bold")

    # (c) CLR fraction vs change in signed AUC error
    axc = fig.add_subplot(gb[1])
    nz = mech[mech.CLRbase_template_Lh > 0]
    z = mech[mech.CLRbase_template_Lh == 0]
    grp_c = {"Renal-leaning": "#c0392b", "Metabolism-leaning": "#1f5fa8",
             "Uptake/Biliary-influenced": "#2e8b57", "Mixed/Other": "#8b8a85"}
    for g, c in grp_c.items():
        s = nz[nz.clearance_group == g]
        axc.scatter(s.CLR_frac_of_obsCL, s.d_fe_auc_rel, s=20, color=c, label=g, zorder=3,
                    edgecolor="white", linewidth=0.4)
    xz = 3e-5
    axc.scatter(np.full(len(z), xz), z.d_fe_auc_rel, s=18, marker="x", color="#555555", zorder=3,
                label="CLRbase = 0 (n = %d)" % len(z))
    axc.set_xscale("log")
    axc.axhline(0, color="#333333", lw=0.7)
    axc.set_xlabel("Template CLRbase / observed total CL")
    axc.set_ylabel("Δ signed AUC log$_2$ FE", fontsize=11)
    r = spd[(spd.x == "CLR_frac_of_obsCL") & (spd.y == "d_fe_auc_rel") & (spd.subset == "all 41")].iloc[0]
    axc.text(0.04, 0.62, "Spearman ρ = %.2f (n = %d)\nexploratory" % (r.spearman_rho, r.n),
             transform=axc.transAxes, ha="left", va="center", fontsize=8.5)
    axc.set_title("Change vs template CLR share", fontsize=11, fontweight="bold")
    axc.legend(fontsize=6.8, loc="lower left", frameon=False, handletextpad=0.2, borderaxespad=0.2)
    axc.text(-0.24, 1.04, "(c)", transform=axc.transAxes, fontsize=13, fontweight="bold")
    style.save(fig, P.FIG_SI, "FigS_clearance_hierarchy")
    plt.close(fig)


if __name__ == "__main__":
    main()
