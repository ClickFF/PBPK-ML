#!/usr/bin/env python3
"""Paired contrasts within each comparison group -> outputs/tables/paired_contrasts.csv

Method carried over unchanged from response_letter/P1/scripts/paired_stats_all_endpoints.py:
Hodges-Lehmann estimate of the paired difference (test - reference, so positive = the test arm
has the larger error), 95% percentile bootstrap over compounds (10 000 resamples, fixed seed),
two-sided Wilcoxon signed-rank p. Added: Holm adjustment within each group x endpoint family, and
the within-2-fold counts as descriptive paired tallies.

Endpoints (Methods): C-T profile by log-NRMSE (n = 40) and RMSE in log10 (n = 41); AUC and Cmax
by absolute log2 fold error (n = 41).

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/02_paired.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import paths as P                                                 # noqa: E402
import arms as A                                                  # noqa: E402

N_BOOT = 10000
RNG = np.random.default_rng(20260920)


def hodges_lehmann(d):
    i, j = np.triu_indices(len(d), k=0)
    return float(np.median((d[i] + d[j]) / 2.0))


def contrast(t, r):
    j = pd.concat([t.rename("t"), r.rename("r")], axis=1).dropna()
    diff = (j["t"] - j["r"]).to_numpy(float)
    n = len(diff)
    est = hodges_lehmann(diff)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    boot = np.array([hodges_lehmann(diff[k]) for k in idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(j["t"], j["r"]).pvalue)
    return dict(n=n, estimate=est, ci_low=float(lo), ci_high=float(hi), p=p,
                n_worse=int(np.sum(diff > 0)), n_better=int(np.sum(diff < 0)))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(o):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    d = pd.read_csv(P.TABLES / "per_compound_metrics.csv")
    val = lambda run, col: d[d.scenario == run].set_index("compound_id")[col]

    rows = []
    for g in ("A", "B"):
        ref_id = A.REFERENCE[g]
        ref = A.run_of(ref_id)
        tests = [(r, n, l) for r, n, l in A.group(g) if n != ref_id]
        for col, elab, fam, _ in A.ENDPOINTS:
            block = []
            for run, nid, lab in tests:
                c = contrast(val(run, col), val(ref, col))
                block.append(dict(group=g, arm_id=nid, run_id=run, label=lab,
                                  reference=ref, endpoint=col, endpoint_label=elab,
                                  family=fam, **c))
            adj = holm([b["p"] for b in block])
            for b, a in zip(block, adj):
                b["p_holm"] = float(a)
                b["detectable"] = bool(b["ci_low"] > 0 or b["ci_high"] < 0)
                b["direction"] = ("worse" if b["ci_low"] > 0 else
                                  "better" if b["ci_high"] < 0 else "none")
            rows += block
            print("  group %s, %s" % (g, col))
    # direct strategy contrasts, referenced to top-down CLsys (h2_run4): is either
    # parameterisation detectably better than the other?
    for col, elab, fam, _ in A.ENDPOINTS:
        block = []
        for run, lab in (("h1_run0_noCLr", "bottom-up CLint (CLR = 0) vs top-down CLsys"),
                         ("h1_run0", "bottom-up CLint (template CLR) vs top-down CLsys"),
                         ("h2_run0", "all-ML inputs vs top-down CLsys")):
            c = contrast(val(run, col), val("h2_run4", col))
            block.append(dict(group="C", arm_id="C", run_id=run, label=lab, reference="h2_run4",
                              endpoint=col, endpoint_label=elab, family=fam, **c))
        adj = holm([b["p"] for b in block])
        for b, a in zip(block, adj):
            b["p_holm"] = float(a)
            b["detectable"] = bool(b["ci_low"] > 0 or b["ci_high"] < 0)
            b["direction"] = ("worse" if b["ci_low"] > 0 else
                              "better" if b["ci_high"] < 0 else "none")
        rows += block
    out = pd.DataFrame(rows)
    out.to_csv(P.TABLES / "paired_contrasts.csv", index=False, float_format="%.5f")

    # within-2-fold tallies: compounds that crossed the 2-fold line relative to the reference
    tal = []
    for g in ("A", "B"):
        ref = A.run_of(A.REFERENCE[g])
        for run, nid, lab in A.group(g):
            if run == ref:
                continue
            for col in ("auc_within_2fold", "cmax_within_2fold"):
                t, r = val(run, col), val(ref, col)
                tal.append(dict(group=g, run_id=run, label=lab, endpoint=col,
                                test_pct=100 * t.mean(), ref_pct=100 * r.mean(),
                                n_lost=int(((r == 1) & (t == 0)).sum()),
                                n_gained=int(((r == 0) & (t == 1)).sum())))
    pd.DataFrame(tal).to_csv(P.TABLES / "within_2fold_paired_counts.csv", index=False,
                             float_format="%.1f")

    show = out[["group", "run_id", "endpoint", "n", "estimate", "ci_low", "ci_high", "p",
                "p_holm", "direction"]]
    print("\n" + show.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
