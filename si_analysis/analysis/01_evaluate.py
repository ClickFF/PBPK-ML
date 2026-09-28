#!/usr/bin/env python3
"""Evaluate every arm once -> outputs/tables/per_compound_metrics.csv and arm_summary.csv.

All later analyses and figures read these two tables and nothing else from the simulations.

Checks before writing:
  1. AUC and Cmax fold errors reproduce the Sep20 evaluation of the same batch to 1e-9.
  2. Profile scores for the 17 arms unchanged between batches reproduce the Sep20 ct_lognrmse
     output; v0_run0 is expected to differ (it was re-simulated) and the differences are listed.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/01_evaluate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import paths as P                                                 # noqa: E402
import arms as A                                                  # noqa: E402
import metrics as M                                               # noqa: E402

P.check()
SEP20 = P.ROOT / "repo" / "claude_new_Sep20"
SEP20_EVAL = SEP20 / "model_architect" / "11_pbpk_evaluation" / "outputs" / "rerun_2026_09_20"
SEP20_P1 = SEP20 / "response_letter" / "P1" / "outputs" / "ct_lognrmse_by_compound.csv"


def load_obs():
    o = pd.read_csv(P.OBS)
    o["Conc_mgl"] = o["Conc_ng_ml"] / 1000
    o = o[o["Conc_mgl"] > 0]
    return o.sort_values(["compound_id", "Time_hr"]), o.sort_values(
        ["compound_id", "pk_curve_id", "Time_hr"])


def load_sim(run):
    s = pd.read_excel(P.SIM / run / "ct_wide_all.xlsx")
    s = s[s["mean"] > 0]
    return s.sort_values(["compound_id", "Time_hr"])


def main():
    pooled, curves = load_obs()
    m = pd.read_csv(P.MASTER)
    names = dict(zip(m.ID_trend_pbpk.astype(int), m.Name_final.str.strip()))

    frames = []
    for run in A.runs():
        frames.append(M.evaluate_arm(pooled, curves, load_sim(run), run))
        print("  evaluated %s" % run)
    d = pd.concat(frames, ignore_index=True)
    d.insert(2, "compound", d["compound_id"].map(names))

    # ---- check 1: AUC / Cmax against the Sep20 evaluation of the same batch ----------------
    worst = 0.0
    for run in A.runs():
        f = SEP20_EVAL / ("%s_evaluation_summary.csv" % run)
        s = pd.read_csv(f).set_index("compound_id").sort_index()
        n = d[d.scenario == run].set_index("compound_id").sort_index()
        for c in ("fe_auc_rel", "fe_cmax_rel"):
            worst = max(worst, float(np.nanmax(np.abs(s[c].values - n[c].values))))
    print("\nAUC/Cmax fold errors vs Sep20 evaluation: worst disagreement %.2e" % worst)
    if worst > 1e-9:
        raise SystemExit("AUC/Cmax do not reproduce the evaluation")

    # ---- check 2: profile scores against the Sep20 P1 output (old batch) --------------------
    old = pd.read_csv(SEP20_P1)
    diff_rows = []
    for run in A.runs():
        o = old[old.scenario == run].set_index("compound_id")
        n = d[d.scenario == run].set_index("compound_id")
        both = o.index.intersection(n.index)
        dv = (n.loc[both, "log_NRMSE"] - o.loc[both, "log_NRMSE"]).abs()
        dr = (n.loc[both, "rmse_log10"] - o.loc[both, "rmse_log10"]).abs()
        bad = sorted(set(dv[dv > 1e-4].index) | set(dr[dr > 1e-4].index))
        diff_rows.append((run, len(bad), bad))
    for run, k, bad in diff_rows:
        if k:
            print("  profile scores differ from the Sep20 (old-batch) output for %s: %s"
                  % (run, ", ".join("%s (%s)" % (names.get(int(b), b), b) for b in bad)))
    unexpected = [r for r, k, _ in diff_rows if k and r != "v0_run0"]
    if unexpected:
        raise SystemExit("profile scores changed for arms that were not re-simulated: %s"
                         % unexpected)
    print("profile scores: all 17 unchanged arms reproduce the Sep20 output; "
          "v0_run0 differs only where re-simulated")

    d.to_csv(P.TABLES / "per_compound_metrics.csv", index=False)

    # ---- arm summary ------------------------------------------------------------------------
    grp = {r: g for r, _, _, g in A.ARMS}
    rows = []
    for run in A.runs():
        x = d[d.scenario == run]
        rows.append(dict(
            scenario=run, group=grp.get(run),
            n=len(x),
            log_NRMSE_n=int(x.log_NRMSE.notna().sum()),
            log_NRMSE_median=x.log_NRMSE.median(), log_NRMSE_mean=x.log_NRMSE.mean(),
            rmse_log10_median=x.rmse_log10.median(), rmse_log10_mean=x.rmse_log10.mean(),
            rmse_log10_tw_median=x.rmse_log10_timeweighted.median(),
            auc_log2_signed_median=x.fe_auc_rel.median(), auc_log2_abs_median=x.fe_auc_abs.median(),
            auc_within_2fold=x.auc_within_2fold.mean(), auc_within_3fold=x.auc_within_3fold.mean(),
            cmax_log2_signed_median=x.fe_cmax_rel.median(),
            cmax_log2_signed_mean=x.fe_cmax_rel.mean(),
            cmax_log2_abs_median=x.fe_cmax_abs.median(),
            cmax_within_2fold=x.cmax_within_2fold.mean(), cmax_within_3fold=x.cmax_within_3fold.mean(),
        ))
    s = pd.DataFrame(rows)
    s.to_csv(P.TABLES / "arm_summary.csv", index=False, float_format="%.4f")
    show = s[s.group.isin(["A", "B"])][["scenario", "log_NRMSE_n", "log_NRMSE_median",
                                        "rmse_log10_median", "auc_log2_abs_median",
                                        "auc_within_2fold", "cmax_log2_signed_mean",
                                        "cmax_within_2fold"]]
    print("\n" + show.round(3).to_string(index=False))
    print("\nwrote per_compound_metrics.csv (%d rows) and arm_summary.csv" % len(d))


if __name__ == "__main__":
    main()
