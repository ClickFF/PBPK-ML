# -*- coding: utf-8 -*-
"""Every exposure metric used in claude_new_Sep21, defined once.

Two families, and which endpoint each applies to is part of the definition (Methods, author's
decision 2026-09-21):

  CONCENTRATION-TIME PROFILE  ->  log-NRMSE (primary) and RMSE in log10 concentration
      A profile is a curve, not a single ratio, so it is scored by the dispersion of its log
      residuals. log-NRMSE divides that dispersion by the curve's own observed log10 range so
      that compounds with short and long concentration spans share one scale. A signed or
      absolute log2 "fold error" is NOT computed for profiles: a mean of pointwise log ratios
      lets positive and negative deviations cancel and weights densely sampled phases.

  AUC AND Cmax  ->  log2 fold error, log2(predicted / observed), and within-k-fold coverage
      Both are single quantities per compound, for which a fold error is the natural measure.

Sources, copied rather than reimplemented:

  * `calc_auc` and `evaluate_one_compound` are verbatim from cell 1 of
    Table3_data/pbpk evaluation/eval supplemental.ipynb, the code behind the published
    evaluation. They define AUC (linear-up / log-down trapezoid over the observed sampling
    window, AUC_0-tlast) and Cmax (maximum of the matched points), both on a compound's pooled
    curves and neither dose-normalised. Their `rel_log2`/`abs_log2` profile outputs are carried
    for traceability only and are never reported.
  * `curve_scores` is the per-curve profile scoring of response_letter/P1/scripts/ct_lognrmse.py:
    PCHIP prediction at the observed times, >= 3 usable points, RMSE of log10 residuals, and
    log-NRMSE = RMSE / observed log10 range with curves below 0.2 log10 units excluded; curve
    scores are averaged within compound and compounds weighted equally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

MIN_POINTS = 3          # a curve needs at least this many usable points
MIN_RANGE = 0.2         # log10 units; below this the NRMSE denominator is not informative


# ==============================================================================================
# BEGIN VERBATIM COPY — eval supplemental.ipynb, code cell 1. Do not modify.
# ==============================================================================================

# ============================================================
# Helper: AUC with linear-up/log-down rule
# ============================================================
def calc_auc(time, conc):
    auc = 0.0
    for i in range(len(time) - 1):
        t1, t2 = time[i], time[i+1]
        c1, c2 = conc[i], conc[i+1]

        if c2 > c1:  # linear up
            auc += (c1 + c2) / 2 * (t2 - t1)
        else:        # log down
            if c1 > 0 and c2 > 0 and c1 != c2:
                auc += (c1 - c2) / np.log(c1 / c2) * (t2 - t1)
            else:
                auc += (c1 + c2) / 2 * (t2 - t1)
    return auc


# ============================================================
# Helper: Evaluate one compound
# folds: tuple/list of fold thresholds for coverage metrics
# ============================================================
def evaluate_one_compound(obs_df_c, sim_df_c, folds=(2, 3, 5, 10)):

    cid = obs_df_c["compound_id"].iloc[0]

    # ---------- interpolate simulated → observed times ----------
    f = PchipInterpolator(sim_df_c["Time_hr"], sim_df_c["mean"])
    pred = f(obs_df_c["Time_hr"].values)

    obs = obs_df_c["Conc_mgl"].values
    time = obs_df_c["Time_hr"].values

    # Remove non-positive
    mask = (pred > 0) & (obs > 0)
    pred = pred[mask]
    obs = obs[mask]
    time = time[mask]

    # ---------- point-wise log2 errors ----------
    log2_ratio = np.log2(pred / obs)
    rel_log2 = np.mean(log2_ratio)
    abs_log2 = np.mean(np.abs(log2_ratio))

    # ---------- Cmax ----------
    cmax_pred = np.max(pred)
    cmax_obs  = np.max(obs)
    fe_cmax_rel = np.log2(cmax_pred / cmax_obs)     # direction (bias)
    fe_cmax_abs = np.abs(fe_cmax_rel)               # magnitude

    # ---------- AUC ----------
    auc_pred = calc_auc(time, pred)
    auc_obs  = calc_auc(time, obs)
    fe_auc_rel = np.log2(auc_pred / auc_obs)
    fe_auc_abs = np.abs(fe_auc_rel)

    # ---------- Tmax (方向性差异即可) ----------
    tmax_pred = time[np.argmax(pred)]
    tmax_obs  = time[np.argmax(obs)]
    delta_tmax = tmax_pred - tmax_obs

    # ---------- Within X-fold coverage metrics ----------
    # 使用 absolute log2 error 来判断是否在 X-fold 内
    coverage = {}
    abs_log2_point = np.abs(log2_ratio)

    for fold in folds:
        thr = np.log2(fold)

        # 1) 时间点 coverage：% time points within X-fold
        pct_time_within = np.mean(abs_log2_point <= thr)  # 0–1 之间
        coverage[f"time_within_{fold}fold"] = pct_time_within

        # 2) Cmax coverage：compound-level 0/1
        coverage[f"cmax_within_{fold}fold"] = float(fe_cmax_abs <= thr)

        # 3) AUC coverage：compound-level 0/1
        coverage[f"auc_within_{fold}fold"] = float(fe_auc_abs <= thr)

    # ---------- assemble result ----------
    result = {
        "compound_id": cid,
        "rel_log2": rel_log2,
        "abs_log2": abs_log2,
        "fe_cmax_rel": fe_cmax_rel,
        "fe_cmax_abs": fe_cmax_abs,
        "fe_auc_rel": fe_auc_rel,
        "fe_auc_abs": fe_auc_abs,
        "delta_tmax": delta_tmax,
    }

    # 把 coverage 字段并进 result
    result.update(coverage)
    return result

# ==============================================================================================
# END VERBATIM COPY
# ==============================================================================================


def exposure_values(obs_df_c, sim_df_c):
    """The AUC and Cmax values behind evaluate_one_compound's fold errors, same matching."""
    pred = PchipInterpolator(sim_df_c["Time_hr"], sim_df_c["mean"])(obs_df_c["Time_hr"].values)
    obs = obs_df_c["Conc_mgl"].values
    time = obs_df_c["Time_hr"].values
    m = (pred > 0) & (obs > 0)
    pred, obs, time = pred[m], obs[m], time[m]
    return dict(auc_obs=calc_auc(time, obs), auc_pred=calc_auc(time, pred),
                cmax_obs=float(np.max(obs)), cmax_pred=float(np.max(pred)),
                n_points=int(m.sum()), t_last=float(time.max()))


def curve_scores(obs, sim, scenario):
    """Per-curve profile scores for one arm (ct_lognrmse.py, unchanged)."""
    sim_by = {c: g for c, g in sim.groupby("compound_id")}
    rows = []
    for (cid, curve), g in obs.groupby(["compound_id", "pk_curve_id"]):
        sg = sim_by.get(cid)
        if sg is None or len(sg) < 2:
            continue
        t = g["Time_hr"].to_numpy(float)
        o = g["Conc_mgl"].to_numpy(float)
        try:
            pred = PchipInterpolator(sg["Time_hr"].to_numpy(float), sg["mean"].to_numpy(float))(t)
        except Exception:
            continue
        ok = np.isfinite(pred) & (pred > 0) & np.isfinite(o) & (o > 0)
        if ok.sum() < MIN_POINTS:
            continue
        lo, lp = np.log10(o[ok]), np.log10(pred[ok])
        tt = t[ok]
        rng = float(lo.max() - lo.min())
        e2 = (lp - lo) ** 2
        rmse = float(np.sqrt(np.mean(e2)))
        if len(tt) >= 2:
            edges = np.concatenate(([tt[0]], (tt[1:] + tt[:-1]) / 2.0, [tt[-1]]))
            w = np.diff(edges)
            w = w / w.sum() if w.sum() > 0 else np.full(len(tt), 1.0 / len(tt))
        else:
            w = np.full(len(tt), 1.0 / max(len(tt), 1))
        rmse_tw = float(np.sqrt(np.sum(w * e2)))
        rows.append(dict(scenario=scenario, compound_id=cid, pk_curve_id=curve,
                         n_points=int(ok.sum()), obs_log_range=rng,
                         rmse_log10=rmse, rmse_log10_timeweighted=rmse_tw,
                         log_NRMSE=rmse / rng if rng >= MIN_RANGE else np.nan,
                         excluded="" if rng >= MIN_RANGE
                         else "observed log range < %.1f" % MIN_RANGE))
    return pd.DataFrame(rows)


def profile_scores(curves):
    """Compound-level profile scores: curve scores averaged within compound."""
    valid = curves.dropna(subset=["log_NRMSE"])
    a = (valid.groupby("compound_id")
         .agg(log_NRMSE=("log_NRMSE", "mean"), n_curves_nrmse=("log_NRMSE", "size"))
         .reset_index())
    b = (curves.groupby("compound_id")
         .agg(rmse_log10=("rmse_log10", "mean"),
              rmse_log10_timeweighted=("rmse_log10_timeweighted", "mean"),
              n_curves=("rmse_log10", "size"), n_points_profile=("n_points", "sum"))
         .reset_index())
    return b.merge(a, on="compound_id", how="left")


def evaluate_arm(obs_pooled, obs_curves, sim, scenario):
    """One row per compound: profile scores + AUC/Cmax fold errors + AUC/Cmax values."""
    rows = []
    obs_use = obs_pooled[obs_pooled["compound_id"].isin(sim["compound_id"].unique())]
    for cid, g in obs_use.groupby("compound_id"):
        sg = sim[sim["compound_id"] == cid]
        r = evaluate_one_compound(g, sg)
        r.update(exposure_values(g, sg))
        rows.append(r)
    ev = pd.DataFrame(rows)
    prof = profile_scores(curve_scores(obs_curves, sim, scenario))
    out = ev.merge(prof, on="compound_id", how="left")
    out.insert(0, "scenario", scenario)
    return out
