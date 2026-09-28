#!/usr/bin/env python3
"""Main-text Figure 2, draft_02: dataset distribution + training and held-out performance.

Three rows (f_u, CL_sys, VD_ss) x three columns:
  col 1  distribution of the observed log10 values, training against held-out  -> panels (a), (d), (g)
  col 2  observed versus predicted, training set                               -> panels (b), (e), (h)
  col 3  observed versus predicted, held-out benchmark set (Test #1)           -> panels (c), (f), (i)

Columns 2 and 3 reproduce the frozen Round 4B Figure 2 exactly (same data, same statistics, same styling);
column 1 is new and restores the dataset-distribution panels the v7.2 Results paragraph refers to, so the
claim about dynamic range and about train/test comparability is supported by a display again. The observed
axis is shared across the three panels of a row, so the distribution lines up with the scatters beside it.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v7.4_final01/fig02_train_test_3x3.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S21 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S21 / "lib"))
import style_draft02 as S                                         # noqa: E402
import paths as P                                                 # noqa: E402

S.use()
RES = "ML+training-raw/DL+ML_trainV4/pred_res/res_v1/res_%s.csv"
ENDPOINTS = [("Fu", "lgFu_train", "lgFu_test", "Plasma unbound fraction (f$_u$)", 4042),
             ("CL", "lgCL_train", "lgCL_test", "Systemic clearance (CL$_{sys}$)", 1287),
             ("VDss", "lgVD_train", "lgVD_test", "Volume of distribution (VD$_{ss}$)", 1287)]
L2, L3 = np.log10(2), np.log10(3)
TRAIN, TEST = S.GREY, S.DLML


def load(stem, tr, te):
    df = pd.read_csv(P.ROOT / (RES % stem))
    return {k: df[df.Dataset.astype(str) == tok][["Actual", "Predicted"]].dropna()
            for k, tok in (("train", tr), ("test", te))}


def label(ax, letter):
    """Panel label a fixed distance from the axes corner, in points, so all nine align the same way."""
    ax.annotate("(%s)" % letter, xy=(0, 1), xycoords="axes fraction", xytext=(-30, 8),
                textcoords="offset points", fontsize=9, fontweight="bold", ha="left", va="bottom",
                annotation_clip=False)


def r2(y, p):
    return 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    fig, axes = S.plt.subplots(3, 3, figsize=(S.DOUBLE, 7.3))
    letters = "abcdefghi"
    rows = []
    for ri, (key, tr, te, name, n_cur) in enumerate(ENDPOINTS):
        d = load("VDss" if key == "VDss" else key, tr, te)
        v = np.concatenate([np.r_[f.Actual, f.Predicted] for f in d.values()])
        lo, hi = v.min(), v.max()
        lo, hi = lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)
        g = np.array([lo, hi])

        # ---- column 1: observed-value distributions
        ax = axes[ri, 0]
        obs_tr = d["train"].Actual.to_numpy(float)
        obs_te = d["test"].Actual.to_numpy(float)
        bins = np.linspace(lo, hi, 34)
        peak = 0.0
        for vals, col, lab in ((obs_tr, TRAIN, "Training"), (obs_te, TEST, "Held-out")):
            ax.hist(vals, bins=bins, density=True, color=col, alpha=0.38, lw=0)
            ax.hist(vals, bins=bins, density=True, histtype="step", color=col, lw=0.9)
            peak = max(peak, float(np.histogram(vals, bins=bins, density=True)[0].max()))
        # headroom so the counts and the colour key sit clear of the tallest bars
        ax.set_ylim(0, peak * 1.46)
        rng = np.r_[obs_tr, obs_te]
        p5, p95 = np.percentile(rng, [5, 95])
        ax.text(0.03, 0.97, ("training N = %d\nheld-out N = %d\nspan %.1f log$_{10}$ units\n"
                             "(central 90%%, %.1f)" % (len(obs_tr), len(obs_te),
                                                       rng.max() - rng.min(), p95 - p5)
                             ).replace("-", "\u2212"),
                transform=ax.transAxes, ha="left", va="top", fontsize=6.6)
        rows.append(dict(endpoint=key, set="observed range", N=len(rng), R2=np.nan,
                         RMSE=np.nan, GMFE=np.nan, FE2=np.nan,
                         span_log10=rng.max() - rng.min(), span_central90=p95 - p5))
        ax.set_xlim(lo, hi)
        ax.set_ylabel("%s\nDensity" % name)
        if ri == 0:
            ax.set_title("Data distribution")
            ax.text(0.97, 0.97, "Training set", transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.6, color=TRAIN, fontweight="bold")
            ax.text(0.97, 0.90, "Held-out test set", transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.6, color=TEST, fontweight="bold")
        if ri == 2:
            ax.set_xlabel("Observed log$_{10}$ value")
        label(ax, letters[ri * 3])

        # ---- columns 2 and 3: observed versus predicted
        for ci, (title, part, col) in enumerate((("Training set", "train", TRAIN),
                                                 ("Held-out test set", "test", TEST)), start=1):
            ax = axes[ri, ci]
            y, p = d[part].Actual.to_numpy(float), d[part].Predicted.to_numpy(float)
            e = p - y
            m = dict(N=len(y), R2=r2(y, p), RMSE=float(np.sqrt(np.mean(e ** 2))),
                     GMFE=float(10 ** np.mean(np.abs(e))), FE2=100 * float(np.mean(np.abs(e) <= L2)))
            rows.append(dict(endpoint=key, set=title, **m))
            ax.fill_between(g, g - L2, g + L2, color=S.LIGHT, lw=0, zorder=0)
            ax.plot(g, g, color=S.OBS, lw=0.7, zorder=2)
            for s in (L3, -L3):
                ax.plot(g, g + s, color=S.GREY, lw=0.5, ls=":", zorder=2)
            big = len(y) > 1000
            ax.scatter(y, p, s=3 if big else 6, color=col, alpha=0.35 if big else 0.7, lw=0,
                       zorder=3, rasterized=big)
            n_txt = "N = %d" % len(y) if (part == "test" or n_cur == len(y)) else \
                    "N = %d (%d plotted)" % (n_cur, len(y))
            ax.text(0.04, 0.96, "%s\nR$^2$ = %.2f\nRMSE = %.2f\nWithin 2-fold = %.0f%%"
                    % (n_txt, m["R2"], m["RMSE"], m["FE2"]),
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.6)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            if ri == 0:
                ax.set_title(title)
            if ri == 2:
                ax.set_xlabel("Observed log$_{10}$ value")
            if ci == 1:
                ax.set_ylabel("Predicted log$_{10}$ value")
            label(ax, letters[ri * 3 + ci])

    fig.tight_layout(w_pad=1.1, h_pad=1.3)
    S.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(S.OUT_TABLES / "figure2_train_test_metrics_draft02.csv", index=False, float_format="%.4f")
    S.save(fig, "main", "Figure2_ml_train_test")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
