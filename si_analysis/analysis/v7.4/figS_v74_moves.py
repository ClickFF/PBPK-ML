#!/usr/bin/env python3
"""v7.4 SI assembly (Round 4).

1. Copies the Round 3 SI figure set (already CLsys-relabelled) into manuscript/v7.4/SI/figure.
2. Renders the ML compound-paired benchmark forest into the SI: in Round 4 the main text keeps the original
   v7.2 ML displays, so all ML paired statistics belong in the SI (v4 fig_ml_forest, main output redirected).
3. Renders the parameter-to-exposure correlation heatmap as an SI figure: it leaves the Round 3 main-text
   propagation figure, which Round 4 does not carry.
4. Copies Figure 1 (executed workflow) from v4 into manuscript/v7.4/figure.

Nothing is recomputed; the heatmap reads outputs/tables/decomposition_routing.csv.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v7.4/figS_v74_moves.py
"""
import re
import runpy
import shutil
import sys
import types
from pathlib import Path

import pandas as pd

S21 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S21 / "lib"))
import matplotlib.text                                            # noqa: E402
import style_v74 as S                                             # noqa: E402

S.use()
V73_SI = S21 / "manuscript" / "v7.3" / "SI" / "figures"
V4_FIG = S21 / "manuscript" / "v4" / "figures"
BARE = re.compile(r"(?<!renal )(?<!Renal )\bCL\b(?![A-Za-z_$])")


def copy_round3_si():
    dst = S.OUT["si"]
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(V73_SI.iterdir()):
        if f.suffix in (".png", ".pdf", ".svg"):
            shutil.copy2(f, dst / f.name)
            n += 1
    return n


def copy_figure1():
    dst = S.OUT["main"]
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf", "svg"):
        src = V4_FIG / ("Figure1_workflow.%s" % ext)
        if src.exists():
            shutil.copy2(src, dst / ("Figure1_workflow.%s" % ext))
            out.append(src.name)
    return out


def ml_forest_to_si():
    """Run the v4 ML forest script with `main` redirected to an SI figure."""
    log = []

    def save(fig, where, name):
        if where == "main":
            name = "FigureS_ml_paired_forest"
        fig.canvas.draw()
        for t in fig.findobj(matplotlib.text.Text):
            s = t.get_text()
            if s:
                n = BARE.sub("CL$_{sys}$", s)
                if n != s:
                    t.set_text(n)
                    log.append(dict(figure=name, before=s, after=n))
        return S.save(fig, "si", name)

    shim = types.ModuleType("style_v4")
    shim.__dict__.update({k: v for k, v in vars(S).items() if not k.startswith("__")})
    shim.save = save
    sys.modules["style_v4"] = shim
    script = S21 / "analysis" / "v4" / "fig_ml_forest.py"
    argv = sys.argv
    sys.argv = [str(script)]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = argv
    return pd.DataFrame(log)


def parameter_heatmap():
    rout = pd.read_csv(S.TABLES / "decomposition_routing.csv")
    ex = ["AUC, signed log2 FE", "Cmax, signed log2 FE", "C–T, RMSE log10", "C–T, log-NRMSE"]
    pars = ["Fu", "VDss", "CLsys"]
    rho = rout.pivot(index="parameter", columns="exposure", values="rho").loc[pars, ex]
    pv = rout.pivot(index="parameter", columns="exposure", values="p").loc[pars, ex]
    fig = S.figure(3.6, 2.6)
    ax = fig.add_axes([0.20, 0.26, 0.58, 0.62])
    im = ax.imshow(rho.abs().to_numpy(), cmap=S.HEATMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(len(pars)):
        for j in range(len(ex)):
            v, p = rho.iloc[i, j], pv.iloc[i, j]
            ax.text(j, i, "%+.2f" % v + ("" if p < 0.05 else "\nn.s."), ha="center", va="center", fontsize=6.6,
                    color="white" if abs(v) < 0.55 else "black")
    ax.set_xticks(range(len(ex)))
    ax.set_xticklabels(["AUC$_{0–t}$", "C$_{max}$", "C–T RMSE", "C–T log-NRMSE"], fontsize=6.6, rotation=35,
                       ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(pars)))
    ax.set_yticklabels(["f$_u$", "VD$_{ss}$", "CL$_{sys}$"])
    ax.set_ylabel("DL–ML input fold error")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
    cb.set_label("|Spearman ρ|", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_linewidth(0.5)
    S.save(fig, "si", "FigureS_parameter_to_exposure")
    return rho


def main():
    n = copy_round3_si()
    f1 = copy_figure1()
    log = ml_forest_to_si()
    rho = parameter_heatmap()
    out = S.OUT_TABLES
    out.mkdir(parents=True, exist_ok=True)
    if len(log):
        log.to_csv(out / "si_forest_relabel_log.csv", index=False)
    print("copied Round 3 SI files:", n)
    print("Figure 1 copied:", f1)
    print("forest relabels:", len(log))
    print(rho.round(2).to_string())


if __name__ == "__main__":
    main()
