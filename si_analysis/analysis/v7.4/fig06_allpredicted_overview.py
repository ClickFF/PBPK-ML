#!/usr/bin/env python3
"""v7.4 Figure 6 (Round 4): end-to-end all-predicted workflow, returning to the structure of v4 Figure 8.

The v4 Figure 8 script is executed unchanged, with `style_v4` bound to a shim built on style_v74 so that the
figure is written to manuscript/v7.4/figure/Figure6_allpredicted_overview.* and the systemic-clearance symbol is
CLsys. Panels are those of v4 Figure 8: (a-c) observed vs predicted dose-normalised concentrations, AUC0-t and
Cmax; (d) per-compound AUC and Cmax fold-error strips ranked by AUC error; (e-g) profiles of the most
underpredicted, median and most overpredicted AUC compounds.

No statistic is recomputed: the coverage table the v4 script writes is compared with the archived
outputs/tables/v4_figure8_gof.csv and the run fails if it differs.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v7.4/fig06_allpredicted_overview.py
"""
import re
import runpy
import sys
import types
from pathlib import Path

S21 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S21 / "lib"))
import matplotlib.text                                            # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
import style_v74 as V74                                           # noqa: E402

PHRASES = [("DL–ML total CL", "DL–ML CL$_{sys}$"), ("total CL", "CL$_{sys}$"), ("systemic CL", "CL$_{sys}$")]
BARE = re.compile(r"(?<!renal )(?<!Renal )\bCL\b(?![A-Za-z_$])")
LOG = []


def clsys(s):
    t = s
    for a, b in PHRASES:
        t = t.replace(a, b)
    return BARE.sub("CL$_{sys}$", t)


def save(fig, where, name):
    if name != "Figure8_dlml_overview":
        V74.plt.close(fig)
        return None
    fig.canvas.draw()
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if s:
            n = clsys(s)
            if n != s:
                t.set_text(n)
                LOG.append(dict(before=s, after=n))
    return V74.save(fig, "main", "Figure6_allpredicted_overview")


def main():
    before = pd.read_csv(V74.TABLES / "v4_figure8_gof.csv", index_col=0)
    shim = types.ModuleType("style_v4")
    shim.__dict__.update({k: v for k, v in vars(V74).items() if not k.startswith("__")})
    shim.save = save
    sys.modules["style_v4"] = shim
    script = S21 / "analysis" / "v4" / "fig08_dlml_overview.py"
    argv = sys.argv
    sys.argv = [str(script)]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = argv
    after = pd.read_csv(V74.TABLES / "v4_figure8_gof.csv", index_col=0)
    assert np.allclose(before.to_numpy(float), after.to_numpy(float), atol=1e-6), "v4 Figure 8 values changed"
    out = V74.OUT_TABLES
    out.mkdir(parents=True, exist_ok=True)
    after.to_csv(out / "figure6_gof.csv", float_format="%.4f")
    pd.DataFrame(LOG).drop_duplicates().to_csv(out / "figure6_relabel_log.csv", index=False)
    print(after.round(3).to_string())
    print("relabelled:", len(pd.DataFrame(LOG).drop_duplicates()))


if __name__ == "__main__":
    main()
