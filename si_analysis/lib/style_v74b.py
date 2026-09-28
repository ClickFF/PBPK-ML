# -*- coding: utf-8 -*-
"""v7.4 Round 4B figure style. Identical to the Round 4 (v7.4) style except for the output tree.

Round 4B writes to manuscript/v7.4/figures_round4b, manuscript/v7.4/SI/figures_round4b and
outputs/tables/v7.4_round4b, so that the Round 4 (and Round 3) artifacts stay intact.
"""
from __future__ import annotations

from pathlib import Path

import style_v74 as _V74
from style_v74 import (  # noqa: F401  (re-exported unchanged)
    BU, CLINT, CLSYS, COLOR, CORE, DLML, DOUBLE, ENDPOINT_LABEL, FU, FULL, GREY, HEATMAP, LABEL, LIGHT,
    MARK, MINIMAL, MISSING, OBS, PAGE_H, REF_LABEL, SHORT, SINGLE, SPLUS, SUPPORTING, TD, VDSS,
    figure, filled, panel_label, pfmt, plt, use,
)

LIB = Path(__file__).resolve().parent
S21 = LIB.parent
V74 = S21 / "manuscript" / "v7.4"
OUT = {"main": V74 / "figures_round4b", "si": V74 / "SI" / "figures_round4b"}
TABLES = _V74.TABLES
OUT_TABLES = TABLES / "v7.4_round4b"


def save(fig, where, name):
    d = OUT[where]
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(d / ("%s.%s" % (name, ext)))
    plt.close(fig)
    return d / ("%s.png" % name)
