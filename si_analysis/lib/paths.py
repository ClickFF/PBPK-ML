# -*- coding: utf-8 -*-
"""Every input and output location in claude_new_Sep21. Nothing here points outside this tree.

The data under data/ are sealed copies:
  simulation_output_batch/  the authoritative 18-arm batch of 2026-09-20 (v0 corrected:
                            phenobarbital CL_IV 0.31 L/h at 1.857 mg/kg IV; posaconazole back to
                            the library default model). Byte-identical to
                            claude_new_Sep20/model_architect/11_pbpk_evaluation/simulation_output_batch.
  observed/                 observed_data_cleaned_deduplicated.csv, as used by the evaluation.
  compounds/                pbpk_physchem_mechanistic_master.csv: names, observed Fu/VDss/CLsys.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parent
S21 = LIB.parent
DATA = S21 / "data"
SIM = DATA / "simulation_output_batch"
OBS = DATA / "observed" / "observed_data_cleaned_deduplicated.csv"
MASTER = DATA / "compounds" / "pbpk_physchem_mechanistic_master.csv"

OUT = S21 / "outputs"
TABLES = OUT / "tables"
FIG_MAIN = OUT / "figures" / "main"
FIG_SI = OUT / "figures" / "si"
MS = S21 / "manuscript" / "v2"

ROOT = S21
while ROOT != ROOT.parent and not (ROOT / "Table3_data").is_dir():
    ROOT = ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))
sys.path.insert(0, str(LIB))

for d in (TABLES, FIG_MAIN, FIG_SI):
    d.mkdir(parents=True, exist_ok=True)


def check():
    missing = [str(p) for p in (SIM, OBS, MASTER) if not p.exists()]
    if missing:
        raise SystemExit("missing sealed input(s):\n  " + "\n  ".join(missing))
