#!/usr/bin/env python3
"""Write the per-endpoint compound identifiers for each of the four data splits.

The paper uses two nested partitions and it is easy to conflate them:

  Training Set #1 / Test Set #1   model development and the held-out benchmark (Table 2)
  Training Set #2 / Test Set #2   the refit used for the PBPK work, with Test #2 held out

Test Set #2 is the 110 concentration-time compounds; Training Set #2 is everything else that carries
a label for that endpoint. Only 90 of the 110 have a measured fraction unbound, which is why the fu
row of Table 1 reads 4585 / 90 while clearance and volume read 1354 / 110.

Run from the repository root:

    python ml/scripts/make_splits.py

Writes ml/splits/{fu,CLsys,VDss}_{train1,test1,train2,test2}.csv and ml/splits/split_set_sizes.csv.
Requires pandas and openpyxl.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
S1 = ROOT / "data" / "S1 ML inputs and pred.xlsx"
OUT = ROOT / "ml" / "splits"

# endpoint -> (label column, split column in the benchmarking sheet)
ENDPOINTS = {"fu": ("lgFu", "Fu_train_test"),
             "CLsys": ("lgCL", "CL_train_test"),
             "VDss": ("lgVD", "VD_train_test")}

EXPECTED = {("fu", "train1"): 4042, ("fu", "test1"): 633,
            ("fu", "train2"): 4585, ("fu", "test2"): 90,
            ("CLsys", "train1"): 1287, ("CLsys", "test1"): 177,
            ("CLsys", "train2"): 1354, ("CLsys", "test2"): 110,
            ("VDss", "train1"): 1287, ("VDss", "test1"): 177,
            ("VDss", "train2"): 1354, ("VDss", "test2"): 110}


def cids(frame, mask):
    s = pd.to_numeric(frame.loc[mask, "PUBCHEM_CID"], errors="coerce").dropna()
    return sorted({int(x) for x in s})


def main():
    if not S1.exists():
        raise SystemExit("cannot find %s - run from the repository root" % S1)
    bench = pd.read_excel(S1, sheet_name="bechmarking_jmc_train_test1")
    retrain = pd.read_excel(S1, sheet_name="modeling_retrain_test2")

    OUT.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for ep, (label, split) in ENDPOINTS.items():
        labelled = pd.to_numeric(retrain[label], errors="coerce").notna()
        in_test2 = retrain["CT_train_test"].astype(str).str.strip().eq("test")
        sets = {
            "train1": cids(bench, bench[split].astype(str).str.strip().eq("train")),
            "test1": cids(bench, bench[split].astype(str).str.strip().eq("test")),
            "train2": cids(retrain, labelled & ~in_test2),
            "test2": cids(retrain, labelled & in_test2),
        }
        for name, ids in sets.items():
            path = OUT / ("%s_%s.csv" % (ep, name))
            pd.DataFrame({"PUBCHEM_CID": ids}).to_csv(path, index=False)
            want = EXPECTED[(ep, name)]
            ok = len(ids) == want
            if not ok:
                failures.append("%s %s: %d, expected %d" % (ep, name, len(ids), want))
            rows.append({"endpoint": ep, "split": name, "n": len(ids), "expected": want,
                         "matches_table1": ok, "file": path.name})
            print("  %-6s %-7s %5d  %s" % (ep, name, len(ids), "ok" if ok else "MISMATCH"))

    pd.DataFrame(rows).to_csv(OUT / "split_set_sizes.csv", index=False)
    print("\nwrote %d split files and split_set_sizes.csv to %s" % (len(rows), OUT))
    if failures:
        print("\nthese counts do not match Table 1:")
        for f in failures:
            print("  " + f)
        return 1
    print("all counts reproduce Table 1 of the paper")
    return 0


if __name__ == "__main__":
    sys.exit(main())
