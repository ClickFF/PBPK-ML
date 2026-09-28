#!/usr/bin/env python3
"""v7.4 final_01 — fill in changelog/v72_paragraph_disposition.csv.

This table is the single spine of the round: it drives the surgical .docx build (keep = XML untouched, so
OMML equations, superscript citations and EndNote fields survive; delete = element removed; edit/insert =
paragraph rebuilt) and the preservation metric. It is authoritative over the markdown draft.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v7.4_final01/set_dispositions.py
"""
from __future__ import annotations

import csv
from pathlib import Path

S21 = Path(__file__).resolve().parents[2]
F = S21 / "manuscript" / "v7.4" / "final_01" / "changelog" / "v72_paragraph_disposition.csv"

# para_idx -> (disposition, unit_id). Paragraphs absent from this map keep disposition "keep".
EDIT = {23: "M-01", 24: "M-02", 26: "M-03", 46: "M-05", 47: "M-06", 54: "M-08", 55: "M-10",
        57: "M-12", 58: "M-13", 59: "M-14", 60: "M-15", 62: "M-16", 69: "M-18",
        86: "R-01", 87: "R-02", 88: "R-03", 93: "R-05", 94: "R-06", 95: "R-07", 98: "R-09",
        100: "R-10", 103: "R-12", 104: "R-13", 105: "R-14", 106: "R-15", 107: "R-15",
        109: "R-16", 119: "R-25", 120: "R-26", 128: "R-30"}
DELETE = {53: "M-07", 63: "M-17", 64: "M-17", 65: "M-17", 66: "M-17", 67: "M-17", 68: "M-17",
          75: "M-19", 78: "M-20", 79: "M-20", 80: "M-20",
          91: "R-04", 110: "R-17", 114: "R-24", 116: "R-24", 121: "R-27", 122: "R-28", 123: "R-28",
          126: "R-29", 129: "R-31",
          # 111 is removed from its position and its revised concluding sentence is reinserted
          # after the new material, where it now closes the subsection (unit R-21)
          111: "R-21"}
RELOCATE = {81: "M-21", 82: "M-21", 83: "M-21"}
# paragraphs after which new material is inserted
INSERT_AFTER = {43: "M-04", 54: "M-09", 55: "M-11", 68: "M-17", 75: "M-19", 83: "M-22",
                95: "R-08", 102: "R-11", 104: "R-13",
                109: "R-18;R-19;R-20;R-22;R-23;R-21", 129: "R-32"}
# stale pointers repaired, for the audit trail
CROSSREF = {47: ("Table Sx", "Table S3"), 60: ("Table S2", "Tables S5 and S6"),
            86: ("Figure 2A, C, E", "TABLE 1"), 87: ("Figure 2B, D, F", "Figure 2"),
            98: ("Figure 3", "Figure S5"), 103: ("Figure S2", "Figures S6 and S7"),
            104: ("Table S2", "Table S8"), 109: ("Figure 4A, 4B", "Figures S8 and S9"),
            120: ("Figure S3", "Figures S11, S12 and S13")}
EVIDENCE = {43: "manuscript_v4.md; ml_benchmark_paired_stats_holm.csv", 46: "manuscript_v4.md; Table S4",
            54: "manuscript_v4.md", 55: "manuscript_v4.md; Table S6", 59: "clearance_hierarchy_contrasts.csv",
            68: "manuscript_v4.md", 75: "paired_contrasts.csv", 83: "manuscript_v4.md",
            87: "figure2_train_test_metrics.csv", 94: "ml_table2_recomputed.csv",
            95: "ml_table2_recomputed.csv", 102: "upstream_parameter_r2.csv", 103: "arm_summary.csv",
            104: "paired_contrasts.csv", 110: "paired_contrasts.csv; clearance_hierarchy_contrasts.csv",
            116: "v4_figureS12_spearman.csv; cmax_bias_by_arm.csv", 120: "v4_figureS9_stats.csv",
            128: "mechanistic_strata_matched_delta.csv", 129: "v4_figure8_gof.csv"}
FLAGS = {53: "M1", 54: "M7", 59: "M5", 60: "M8", 63: "M2", 66: "M2", 75: "M3", 78: "M4", 80: "M4",
         86: "R11", 87: "R12;N6", 88: "R13", 91: "R14", 94: "R15;N1;N2;N3;N4", 95: "N1;N5",
         98: "N1", 100: "R1", 103: "R16", 104: "R17", 110: "R2;R3;R19", 111: "R4", 119: "R5",
         120: "R6;R7", 121: "R7", 128: "R8;R9", 129: "R10"}


def main():
    rows = list(csv.DictReader(F.open()))
    counts = {"keep": 0, "edit": 0, "delete": 0, "relocate": 0, "insert_after": 0}
    for r in rows:
        i = int(r["para_idx"])
        if i in DELETE:
            r["disposition"], r["unit_id"] = "delete", DELETE[i]
        elif i in RELOCATE:
            r["disposition"], r["unit_id"] = "relocate", RELOCATE[i]
        elif i in EDIT:
            r["disposition"], r["unit_id"] = "edit", EDIT[i]
        else:
            r["disposition"] = "keep"
        if i in INSERT_AFTER:
            r["disposition"] = (r["disposition"] + "+insert_after").replace("keep+", "")
            r["unit_id"] = ";".join(x for x in (r["unit_id"], INSERT_AFTER[i]) if x)
            counts["insert_after"] += 1
        if i in CROSSREF:
            r["crossref_from"], r["crossref_to"] = CROSSREF[i]
        r["evidence_source"] = EVIDENCE.get(i, "")
        r["flag_ids"] = FLAGS.get(i, "")
        base = r["disposition"].split("+")[0]
        if base in counts:
            counts[base] += 1

    with F.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    scope = [r for r in rows if 20 <= int(r["para_idx"]) <= 129]
    for k in ("keep", "edit", "delete", "relocate", "insert_after"):
        n = sum(1 for r in scope if r["disposition"].split("+")[0] == k) if k != "insert_after" \
            else sum(1 for r in scope if "insert_after" in r["disposition"])
        print("%-12s %3d paragraphs in scope" % (k, n))
    out_of_scope = [r for r in rows if not (20 <= int(r["para_idx"]) <= 129)
                    and r["disposition"] != "keep"]
    print("out-of-scope paragraphs marked for change:", len(out_of_scope), "(must be 0)")


if __name__ == "__main__":
    main()
