#!/usr/bin/env python3
"""v7.4 final_01 — Step 0: freeze every source the Methods+Results rewrite reads.

Copies the frozen Round 4B/4C display package, the Round 4B main-text tables, the v4 SI markdown and
tables, the v7.2 baselines and every archived analysis CSV into manuscript/v7.4/final_01/sources/, then
writes BACKUP_MANIFEST.csv (role, original path, copied path, bytes, sha256, mtime, freezing round).

The repository has no git commits, so this manifest is the only provenance record and the only rollback
guarantee. Nothing under sources/ is ever edited afterwards.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v7.4_final01/backup_sources.py
"""
from __future__ import annotations

import csv
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

S21 = Path(__file__).resolve().parents[2]
V74 = S21 / "manuscript" / "v7.4"
V4 = S21 / "manuscript" / "v4"
QC = V74 / "round4c_qc"
FINAL = V74 / "final_01"
SRC = FINAL / "sources"

# (role, source path, destination relative to sources/, freezing round)
PLAN: list[tuple[str, Path, str, str]] = []


def add(role, src, dest, frozen_by):
    PLAN.append((role, src, dest, frozen_by))


def add_glob(role, folder, pattern, dest_dir, frozen_by, skip_lock=True):
    for p in sorted(folder.glob(pattern)):
        if skip_lock and p.name.startswith("~$"):
            continue
        add(role, p, "%s/%s" % (dest_dir, p.name), frozen_by)


# --- baselines -------------------------------------------------------------------------------------
add("baseline manuscript", S21 / "manuscript" / "revised ACS JCIM v7.2.docx",
    "baseline/revised ACS JCIM v7.2.docx", "v7.2")
add("baseline SI", S21 / "manuscript" / "Supplementary Information v7.2.docx",
    "baseline/Supplementary Information v7.2.docx", "v7.2")

# --- main-text figures (Round 4B) -------------------------------------------------------------------
add_glob("main figure", V74 / "figures_round4b", "Figure*.*", "figures", "round4b")
add("figure architecture", V74 / "figures_round4b" / "v7.4_round4b_final_figure_architecture.md",
    "captions/v7.4_round4b_final_figure_architecture.md", "round4b")

# --- frozen captions and cross-reference freeze (Round 4C) ------------------------------------------
add("frozen captions 3-6", QC / "v7.4_display_package_round4c.md",
    "captions/v7.4_display_package_round4c.md", "round4c")
add("round4b display package", V74 / "v7.4_display_package_round4b.md",
    "captions/v7.4_display_package_round4b.md", "round4b")
add("crossref audit", QC / "v7.4_round4c_crossref_audit.md",
    "captions/v7.4_round4c_crossref_audit.md", "round4c")
add("SI numbering freeze", QC / "v7.4_round4c_si_numbering_crossref_freeze.md",
    "si/v7.4_round4c_si_numbering_crossref_freeze.md", "round4c")

# --- main-text tables --------------------------------------------------------------------------------
add("main tables 1-3", V74 / "tables_round4b" / "TABLE_maintext.docx",
    "tables/TABLE_maintext.docx", "round4b")
add("Table 3 scenario record", QC / "Table3_core_pbpk_scenarios_round4c.md",
    "tables/Table3_core_pbpk_scenarios_round4c.md", "round4c")

# --- SI: numbering map, final captions, final figure files -------------------------------------------
add("SI numbering map", QC / "si_figure_numbering_map.csv", "si/si_figure_numbering_map.csv", "round4c")
add("SI captions", QC / "SI" / "captions_final.md", "si/captions_final.md", "round4c")
add_glob("SI figure", QC / "SI" / "figures_final", "*", "si/figures_final", "round4c")
add("SI prose + tables S1-S13", V4 / "SI" / "Supplementary_Information_v4.md",
    "si/Supplementary_Information_v4.md", "v4")
add_glob("SI table source", V4 / "SI" / "tables", "*", "si/tables", "v4")

# --- evidence bank ------------------------------------------------------------------------------------
add("v4 manuscript (evidence)", V4 / "manuscript_v4.md", "evidence/manuscript_v4.md", "v4")
add_glob("archived CSV", S21 / "outputs" / "tables", "*.csv", "evidence/outputs_tables", "v4")
add_glob("round4b CSV", S21 / "outputs" / "tables" / "v7.4_round4b", "*.csv",
         "evidence/outputs_tables/v7.4_round4b", "round4b")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    missing = [(r, p) for r, p, _, _ in PLAN if not p.exists()]
    if missing:
        raise SystemExit("missing sources:\n" + "\n".join("  %s  %s" % (r, p) for r, p in missing))

    rows, total = [], 0
    for role, src, dest, frozen_by in PLAN:
        out = SRC / dest
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        size = src.stat().st_size
        total += size
        rows.append({
            "role": role,
            "original_path": str(src.relative_to(S21)),
            "copied_path": str(out.relative_to(FINAL)),
            "bytes": size,
            "sha256": sha256(src),
            "mtime_utc": datetime.fromtimestamp(src.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "frozen_by": frozen_by,
        })

    # The Round 4B SI figure tree is superseded by round4c_qc/SI/figures_final (same images, final
    # numbering), so it is recorded by checksum only rather than copied a second time.
    for p in sorted((V74 / "SI" / "figures_round4b").glob("*")):
        rows.append({
            "role": "SI figure (superseded, not copied)",
            "original_path": str(p.relative_to(S21)),
            "copied_path": "",
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
            "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "frozen_by": "round4b",
        })

    FINAL.mkdir(parents=True, exist_ok=True)
    with (FINAL / "BACKUP_MANIFEST.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    copied = sum(1 for r in rows if r["copied_path"])
    print("copied %d files (%.1f MB) into %s" % (copied, total / 1e6, SRC.relative_to(S21)))
    print("recorded %d superseded files by checksum only" % (len(rows) - copied))
    print("manifest: %s" % (FINAL / "BACKUP_MANIFEST.csv").relative_to(S21))


if __name__ == "__main__":
    main()
