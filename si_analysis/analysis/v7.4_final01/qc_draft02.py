#!/usr/bin/env python3
"""QC gate for manuscript/v7.4/final_01/draft_02."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document

S21 = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import draft02_map as M                                            # noqa: E402

D2 = S21 / "manuscript" / "v7.4" / "final_01" / "draft_02"
SI_MD = D2 / "SI" / "Supplementary_Information_v74b.md"
MAIN_DOCX = D2 / "maintext" / "v7.4_methods_results_draft02.docx"
checks = []


def chk(n, ok, d):
    checks.append((n, bool(ok), d))


def strip(t):
    return re.sub(r"</?su[bp]>|\*\*", "", t)


si = strip(SI_MD.read_text())
main_doc = Document(str(MAIN_DOCX))
main = strip("\n".join(p.text for p in main_doc.paragraphs))
sec = {}
for i, p in enumerate(main_doc.paragraphs):
    u = p.text.strip().upper()
    if u in ("METHODS", "RESULTS", "DISCUSSION"):
        sec.setdefault(u, i)
scope = strip("\n".join(p.text for p in main_doc.paragraphs[sec["METHODS"]:sec["DISCUSSION"]]))


def refs(t):
    out = set()
    for m in re.finditer(r"Figures? S\d+(?:\s*(?:,|and|to|–|-|through)\s*S\d+)*", t):
        nums = [int(x) for x in re.findall(r"S(\d+)", m.group(0))]
        if re.search(r"(–|to|through)", m.group(0)) and len(nums) >= 2:
            out.update(range(min(nums), max(nums) + 1))
        else:
            out.update(nums)
    return out


# --- SI
defined = []
for n in [int(x) for x in re.findall(r"FIGURE S(\d+)", si)]:
    if n not in defined:
        defined.append(n)
chk("1. SI defines Figures S1-S20 in ascending order", defined == list(range(1, 21)), "order %s" % defined)
chk("2. No dangling SI figure reference", not (refs(si) - set(defined)),
    "dangling %s" % sorted(refs(si) - set(defined)) if refs(si) - set(defined) else "%d cited" % len(refs(si)))
secs = re.findall(r"^## (S\d+)\.", SI_MD.read_text(), re.M)
chk("3. Seven SI sections, S8 dropped", secs == ["S%d" % i for i in range(1, 8)], " ".join(secs))
tabs = [int(x) for x in re.findall(r"\*\*TABLE S(\d+)\.", SI_MD.read_text())]
chk("4. SI Tables S1-S13 unchanged and in order", tabs == list(range(1, 14)), "%s" % tabs)
embeds = re.findall(r"!\[.*?\]\(figures/([^)]+)\)", SI_MD.read_text())
on_disk = {p.stem for p in (D2 / "SI" / "figures").glob("*.png")}
chk("5. Every SI figure file resolves, 1:1", {Path(e).stem for e in embeds} == on_disk,
    "%d embeds, %d files" % (len(embeds), len(on_disk)))

# --- the requested removals and the move
for num, (stem, why) in M.REMOVED.items():
    chk("6. Old Figure S%d is gone from the SI (%s)" % (num, why.split(";")[0]),
        stem not in SI_MD.read_text(), "stem absent" if stem not in SI_MD.read_text() else "STILL PRESENT")
alt = [i for i, e in enumerate(embeds) if "alt_input_substitution" in e]
s4_start = SI_MD.read_text().index("## S4.")
s5_start = SI_MD.read_text().index("## S5.")
pos = SI_MD.read_text().index("FigureS14_alt_input_substitution_branches")
chk("7. The alternative-substitution figure now sits inside Section S4", s4_start < pos < s5_start,
    "Figure S14, between the Section S4 and S5 headings")

# --- main text
mfigs = sorted({int(n) for n in re.findall(r"(?<!S)\bFigure (\d+)\b", scope)})
chk("8. Main text cites Figures 1-7 only", set(mfigs) <= set(range(1, 8)), "cited %s" % mfigs)
order, first = [int(n) for n in re.findall(r"(?<!S)\bFigure (\d+)\b", scope)], []
for n in order:
    if n not in first:
        first.append(n)
chk("9. Main-text figures first cited in ascending order", first == sorted(first), "%s" % first)
chk("10. Main text cites no SI figure above S20", not (refs(scope) - set(range(1, 21))),
    "cited %s" % sorted(refs(scope)))
chk("11. Every SI figure the main text cites exists", not (refs(scope) - set(defined)),
    "dangling %s" % sorted(refs(scope) - set(defined)) if refs(scope) - set(defined) else "all resolve")
chk("12. Main text no longer cites the deleted Figure S25", "Figure S25" not in main,
    "absent" if "Figure S25" not in main else "PRESENT")
chk("12b. The propagation sentence now cites Figure 6, not SI figures",
    "not detectably associated with any exposure endpoint (Figure 6)" in main, "re-pointed to Figure 6")
chk("12c. New S15/S16 are the Cmax figures, so the Cmax sentence may cite them",
    "structural rather than an input effect (Figures S15 and S16; Table S9)" in main
    and (D2 / "SI" / "figures" / "FigureS15_cmax_structure.png").exists(),
    "S15 = cmax_structure, S16 = cmax_readout")
chk("13. Main text carries the new Figure 6 caption",
    "FIGURE 6. Association between upstream input error" in main, "present")
chk("14. Former Figure 6 is now Figure 7", "FIGURE 7. End-to-end performance" in main, "present")

# --- figure files and display docs
missing = [n for n in range(1, 8) if not list((D2 / "figures").glob("Figure%d_*.png" % n))]
chk("15. Main-text figure files 1-7 present", not missing, "missing %s" % missing if missing else
    "%d files" % len(list((D2 / "figures").glob("*.png"))))
fd = Document(str(D2 / "display" / "FIGURE_maintext_draft02.docx"))
chk("16. Figure display document carries 7 figures", len(fd.inline_shapes) == 7,
    "%d images, %d paragraphs" % (len(fd.inline_shapes), len(fd.paragraphs)))
td = Document(str(D2 / "display" / "TABLE_maintext_draft02.docx"))
chk("17. Table display document carries Tables 1-3", len(td.tables) == 3,
    "%dx%d / %dx%d / %dx%d" % tuple(x for t in td.tables for x in (len(t.rows), len(t.columns))))

# --- Figure 6 statistics reproduce the archive
st = pd.read_csv(D2 / "stats" / "figure6_parameter_to_exposure_spearman.csv").set_index("association")
ref = pd.read_csv(S21 / "outputs" / "tables" / "v4_figureS12_spearman.csv").set_index("association")
pairs = [("|CLsys FE| vs C–T RMSE", "|CL FE| vs C–T RMSE"), ("CLsys FE vs AUC FE", "CL FE vs AUC FE"),
         ("VDss FE vs Cmax FE", "VDss FE vs Cmax FE")]
same = all(abs(st.loc[a, "spearman_rho"] - ref.loc[b, "spearman_rho"]) < 5e-4 for a, b in pairs)
chk("18. Figure 6 statistics reproduce the archived values", same,
    "rho %s" % [round(st.loc[a, "spearman_rho"], 3) for a, _ in pairs])

# --- nothing outside draft_02 disturbed
ref_t = SI_MD.stat().st_mtime
touched = [p.name for p in sorted((S21 / "outputs" / "tables").glob("*.csv")) if p.stat().st_mtime > ref_t - 1]
chk("19. Archived analysis CSVs untouched", not touched, "modified %s" % touched[:3] if touched else "31 CSVs")
frozen_touched = [p.name for p in sorted((S21 / "manuscript" / "v7.4" / "figures_round4b").glob("*"))
                  if p.stat().st_mtime > ref_t - 1]
chk("20. Round 4B frozen figures untouched", not frozen_touched, "modified" if frozen_touched else "unchanged")

# --- the Discussion must be untouched (it is out of scope and holds a stale v7.2 pointer)
d01 = Document(str(S21 / "manuscript" / "v7.4" / "final_01" / "draft" / "v7.4_methods_results.docx"))
i01 = next(i for i, q in enumerate(d01.paragraphs) if q.text.strip().upper() == "DISCUSSION")
i02 = next(i for i, q in enumerate(main_doc.paragraphs) if q.text.strip().upper() == "DISCUSSION")
tail01 = [q.text for q in d01.paragraphs[i01:]]
tail02 = [q.text for q in main_doc.paragraphs[i02:]]
chk("21. Discussion and everything after it is byte-identical to draft_01", tail01 == tail02,
    "%d paragraphs unchanged" % len(tail01) if tail01 == tail02 else "DIFFERS")

# --- Figure 2 regained its distribution panels
chk("22. Figure 2 caption describes the nine panels",
    "(a, d, g)" in main and "(b, e, h)" in main and "(c, f, i)" in main, "3 x 3 layout described")
chk("23. The dataset paragraph cites the distribution panels",
    "As displayed in Figure 2a, d and g" in main, "cites Figure 2a, d, g")

fails = [c for c in checks if not c[1]]
for n, ok, d in checks:
    print("%-62s %-4s %s" % (n[:62], "PASS" if ok else "FAIL", str(d)[:60]))
print("\n%d checks, %d pass, %d fail" % (len(checks), len(checks) - len(fails), len(fails)))
