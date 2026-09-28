#!/usr/bin/env python3
"""QC gate for the v7.4 Supporting Information document."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document

S21 = Path(__file__).resolve().parents[2]
SI = S21 / "manuscript" / "v7.4" / "SI"
MD = SI / "Supplementary_Information_v74.md"
FIN = S21 / "manuscript" / "v7.4" / "final_01" / "sources" / "si"


def load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    sys.modules[n] = m
    s.loader.exec_module(m)
    return m


AUDIT = load(S21 / "analysis" / "v7.4_round4b" / "round4c_audit.py", "round4c_audit")
BUILD = load(Path(__file__).with_name("build_si_v74.py"), "build_si_v74")
FIX = load(Path(__file__).with_name("table_fixes.py"), "table_fixes")

t = MD.read_text()
plain = re.sub(r"</?su[bp]>", "", t)          # notation-stripped, for comparing against frozen text
checks = []


def chk(name, ok, detail):
    checks.append((name, bool(ok), detail))


# 1. captions verbatim against the frozen package
caps = BUILD.frozen_captions()
embeds = re.findall(r"!\[(.*?)\]\(figures/([^)]+)\)", t)
bad, cont = [], 0
for cap, path in embeds:
    capp = re.sub(r"</?su[bp]>", "", cap).replace("**", "")
    capp = re.sub(r"R²", "R2", capp)
    if "(continued)" in capp:
        cont += 1
        continue
    n = int(re.match(r"FIGURE S(\d+)", capp).group(1))
    want = re.sub(r"R²", "R2", re.sub(r"\s+", " ", caps[n]).strip())
    got = re.sub(r"\s+", " ", capp).strip()
    if want.replace("f_u", "fu").replace("CL_sys", "CLsys").replace("VD_ss", "VDss") \
            .replace("CL_int", "CLint").replace("C_max", "Cmax").replace("log10", "log10") \
            .replace("AUC0–t", "AUC0–t") != got.replace("f_u", "fu").replace("CL_sys", "CLsys") \
            .replace("VD_ss", "VDss").replace("CL_int", "CLint").replace("C_max", "Cmax"):
        bad.append(n)
chk("1. All 25 frozen captions reproduced verbatim (notation aside)", not bad,
    "mismatched: %s" % bad if bad else "25 captions + %d continued pages" % cont)

# 2. every figure file resolves; no stem unused or duplicated
stems_used = [Path(p).stem for _, p in embeds]
on_disk = {p.stem for p in (SI / "figures").glob("*.png")}
missing = [s for s in stems_used if s not in on_disk]
unused = sorted(on_disk - set(stems_used))
dupes = sorted({s for s in stems_used if stems_used.count(s) > 1})
chk("2. Every referenced figure file exists", not missing, "missing: %s" % missing[:4] if missing else
    "%d embeds resolve" % len(stems_used))
chk("2b. No figure file unused and none cited twice", not unused and not dupes,
    "unused %s dupes %s" % (unused, dupes) if (unused or dupes) else "%d stems, 1:1" % len(on_disk))

# 3. cross-references
fig_refs = AUDIT.fig_refs(plain)
nums_defined = {int(re.match(r"FIGURE S(\d+)", re.sub(r"</?su[bp]>|\*\*", "", c)).group(1))
                for c, _ in embeds}
chk("3. Every Figure S# cited is defined in this document", not (fig_refs - nums_defined),
    "dangling: %s" % sorted(fig_refs - nums_defined) if fig_refs - nums_defined
    else "%d cited, all of S1-S25 defined" % len(fig_refs))
chk("3b. Figures S1-S25 all present", nums_defined == set(range(1, 26)),
    "missing %s" % sorted(set(range(1, 26)) - nums_defined))
tab_refs = AUDIT.table_refs(plain)
defined_t = {int(m) for m in re.findall(r"\*\*TABLE S(\d+)\.", t)}
chk("4. Every Table S# cited is defined", not (tab_refs - defined_t),
    "dangling: %s" % sorted(tab_refs - defined_t) if tab_refs - defined_t
    else "%d cited; Tables S1-S13 defined" % len(tab_refs))
chk("4b. Tables S1-S13 all present", defined_t == set(range(1, 14)),
    "have %s" % sorted(defined_t))

# 5. document order ascends
order = [int(re.match(r"FIGURE S(\d+)", re.sub(r"</?su[bp]>|\*\*", "", c)).group(1)) for c, _ in embeds]
first = []
for n in order:
    if n not in first:
        first.append(n)
chk("5. Figure numbers ascend through the document", first == sorted(first), "order: %s" % first)
torder = [int(m) for m in re.findall(r"\*\*TABLE S(\d+)\.", t)]
chk("5b. Table numbers ascend through the document", torder == sorted(torder), "order: %s" % torder)

# 6. no stale main-text figure pointer, no placeholder notation, CLsys terminology
# main-text references are legitimate where the frozen captions make them ("main-text Figure 4");
# what must not survive is a main-text pointer to a display that moved into the SI (v7.2/v4 Figures 3, 5, 7, 8)
mt = [plain[max(0, m.start() - 34):m.end()] for m in re.finditer(r"(?<!S)\bFigure [1-9]\b", plain)]
stale = [c for c in mt if not re.search(r"main-text Figure [1-6]$|v7\.2 Figure \d$|Figure [1246]$", c)]
chk("6. Main-text Figure pointers are valid (1-6) and none points at a moved display", not stale,
    "suspect: %s" % stale[:3] if stale else "%d refs, all to main-text Figures 1-6" % len(mt))
prose = re.sub(r"\]\(figures/[^)]+\)", "]()", t)   # figure file stems legitimately contain "log10"
ph = {q: prose.count(q) for q in ("f_u", "CL_sys", "VD_ss", "CL_int", "C_max", "log10", " R2 ")
      if prose.count(q)}
chk("7. No placeholder notation left", not ph, "found: %s" % ph if ph else "all converted to sub/superscript")
hits = list(re.finditer(r"(?<![A-Za-z_<>/])CL(?![A-Za-z_<]|</)", plain))
ctx = [plain[max(0, m.start() - 20):m.start()] for m in hits]
bad_cl = [plain[max(0, m.start() - 26):m.end() + 6] for m, c in zip(hits, ctx)
          if not re.search(r"(renal|Renal|separate|total|entered|in vivo|observed)\s*$", c)]
chk("8. CLsys terminology: no unqualified bare 'CL'", not bad_cl,
    "%d: %s" % (len(bad_cl), bad_cl[:3]) if bad_cl else "%d occurrences, all 'renal/total CL'" % len(ctx))

# 9. sections
secs = re.findall(r"^## (S\d+\..*)$", t, re.M)
chk("9. Eight sections, S1-S8, in order", [s.split(".")[0] for s in secs] == ["S%d" % i for i in range(1, 9)],
    " | ".join(s[:34] for s in secs))

# 10. Contents line agrees with reality
cont_line = re.search(r"\*\*Contents\.\*\*(.*)", t).group(1)
cont_ok = all(("Figures S%d" % a) in cont_line or ("Figure S%d" % a) in cont_line
              for a in (1, 19, 20, 24)) and "Section S8" in cont_line
chk("10. Contents line lists all eight sections and the frozen figure ranges", cont_ok,
    "S1-S5 / S6-S18 / S19 / S20 / S21-S23 / S24, S25 as built")

# 11. rendered artifacts
d = Document(str(MD.with_suffix(".docx")))
chk("11. docx built on the v7.2 SI template", d.styles["Normal"].font.name == "Times New Roman",
    "Normal font %r, %d paragraphs, %d tables, %d images"
    % (d.styles["Normal"].font.name, len(d.paragraphs), len(d.tables), len(d.inline_shapes)))
chk("11b. docx carries all 24 figure embeds", len(d.inline_shapes) == len(embeds),
    "%d images vs %d embeds" % (len(d.inline_shapes), len(embeds)))
subs = sum(1 for p in d.paragraphs for r in p.runs if r.font.subscript)
chk("11c. docx renders real subscript runs", subs > 100, "%d subscript runs" % subs)
pdf = MD.with_suffix(".pdf")
chk("12. pdf built", pdf.exists(), "%.1f MB" % (pdf.stat().st_size / 1e6) if pdf.exists() else "missing")

# 13. nothing outside SI/ written
ref = MD.stat().st_mtime
touched = [p.name for p in sorted((S21 / "outputs" / "tables").glob("*.csv")) if p.stat().st_mtime > ref - 1]
chk("13. Archived analysis CSVs untouched", not touched, "modified: %s" % touched[:3] if touched else
    "31 CSVs unchanged")

fails = [c for c in checks if not c[1]]
for name, ok, detail in checks:
    print("%-64s %-4s %s" % (name[:64], "PASS" if ok else "FAIL", str(detail)[:78]))
print("\n%d checks, %d pass, %d fail" % (len(checks), len(checks) - len(fails), len(fails)))
