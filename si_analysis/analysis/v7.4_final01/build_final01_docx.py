#!/usr/bin/env python3
"""v7.4 final_01 — build the submission-facing .docx by surgical edit of the v7.2 baseline.

25 v7.2 body paragraphs carry Word OMML equation objects (~31-42, 50, 63-75, 79), several of them inline
inside ordinary sentences, and the body also carries EndNote fields and superscript citations. Regenerating
the section from markdown destroys all of that silently. So this builder opens a copy of the frozen baseline
and touches only what changed:

  keep      paragraph XML untouched  -> OMML, citations and fields survive
  edit      runs replaced, paragraph style and numbering kept
  delete    <w:p> removed from the body
  relocate  <w:p> removed (the text moves to the SI in a later round, it is not lost)
  insert    new <w:p> cloned from its anchor's formatting, inserted after it

Tables 1-3 are replaced by deep copies from the frozen tables_round4b file, and the Table 4 object is
removed. Every mapped block is guarded by a text prefix, so any drift in the draft raises instead of
silently misaligning.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v7.4_final01/build_final01_docx.py
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import re
import shutil
from pathlib import Path

from docx import Document
from docx.text.paragraph import Paragraph

S21 = Path(__file__).resolve().parents[2]
FINAL = S21 / "manuscript" / "v7.4" / "final_01"
SRC = FINAL / "src"
DRAFT = FINAL / "draft"
BASE = FINAL / "sources" / "baseline" / "revised ACS JCIM v7.2.docx"
TABLES = FINAL / "sources" / "tables" / "TABLE_maintext.docx"
OUT = DRAFT / "v7.4_methods_results.docx"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MATH = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MD = load(S21 / "manuscript" / "v4" / "scripts" / "md2docx.py", "md2docx")
BUILD = load(Path(__file__).with_name("build_final01.py"), "build_final01")
FIX = load(Path(__file__).with_name("table_fixes.py"), "table_fixes")

# (block index in the body file, v7.2 paragraph index or None, guard prefix)
# None = inserted material; the anchor is given by INSERTS below.
METHODS_MAP = [
    (2, 22, "As displayed in TABLE 1"), (3, 23, "In this study, we also"), (4, 24, "For endpoint-level"),
    (5, 25, "TABLE 1. Dataset"), (6, 26, "Training Set #2 was used only"),
    (8, 29, "Model performance was evaluated"), (10, 34, "where denote the observed"),
    (11, 35, "To account for the wide"), (12, 36, "For each compound"),
    (14, 38, "The geometric mean"), (16, 40, "Prediction reliability"),
    (18, 42, "where is an indicator"), (19, 43, "Together, these metrics"),
    (20, None, "For matched-compound comparisons"),
    (22, 45, "The DL encoders"), (23, 46, "Molecular embeddings provide"),
    (24, 47, "Encoder architectures and tuned"),
    (26, 49, "To benchmark the predictive"), (27, 50, "Overall, for each PK endpoint"),
    (29, 52, "All PBPK model simulations"), (30, 54, "Whole-body (full) PBPK scenarios"),
    (31, None, "The primary scenarios used"), (32, 55, "For the 41 compounds with"),
    (33, None, "Clearance was entered by one"),
    (35, 57, "To enable a controlled"), (36, 58, "(i) Sensitivity analysis"),
    (37, 59, "(ii) Hybrid clearance"), (38, 60, 'For clarity, the term "top-down"'),
    (40, 62, "Hybrid-PBPK model performance"), (41, None, "A C–T profile is a curve"),
    (42, 69, "In addition to full C–T profile"), (44, 72, "where = observed PK endpoint"),
    (45, 73, "To quantify practical"), (47, None, "Because every scenario was simulated"),
    (49, 77, "A compound-level mechanistic"), (50, None, "Three analyses attribute"),
]
RESULTS_MAP = [
    (2, 86, "Herein, DL–ML framework"), (3, 87, "Observed-versus-predicted relationships"),
    (4, 88, "**FIGURE 2. Observed versus predicted"),
    (6, 93, "We next evaluated whether"), (7, 94, "For systemic clearance (CLsys)"),
    (8, 95, "In addition, Table S1 includes"), (9, None, "The descriptive comparison was"),
    (10, 97, "TABLE 2. Performance and methods"), (11, 98, "ML Group A, our merged"),
    (12, 100, "Taken together, these results"),
    (14, 102, "Among the 110 compounds reserved"),
    (15, None, "The inputs these scenarios depend on"), (16, None, "**FIGURE 3."),
    (17, 103, "To establish a controlled evaluation"), (18, 104, "To assess model sensitivity"),
    (19, 105, "Based on these observations"), (20, 106, "TABLE 3. Configuration"),
    (21, 107, "Input source and parameterization"),
    (23, 109, "Following confirmation of the minimal"),
    (24, None, "Replacing observed clearance"), (25, None, "**FIGURE 4."),
    (26, None, "The two clearance strategies were"), (27, None, "**FIGURE 5."),
    (28, None, "In the renal-clearance sensitivity"), (29, None, "To locate the source"),
    (30, None, "The systematic Cmax underprediction"), (31, None, "Collectively, on this compound set"),
    (33, 119, "Global summaries conceal"), (34, 120, "Predicting clearance increased"),
    (35, 128, "Stratifying the information-matched"),
    (36, None, "Finally, the all-predicted scenario"), (37, None, "**FIGURE 6."),
]
# anchor paragraph -> ordered block indices inserted after it
INSERTS = {"methods": {43: [20], 54: [31], 55: [33], 62: [41], 74: [47], 77: [50]},
           "results": {95: [9], 102: [15, 16], 109: [24, 25, 26, 27, 28, 29, 30, 31],
                       128: [36, 37]}}
DELETE = [53, 63, 64, 65, 66, 67, 68, 75, 78, 79, 80,
          91, 110, 111, 114, 116, 121, 122, 123, 126, 129]
RELOCATE = [81, 82, 83]
# v7.2 table index -> frozen table index, or None to delete
TABLE_ACTION = {0: 0, 1: 1, 2: 2, 3: None}


def blocks(name):
    text = re.sub(r"<!--.*?-->", "", (SRC / name).read_text(), flags=re.S)
    caps = BUILD.frozen_captions()
    text = re.sub(r"<<<FROZEN_CAPTION:(\d+)>>>", lambda m: caps[int(m.group(1))], text)
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def set_text(par, text):
    """Replace a paragraph's runs, keeping its <w:pPr> (style, spacing, numbering)."""
    for r in list(par._p.findall(W + "r")) + list(par._p.findall(W + "hyperlink")):
        par._p.remove(r)
    MD.add_runs(par, text)


def clone_after(anchor_p, text):
    new = copy.deepcopy(anchor_p)
    for child in list(new):
        if child.tag != W + "pPr":
            new.remove(child)
    anchor_p.addnext(new)
    set_text(Paragraph(new, None), text)
    return new


def main():
    mb, rb = blocks("body_methods.md"), blocks("body_results.md")
    bl = {"methods": mb, "results": rb}

    shutil.copy2(BASE, OUT)
    doc = Document(str(OUT))
    paras = doc.paragraphs
    body = doc.element.body

    # 0. guards: every mapped block must still start with its expected text
    bad = []
    for sec, mp in (("methods", METHODS_MAP), ("results", RESULTS_MAP)):
        for bi, pi, guard in mp:
            if bi >= len(bl[sec]) or not bl[sec][bi].startswith(guard):
                bad.append((sec, bi, guard, bl[sec][bi][:50] if bi < len(bl[sec]) else "<missing>"))
    if bad:
        raise SystemExit("draft drifted from the docx map:\n" + "\n".join(map(str, bad)))

    # 1. edits, in place. Driven by the disposition table, never by comparing text: paragraphs that
    # carry inline OMML read differently in the markdown draft (which uses [OMML ...] placeholders), so a
    # text comparison would "edit" a kept equation paragraph and write the placeholder into the document.
    dispo = {int(r["para_idx"]): r["disposition"] for r in
             csv.DictReader((FINAL / "changelog" / "v72_paragraph_disposition.csv").open())}
    n_edit, kept_omml_intact = 0, 0
    for sec, mp in (("methods", METHODS_MAP), ("results", RESULTS_MAP)):
        for bi, pi, _ in mp:
            if pi is None:
                continue
            if not dispo.get(pi, "keep").startswith("edit"):
                if paras[pi]._p.findall(".//" + MATH + "oMath"):
                    kept_omml_intact += 1
                continue
            set_text(paras[pi], bl[sec][bi])
            # an edited paragraph's replacement text supersedes any inline equation it used to carry
            for om in paras[pi]._p.findall(".//" + MATH + "oMath"):
                om.getparent().remove(om)
            n_edit += 1

    # 2. insertions, after their anchors (reverse order within an anchor keeps the sequence)
    n_ins = 0
    for sec, mapping in INSERTS.items():
        for anchor, idxs in mapping.items():
            prev = paras[anchor]._p
            for bi in idxs:
                prev = clone_after(prev, bl[sec][bi])
                n_ins += 1

    # 3. deletions and relocations
    n_del = 0
    for pi in DELETE + RELOCATE:
        p = paras[pi]._p
        p.getparent().remove(p)
        n_del += 1

    # 4. tables: replace 1-3 from the frozen file, remove Table 4
    frozen = Document(str(TABLES))
    tbls = [el for el in body.iterchildren() if el.tag == W + "tbl"]
    n_tbl = 0
    for ti, el in enumerate(tbls):
        act = TABLE_ACTION.get(ti)
        if act is None:
            el.getparent().remove(el)
        else:
            new = copy.deepcopy(frozen.tables[act]._tbl)
            el.addnext(new)
            el.getparent().remove(el)
        n_tbl += 1

    # 5. Table 2: the archived CSV is authoritative for the ML Group A CLsys within-twofold cell
    fixed = 0
    for t in doc.tables:
        if len(t.columns) == 10 and t.rows[0].cells[0].text.strip().startswith("Model Group"):
            for row in t.rows:
                labels = [c.text.strip() for c in row.cells]
                if labels[0] == "A" and any("CL" == x for x in labels):
                    for cell in row.cells:
                        if cell.text.strip() in ("66", "66%"):
                            for par in cell.paragraphs:
                                set_text(par, cell.text.strip().replace("66", "65"))
                            fixed += 1
    # 6. shared table fixes: CLsys terminology, Table 3 group labels, subscript formatting
    tstats = FIX.fix_tables(doc)
    left = FIX.bare_cl(doc)
    if left:
        raise SystemExit("bare 'CL' left in the tables: %s" % left)
    doc.save(str(OUT))

    print("table cells rewritten  %3d  (CLsys terminology, group labels, subscripts)" % tstats["cells"])
    print("edits applied          %3d" % n_edit)
    print("kept OMML paragraphs untouched %3d" % kept_omml_intact)
    print("paragraphs inserted    %3d" % n_ins)
    print("paragraphs removed     %3d  (%d deleted, %d relocated to SI)" % (n_del, len(DELETE), len(RELOCATE)))
    print("tables handled         %3d  (1-3 replaced from the frozen file, Table 4 removed)" % n_tbl)
    print("Table 2 66%%->65%% cells  %3d" % fixed)
    out = Document(str(OUT))
    omml = sum(1 for p in out.paragraphs if p._p.findall(".//" + MATH + "oMath"))
    print("OMML paragraphs surviving in the output: %d" % omml)
    print("wrote", OUT.relative_to(S21))


if __name__ == "__main__":
    main()
