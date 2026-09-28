#!/usr/bin/env python3
"""v7.4 final_01 — terminology and formatting fixes applied to main-text Tables 1-3.

Shared by build_final01_docx.py (the manuscript) and build_display_docs.py (the paste-ready documents), so
the tables are identical wherever they appear. Two author decisions are implemented here:

  1. bare "CL" -> "CLsys" for systemic clearance (the terminology sweep; "renal CL" and "CL = 0" are left
     alone), and Table 3's block headers relabelled from "PBPK Group I./II./III" to the "PBPK Group A /
     PBPK Group B / Structural sensitivity" names the revised Methods, Results and Tables S5/S8 use;
  2. real Word subscript runs for Fu, VDss, CLsys, CLint, CLR, Cmax, AUC0-t, Kp and pKa, so the tables match
     the frozen figure captions when pasted beside them.

No cell value is changed by either step. Every paragraph in these tables has uniform bold/italic runs
(verified), so rebuilding a paragraph's runs preserves its formatting.
"""
from __future__ import annotations

import re

# text substitutions, applied before formatting; order matters
TEXT = [
    (r"(?<![A-Za-z_])observed CL(?![A-Za-z_])", "observed CLsys"),
    (r"(?<![A-Za-z_])DL–ML CL(?![A-Za-z_])", "DL–ML CLsys"),
    (r"(?<![A-Za-z_])PBPK_v3_CL(?![A-Za-z_])", "PBPK_v3_CLsys"),
    (r"different CL inputs", "different CLsys inputs"),
    (r"^CL$", "CLsys"),
]
# Table 3 block headers -> the names used by the revised prose and by Tables S5/S8
GROUPS = [
    (r"^PBPK Group I\.?\s*:?\s*Sensitivity to other parameters substitution \(clearance observed\)\s*$",
     "PBPK Group A: sensitivity to non-clearance parameter substitution (clearance observed)"),
    (r"^PBPK Group II\.?\s*:?\s*Sensitivity to different CL(?:sys)? inputs methods \(clearance predicted\)\s*$",
     "PBPK Group B: sensitivity to the clearance input (clearance predicted)"),
    (r"^Group III\s*:?\s*PBPK Model Structural sensitivity\s*$",
     "Structural sensitivity analysis"),
]
# (regex, base text, subscript text)
SUBS = [
    (r"CLsys", "CL", "sys"), (r"CLint", "CL", "int"), (r"CLR(?![a-z])", "CL", "R"),
    (r"VDss", "VD", "ss"), (r"AUC0[–-]t", "AUC", "0–t"), (r"Cmax", "C", "max"),
    (r"Fu", "F", "u"), (r"fu", "f", "u"), (r"Kp", "K", "p"), (r"pKa", "pK", "a"),
]
TOKEN = re.compile(r"\b(" + "|".join(p for p, _, _ in SUBS) + r")\b")


def _sub_for(tok):
    for pat, base, sub in SUBS:
        if re.fullmatch(pat, tok):
            return base, sub
    return tok, ""


def _rewrite(par) -> bool:
    """Apply text fixes and subscript formatting to one paragraph. Returns True if it changed."""
    old = par.text
    if not old.strip():
        return False
    new = old
    for pat, rep in GROUPS:
        new = re.sub(pat, rep, new)
    for pat, rep in TEXT:
        new = re.sub(pat, rep, new)
    parts = TOKEN.split(new)
    if new == old and len(parts) == 1:
        return False
    src = par.runs[0] if par.runs else None
    bold = bool(src.bold) if src is not None else False
    italic = bool(src.italic) if src is not None else False
    name = src.font.name if src is not None else None
    size = src.font.size if src is not None else None
    for r in list(par.runs):
        r._element.getparent().remove(r._element)
    for i, seg in enumerate(parts):
        if not seg:
            continue
        is_tok = i % 2 == 1
        base, sub = _sub_for(seg) if is_tok else (seg, "")
        for text, is_sub in ((base, False), (sub, True)):
            if not text:
                continue
            r = par.add_run(text)
            r.bold = True if bold else None
            r.italic = True if italic else None
            if name:
                r.font.name = name
            if size:
                r.font.size = size
            if is_sub:
                r.font.subscript = True
    return True


def fix_tables(doc, notes=()) -> dict:
    """Fix every cell of every table in `doc`, plus any extra paragraphs given in `notes`."""
    n_cells = n_notes = 0
    for t in doc.tables:
        seen = set()
        for row in t.rows:
            for cell in row.cells:
                if cell._tc in seen:          # merged cells repeat across the row
                    continue
                seen.add(cell._tc)
                for par in cell.paragraphs:
                    if _rewrite(par):
                        n_cells += 1
    for par in notes:
        if _rewrite(par):
            n_notes += 1
    return {"cells": n_cells, "notes": n_notes}


def bare_cl(doc) -> list:
    """Standalone 'CL' meaning systemic clearance, which the terminology rule forbids."""
    hits = []
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for m in re.finditer(r"(?<![A-Za-z_])CL(?![A-Za-z_])", cell.text):
                    ctx = cell.text[max(0, m.start() - 12):m.start()]
                    if re.search(r"(renal|Renal|separate)\s*$", ctx):
                        continue
                    if re.match(r"\s*(=|R\b)", cell.text[m.end():]):
                        continue
                    hits.append(cell.text.strip()[:44])
    return sorted(set(hits))
