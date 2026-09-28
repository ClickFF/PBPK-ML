#!/usr/bin/env python3
"""Round 5C cleanup applied to the draft_02 sources and display package.

Keeps draft_02 consistent with the Round 5C manuscript: the same terminology sweep, plus the two citation
corrections the author asked for (Figure 5 caption -> Table S11; the C_max template stratification ->
Figure S15b). "bottom-up" is retained wherever it denotes IVIVE of CL_int and dropped where it does not.

Every substitution is an exact string and must land at least once across the tree, or the script refuses.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v7.4_final01/fix_sources_round5c.py
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

from docx import Document

S21 = Path(__file__).resolve().parents[2]
D2 = S21 / "manuscript" / "v7.4" / "final_01" / "draft_02"

MD = ["maintext/body_methods.md", "maintext/body_results.md",
      "maintext/v7.4_methods_results_draft02.md"]
DOCX = ["maintext/v7.4_methods_results_draft02.docx",
        "display/FIGURE_maintext_draft02.docx", "display/TABLE_maintext_draft02.docx"]

APOLOGIA = ('For clarity, the term "top-down" is used operationally throughout this manuscript to denote '
            "PBPK models parameterized using predicted CLsys, rather than parameters estimated by fitting "
            "clinical PK observations. This terminology is intended solely to distinguish the CLsys-based "
            "parameterization strategy from the IVIVE-based CLint strategy and should not be interpreted as "
            "representing a classical top-down PBPK, population pharmacokinetic, or model-calibration "
            "workflow.")

# (old, new) — applied to markdown as plain text and to .docx through run-span-aware replacement
SUBS = [
    # --- terminology: the predicted-CLsys workflow is never called top-down ---
    ("In the observed-input and top-down scenarios",
     "In the observed-input and predicted-CLsys scenarios"),
    ("comparison with the top-down workflow could be made",
     "comparison with the predicted-CLsys workflow could be made"),
    ("(b) a top-down approach using ML-predicted systemic clearance (CLsys).",
     "(b) a CLsys-based approach entering ML-predicted systemic clearance (CLsys) directly."),
    (APOLOGIA,
     "As defined in the Introduction, the predicted-CLsys workflow enters a predicted systemic clearance "
     "directly, rather than estimating parameters by fitting clinical PK observations."),
    ("PBPK_v3_CLsys (top-down clearance based on ML-predicted CLsys)",
     "PBPK_v3_CLsys (CLsys-based parameterization using ML-predicted CLsys)"),
    ("0.562 for top-down DL–ML CLsys", "0.562 for predicted DL–ML CLsys"),
    ("against the top-down workflow with predicted DL–ML CLsys",
     "against the predicted-CLsys workflow using DL–ML CLsys"),
    # the same Figure 5 caption in markdown carries <sub> markup, so it needs its own literal
    ("against the top-down workflow with predicted DL–ML CL<sub>sys</sub>",
     "against the predicted-CL<sub>sys</sub> workflow using DL–ML CL<sub>sys</sub>"),
    ("PBPK_v3_CLsys (top-down)", "PBPK_v3_CLsys (predicted CLsys)"),
    ("full PBPK, bottom-up / top-down", "full PBPK, CLint / CLsys"),
    # note: the SI still labels scenarios "top-down, full PBPK" in its definition tables. The SI is a
    # separate document and was not in this round's scope; see CHANGELOG_discussion.md.
    # --- the Abstract, Introduction and Discussion carrier text of v7.4_methods_results_draft02.docx.
    #     That file's Methods+Results are the deliverable; its other sections are unrevised carrier,
    #     superseded by discussion/v7.4_discussion_round5c.docx. Swept so no stale label survives a grep.
    ("we compared bottom-up (intrinsic clearance–based) and top-down (systemic clearance–based) "
     "clearance parameterizations",
     "we compared bottom-up (IVIVE-scaled intrinsic clearance, CLint) and CLsys-based "
     "(predicted systemic clearance) parameterizations"),
    ("the top-down approach reduced directional bias",
     "the predicted-CLsys workflow reduced directional bias"),
    (', whereas "top-down" refers to parameterization using machine-learning–predicted CL',
     ", whereas the predicted-CL"),
    (". We acknowledge that the latter does not represent classical top-down fitting to clinical "
     "concentration–time data; the terminology is used here solely to distinguish systemic "
     "clearance–based (top-down) versus mechanistic intrinsic clearance–based (bottom-up) "
     "parameterization strategies.",
     " workflow, also referred to as CLsys-based parameterization, refers to entering a "
     "machine-learning–predicted systemic clearance directly as the total clearance. This is not "
     "classical top-down PBPK: no parameter is fitted to clinical concentration–time data. The two "
     "strategies are distinguished here by the clearance quantity each supplies to the model."),
    ("how bottom-up versus top-down clearance parameterizations affect exposure predictions",
     "how bottom-up (IVIVE-scaled CLint) versus CLsys-based clearance parameterizations affect "
     "exposure predictions"),
    ("evaluating systemic clearance–based (top-down) parameterization strategies",
     "evaluating systemic clearance–based parameterization strategies"),
    ("we directly contrasted bottom-up (intrinsic clearance–based) and top-down "
     "(systemic clearance–based) approaches",
     "we directly contrasted bottom-up (IVIVE-scaled CLint) and CLsys-based approaches"),

    # --- "bottom-up" dropped here: the Simcyp library baseline is not IVIVE of CL_int ---
    ("mechanistically optimized, bottom-up PBPK models", "mechanistically optimized PBPK models"),
    # --- citation corrections ---
    ("The renal-clearance sensitivity analysis is Figure S20, and the secondary comparison of the "
     "workflows as practically configured is in Table S11.",
     "The renal-clearance sensitivity analysis and the secondary comparison of the workflows as "
     "practically configured are both in Table S11."),
    ("templates ship a full PBPK distribution model, so it is a structural rather than an input effect "
     "(Figures S15 and S16; Table S9)",
     "templates ship a full PBPK distribution model (Figure S15b), so it is a structural rather than an "
     "input effect (Figures S15 and S16)"),
]

# the .docx files carry CL_sys / CL_int as real subscript runs; matching is on flattened text, so the
# search strings above work unchanged, but the replacements must re-create the subscripts
SUBMAP = [("CLsys", "CL<sub>sys</sub>"), ("CLint", "CL<sub>int</sub>")]


def main():
    spec = importlib.util.spec_from_file_location(
        "bd", Path(__file__).with_name("build_discussion_round5c.py"))
    BD = importlib.util.module_from_spec(spec)
    sys.modules["bd"] = BD
    spec.loader.exec_module(BD)

    hits = {old: 0 for old, _ in SUBS}

    for rel in MD:
        f = D2 / rel
        shutil.copy2(f, f.with_suffix(f.suffix + ".pre_round5c"))
        txt = f.read_text()
        for old, new in SUBS:
            n = txt.count(old)
            if n:
                txt = txt.replace(old, new)
                hits[old] += n
        f.write_text(txt)
        print("  %-48s rewritten" % rel)

    for rel in DOCX:
        f = D2 / rel
        shutil.copy2(f, f.with_suffix(".pre_round5c.docx"))
        d = Document(str(f))
        targets = list(d.paragraphs)
        for t in d.tables:
            for r in t.rows:
                for c in r.cells:
                    targets += list(c.paragraphs)
        for old, new in SUBS:
            rich = new
            for a, b in SUBMAP:
                rich = rich.replace(a, b)
            for par in targets:
                while old in par.text:
                    if not BD.rich_replace(par, old, rich, "clean"):
                        break
                    hits[old] += 1
        d.save(str(f))
        print("  %-48s rewritten" % rel)

    print()
    dead = [o[:70] for o, n in hits.items() if n == 0]
    for old, n in hits.items():
        print("  %3d x  %s" % (n, old[:88]))
    if dead:
        raise SystemExit("\nsubstitutions that never matched:\n  " + "\n  ".join(dead))
    print("\nall %d substitutions landed; .pre_round5c backups written beside each file" % len(SUBS))


if __name__ == "__main__":
    main()
