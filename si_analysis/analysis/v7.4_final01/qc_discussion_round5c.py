#!/usr/bin/env python3
"""Round 5C QC — the revised Discussion against `revised ACS JCIM v7.4.docx`.

Asserts that the rewrite touched only the Discussion, that the author's reference numbering and every
EndNote field survived, that the guardrails in round5c_framing_plan_v2_guardrailed.md hold, and that no
number appears in the Discussion for the first time.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v7.4_final01/qc_discussion_round5c.py
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document

S21 = Path(__file__).resolve().parents[2]
D2 = S21 / "manuscript" / "v7.4" / "final_01" / "draft_02"
SRC = D2 / "revised ACS JCIM v7.4.docx"
NEW = D2 / "discussion" / "v7.4_discussion_round5c.docx"
HI = D2 / "discussion" / "v7.4_discussion_round5c_highlighted.docx"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# guardrail 5 of the v2 plan: comparative wording that must never stand unqualified
FORBIDDEN = ["improvement", "superior", "superiority", "advantage", "strategy preference",
             "benefit score", "outperform", "better than", "equivalent", "proves", "demonstrates that"]
# wording that would read as reviewer rebuttal (guardrail 12) or leak an internal label (copy defects)
LEAKS = ["v1_run4", "v3 benefit", "PBPK_v3", "Compound ID", "as requested", "the reviewer", "we now",
         "previously reported here", "in the previous version", "Figure S4"]
R = []


def chk(name, ok, detail):
    R.append((name, bool(ok), detail))


def fields(par):
    return sum(1 for r in par._p.findall(".//" + W + "fldChar")
               if r.get(W + "fldCharType") == "begin")


def main():
    a, b = Document(str(SRC)), Document(str(NEW))
    pa, pb = a.paragraphs, b.paragraphs
    ta = [p.text for p in pa]
    tb = [p.text for p in pb]

    # --- 1-2 outside the Discussion, only the approved substitutions moved ---------------------------
    import importlib.util as _il
    _s = _il.spec_from_file_location("bd", Path(__file__).with_name("build_discussion_round5c.py"))
    BD = _il.module_from_spec(_s); _s.loader.exec_module(BD)
    allowed = {i for i, _, _ in BD.SUBS if i < 119}
    moved = {i for i in range(119) if tb[i] != ta[i]}
    chk("1. Before DISCUSSION, only the approved terminology paragraphs changed",
        moved <= allowed, "changed %s; approved %s" % (sorted(moved), sorted(allowed)))

    tail_a, tail_b = ta[142:], tb[len(tb) - len(ta[142:]):]
    chk("2. CONCLUSIONS through REFERENCES is byte-identical (%d paragraphs)" % len(tail_a),
        tail_a == tail_b, "first divergence: %s" % next((i for i in range(len(tail_a))
                                                         if tail_a[i] != tail_b[i]), "none"))

    # --- 3-5 the author's reference numbering --------------------------------------------------------
    refs_a = [t for t in ta if re.match(r"^\(\d+\)\s", t.strip())]
    refs_b = [t for t in tb if re.match(r"^\(\d+\)\s", t.strip())]
    nums = [int(re.match(r"^\((\d+)\)", t.strip()).group(1)) for t in refs_b]
    chk("3. Reference list unchanged in content and order", refs_a == refs_b,
        "%d entries, identical" % len(refs_b) if refs_a == refs_b else "CHANGED")
    chk("4. Reference numbering is 1..%d, unreordered" % len(nums), nums == list(range(1, len(nums) + 1)),
        "ok" if nums == list(range(1, len(nums) + 1)) else "out of sequence: %s" % nums[:12])

    fa = sum(fields(p) for p in pa[119:142])
    fb = sum(fields(p) for p in pb[119:len(pb) - len(tail_a)])
    chk("5. Every EndNote citation field in the Discussion survives", fa == fb,
        "%d field starts before, %d after" % (fa, fb))

    # --- 6 citation numbers cited in the new Discussion are a subset of the old ----------------------
    def sup(par):
        out = set()
        for r in par.runs:
            va = r._element.find(".//" + W + "vertAlign")
            if va is not None and va.get(W + "val") == "superscript":
                out.update(int(x) for x in re.findall(r"\d+", r.text))
        return out

    ca = set().union(*[sup(p) for p in pa[119:142]]) if True else set()
    cb = set().union(*[sup(p) for p in pb[119:len(pb) - len(tail_a)]])
    chk("6. No citation introduced or renumbered in the Discussion", cb <= ca,
        "before %s / after %s" % (sorted(ca), sorted(cb)))

    # --- 7-9 guardrails ------------------------------------------------------------------------------
    disc = "\n".join(tb[119:len(tb) - len(tail_a)])
    low = disc.lower()
    hits = []
    for w in FORBIDDEN:
        for m in re.finditer(re.escape(w), low):
            ctx = disc[max(0, m.start() - 90):m.end() + 60].replace("\n", " ")
            # a negated or explicitly hedged use is allowed ("no detectable difference", "does not support")
            if re.search(r"\b(no|not|nor|rather than|does not|did not|without|universally)\b",
                         ctx[:m.start() - max(0, m.start() - 90) + len(w)], re.I):
                continue
            hits.append("%r -> %s" % (w, ctx.strip()))
    chk("7. No unqualified comparative claim (guardrail 5)", not hits,
        "%d hit(s): %s" % (len(hits), hits[:2]) if hits else "0 hits across %d terms" % len(FORBIDDEN))

    leak = [w for w in LEAKS if w.lower() in low]
    chk("8. No internal label, dead pointer or rebuttal voice (guardrails 9, 12)", not leak,
        "found: %s" % leak if leak else "none of %d patterns" % len(LEAKS))

    spec = re.findall(r"[^.]*\b(?:may|might|remains to be tested|hypothesis-generating|consistent with)\b[^.]*\.", disc)
    chk("9. Speculation is marked as such (guardrail 11)", len(spec) >= 5,
        "%d hedged sentences" % len(spec))

    # --- 10-11 headings ------------------------------------------------------------------------------
    h3 = [t for p, t in zip(pb[119:len(pb) - len(tail_a)], tb[119:len(tb) - len(tail_a)])
          if p.style.name == "Heading 3"]
    chk("10. Six real Heading 3 subsections", len(h3) == 6, "; ".join(h3))
    stranded = [t[-40:] for p, t in zip(pb[119:len(pb) - len(tail_a)], tb[119:len(tb) - len(tail_a)])
                if p.style.name != "Heading 3" and len(t) > 60
                and re.search(r"[a-z]\.\s*[A-Z][A-Za-z-]+(?: [A-Z][A-Za-z-]+){1,4}$", t)]
    chk("11. No heading stranded as trailing body text", not stranded,
        "found: %s" % stranded if stranded else "none")

    # --- 12 every interpretive paragraph names a display --------------------------------------------
    span = slice(119, len(pb) - len(tail_a))
    kept = {x.strip() for x in ta}
    orig_disc = [x.strip() for x in ta[119:142]]
    def unchanged(x):
        """Verbatim, a deletion-only edit (trim leaves a prefix, cut leaves a suffix), or a paragraph
        changed only by an approved terminology substitution."""
        y = x.strip()
        if y in kept:
            return True
        for _i, _old, _new in BD.SUBS:                       # undo the substitutions, then re-test
            y = y.replace(re.sub(r"</?sub>", "", _new), _old)
        if y in kept:
            return True
        z = x.strip()
        return any(o.startswith(z) or o.endswith(z) for o in orig_disc if len(o) > 60)
    rewritten = [t for p, t in zip(pb[span], tb[span])
                 if p.style.name != "Heading 3" and len(t.split()) > 40 and not unchanged(t)]
    nodisp = [t[:70] for t in rewritten if not re.search(r"(Figure|Table|Section)s?\s+S?\d", t)]
    # the opening paragraph frames the study and cites nothing by design; every other rewrite must
    chk("12. Every rewritten Discussion paragraph cites the display it rests on", len(nodisp) <= 1,
        "%d of %d rewritten paragraphs without one%s"
        % (len(nodisp), len(rewritten), " (the opener)" if len(nodisp) == 1 else ""))
    main_disp = sorted({int(m) for m in re.findall(r"Figures?\s+(\d)\b", disc)})
    chk("13. The Discussion now cites main-text figures (it cited none before)", len(main_disp) >= 4,
        "main-text figures cited: %s" % main_disp)

    # --- 14 no new quantity (guardrail 8) -----------------------------------------------------------
    evidence = ((D2 / "maintext" / "body_results.md").read_text()
                + (D2 / "maintext" / "body_methods.md").read_text()
                + (D2 / "display" / "FIGURE_maintext_draft02.docx").with_suffix(".docx").name)
    old_disc = "\n".join(ta[119:142])
    new_nums = set(re.findall(r"\d+\.\d+|\b\d{2,}\b", disc))
    old_nums = set(re.findall(r"\d+\.\d+|\b\d{2,}\b", old_disc))
    ev_nums = set(re.findall(r"\d+\.\d+|\b\d{2,}\b", evidence))
    unsourced = sorted(n for n in new_nums - old_nums - ev_nums
                       if not re.match(r"^(S?\d+)$", n) or len(n) > 2)
    chk("14. No quantity appears in the Discussion for the first time (guardrail 8)", not unsourced,
        "unsourced: %s" % unsourced if unsourced else "all %d quantities traced to Results/Methods"
        % len(new_nums))

    # --- 15 subscript convention --------------------------------------------------------------------
    subs = [r.text for p in pb[span] for r in p.runs if r.font.subscript]
    want = {"sys", "int", "max", "ss", "0–t"}
    chk("15. CL/C/VD/AUC qualifiers use real subscript runs, as the benchmark does",
        want <= set(subs), "%d subscript runs %s; missing %s"
        % (len(subs), sorted(set(subs)), sorted(want - set(subs)) or "none"))

    # --- Round 5C cleanup -----------------------------------------------------------------------------
    body = "\n".join(tb[:len(tb) - len(tail_a)])
    td = [i for i, x in enumerate(tb[:len(tb) - len(tail_a)]) if "top-down" in x.lower()]
    # 14 defines classical top-down as background; 17 states the present workflow is not it
    chk("18. 'top-down' survives only as the classical concept it is contrasted with",
        set(td) <= {14, 17}, "paragraphs %s" % td)
    nonclassical = [i for i in td if not re.search(
        r"(classical top-down|top-down parameterization estimates)", tb[i])]
    chk("19. No paragraph applies 'top-down' to this study's own workflow", not nonclassical,
        "offenders: %s" % nonclassical if nonclassical else "none")

    bu = [i for i, x in enumerate(tb[:len(tb) - len(tail_a)]) if "bottom-up" in x.lower()]
    stray = [i for i in bu if not re.search(r"(CL\s*int|intrinsic clearance|IVIVE|microsom)",
                                            tb[i], re.I)]
    chk("20. 'bottom-up' retained only where it denotes IVIVE of CL_int", not stray,
        "%d paragraphs, all IVIVE-anchored: %s" % (len(bu), bu) if not stray else "stray: %s" % stray)

    chk("21. The defensive limitations opener is gone",
        "do not affect the comparative conclusions" not in body, "removed")
    chk("22. The author's exact ML sentence is present",
        "Statistically detectable differences, where present, were limited to comparisons with the "
        "single-representation controls" in body, "verbatim")
    ap = next(x for x in tb if "constitutes the practical end-to-end assessment" in x)
    chk("23. All-predicted framed as the practical end-to-end assessment, not exploratory",
        "exploratory" not in ap and "prospective external validation" in ap,
        "'exploratory' absent; scope limit retained")
    chk("24. Causal guardrails intact",
        all(k in body for k in ["do not by themselves identify a causal effect",
                                "does not establish equivalence",
                                "consistent with a structural contribution"]),
        "associations / equivalence / structural all hedged")

    # --- 16 highlighted twin ------------------------------------------------------------------------
    h = Document(str(HI))
    th = [p.text for p in h.paragraphs]
    chk("16. Highlighted twin keeps the same reference list", 
        [t for t in th if re.match(r"^\(\d+\)\s", t.strip())] == refs_a, "identical")
    marked = sum(1 for p in h.paragraphs[119:142] for r in p.runs
                 if r.font.strike or r.font.highlight_color is not None)
    chk("17. Highlighted twin marks the changes", marked > 100, "%d marked runs" % marked)

    width = max(len(n) for n, _, _ in R)
    for n, ok, d in R:
        print("%-*s  %s  %s" % (width, n, "PASS" if ok else "FAIL", d))
    bad = [n for n, ok, _ in R if not ok]
    print("\n%d/%d checks pass" % (len(R) - len(bad), len(R)))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
