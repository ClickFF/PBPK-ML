#!/usr/bin/env python3
"""Round 5C QC — Introduction, Abstract and Conclusions, against the frozen Discussion manuscript.

Asserts that the frozen Discussion, the Methods, the Results and the reference list are untouched, that
every EndNote field in the Introduction survived, and that the three rewritten sections obey the
guardrails in round5c_framing_plan_v2_guardrailed.md.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v7.4_final01/qc_framing_round5c_final.py
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document

S21 = Path(__file__).resolve().parents[2]
D2 = S21 / "manuscript" / "v7.4" / "final_01" / "draft_02"
FROZEN = D2 / "discussion" / "v7.4_discussion_round5c.docx"
NEW = D2 / "framing_draft" / "v7.4_framing_round5c.docx"
HI = D2 / "framing_draft" / "v7.4_framing_round5c_highlighted.docx"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

FORBIDDEN = ["improvement", "superior", "superiority", "advantage", "strategy preference",
             "benefit score", "outperform", "better than", "proves", "demonstrates that",
             "uniformly superior", "modest numerical"]
R = []


def chk(name, ok, detail):
    R.append((name, bool(ok), detail))


def fields(p):
    return sum(1 for r in p._p.findall(".//" + W + "fldChar") if r.get(W + "fldCharType") == "begin")


def main():
    a, b = Document(str(FROZEN)), Document(str(NEW))
    pa, pb = a.paragraphs, b.paragraphs
    ta, tb = [p.text for p in pa], [p.text for p in pb]

    # --- 1-3 the frozen material ---------------------------------------------------------------------
    da = next(i for i, x in enumerate(ta) if x.strip().upper() == "DISCUSSION")
    db = next(i for i, x in enumerate(tb) if x.strip().upper() == "DISCUSSION")
    ca = next(i for i, x in enumerate(ta) if x.strip().upper() == "CONCLUSIONS")
    cb = next(i for i, x in enumerate(tb) if x.strip().upper() == "CONCLUSIONS")
    chk("1. The frozen Discussion is byte-identical (%d paragraphs)" % (ca - da),
        ta[da:ca] == tb[db:cb], "offset %d; first divergence %s" % (db - da,
        next((k for k in range(ca - da) if ta[da + k] != tb[db + k]), "none")))

    ms = next(i for i, x in enumerate(ta) if x.strip().upper() == "METHODS")
    chk("2. METHODS through the end of RESULTS is byte-identical (%d paragraphs)" % (da - ms),
        ta[ms:da] == tb[ms:db], "first divergence: %s"
        % next((k + ms for k in range(da - ms) if ta[ms + k] != tb[ms + k]), "none"))

    ra = [x for x in ta if re.match(r"^\(\d+\)\s", x.strip())]
    rb = [x for x in tb if re.match(r"^\(\d+\)\s", x.strip())]
    nums = [int(re.match(r"^\((\d+)\)", x.strip()).group(1)) for x in rb]
    chk("3. Reference list unchanged, %d entries, numbering 1..%d unreordered" % (len(rb), len(rb)),
        ra == rb and nums == list(range(1, len(nums) + 1)), "identical" if ra == rb else "CHANGED")

    # --- 4 the Introduction's live citations ----------------------------------------------------------
    fa = sum(fields(p) for p in pa[12:20])
    fb = sum(fields(p) for p in pb[12:21])
    chk("4. Every EndNote field in the Introduction survives", fa == fb,
        "%d field starts before, %d after" % (fa, fb))
    sa = set(re.findall(r"\d+", " ".join(
        r.text for p in pa[12:20] for r in p.runs
        if (r._element.find(".//" + W + "vertAlign") is not None))))
    sb = set(re.findall(r"\d+", " ".join(
        r.text for p in pb[12:21] for r in p.runs
        if (r._element.find(".//" + W + "vertAlign") is not None))))
    chk("5. No Introduction citation added or renumbered", sb <= sa,
        "%d distinct numbers, unchanged" % len(sb) if sb <= sa else "new: %s" % sorted(sb - sa))

    # --- 6-9 the Abstract -----------------------------------------------------------------------------
    ab = tb[11]
    chk("6. Abstract within the JCIM 250-word cap", len(ab.split()) <= 250, "%d words" % len(ab.split()))
    anchors = re.findall(r"\d+\.?\d*%|\b41\b|\b54\b", ab)
    chk("7. Abstract carries anchor numbers (it had none)", len(anchors) >= 6,
        "%d numeric anchors: %s" % (len(anchors), anchors))
    gone = [c for c in ["improvement observed for systemic clearance", "reduced directional bias",
                        "more consistent twofold error coverage",
                        "heterogeneity in clearance strategy preference"] if c in ab]
    chk("8. The four contradicted Abstract claims are gone", not gone, "found: %s" % gone if gone
        else "all four removed")
    chk("9. Abstract keeps the no-equivalence guardrail",
        "does not establish equivalence" in ab, "present")

    # --- 10-12 the Conclusions ------------------------------------------------------------------------
    con = tb[cb + 1] + "\n" + tb[cb + 2]
    # the baseline fused both blocks into one paragraph; the second must now start its own
    chk("10. Conclusions split into two paragraphs",
        bool(tb[cb + 1].strip()) and tb[cb + 2].strip().startswith("Hybrid ML–PBPK strategies")
        and len(ta[ca + 1].split()) > len(tb[cb + 1].split()),
        "%d + %d words (was one paragraph of %d)"
        % (len(tb[cb + 1].split()), len(tb[cb + 2].split()), len(ta[ca + 1].split())))
    chk("11. Conclusions drops 'modest numerical advantage' and 'uniformly superior'",
        "modest numerical" not in con and "uniformly superior" not in con, "both removed")
    chk("12. Conclusions narrows the early-discovery translation claim",
        "curated Simcyp compound files already exist" in con, "scope stated")

    # --- 13-16 guardrails across all three sections ---------------------------------------------------
    new_text = ab + "\n" + tb[15] + "\n" + tb[16] + "\n" + tb[17] + "\n" + tb[18] + "\n" + con
    low = new_text.lower()
    hits = []
    for w in FORBIDDEN:
        for m in re.finditer(re.escape(w), low):
            lo = max(0, m.start() - 90)
            if re.search(r"\b(no|not|nor|rather than|does not|did not|without)\b",
                         new_text[lo:m.end()], re.I):
                continue
            hits.append("%r -> %s" % (w, new_text[lo:m.end() + 50].replace("\n", " ")))
    chk("13. No unqualified comparative claim (guardrail 5)", not hits,
        "%d hit(s): %s" % (len(hits), hits[:2]) if hits else "0 hits across %d terms" % len(FORBIDDEN))

    body = "\n".join(tb[:cb + 3])
    td = [i for i, x in enumerate(tb[:160]) if "top-down" in x.lower()]
    bad = [i for i in td if not re.search(r"(classical top-down|top-down parameterization estimates)", tb[i])]
    chk("14. 'top-down' still never applied to this study's workflow", not bad, "paragraphs %s" % td)
    chk("15. Vd renamed to VD_ss throughout (R3 Minor 13)",
        not re.search(r"(?<![A-Za-z])Vd(?![A-Za-z])", body), "no bare 'Vd' remains")

    ev = ((D2 / "maintext" / "body_results.md").read_text()
          + (D2 / "maintext" / "body_methods.md").read_text())
    q = set(re.findall(r"\d+\.\d+|\b\d{2,}\b", ab + con))
    unsourced = sorted(n for n in q if n not in ev)
    chk("16. Every Abstract/Conclusions quantity appears in the Results or Methods (guardrail 8)",
        not unsourced, "unsourced: %s" % unsourced if unsourced else "all %d traced" % len(q))

    # --- 17-19 structure ------------------------------------------------------------------------------
    fig1 = next(i for i, x in enumerate(tb) if x.strip().startswith("FIGURE 1."))
    chk("17. Figure 1 caption follows the paragraph that introduces it",
        "as shown in Figure 1" in tb[fig1 - 1], "caption at paragraph %d" % fig1)
    asks = [i for i in range(12, 21) if re.search(r"\(i\)\s*whether|First, we ask", tb[i])]
    chk("18. The two research questions are stated once, not twice", len(asks) == 1,
        "stated in paragraph(s) %s" % asks)
    chk("19. The comma splice is repaired",
        "platform., we investigated" not in body and " , which is referred as" not in body, "both fixed")

    h = Document(str(HI))
    th = [p.text for p in h.paragraphs]
    chk("20. Highlighted twin keeps the same reference list",
        [x for x in th if re.match(r"^\(\d+\)\s", x.strip())] == ra, "identical")
    ins = sum(1 for r in h.paragraphs[11].runs if r.font.highlight_color is not None)
    dele = sum(1 for r in h.paragraphs[11].runs if r.font.strike)
    intro = sum(1 for p in h.paragraphs[12:21] for r in p.runs
                if r.font.strike or r.font.highlight_color is not None)
    chk("21. Highlighted twin marks both insertions and deletions",
        ins and dele and intro, "Abstract %d added / %d removed; Introduction %d marked runs"
        % (ins, dele, intro))

    width = max(len(n) for n, _, _ in R)
    for n, ok, d in R:
        print("%-*s  %s  %s" % (width, n, "PASS" if ok else "FAIL", d))
    bad_ = [n for n, ok, _ in R if not ok]
    print("\n%d/%d checks pass" % (len(R) - len(bad_), len(R)))
    if bad_:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
