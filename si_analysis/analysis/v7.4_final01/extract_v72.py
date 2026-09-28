#!/usr/bin/env python3
"""v7.4 final_01 — extract the v7.2 body inventory and the untouched carrier text.

Reads the frozen baseline copy under final_01/sources/baseline/ and writes:

  changelog/v72_paragraph_disposition.csv  one row per body paragraph, pre-filled disposition=keep
  src/carrier_front.md                     title page through end of Introduction (never edited)
  src/carrier_back.md                      Discussion through References (never edited)
  qc/v72_baseline_stats.csv                sentence counts per subsection, the preservation denominators

Paragraph indices are python-docx body-paragraph indices, the same ones the migration map and the
change-log cite. OMML (Word equation) paragraphs are flagged because they cannot survive a markdown
round trip: the submission .docx is built by editing a copy of this file, not by regenerating it.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v7.4_final01/extract_v72.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from docx import Document

S21 = Path(__file__).resolve().parents[2]
FINAL = S21 / "manuscript" / "v7.4" / "final_01"
BASE = FINAL / "sources" / "baseline" / "revised ACS JCIM v7.2.docx"

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

METHODS = range(20, 84)
RESULTS = range(84, 130)


def has_omml(p) -> bool:
    return bool(p._p.findall(".//{%s}oMath" % M)) or bool(p._p.findall(".//{%s}oMathPara" % M))


def superscripts(p) -> str:
    """Superscript run text — the EndNote citation numbers, e.g. '44' or '44,45'."""
    out = []
    for r in p.runs:
        va = r._element.find(".//{%s}vertAlign" % W)
        if va is not None and va.get("{%s}val" % W) == "superscript":
            t = r.text.strip()
            if t:
                out.append(t)
    return " ".join(out)


def sentences(text: str) -> list[str]:
    """Split on sentence-final punctuation followed by a capital, protecting common abbreviations."""
    t = re.sub(r"\b(e\.g|i\.e|vs|et al|cf|approx|ca|Fig|Eq|Ref|Dr|No)\.", lambda m: m.group(1) + "<DOT>", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", t)
    return [s.replace("<DOT>", ".").strip() for s in parts if s.replace("<DOT>", ".").strip()]


def main():
    doc = Document(str(BASE))
    paras = doc.paragraphs

    section, subsection = "front matter", ""
    rows = []
    for i, p in enumerate(paras):
        text = p.text.strip()
        style = p.style.name
        upper = text.upper()
        if style.startswith("Heading") or style == "Style1":
            if upper in ("METHODS", "RESULTS", "DISCUSSION", "CONCLUSIONS", "INTRODUCTION",
                         "ABSTRACT", "ASSOCIATED CONTENT", "REFERENCES"):
                section, subsection = text.lower(), ""
            elif text:
                subsection = text
        rows.append({
            "para_idx": i,
            "section": section,
            "subsection": subsection,
            "style": style,
            "is_heading": int(style.startswith("Heading") or style == "Style1"),
            "has_omml": int(has_omml(p)),
            "citations": superscripts(p),
            "n_sentences": len(sentences(text)) if text and not style.startswith("Heading") else 0,
            "n_words": len(text.split()),
            "disposition": "keep",
            "unit_id": "",
            "sentences_kept_verbatim": "",
            "evidence_source": "",
            "flag_ids": "",
            "crossref_from": "",
            "crossref_to": "",
            "text": text,
        })

    (FINAL / "changelog").mkdir(parents=True, exist_ok=True)
    (FINAL / "src").mkdir(parents=True, exist_ok=True)
    (FINAL / "qc").mkdir(parents=True, exist_ok=True)

    with (FINAL / "changelog" / "v72_paragraph_disposition.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def carrier(idxs, title):
        out = ["<!-- %s: extracted verbatim from the v7.2 baseline, never edited. Present only so the"
               " highlighter can diff a complete document. -->" % title, ""]
        for i in idxs:
            r = rows[i]
            if not r["text"]:
                continue
            if r["is_heading"]:
                out += ["", "## " + r["text"] if r["style"] == "Style1" else "### " + r["text"], ""]
            else:
                out.append(r["text"])
                out.append("")
        return "\n".join(out) + "\n"

    (FINAL / "src" / "carrier_front.md").write_text(carrier(range(0, 20), "carrier front"))
    (FINAL / "src" / "carrier_back.md").write_text(carrier(range(130, len(rows)), "carrier back"))

    # preservation denominators, per subsection
    stats = {}
    for i in list(METHODS) + list(RESULTS):
        r = rows[i]
        if r["is_heading"] or not r["n_sentences"]:
            continue
        key = (r["section"], r["subsection"])
        s = stats.setdefault(key, {"paragraphs": 0, "sentences": 0, "words": 0, "omml": 0, "first": i, "last": i})
        s["paragraphs"] += 1
        s["sentences"] += r["n_sentences"]
        s["words"] += r["n_words"]
        s["omml"] += r["has_omml"]
        s["last"] = i

    with (FINAL / "qc" / "v72_baseline_stats.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "subsection", "first_para", "last_para", "paragraphs", "sentences",
                    "words", "omml_paragraphs", "words_per_sentence"])
        for (sec, sub), s in stats.items():
            w.writerow([sec, sub, s["first"], s["last"], s["paragraphs"], s["sentences"], s["words"],
                        s["omml"], round(s["words"] / s["sentences"], 1)])

    for sec in ("methods", "results"):
        sub = {k: v for k, v in stats.items() if k[0] == sec}
        print("%-8s %2d subsections  %3d paragraphs  %3d sentences  %d OMML paragraphs"
              % (sec, len(sub), sum(v["paragraphs"] for v in sub.values()),
                 sum(v["sentences"] for v in sub.values()), sum(v["omml"] for v in sub.values())))
    omml = [r["para_idx"] for r in rows if r["has_omml"]]
    print("OMML paragraphs overall:", omml)
    cited = {}
    for r in rows:
        for n in re.findall(r"\d+", r["citations"]):
            cited.setdefault(int(n), []).append(r["para_idx"])
    print("reference 47 cited in paragraphs:", cited.get(47, "NOT FOUND"))


if __name__ == "__main__":
    main()
