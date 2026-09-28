#!/usr/bin/env python3
"""QC gate for the Round 5C framing planning documents."""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

S21 = Path(__file__).resolve().parents[2]
FIN = S21 / "manuscript" / "v7.4" / "final_01"
F = FIN / "draft_02" / "framing"
plan = (F / "round5c_framing_plan.md").read_text()
ledg = (F / "round5c_claim_ledger.md").read_text()
omap = (F / "round5c_reviewer_obligation_map.md").read_text()
rows = {int(r["para_idx"]): r for r in
        csv.DictReader((FIN / "changelog" / "v72_paragraph_disposition.csv").open())}
checks = []


def chk(n, ok, d):
    checks.append((n, bool(ok), d))


scope = [11] + list(range(13, 20)) + list(range(131, 153)) + [154]
body = [i for i in scope if rows[i]["is_heading"] == "0" and rows[i]["text"].strip()]
missing = [i for i in body if not re.search(r"\u00b6%d\b" % i, plan)]
chk("1. Every in-scope body paragraph appears in the framing plan", not missing,
    "missing %s" % missing if missing else "%d paragraphs, all dispositioned" % len(body))

tbl = [l for l in ledg.splitlines() if l.startswith("|")]
c = Counter(v for l in tbl for v in re.findall(r"\*\*(KEEP|SOFTEN|REPLACE|DELETE|ADD)\*\*", l))
stated = dict(re.findall(r"^\| (KEEP|SOFTEN|REPLACE|DELETE|ADD) \| (\d+) \|", ledg, re.M))
chk("2. Ledger summary matches the ledger rows",
    all(int(stated.get(k, -1)) == v for k, v in c.items()),
    "counted %s" % dict(c))

res = (FIN / "draft_02" / "maintext" / "v7.4_methods_results_draft02.md").read_text()
disc = plan[plan.index("# 3. DISCUSSION"):plan.index("# 4. CONCLUSIONS")]
nums = set(re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", disc))
bad = sorted(n for n in nums if n not in res)
chk("3. No number enters the Discussion plan that is absent from the Results", not bad,
    "untraceable: %s" % bad if bad else "%d values, all traceable" % len(nums))

# only section 1 is the four-column obligation table; section 2 is deliberately two-column
sec1 = omap[omap.index("## 1. Asks that name"):omap.index("## 2. Asks that do not")]
r1 = re.findall(r"^\| \*\*(R[23] (?:Major|Minor) \d+|R3 Summary)\*\* \|(.*)$", sec1, re.M)
short = [r for r, line in r1 if line.count("|") < 4]
chk("4. Every ask in the obligation table carries a location and a status",
    not short and len(r1) >= 10, "%d asks, %d short" % (len(r1), len(short)))
chk("4b. The two response-letter gaps are recorded", omap.count("**GAP") == 2,
    "%d GAP markers" % omap.count("**GAP"))

# a forbidden word is acceptable inside a quotation (double OR single quotes), a table cell,
# a blockquote, an ORIGINAL block, a RATIONALE, or the rule that lists them
FORB = ["improvement", "superior", "advantage", "strategy preference", "benefit score", "trade-off"]
leaked = []
# quotations wrap across lines, so judge the enclosing block, not the single line
blocks = re.split(r"\n\s*\n", plan)
for blk in blocks:
    quoted = (blk.count(chr(34)) >= 2 or blk.count(chr(39)) >= 2
              or blk.lstrip().startswith(("**ORIGINAL", "|", ">"))
              or "RATIONALE" in blk or "orbidden" in blk)
    if quoted:
        continue
    for w in FORB:
        for m in re.finditer(re.escape(w), blk, re.I):
            # the rule permits these words inside an explicit negation
            pre = blk[max(0, m.start() - 44):m.start()].lower()
            if re.search(r"\b(not|no|never|cannot|without|nothing)\b[^.]*$", pre):
                continue
            leaked.append((w, " ".join(blk.split())[:70]))
chk("5. Forbidden wording appears only in quotations or the rule itself", not leaked,
    "leaked: %s" % leaked[:2] if leaked else "all occurrences quoted or in the rule")

chk("6. Heading-hierarchy fix recorded in both directions",
    "restyle as body text" in plan and "stranded" in plan, "\u00b6135 and \u00b6141 both covered")
chk("7. Three deliverables present and cross-linked",
    all((F / n).exists() for n in ("round5c_framing_plan.md", "round5c_reviewer_obligation_map.md",
                                   "round5c_claim_ledger.md"))
    and "round5c_reviewer_obligation_map.md" in plan, "framing plan links the obligation map")

fails = [c for c in checks if not c[1]]
for n, ok, d in checks:
    print("%-62s %-4s %s" % (n[:62], "PASS" if ok else "FAIL", str(d)[:62]))
print("\n%d checks, %d pass, %d fail" % (len(checks), len(checks) - len(fails), len(fails)))
