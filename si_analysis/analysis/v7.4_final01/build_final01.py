#!/usr/bin/env python3
"""v7.4 final_01 — assemble the Methods+Results markdown draft.

Concatenates src/body_methods.md and src/body_results.md into draft/v7.4_methods_results.md, substituting
each <<<FROZEN_CAPTION:n>>> marker with the caption for main-text Figure n taken verbatim from
sources/captions/v7.4_display_package_round4c.md. Also writes draft/v7.4_carrier_full.md (front carrier +
bodies + back carrier) so the highlighter can diff a complete document and so any highlight mark outside
METHODS/RESULTS is visible as a scope violation.

Writes only inside manuscript/v7.4/final_01/draft/.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v7.4_final01/build_final01.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

S21 = Path(__file__).resolve().parents[2]
FINAL = S21 / "manuscript" / "v7.4" / "final_01"
SRC = FINAL / "src"
DRAFT = FINAL / "draft"
FROZEN = FINAL / "sources" / "captions" / "v7.4_display_package_round4c.md"


def frozen_captions() -> dict[int, str]:
    """Figure number -> caption text, de-blockquoted, exactly as Round 4C froze it."""
    out, cur, buf = {}, None, []
    for line in FROZEN.read_text().splitlines():
        m = re.match(r"^## Figure (\d+)\s*$", line)
        if m:
            if cur is not None:
                out[cur] = " ".join(buf).strip()
            cur, buf = int(m.group(1)), []
            continue
        if cur is None:
            continue
        if line.startswith("> "):
            buf.append(line[2:].strip())
        elif line.startswith(">"):
            buf.append(line[1:].strip())
        elif line.startswith("*Round 4C change:"):
            if buf:
                out[cur] = " ".join(buf).strip()
            cur, buf = None, []
    if cur is not None and buf:
        out[cur] = " ".join(buf).strip()
    return out


def main():
    caps = frozen_captions()
    missing = [n for n in (3, 4, 5, 6) if n not in caps]
    if missing:
        raise SystemExit("frozen captions not found for figures %s" % missing)

    bodies = []
    for name in ("body_methods.md", "body_results.md"):
        text = (SRC / name).read_text()

        def sub(m):
            n = int(m.group(1))
            if n not in caps:
                raise SystemExit("no frozen caption for Figure %d (in %s)" % (n, name))
            return caps[n]

        text, n_sub = re.subn(r"<<<FROZEN_CAPTION:(\d+)>>>", sub, text)
        left = re.findall(r"<<<[^>]*>>>", text)
        if left:
            raise SystemExit("unsubstituted markers in %s: %s" % (name, left))
        print("%-18s %d frozen captions substituted" % (name, n_sub))
        bodies.append(text.rstrip() + "\n")

    DRAFT.mkdir(parents=True, exist_ok=True)
    draft = "\n".join(bodies)
    (DRAFT / "v7.4_methods_results.md").write_text(draft)

    full = "\n".join([(SRC / "carrier_front.md").read_text().rstrip(), "", draft.rstrip(), "",
                      (SRC / "carrier_back.md").read_text().rstrip(), ""])
    (DRAFT / "v7.4_carrier_full.md").write_text(full)

    words = len(draft.split())
    print("draft/v7.4_methods_results.md  %d words" % words)
    print("draft/v7.4_carrier_full.md     %d words" % len(full.split()))
    for n in sorted(caps):
        print("  Figure %d caption: %d words" % (n, len(caps[n].split())))


if __name__ == "__main__":
    sys.exit(main())
