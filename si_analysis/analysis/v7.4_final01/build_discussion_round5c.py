#!/usr/bin/env python3
"""Round 5C — revised DISCUSSION only, built on `revised ACS JCIM v7.4.docx`.

Scope is the Discussion (paragraphs 119-141 of that file). Everything else — Abstract, Introduction, Methods,
Results, Conclusions, references — is left byte-identical, so the author's reference numbering is untouched.

EndNote citations are live fields (fldChar / instrText). A rebuilt paragraph would destroy them, so each
citation's run block is deep-copied out of its original paragraph and re-inserted into the new text at a
{{CITE:n}} marker. No citation is created, renumbered or reordered.

Writes:
  draft_02/discussion/v7.4_discussion_round5c.docx              clean
  draft_02/discussion/v7.4_discussion_round5c_highlighted.docx  word-level diff against the benchmark

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v7.4_final01/build_discussion_round5c.py
"""
from __future__ import annotations

import copy
import importlib.util
import re
import shutil
import sys
from pathlib import Path

from docx import Document
from docx.text.paragraph import Paragraph
from docx.text.run import Run

S21 = Path(__file__).resolve().parents[2]
D2 = S21 / "manuscript" / "v7.4" / "final_01" / "draft_02"
SRC = D2 / "revised ACS JCIM v7.4.docx"
OUT = D2 / "discussion"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

HL = None  # loaded in main


# ----------------------------------------------------------------- the revised Discussion
# keep      : paragraph untouched (its EndNote fields survive untouched)
# edit      : rebuilt from the template below
# delete    : removed
# heading   : new Heading 3 inserted after the given paragraph
# insert    : new body paragraph inserted after the given paragraph
PLAN = [
    ("edit", 120, """Human PK prediction remains a central challenge in drug discovery, particularly when \
early-stage compounds lack experimentally determined ADME parameters. In this study, we developed an \
integrated DL–ML framework for predicting key PK parameters and systematically evaluated its integration \
into a hybrid ML–PBPK workflow within the Simcyp platform. The objective was not only to compare predictive \
accuracy at the PK parameter level but also to assess how structure-derived estimates influence \
concentration–time (C–T) profile predictions under different clearance parameterizations and model \
complexity levels, i.e., minimal structure vs. whole-body PBPK. The contribution of this work is therefore \
not limited to developing another ML model for PK prediction. Rather, it provides an end-to-end evaluation \
of how molecular structure-derived PK predictions behave when embedded within a mechanistic PBPK workflow. \
The results indicate that graph-based embeddings are best viewed as complementary to conventional \
descriptors, and that in this dataset downstream PBPK performance was associated primarily with the accuracy \
of the predicted clearance input, together with a residual C<sub>max</sub> offset that was not explained by \
input prediction error alone."""),

    ("keep", 122, None),

    ("edit", 123, """On standardized benchmark datasets, the merged GNN+RDKit DL–ML models achieved \
performance broadly comparable to previously reported descriptor-based ML models. For CL<sub>sys</sub>, the \
merged representation was competitive with the reported descriptor-based model in the descriptive \
comparison, and in the compound-paired analysis no difference was detectable against the re-implemented and \
reproduced descriptor models. Statistically detectable differences, where present, were limited to \
comparisons with the single-representation controls (Table 2; Figures S2–S4). For Fu and VD<sub>ss</sub>, performance was similar \
or slightly less favorable compared with descriptor-based benchmarks. Overall, these results support a \
conservative interpretation: graph-based embeddings provide complementary molecular information when \
integrated with RDKit descriptors, rather than serving as a universally superior replacement for \
handcrafted representations."""),

    ("edit", 124, """This conclusion is consistent with endpoint-dependent behavior observed in the control \
analyses. The merged representation differed most from the single-representation controls for \
CL<sub>sys</sub>, the most challenging endpoint, whereas embedding-only and RDKit-only configurations showed \
differing performance patterns across endpoints (Figure S1 and Table S1). These results indicate that \
representation choice is not the sole limiting factor across PK endpoints. Rather, intrinsic biological \
variability, heterogeneous clearance mechanisms, and experimental noise likely impose fundamental \
constraints on achievable predictive accuracy, particularly for clearance-related endpoints. From a \
methodological standpoint, the two-stage DL–ML framework enables transparent comparison of representation \
strategies while providing a competitive and internally consistent feature set for downstream PBPK \
simulations."""),

    ("keep", 126, None),
    ("keep", 128, None),
    ("keep", 129, None),

    ("edit", 130, """Replacing an observed clearance value with a predicted one was associated with the \
largest change in exposure performance observed in this study. Across the scenarios in which clearance was \
predicted, the proportion of compounds whose AUC<sub>0–t</sub> fell within twofold of the observed value \
decreased from 82.9% to between 41.5% and 53.7%, and the concentration–time profile error increased \
correspondingly (Figure 4). By contrast, when the two predicted-clearance representations were compared \
under matched renal-clearance information — hepatic CL<sub>int</sub> scaled within the PBPK model against \
directly predicted CL<sub>sys</sub> — no difference was detectable on any of the four accuracy endpoints \
(Figure 5). An interval covering zero indicates no detectable difference; with 41 compounds it \
does not establish equivalence, and these data do not support a claim that either representation is \
preferable. Both arms are structure-derived predictions generated by different software from different \
training data, so the comparison concerns two practical prediction-and-parameterization workflows rather \
than an intrinsic property of either clearance formulation."""),

    ("heading", 130, "Compound-Level Heterogeneity"),

    ("edit", 131, """Compound-level analyses indicate that the relative difficulty of a compound was largely \
preserved regardless of the source of the clearance input. Predicting clearance increased the profile error \
for most compounds, yet the per-compound errors obtained with observed and with predicted clearance remained \
correlated (Figure S10), so compounds that were difficult to reproduce with an observed clearance value \
generally remained difficult when clearance was predicted. This pattern is consistent with a substantial \
part of the residual error being a property of the compound and its PBPK representation rather than of the \
clearance input, and it is in line with prior ML-driven hybrid PBPK investigations in which compound-level \
behavior varied within a fixed structural configuration.{{CITE:34, 39}}"""),

    ("edit", 132, """Stratifying the information-matched contrast by simplified ECCS class did not reveal a \
class-dependent difference between the two workflows. That analysis, together with the related mechanistic \
strata, is reported in the Supporting Information (Section S7) and is regarded as hypothesis-generating \
given the small number of compounds in each class.{{CITE:51}}"""),

    ("delete", 133, None),

    ("insert", 132, """To examine how input error relates to exposure error, the fold errors of the \
predicted inputs were compared with the exposure errors of the scenario in which Fu, VD<sub>ss</sub> and \
CL<sub>sys</sub> were all predicted (Figure 6). The fold error of CL<sub>sys</sub> was strongly associated \
with the AUC<sub>0–t</sub> fold error and was the principal correlate of the profile error, whereas the fold \
error of VD<sub>ss</sub> was more strongly associated with C<sub>max</sub>, and the fold error of Fu was not \
detectably associated with any exposure endpoint. These associations are correlational and were computed \
within a single scenario. They are consistent with an endpoint-dependent contribution of each input to the \
corresponding exposure measure, but they do not by themselves identify a causal effect."""),

    ("insert", 132, """The scenario in which Fu, VD<sub>ss</sub> and CL<sub>sys</sub> were all supplied by \
the DL–ML models constitutes the practical end-to-end assessment of the structure-to-PBPK workflow as it is \
currently configured. In that configuration, 41% of compounds had an AUC<sub>0–t</sub> within twofold of the \
observed value and 59% had a C<sub>max</sub> within twofold (Figure 7). This is the level of agreement the \
workflow delivers when every clearance- and distribution-related input is derived from molecular structure, \
and it is the figure against which its use for exposure assessment and compound ranking should be judged. \
Because the non-clearance parameters of these compound files encode prior clinical information, it \
characterizes the assembled workflow rather than constituting a prospective external validation on novel \
chemical matter."""),

    ("heading-retitle", 134, "PBPK Model Structure and the C<sub>max</sub> Offset"),
    ("keep", 135, None),
    # the third sentence restates the second and carries an unpaired em dash; its runs are deleted
    # individually so the two EndNote fields earlier in the paragraph are not disturbed
    ("trim", 136, "When parameter uncertainty remains "),

    ("edit", 137, """A systematic underprediction of C<sub>max</sub> was present in the minimal-structure \
scenarios and persisted when every input was set to its observed value, indicating that it is not \
attributable to input prediction error alone. Within the Simcyp library baseline, the offset was smaller for \
compounds whose template ships a full PBPK distribution model than for those using a minimal model, and when \
VD<sub>ss</sub> was observed the residual bias showed no detectable association with VD<sub>ss</sub>, whereas \
a VD<sub>ss</sub>-dependent component appeared once VD<sub>ss</sub> was predicted (Figure S15). Together \
these observations are consistent with a structural contribution from the representation of distribution, \
onto which error in predicted distribution adds, rather than with an effect of the clearance input. The \
offset is also not an artifact of how the peak is read from the simulated profile: restricting the \
comparison to observed times inside the simulated window, or taking the simulated peak instead of the value \
at the observed sampling times, moves the bias without removing it (Figure S16). Whether a \
compound-specific distribution model would reduce this offset in prospective use remains to be tested."""),

    ("edit", 139, """Overall, the heterogeneous behavior observed across compounds is consistent with the \
existence of an applicability domain for hybrid ML–PBPK modeling, although the present analysis \
characterizes it only in part. An embedding-distance analysis on the benchmark test set showed an \
informative relationship between distance to the training set and prediction error for Fu, but did not \
establish a validated standalone uncertainty measure across endpoints (Figure S17; Table S10). It therefore \
supports a fit-for-purpose–driven modeling philosophy, consistent with previously reported tiered PBPK \
frameworks in translational and high-throughput applications.{{CITE:53, 54}} The present PBPK evaluation was \
conducted on a small set of molecules (N = 41), which is insufficient to formally characterize the \
mechanistic determinants of compound-level performance. Expanding the evaluation dataset and stratifying \
compounds by elimination route, permeability class, or transporter involvement will be necessary to define \
such decision rules more precisely."""),

    ("edit", 140, """Several limitations should be acknowledged. PBPK simulations were restricted to the \
representative PK curves based on intravenous dosing in healthy adults, thereby minimizing absorption and \
formulation-related variability. Transporter-mediated and nonlinear processes were not explicitly \
subclassified. In addition, the size of the exclusive PBPK evaluation dataset (Test #2) limits statistical \
inference regarding compound-level behavior. Because the PBPK-oriented evaluation relied on compounds with \
sufficiently complete Simcyp-ready model inputs and templates, the resulting dataset represents a selected, \
well-characterized subset rather than an unbiased sample of early discovery chemical space. The \
non-clearance parameters of these template files encode prior clinical information, so the evaluation tests \
the propagation of predicted inputs through otherwise calibrated compound files rather than the \
construction of a model for a genuinely novel compound, and the early-discovery interpretation is narrowed \
accordingly. The primary profile statistic is computed without time weighting, so compounds sampled densely \
in the early phase contribute disproportionately to it; a time-weighted form is reported as a sensitivity \
analysis and does not change the ranking of scenarios (Table S9). In addition, although Test #2 compounds \
were excluded from the final refitting stage, the PBPK results should be interpreted as a secondary \
held-out assessment under fixed model settings rather than a fully independent external validation. Another \
limitation of the workflow is that the full re-execution of PBPK simulations requires access to Simcyp. To \
enhance transparency, we provide model input summaries, scenario definitions, simulation outputs, and \
evaluation scripts in the GitHub repository. Accordingly, all reported PBPK metrics and figures can be \
reproduced from archived outputs, while users with Simcyp access can reconstruct the corresponding \
simulation scenarios from the provided inputs."""),

    ("cut", 141, "These limitations do not affect the comparative conclusions but indicate areas for "
                 "further investigation. "),
]



# ------------------------------------------------------- Round 5C cleanup: reader-facing terminology sweep
# "bottom-up" is RETAINED wherever it denotes IVIVE of CL_int, and dropped where it does not.
# "top-down" is removed from reader-facing prose; the distinction is defined once, in the Introduction (17).
# Reference titles (paragraph 160 onward) are never touched: reference 23 is literally "... Bottom-Up
# Approach Using In Vitro Assay". Each entry is (paragraph index, exact old text, new text with <sub> markup).
SUBS = [
    # --- ABSTRACT (its contradicted claims are a later round; only the terminology changes here) ---
    (11, "we compared bottom-up (intrinsic clearance–based) and top-down (systemic clearance–based) "
         "clearance parameterizations",
         "we compared bottom-up (IVIVE-scaled intrinsic clearance, CL<sub>int</sub>) and CL<sub>sys</sub>-based "
         "(predicted systemic clearance) parameterizations"),
    (11, "the top-down approach reduced directional bias",
         "the predicted-CL<sub>sys</sub> workflow reduced directional bias"),

    # --- INTRODUCTION: this is where the distinction is defined, once ---
    (17, ', whereas "top-down" refers to parameterization using machine-learning–predicted CL',
         ", whereas the predicted-CL"),
    (17, ". We acknowledge that the latter does not represent classical top-down fitting to clinical "
         "concentration–time data; the terminology is used here solely to distinguish systemic "
         "clearance–based (top-down) versus mechanistic intrinsic clearance–based (bottom-up) "
         "parameterization strategies.",
         " workflow, also referred to as CL<sub>sys</sub>-based parameterization, refers to entering a "
         "machine-learning–predicted systemic clearance directly as the total clearance. This is not "
         "classical top-down PBPK: no parameter is fitted to clinical concentration–time data. The two "
         "strategies are distinguished here by the clearance quantity each supplies to the model."),
    (19, "how bottom-up versus top-down clearance parameterizations affect exposure predictions",
         "how bottom-up (IVIVE-scaled CL<sub>int</sub>) versus CL<sub>sys</sub>-based clearance "
         "parameterizations affect exposure predictions"),

    # --- METHODS: swept for consistency; leaving "top-down" here would contradict every other section ---
    (58, "In the observed-input and top-down scenarios",
         "In the observed-input and predicted-CL<sub>sys</sub> scenarios"),
    (58, "so that a comparison with the top-down workflow could be made",
         "so that a comparison with the predicted-CL<sub>sys</sub> workflow could be made"),
    (62, "and (b) a top-down approach using ML-predicted systemic clearance (CLsys)",
         "and (b) a CL<sub>sys</sub>-based approach entering ML-predicted systemic clearance "
         "(CL<sub>sys</sub>) directly"),
    # 63 defined the term a second time and apologised for it; the Introduction now carries the definition
    (63, 'For clarity, the term "top-down" is used operationally throughout this manuscript to denote '
         "PBPK models parameterized using predicted CL",
         "As defined in the Introduction, the predicted-CL"),
    (63, ", rather than parameters estimated by fitting clinical PK observations. This terminology is "
         "intended solely to distinguish the CLsys-based parameterization strategy from the IVIVE-based "
         "CLint strategy and should not be interpreted as representing a classical top-down PBPK, "
         "population pharmacokinetic, or model-calibration workflow.",
         " workflow enters a predicted systemic clearance directly, rather than estimating parameters by "
         "fitting clinical PK observations."),

    # --- RESULTS ---
    # the Simcyp library baseline is mechanistically parameterized but is not IVIVE of CL_int
    (98, "represent mechanistically optimized, bottom-up PBPK models",
         "represent mechanistically optimized PBPK models"),

    # --- DISCUSSION (approved; only the two terminology parentheticals change) ---
    (128, "evaluating systemic clearance–based (top-down) parameterization strategies",
          "evaluating systemic clearance–based parameterization strategies"),
    (129, "we directly contrasted bottom-up (intrinsic clearance–based) and top-down "
          "(systemic clearance–based) approaches",
          "we directly contrasted bottom-up (IVIVE-scaled CL<sub>int</sub>) and CL<sub>sys</sub>-based "
          "approaches"),
]


def citation_blocks(par):
    """Every EndNote citation in a paragraph, as (visible number text, deep-copied run elements)."""
    out, kids, i = [], list(par._p), 0
    while i < len(kids):
        el = kids[i]
        fc = el.find(W + "fldChar") if el.tag == W + "r" else None
        if fc is not None and fc.get(W + "fldCharType") == "begin":
            depth, j = 0, i
            while j < len(kids):
                e = kids[j]
                f = e.find(W + "fldChar") if e.tag == W + "r" else None
                if f is not None:
                    t = f.get(W + "fldCharType")
                    if t == "begin":
                        depth += 1
                    elif t == "end":
                        depth -= 1
                        if depth == 0:
                            break
                j += 1
            group = kids[i:j + 1]
            shown = "".join(e.find(W + "t").text for e in group
                            if e.tag == W + "r" and e.find(W + "t") is not None
                            and e.find(".//" + W + "vertAlign") is not None)
            out.append((shown.strip(), [copy.deepcopy(e) for e in group]))
            i = j + 1
            continue
        i += 1
    return out


def plain(t):
    """The template as running text: subscript markup dropped, citation numbers left where they sit."""
    t = re.sub(r"\{\{CITE:([^}]+)\}\}", r"\1", t)
    return re.sub(r"</?sub>", "", t)



def rich_replace(par, old, new, mode="clean"):
    """Replace `old` with `new` across run boundaries. `new` may carry <sub> markup.

    Matching is done on the paragraph's flattened run text, so a phrase split across runs (or across a
    subscript) still matches. Field-code runs contribute no text and are never disturbed, so EndNote
    citations survive. In diff mode the old text is kept, struck through, before the new runs.
    """
    from docx.enum.text import WD_COLOR_INDEX
    from docx.shared import RGBColor

    runs, spans, full = list(par.runs), [], ""
    for r in runs:
        t = r.text or ""
        spans.append((r, len(full), len(full) + len(t)))
        full += t
    i = full.find(old)
    if i < 0:
        return False
    j = i + len(old)
    affected = [(r, a, b) for r, a, b in spans if not (b <= i or a >= j)]
    first = affected[0][0]

    tail = ""
    for k, (r, a, b) in enumerate(affected):
        lo, hi = max(i, a) - a, min(j, b) - a
        t = r.text or ""
        if k == 0:
            tail = t[hi:] if len(affected) == 1 else ""
            r.text = t[:lo]
        elif k == len(affected) - 1:
            r.text = t[hi:]
        else:
            r.text = ""

    anchor = first._element
    def emit(text, script=None, kind=None):
        nonlocal anchor
        e = copy.deepcopy(first._element)
        for ch in list(e):
            if ch.tag != W + "rPr":
                e.remove(ch)
        anchor.addnext(e)
        anchor = e
        run = Run(e, par)
        run.text = text
        # subscript and superscript share one w:vertAlign element, so setting the other to None
        # would wipe the one just set; only ever touch the one that applies
        if script == "sub":
            run.font.subscript = True
        elif script == "sup":
            run.font.superscript = True
        if kind == "del":
            run.font.strike = True
            run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
        elif kind == "ins":
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW

    if mode == "diff":
        emit(old, kind="del")
    for seg in re.split(r"(<su[bp]>.*?</su[bp]>)", new):
        if not seg:
            continue
        m = re.fullmatch(r"<(su[bp])>(.*?)</su[bp]>", seg)
        emit(m.group(2) if m else seg, script=m.group(1) if m else None,
             kind="ins" if mode == "diff" else None)
    if tail:
        emit(tail)
    return True


def clear(par):
    for ch in list(par._p):
        if ch.tag not in (W + "pPr",):
            par._p.remove(ch)


def write(par, template, cites, mark=None):
    """Render a template with <sub> and {{CITE:n}} into `par`; `mark` = 'ins' highlights everything."""
    for chunk in re.split(r"(\{\{CITE:[^}]+\}\})", template):
        if not chunk:
            continue
        m = re.fullmatch(r"\{\{CITE:([^}]+)\}\}", chunk)
        if m:
            key = m.group(1).strip()
            block = cites.get(key)
            if block is None:
                raise SystemExit("citation %r not found among %s" % (key, sorted(cites)))
            for e in block:
                par._p.append(copy.deepcopy(e))
            continue
        for seg in re.split(r"(<su[bp]>.*?</su[bp]>)", chunk):
            if not seg:
                continue
            sm = re.fullmatch(r"<(su[bp])>(.*?)</su[bp]>", seg)
            r = par.add_run(sm.group(2) if sm else seg)
            if sm:
                if sm.group(1) == "sub":
                    r.font.subscript = True
                else:
                    r.font.superscript = True
            if mark == "ins":
                from docx.enum.text import WD_COLOR_INDEX
                r.font.highlight_color = WD_COLOR_INDEX.YELLOW


def build(highlight: bool):
    out = OUT / ("v7.4_discussion_round5c_highlighted.docx" if highlight
                 else "v7.4_discussion_round5c.docx")
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, out)
    doc = Document(str(out))
    paras = doc.paragraphs

    # harvest every citation in the Discussion before anything is rebuilt
    cites = {}
    for i in range(119, 142):
        for shown, block in citation_blocks(paras[i]):
            cites.setdefault(shown, block)

    if not highlight:
        print("  citation fields harvested:", ", ".join("[%s]" % k for k in cites))
    originals = {i: paras[i].text for i in range(119, 142)}
    stats = {"edit": 0, "delete": 0, "insert": 0, "heading": 0, "keep": 0}
    anchors = {}

    for action, idx, text in PLAN:
        p = paras[idx]
        if action == "keep":
            stats["keep"] += 1
        elif action == "edit":
            clear(p)
            if highlight:
                for t, kind in HL.word_diff(originals[idx], plain(text)):
                    HL.docx_runs(p, [(t, kind)])
            else:
                write(p, text, cites)
            stats["edit"] += 1
        elif action == "delete":
            if highlight:
                clear(p)
                HL.docx_runs(p, [(originals[idx], "del")])
            else:
                p._p.getparent().remove(p._p)
            stats["delete"] += 1
        elif action in ("heading", "insert"):
            anchor = anchors.get(idx, paras[idx])
            new = copy.deepcopy(anchor._p)
            for ch in list(new):
                if ch.tag != W + "pPr":
                    new.remove(ch)
            anchor._p.addnext(new)
            np_ = Paragraph(new, anchor._parent)
            if action == "heading":
                np_.style = doc.styles["Heading 3"]
            write(np_, text, cites, mark="ins" if highlight else None)
            anchors[idx] = np_
            stats["heading" if action == "heading" else "insert"] += 1
        elif action == "cut":
            if not rich_replace(p, text, "", "diff" if highlight else "clean"):
                raise SystemExit("cut anchor not found in paragraph %d" % idx)
            stats["edit"] += 1

        elif action == "trim":
            kids = list(p._p)
            start = next(i for i, e in enumerate(kids)
                         if e.tag == W + "r" and e.find(W + "t") is not None
                         and (e.find(W + "t").text or "").startswith(text))
            while start and kids[start - 1].tag == W + "r":
                t0 = kids[start - 1].find(W + "t")
                if t0 is None or (t0.text or "").strip():
                    break
                start -= 1
            for e in kids[start:]:
                if highlight:
                    from docx.shared import RGBColor
                    rpr = e.get_or_add_rPr() if hasattr(e, "get_or_add_rPr") else None
                    r = next(r for r in p.runs if r._element is e)
                    r.font.strike = True
                    r.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
                else:
                    p._p.remove(e)
            stats["edit"] += 1

        elif action == "heading-retitle":
            clear(p)
            if highlight:
                for t, kind in HL.word_diff(originals[idx], plain(text)):
                    HL.docx_runs(p, [(t, kind)])
            else:
                write(p, text, cites)
            stats["edit"] += 1

    missed = []
    for idx, old, new in SUBS:
        if idx >= 160:
            raise SystemExit("refusing to edit the reference list (paragraph %d)" % idx)
        if not rich_replace(paras[idx], old, new, "diff" if highlight else "clean"):
            missed.append((idx, old[:60]))
    if missed:
        raise SystemExit("substitution anchors not found: %s" % missed)
    stats["terms"] = len(SUBS)

    doc.save(str(out))
    return out, stats



def write_md(clean_docx):
    """Emit the Discussion body as markdown, read back from the built .docx.

    Subscript runs become <sub> markup and EndNote result text becomes <sup> markup, so the file carries
    the same typography as the manuscript. Reference numbers are the author's own, unchanged.
    """
    doc = Document(str(clean_docx))
    paras = doc.paragraphs
    end = next(i for i, p in enumerate(paras)
               if p.text.strip().upper() == "CONCLUSIONS" and i > 119)

    out = ["<!-- Round 5C DISCUSSION body, read back from %s." % clean_docx.name,
           "     Reference numbers are those of 'revised ACS JCIM v7.4.docx' and are not reordered.",
           "     Built by analysis/v7.4_final01/build_discussion_round5c.py; QC: qc_discussion_round5c.py. -->",
           ""]
    for i in range(119, end):
        p = paras[i]
        if not p.text.strip():
            continue
        if p.style.name == "Heading 3":
            out += ["### " + p.text.strip(), ""]
            continue
        if p.text.strip().upper() == "DISCUSSION":
            out += ["## DISCUSSION", ""]
            continue
        buf = []
        for r in p.runs:
            t = r.text
            if not t:
                continue
            if r.font.subscript:
                buf.append("<sub>%s</sub>" % t)
            elif r.font.superscript:
                buf.append("<sup>%s</sup>" % t)
            else:
                buf.append(t)
        line = "".join(buf)
        line = re.sub(r"</sub><sub>", "", line)
        line = re.sub(r"</sup><sup>", "", line)
        out += [line.strip(), ""]

    md = OUT / "v7.4_discussion_round5c.md"
    md.write_text("\n".join(out))
    words = sum(len(x.split()) for x in out if not x.startswith(("<!--", "#", " ")))
    print("%-46s %d paragraphs, ~%d words" % (md.name, sum(1 for x in out if x and not x.startswith(("<!--", "#"))), words))
    return md


def main():
    global HL
    spec = importlib.util.spec_from_file_location(
        "hl", S21 / "manuscript" / "v4" / "scripts" / "build_highlighted_v4.py")
    HL = importlib.util.module_from_spec(spec)
    sys.modules["hl"] = HL
    spec.loader.exec_module(HL)

    clean = None
    for hi in (False, True):
        out, stats = build(hi)
        print("%-46s %s" % (out.name, stats))
        if not hi:
            clean = out
    write_md(clean)


if __name__ == "__main__":
    main()
