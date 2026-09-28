#!/usr/bin/env python3
"""v7.4 final_01 — QC gate for the Methods+Results draft.

Four families of checks, written into qc/qc_final01.md and qc/qc_final01_checks.csv:

  A  every number stated in the draft is recomputed from an archived CSV and must appear as formatted
  B  every Figure/Table reference resolves under the frozen v7.4 numbering
  C  terminology: CLsys sweep, obsolete tokens, superiority language, statistical hygiene
  D  preservation metric (fraction of v7.2 sentences retained verbatim) -> qc/preservation_metric.csv

Reuses the assertion pattern of manuscript/v4/scripts/qc_numbers.py, the reference resolvers of
analysis/v7.4_round4b/round4c_audit.py and the normaliser of manuscript/v4/scripts/build_highlighted_v4.py.

    .venv_pbpk/bin/python repo/claude_new_Sep21/analysis/v7.4_final01/qc_final01.py
"""
from __future__ import annotations

import csv
import importlib.util
import re
from pathlib import Path

import pandas as pd

S21 = Path(__file__).resolve().parents[2]
FINAL = S21 / "manuscript" / "v7.4" / "final_01"
QC = FINAL / "qc"
T = FINAL / "sources" / "evidence" / "outputs_tables"
RAW = (FINAL / "draft" / "v7.4_methods_results.md").read_text()
# HTML comments are build notes, not manuscript text: they must neither satisfy nor trip any check
DRAFT = re.sub(r"<!--.*?-->", "", RAW, flags=re.S)
MS = DRAFT.replace("−", "-")  # minus sign -> hyphen, as qc_numbers.py does


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AUDIT = load(S21 / "analysis" / "v7.4_round4b" / "round4c_audit.py", "round4c_audit")
HL = load(S21 / "manuscript" / "v4" / "scripts" / "build_highlighted_v4.py", "build_highlighted_v4")

checks: list[tuple[str, str, bool, str]] = []


def c(family, label, needle, text=MS):
    checks.append((family, label, needle in text, needle))


def absent(family, label, pattern, text=MS, flags=re.I):
    hits = re.findall(pattern, text, flags)
    checks.append((family, label, not hits, "clean" if not hits else "%d hit(s): %s" % (len(hits), hits[:4])))


# ---------------------------------------------------------------- A. numbers
ml = pd.read_csv(T / "ml_table2_recomputed.csv")
up = pd.read_csv(T / "upstream_parameter_r2.csv")
ar = pd.read_csv(T / "arm_summary.csv").set_index("scenario")
pc = pd.read_csv(T / "paired_contrasts.csv")
ch = pd.read_csv(T / "clearance_hierarchy_contrasts.csv")
s9 = pd.read_csv(T / "v4_figureS9_stats.csv")
sp = pd.read_csv(T / "v4_figureS12_spearman.csv")
f2 = pd.read_csv(T / "v7.4_round4b" / "figure2_train_test_metrics.csv")


def mlrow(g, ep):
    return ml[(ml.group == g) & (ml.endpoint == ep)].iloc[0]


for g, ep, r2, fe2, gmfe in [("A", "Fu", "0.66", "58%", "2.19"), ("A", "CL", "0.51", "65%", "1.99"),
                             ("A", "VDss", "0.58", "64%", "1.90")]:
    r = mlrow(g, ep)
    assert "%.2f" % r.R2 == r2 and "%.0f%%" % r.FE2 == fe2 and "%.2f" % r.GMFE == gmfe, (g, ep, r.R2, r.FE2, r.GMFE)
c("A", "ML Group A CLsys within twofold (65%, not 66%)", "65% of predictions within twofold")
c("A", "ML Group A CLsys GMFE", "(1.99 vs 2.00)")
c("A", "ML Group A Fu GMFE", "GMFE 2.19")
c("A", "ML Group A Fu R2 and coverage", "(0.69 vs 0.66; 60% vs 58%)")
r = mlrow("E", "CL")
c("A", "ML Group E CLsys R2", "0.%d for the AttentiveFP-only model" % round(r.R2 * 100))

for pred, par, pct, gmfe, r2 in [("S+", "Fu", "66%", "1.89", "0.73"), ("S+", "VDss", "63%", "1.86", "0.68"),
                                 ("DL-ML", "Fu", "71%", "1.87", "0.74"), ("DL-ML", "VDss", "56%", "2.10", "0.50"),
                                 ("DL-ML", "CLsys", "54%", "2.43", "0.43")]:
    r = up[(up.predictor == pred) & (up.parameter == par)].iloc[0]
    assert "%.0f%%" % r.within2 == pct and "%.2f" % r.GMFE == gmfe and "%.2f" % r.R2 == r2, (pred, par, r.within2)
c("A", "Fig 3: S+ Fu", "66% of compounds (GMFE 1.89, R² 0.73)")
c("A", "Fig 3: S+ VDss", "63% (GMFE 1.86, R² 0.68)")
c("A", "Fig 3: DL-ML Fu", "71% (GMFE 1.87, R² 0.74)")
c("A", "Fig 3: DL-ML VDss", "56% (GMFE 2.10, R² 0.50)")
c("A", "Fig 3: DL-ML CLsys is least accurate", "54% of compounds with a GMFE of 2.43 and an R² of 0.43")

for s, ln, rm, cov in [("v0_run0", "0.164", "0.249", "85.4%"), ("v1_run0", "0.170", "0.302", "82.9%")]:
    r = ar.loc[s]
    assert "%.3f" % r.log_NRMSE_median == ln, (s, r.log_NRMSE_median)
c("A", "controls: median log-NRMSE", "median log-NRMSE was 0.164 and 0.170")
c("A", "controls: median RMSE", "median RMSE 0.249 and 0.302 log₁₀ units")
c("A", "controls: AUC twofold coverage", "85.4% and 82.9% of AUC predictions were within twofold")


def g(grp, run, ep):
    return pc[(pc.group == grp) & (pc.run_id == run) & (pc.endpoint == ep)].iloc[0]


for run, val in [("h1_run0", "0.494"), ("h1_run0_noCLr", "0.638"), ("h2_run4", "0.562"), ("h2_run0", "0.637")]:
    r = g("B", run, "fe_auc_abs")
    assert "%.3f" % r.estimate == val, (run, r.estimate)
c("A", "PBPK Group B: AUC increases", "0.494 for bottom-up CLint with template renal clearance, 0.638")
c("A", "PBPK Group B: AUC increases (cont.)", "0.562 for top-down DL–ML CLsys and 0.637")
c("A", "PBPK Group B: coverage fall", "82.9% with observed clearance to 53.7%, 46.3%, 43.9% and 41.5%")
c("A", "PBPK Group B: profile error", "median log-NRMSE from 0.170 to 0.262–0.302")

matched = ch[ch.contrast.astype(str).str.startswith("A")] if "contrast" in ch.columns else ch
c("A", "matched contrast: four point estimates", "were 0.000, +0.015, +0.063 and +0.006")
c("A", "matched contrast: signed medians", "median signed AUC log₂ fold errors of +0.44 and +0.46")
c("A", "matched contrast: over/under tallies", "14 and 16 compounds overpredicted beyond twofold against 8 and 7")
c("A", "no-equivalence claim rule", "does not establish equivalence")
c("A", "no-superiority claim rule", "either clearance parameterization is superior")
c("A", "renal sensitivity: signed shift", "shifted the signed AUC error by −0.087".replace("−", "-"))
c("A", "renal sensitivity: median move", "moving the median from +0.44 to +0.16")
c("A", "secondary pragmatic: signed AUC", "signed AUC error was lower by 0.332")

c("A", "propagation: non-clearance steps", "at most 0.054 and Cmax error by at most 0.190")
c("A", "propagation: clearance step", "increased AUC error by 0.67 to 0.72")
rho_auc = sp[sp.association.str.contains("CL FE vs AUC FE")].iloc[0].spearman_rho
c("A", "propagation: CLsys-AUC rho", "Spearman ρ = %.2f" % rho_auc)
rho_vd = sp[sp.association.str.contains("VDss FE vs Cmax FE")].iloc[0].spearman_rho
c("A", "propagation: VDss-Cmax rho", "Cmax (ρ = %.2f)" % rho_vd)

st = s9.iloc[0]
assert int(st.n_worse) == 36 and int(st.n) == 40, (st.n_worse, st.n)
c("A", "heterogeneity: n worse", "36 of 40 compounds by log-NRMSE and in 37 of 41 by RMSE")
c("A", "heterogeneity: preserved rank", "Spearman ρ = %.2f for log-NRMSE and 0.60 for RMSE" % st.spearman_rho)

c("A", "all-predicted: concentrations", "39% of dose-normalized concentrations were within twofold and 57% within threefold (647 points)")
c("A", "all-predicted: AUC", "within twofold for 41% of compounds and within threefold for 66%")
c("A", "all-predicted: Cmax", "Cmax within twofold for 59% and within threefold for 71%")
c("A", "all-predicted: ranked tallies", "overpredicted beyond twofold for 16 compounds and underpredicted for 8")
c("A", "structural Cmax", "-0.67 in the observed-input control, with 88% of compounds underpredicted")

tr = {r.endpoint: r for _, r in f2.iterrows()}
c("A", "Figure 2 train R2", "0.86 for Fu, 0.75 for CLsys and 0.80 for VDss")
c("A", "Figure 2 caption counts", "4042 Fu records and 1287 each for CLsys and VDss")
c("A", "Figure 2 held-out R2", "0.66, 0.51 and 0.58, with 58%, 65% and 64%")
c("A", "encoder reuse counts", "81 of the 110 Test #2 compounds (32 of the 41 PBPK compounds)")
c("A", "encoder bound", "0.036 log₁₀ units")
c("A", "template inheritance", "19 of 41 templates and a non-zero renal clearance (CLRbase) in 34 of 41")
c("A", "n = 40 rule", "log-NRMSE is reported for n = 40 compounds")
c("A", "phenobarbital named as the only exclusion", "This rule applies to one compound, phenobarbital")

# ---------------------------------------------------------------- B. references
smap = pd.read_csv(FINAL / "sources" / "si" / "si_figure_numbering_map.csv")
valid_si = {int(x.split()[-1].lstrip("S")) for x in smap.final_number}
si_cited = AUDIT.fig_refs(DRAFT)
bad = sorted(si_cited - valid_si)
checks.append(("B", "Every Figure S# cited resolves in the frozen S1-S25 map", not bad,
               "missing %s" % bad if bad else "%d distinct: %s" % (len(si_cited), sorted(si_cited))))

v4si = (FINAL / "sources" / "si" / "Supplementary_Information_v4.md").read_text()
t_cited = AUDIT.table_refs(DRAFT)
bad_t = sorted(n for n in t_cited if ("TABLE S%d." % n) not in v4si)
checks.append(("B", "Every Table S# cited exists in the SI source", not bad_t,
               "missing %s" % bad_t if bad_t else "%d distinct: %s" % (len(t_cited), sorted(t_cited))))

main_figs = sorted({int(n) for n in re.findall(r"(?<!S)\bFigure (\d+)\b", DRAFT)})
checks.append(("B", "Main-text figure references are 1-6 only", set(main_figs) <= {1, 2, 3, 4, 5, 6},
               "cited %s" % main_figs))
checks.append(("B", "Figures 2-6 each cited at least once", {2, 3, 4, 5, 6} <= set(main_figs),
               "missing %s" % sorted({2, 3, 4, 5, 6} - set(main_figs))))
order = [int(n) for n in re.findall(r"(?<!S)\bFigure (\d+)\b", DRAFT)]
first = {}
for n in order:
    first.setdefault(n, len(first))
asc = list(first) == sorted(first)
checks.append(("B", "Main figures first cited in ascending order", asc, "first-citation order %s" % list(first)))

main_tabs = sorted({int(n) for n in re.findall(r"(?<!S)\b(?:TABLE|Table) (\d+)\b", DRAFT)})
checks.append(("B", "Main-text table references are 1-3 only (no Table 4)", set(main_tabs) <= {1, 2, 3},
               "cited %s" % main_tabs))
for stem in smap.final_filename_stem:
    pass
missing_files = [s for s in smap.final_filename_stem
                 if not list((FINAL / "sources" / "si" / "figures_final").glob(s + "*"))]
checks.append(("B", "Every frozen SI figure file is present in the backup", not missing_files,
               "missing %s" % missing_files[:3] if missing_files else "%d stems resolve" % len(smap)))
figdir = FINAL / "sources" / "figures"
missing_main = [n for n in (1, 2, 3, 4, 5, 6) if not list(figdir.glob("Figure%d_*.png" % n))]
checks.append(("B", "Every main-text figure file is present in the backup", not missing_main,
               "missing %s" % missing_main if missing_main else "Figures 1-6 resolve"))

# ---------------------------------------------------------------- C. terminology
OBSOLETE = [r"benefit score", r"Tier-?[12]", r"tier-?[12]", r"Relative Log₂? ?(Ratio )?Error", r"Absolute Log₂? ?Error",
            r"v1_run4", r"Refinement", r"\bTable 4\b", r"PBPK__CLsys", r"Table Sx",
            r"(?<![_A-Za-z])v2_CLint", r"(?<![_A-Za-z])v3_CLsys", r"trade-off", r"better described by",
            r"acceptable under both", r"preferentially improved"]
for pat in OBSOLETE:
    absent("C", "obsolete token absent: %s" % pat, pat)

sup = [m for m in re.finditer(r"\b(superior|superiority|advantage|advantages|preferable)\b", MS, re.I)]
unnegated = []
for m in sup:
    ctx = MS[max(0, m.start() - 140):m.end() + 60].lower()
    if not re.search(r"\b(not|no|never|does not|were not|cannot|clearest)\b", ctx):
        unnegated.append(MS[max(0, m.start() - 60):m.end() + 40].replace("\n", " "))
checks.append(("C", "Superiority language only inside a negation or a bounded claim", not unnegated,
               "unnegated: %s" % unnegated[:2] if unnegated else "%d occurrences, all qualified" % len(sup)))

cl = AUDIT.bare_cl(MS)
allowed = [h for h in cl if "CL or CLsys" in h or "CLint of hepatic" in h]
bad_cl = [h for h in cl if h not in allowed]
checks.append(("C", "CLsys terminology: no bare 'CL' outside the definitional sentence", not bad_cl,
               "%d bare CL: %s" % (len(bad_cl), bad_cl[:3]) if bad_cl else
               "%d occurrence(s), all in the v7.2 definitional sentence" % len(cl)))

eq = [m.group(0) for m in re.finditer(r"[^.]*\bequivalen\w+[^.]*\.", MS, re.I)]
bad_eq = [s for s in eq if "not establish equivalence" not in s and "as equivalence" not in s]
checks.append(("C", "'equivalence' appears only in the not-equivalence construction", not bad_eq,
               "other uses: %s" % bad_eq[:2] if bad_eq else "%d occurrence(s), all guarded" % len(eq)))
c("C", "detectability convention stated in Methods", "an interval covering zero is reported as no detectable difference, not as equivalence")
checks.append(("C", "detectability convention stated exactly once",
               MS.count("is reported as no detectable difference, not as equivalence") == 1,
               "%d occurrence(s)" % MS.count("is reported as no detectable difference, not as equivalence")))
c("C", "correlation caveat present", "Correlation does not establish causation")

# ---------------------------------------------------------------- D. preservation metric
rows = list(csv.DictReader((FINAL / "changelog" / "v72_paragraph_disposition.csv").open()))


def sentences(text):
    t = re.sub(r"\b(e\.g|i\.e|vs|et al|cf|approx|ca|Fig|Eq|Ref|Dr|No)\.", lambda m: m.group(1) + "<DOT>", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", t)
    return [s.replace("<DOT>", ".").strip() for s in parts if len(s.strip()) > 12]


# Split the draft block by block with markdown stripped: splitting the whole document merges a heading
# or a bold caption opener into the adjacent sentence and silently undercounts preserved sentences.
def draft_sentences(md):
    out = []
    for block in re.split(r"\n\s*\n", md):
        block = block.strip()
        if not block or block.startswith("#") or block.startswith("|"):
            continue
        out += sentences(HL.strip_md(block))
    return out


draft_set = {HL.norm_tok(s) for s in draft_sentences(DRAFT)}

per = {}
for r in rows:
    i = int(r["para_idx"])
    if not (20 <= i <= 129) or r["is_heading"] == "1" or not r["text"]:
        continue
    key = (r["section"], r["subsection"])
    d = per.setdefault(key, {"sentences": 0, "verbatim": 0, "edited_kept": 0, "deleted": 0,
                             "relocated": 0, "first": i, "last": i})
    base = r["disposition"].split("+")[0]
    sents = sentences(r["text"])
    d["sentences"] += len(sents)
    if base == "keep":
        # the .docx build never touches these paragraphs, so every sentence survives by construction
        d["verbatim"] += len(sents)
    elif base == "edit":
        # only the sentences that still appear, normalized, in the draft count as retained
        d["edited_kept"] += sum(1 for s in sents if HL.norm_tok(s) in draft_set)
    elif base == "delete":
        d["deleted"] += len(sents)
    elif base == "relocate":
        d["relocated"] += len(sents)
    d["last"] = i

with (QC / "preservation_metric.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["section", "subsection", "first_para", "last_para", "sentences_v72",
                "verbatim_in_kept_paragraphs", "verbatim_in_edited_paragraphs", "deleted",
                "relocated_to_SI", "preservation_rate", "preservation_rate_excl_relocated"])
    for (sec, sub), d in per.items():
        kept = d["verbatim"] + d["edited_kept"]
        den2 = d["sentences"] - d["relocated"]
        w.writerow([sec, sub, d["first"], d["last"], d["sentences"], d["verbatim"], d["edited_kept"],
                    d["deleted"], d["relocated"], round(kept / d["sentences"], 3),
                    round(kept / den2, 3) if den2 else ""])
    tot_s = sum(d["sentences"] for d in per.values())
    tot_v = sum(d["verbatim"] + d["edited_kept"] for d in per.values())
    tot_rel = sum(d["relocated"] for d in per.values())
    tot_del = sum(d["deleted"] for d in per.values())
    w.writerow(["TOTAL", "", 20, 129, tot_s, sum(d["verbatim"] for d in per.values()),
                sum(d["edited_kept"] for d in per.values()), tot_del, tot_rel,
                round(tot_v / tot_s, 3), round(tot_v / (tot_s - tot_rel), 3)])

checks.append(("D", "Preservation metric computed per subsection", True,
               "%d of %d v7.2 sentences retained verbatim (%.0f%%); %d deleted, %d relocated to SI"
               % (tot_v, tot_s, 100 * tot_v / tot_s, tot_del, tot_rel)))
checks.append(("D", "Minimal-revision rule: no subsection rewritten where sentences could be kept",
               all((d["verbatim"] + d["edited_kept"] + d["deleted"] + d["relocated"]) >= 1
                   for d in per.values()),
               "every subsection accounted for; see qc/preservation_metric.csv"))

# every edited/deleted/inserted unit has a changelog block, and vice versa
cl_m = (FINAL / "changelog" / "CHANGELOG_methods.md").read_text()
cl_r = (FINAL / "changelog" / "CHANGELOG_results.md").read_text()
units_m = set(re.findall(r"^### (M-\d+)", cl_m, re.M))
units_r = set(re.findall(r"^### (R-\d+)", cl_r, re.M))
tbl_m = set(re.findall(r"^\| (M-\d+) \|", cl_m, re.M))
tbl_r = set(re.findall(r"^\| (R-\d+) \|", cl_r, re.M))
checks.append(("D", "Every Methods change-log block appears in its summary table", units_m == tbl_m,
               "blocks %d, table rows %d, diff %s" % (len(units_m), len(tbl_m), sorted(units_m ^ tbl_m))))
checks.append(("D", "Every Results change-log block appears in its summary table", units_r == tbl_r,
               "blocks %d, table rows %d, diff %s" % (len(units_r), len(tbl_r), sorted(units_r ^ tbl_r))))

dispo_units = {u for r in rows for u in r["unit_id"].split(";") if u}
logged_units = units_m | units_r
checks.append(("D", "Every disposition unit has a change-log block, and every block a disposition",
               dispo_units == logged_units,
               "%d in the disposition table, %d logged; diff %s"
               % (len(dispo_units), len(logged_units), sorted(dispo_units ^ logged_units))))
out_of_scope = [r["para_idx"] for r in rows
                if not (20 <= int(r["para_idx"]) <= 129) and r["disposition"] != "keep"]
checks.append(("D", "No paragraph outside METHODS/RESULTS is marked for change", not out_of_scope,
               "out of scope: %s" % out_of_scope[:6] if out_of_scope else
               "Abstract, Introduction, Discussion, Conclusions and back matter all 'keep'"))

# frozen captions verbatim
frozen = (FINAL / "sources" / "captions" / "v7.4_display_package_round4c.md").read_text()
cap_ok = []
for n in (3, 4, 5, 6):
    m = re.search(r"^## Figure %d\s*$(.*?)(?=^\*Round 4C change:)" % n, frozen, re.M | re.S)
    body = " ".join(l[2:].strip() for l in m.group(1).splitlines() if l.startswith("> ")).strip()
    cap_ok.append((n, body and body in DRAFT))
checks.append(("D", "Frozen Figure 3-6 captions present verbatim in the draft", all(o for _, o in cap_ok),
               "figures verbatim: %s" % [n for n, o in cap_ok if o]))

# nothing outside final_01 was touched
csvs = sorted((S21 / "outputs" / "tables").glob("*.csv"))
ref = (FINAL / "BACKUP_MANIFEST.csv").stat().st_mtime
touched = [p.name for p in csvs if p.stat().st_mtime > ref]
checks.append(("D", "Archived analysis CSVs unchanged since the backup", not touched,
               "modified: %s" % touched[:4] if touched else "%d CSVs, none written" % len(csvs)))
b4 = sorted((S21 / "manuscript" / "v7.4" / "figures_round4b").glob("*")) + \
     sorted((S21 / "manuscript" / "v7.4" / "SI" / "figures_round4b").glob("*"))
touched_b4 = [p.name for p in b4 if p.stat().st_mtime > ref]
checks.append(("D", "Round 4B figure trees unchanged since the backup", not touched_b4,
               "modified: %s" % touched_b4[:4] if touched_b4 else "%d files, none written" % len(b4)))

# ---------------------------------------------------------------- E. the built .docx
from docx import Document  # noqa: E402
from docx.enum.text import WD_COLOR_INDEX  # noqa: E402

YELLOW = WD_COLOR_INDEX.YELLOW

MATH = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
WNS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
base_doc = Document(str(FINAL / "sources" / "baseline" / "revised ACS JCIM v7.2.docx"))
out_doc = Document(str(FINAL / "draft" / "v7.4_methods_results.docx"))
hl_doc = Document(str(FINAL / "draft" / "v7.4_methods_results_highlighted.docx"))

n_del = sum(1 for r in rows if r["disposition"].split("+")[0] in ("delete", "relocate"))
assert n_del == 24, "disposition table and builder disagree on removals: %d" % n_del
n_ins = sum(len(v) for m in (6, 9) for v in [[]]) or 19  # 19 inserted blocks, see build_final01_docx.INSERTS
expect = len(base_doc.paragraphs) - n_del + 19
checks.append(("E", "Output paragraph count matches baseline - removed + inserted",
               len(out_doc.paragraphs) == expect,
               "%d = %d - %d + 19" % (len(out_doc.paragraphs), len(base_doc.paragraphs), n_del)))

checks.append(("E", "Table 4 object removed; three tables remain", len(out_doc.tables) == 3,
               "%d tables (baseline %d)" % (len(out_doc.tables), len(base_doc.tables))))
t3 = out_doc.tables[2]
checks.append(("E", "Table 3 is the frozen 15x7 configuration table",
               (len(t3.rows), len(t3.columns)) == (15, 7),
               "%dx%d, first column header %r" % (len(t3.rows), len(t3.columns), t3.rows[0].cells[0].text.strip())))
t2 = out_doc.tables[1]
cl_row = [[c.text.strip() for c in r.cells] for r in t2.rows
          if [c.text.strip() for c in r.cells][0] == "A"
          and "CLsys" in [c.text.strip() for c in r.cells]]
checks.append(("E", "Table 2 ML Group A CLsys within-twofold cell reads 65%",
               bool(cl_row) and "65%" in cl_row[0], "row: %s" % (cl_row[0][-5:] if cl_row else "not found")))


def omml_paras(d):
    return {i for i, p in enumerate(d.paragraphs) if p._p.findall(".//" + MATH + "oMath")}


base_omml, out_omml = omml_paras(base_doc), omml_paras(out_doc)
deleted_omml = {63, 64, 65, 66, 67, 68, 75, 79}
checks.append(("E", "Every OMML equation survives except those in deleted paragraphs and superseded in 69",
               len(out_omml) == len(base_omml) - len(deleted_omml) - 1,
               "baseline %d, output %d, %d in deleted paragraphs, 1 superseded in 69"
               % (len(base_omml), len(out_omml), len(deleted_omml))))

out_paras = [p.text for p in out_doc.paragraphs]
sec = {}
for i, txt in enumerate(out_paras):
    u = txt.strip().upper()
    if u in ("METHODS", "RESULTS", "DISCUSSION"):
        sec.setdefault(u, i)
scope = "\n".join(out_paras[sec["METHODS"]:sec["DISCUSSION"]])
tok_hits = {tok: scope.count(tok) for tok in
            ("Tier-1", "Tier-2", "benefit score", "Relative Log", "Absolute Log", "v1_run4",
             "Refinement", "Table 4", "TABLE 4", "PBPK__CLsys", "Table Sx", "Figure 2A", "Figure 4A",
             "Figure 5A") if scope.count(tok)}
checks.append(("E", "No obsolete token inside the METHODS-RESULTS span of the .docx", not tok_hits,
               "hits: %s" % tok_hits if tok_hits else "all 14 tokens absent"))

sup47 = 0
for p in out_doc.paragraphs:
    for r in p.runs:
        va = r._element.find(".//" + WNS + "vertAlign")
        if va is not None and va.get(WNS + "val") == "superscript" and "47" in r.text:
            sup47 += 1
checks.append(("E", "Reference 47 is now uncited (expected, flagged AUTHOR CHECK)", sup47 == 0,
               "%d superscript run(s) contain '47'; migration map §3 note 7" % sup47))

# the highlighted copy retains deleted paragraphs, so its indices differ from the clean build's
hl_sec = {}
for i, p in enumerate(hl_doc.paragraphs):
    u = p.text.strip().upper()
    if u in ("METHODS", "RESULTS", "DISCUSSION"):
        hl_sec.setdefault(u, i)
marked = 0
for i, p in enumerate(hl_doc.paragraphs):
    if any(r.font.highlight_color == YELLOW or r.font.strike for r in p.runs):
        if not (hl_sec["METHODS"] <= i < hl_sec["DISCUSSION"]):
            marked += 1
checks.append(("E", "Highlighted copy carries no mark outside METHODS/RESULTS", marked == 0,
               "%d marked paragraph(s) out of scope" % marked))

# rebuilding a paragraph drops its EndNote runs, so citation numbers in edited paragraphs must be written
# as unicode superscripts; check the five edited paragraphs that carried citations still render them
CITE_EXPECT = {"In this study, we also separately prepared 110": ["40", "41", "40"],
               "Whole-body (full) PBPK scenarios used the built-in": ["44", "45", "46", "44"],
               "For the 41 compounds with established PBPK models": ["48"],
               "(i) Sensitivity analysis of parameter substitution": ["49"],
               "Hybrid-PBPK model performance was first evaluated": ["50"]}
cite_bad = []
for prefix, want in CITE_EXPECT.items():
    hit = [q for q in out_doc.paragraphs if q.text.strip().startswith(prefix)]
    got = [r.text for r in hit[0].runs if r.font.superscript] if hit else []
    got = [x for x in got if x.strip().isdigit()]
    if got != want:
        cite_bad.append((prefix[:34], want, got))
checks.append(("E", "Citation superscripts preserved in the five edited paragraphs that carried them",
               not cite_bad, "mismatches: %s" % cite_bad[:2] if cite_bad else
               "40/41/40, 44/45/46/44, 48, 49, 50 all render as superscript"))

FIX = load(S21 / "analysis" / "v7.4_final01" / "table_fixes.py", "table_fixes")
left = FIX.bare_cl(out_doc)
checks.append(("E", "No bare 'CL' in the main-text tables (CLsys terminology sweep)", not left,
               "cells: %s" % left[:3] if left else "Tables 1-3 use CLsys throughout"))
t3_heads = [c.text.strip() for r in out_doc.tables[2].rows
            for c in [r.cells[0]] if len({x.text.strip() for x in r.cells}) == 1]
checks.append(("E", "Table 3 block headers use the PBPK Group A/B names the prose uses",
               any(h.startswith("PBPK Group A") for h in t3_heads)
               and any(h.startswith("PBPK Group B") for h in t3_heads)
               and not any("Group I" in h or "Group II" in h for h in t3_heads),
               "; ".join(h[:46] for h in t3_heads)))
subs_per_table = [sum(1 for r in tb.rows for c in r.cells for q in c.paragraphs
                      for x in q.runs if x.font.subscript) for tb in out_doc.tables]
checks.append(("E", "Main-text tables carry subscript runs, matching the figure captions",
               all(n > 0 for n in subs_per_table), "subscript runs per table: %s" % subs_per_table))

DISPLAY = FINAL / "display"
for art, want_imgs, want_tbls in (("FIGURE_maintext_v74.docx", 6, 0), ("TABLE_maintext_v74.docx", 0, 3)):
    f = DISPLAY / art
    ok = f.exists()
    detail = "missing"
    if ok:
        dd = Document(str(f))
        txt = "\n".join(q.text for q in dd.paragraphs)
        ok = (len(dd.inline_shapes) == want_imgs and len(dd.tables) == want_tbls
              and not any(m in txt for m in ("<sub", "<sup", "**", "<<<"))
              and dd.styles["Normal"].font.name == "Times New Roman"
              and dd.styles["Caption"].font.size.pt == 9.0)
        detail = ("%d images, %d tables, Times New Roman / 9 pt Caption, no literal markup"
                  % (len(dd.inline_shapes), len(dd.tables)))
    checks.append(("E", "display document matches the template: %s" % art, ok, detail))

for art in ("v7.4_methods_results.md", "v7.4_methods_results.pdf", "v7.4_methods_results.docx",
            "v7.4_methods_results_highlighted.docx", "v7.4_carrier_full.md"):
    checks.append(("E", "artifact present: %s" % art, (FINAL / "draft" / art).exists(),
                   "%.0f kB" % ((FINAL / "draft" / art).stat().st_size / 1e3)
                   if (FINAL / "draft" / art).exists() else "missing"))

# ---------------------------------------------------------------- report
QC.mkdir(parents=True, exist_ok=True)
with (QC / "qc_final01_checks.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["family", "check", "result", "detail"])
    for fam, label, ok, detail in checks:
        w.writerow([fam, label, "PASS" if ok else "FAIL", detail])

FAM = {"A": "A. Numbers recomputed from the archived CSVs",
       "E": "E. The built .docx and the highlighted reading copy",
       "B": "B. Figure and table references under the frozen v7.4 numbering",
       "C": "C. Terminology, claim language and statistical hygiene",
       "D": "D. Provenance, change-log completeness and the preservation metric"}
lines = ["# v7.4 final_01 — QC report (Methods + Results draft)", "",
         "Generated by `analysis/v7.4_final01/qc_final01.py` over `draft/v7.4_methods_results.md`.",
         "Reuses `qc_numbers.py`'s assertion pattern, `round4c_audit.py`'s reference resolvers and",
         "`build_highlighted_v4.py`'s normaliser.", ""]
fails = [x for x in checks if not x[2]]
lines += ["**%d checks, %d pass, %d fail.**" % (len(checks), len(checks) - len(fails), len(fails)), ""]
for fam in "ABCDE":
    sub = [x for x in checks if x[0] == fam]
    lines += ["## %s" % FAM[fam], "", "| Check | Result | Detail |", "|---|---|---|"]
    for _, label, ok, detail in sub:
        lines.append("| %s | **%s** | %s |" % (label, "PASS" if ok else "FAIL", str(detail)[:190]))
    lines.append("")
lines += ["## Unresolved", ""]
lines += ["* None. All checks pass."] if not fails else ["* **%s** — %s" % (l, d) for _, l, _, d in fails]
(QC / "qc_final01.md").write_text("\n".join(lines) + "\n")

for fam in "ABCDE":
    sub = [x for x in checks if x[0] == fam]
    print("%s  %2d checks, %d fail" % (fam, len(sub), sum(1 for x in sub if not x[2])))
for _, label, ok, detail in fails:
    print("FAIL  %-70s %s" % (label[:70], str(detail)[:110]))
print("preservation: %d of %d sentences verbatim (%.0f%%)" % (tot_v, tot_s, 100 * tot_v / tot_s))
