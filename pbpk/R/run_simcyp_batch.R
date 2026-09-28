#!/usr/bin/env Rscript
# =====================================================================
# Batch Simcyp driver — replaces the five near-identical h2_run*.Rmd
#
# Command line (from this folder):
#     Rscript run_simcyp_batch.R --test h1_run0      # smoke test (test_drugs of that run)
#     Rscript run_simcyp_batch.R --all               # every ENABLED run in runs_config.csv
#     Rscript run_simcyp_batch.R h2_run1 h2_run3     # only these runs
#     (no arguments -> DEFAULT_ARGS below, for RStudio)
#
# RStudio: open 003_ml_inputs.Rproj, set DEFAULT_ARGS below, then either
#     Source (Ctrl+Shift+S), or Background Jobs -> Start Background Job.
#
# Design notes
#   * The workspace is loaded ONCE per compound. Parameters are applied to the
#     loaded workspace and simulated without reloading. (Calling SetWorkspace
#     again would re-read the .wksz from disk and discard every parameter.)
#   * The SetWorkspace dump is the TEMPLATE state and is logged as such.
#     The applied state is recorded by reading each parameter back with
#     GetParameter -> applied_vs_intended.csv.
#   * Dose / route / infusion are not set by this script; they come from the
#     .wksz. They are parsed from the template dump into design_from_template.csv
#     for every run, and compared across runs at the end, so a compound dosed
#     differently in two arms is flagged automatically.
#   * Per-run overrides, template folders and smoke-test compounds live in runs_config.csv.
#   * Clearance sentinel. A read-back only proves a value was stored, not that the
#     model used it. For every compound the script also records the template's own
#     ClearanceSwitch / CL_MetBas / CLRbase (template_watch.csv) and the clearance the
#     simulation actually produced, Dose / AUC0-inf (clearance_sentinel.csv). A CLint
#     run whose simulated clearance tracks the template's CL_MetBas — the observed
#     value in the v1-derived templates — is flagged: that is the signature of an
#     elimination switch that did not take effect.
#   * Output layout matches eval supplemental.ipynb: <out_root>/<run_id>/ct_wide_all.xlsx
# =====================================================================

# Used only when no command-line arguments are given (RStudio Source / Background Job).
# Examples: c("--test", "h1_run0")   c("h1_run1", "h1_run2", "h1_run3")   c("--all")
DEFAULT_ARGS <- c("--test", "h1_run0")

CFG <- list(
  runs_config = "runs_config.csv",
  out_root    = "output_batch",
  test_root   = "output_test",
  test_drugs  = c(52, 323, 1001),   # default; a run can override via the test_drugs column
  systemfiles = "C:\\Program Files\\Simcyp Simulator V24\\Screens\\SystemFiles",
  version     = 24
)

# Template values read back BEFORE any parameter is applied (never set by this script).
WATCH <- c("ClearanceSwitch", "CL_MetBas", "CLRbase")

suppressPackageStartupMessages({
  library(Simcyp); library(RSQLite); library(dplyr); library(tidyr)
  library(readr); library(stringr); library(writexl)
})

args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) args <- DEFAULT_ARGS
TEST <- "--test" %in% args
ALL  <- "--all"  %in% args
sel  <- setdiff(args, c("--test", "--all"))
if (!ALL && !length(sel)) stop("Name the run(s) to execute, or pass --all.", call. = FALSE)

ROOT <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
runs <- read_csv(file.path(ROOT, CFG$runs_config), show_col_types = FALSE)
if (!"enabled" %in% names(runs)) runs$enabled <- TRUE
runs$enabled <- as.logical(runs$enabled)
if (ALL) {
  skipped <- runs$run_id[!runs$enabled]
  if (length(skipped)) message("--all skips disabled runs: ", paste(skipped, collapse = ", "))
  runs <- runs[runs$enabled, ]
} else {
  unknown <- setdiff(sel, runs$run_id)
  if (length(unknown)) stop("Not in runs_config.csv: ", paste(unknown, collapse = ", "), call. = FALSE)
  runs <- runs[runs$run_id %in% sel, ]
  off <- runs$run_id[!runs$enabled]
  if (length(off)) message("[note] running disabled run(s) because they were named explicitly: ", paste(off, collapse = ", "))
}
if (!nrow(runs)) stop("No matching run_id in runs_config.csv: ", paste(sel, collapse = ", "))
if (!"test_drugs" %in% names(runs)) runs$test_drugs <- NA_character_
OUT_ROOT <- file.path(ROOT, if (TEST) CFG$test_root else CFG$out_root)

# Fail fast, before Simcyp starts, if any input file or template folder is missing.
miss_in <- runs$input_csv[!file.exists(file.path(ROOT, runs$input_csv))]
miss_ws <- unique(runs$wksz_dir[!dir.exists(file.path(ROOT, runs$wksz_dir))])
if (length(miss_in) || length(miss_ws))
  stop("Missing before start:\n",
       if (length(miss_in)) paste0("  input_csv: ", miss_in, collapse = "\n"), "\n",
       if (length(miss_ws)) paste0("  wksz_dir:  ", miss_ws, collapse = "\n"), call. = FALSE)

message(sprintf("Mode: %s | runs: %s | output: %s",
                if (TEST) "TEST (test_drugs only)" else "FULL",
                paste(runs$run_id, collapse = ", "), OUT_ROOT))

# ----------------------------- helpers -------------------------------

capture_simcyp_output <- function(expr) {
  collected <- character()
  out <- capture.output(withCallingHandlers(
    expr,
    message = function(m) { collected <<- c(collected, conditionMessage(m)); invokeRestart("muffleMessage") },
    warning = function(w) { collected <<- c(collected, conditionMessage(w)); invokeRestart("muffleWarning") }
  ))
  unique(c(collected, out))
}

field <- function(txt, key) {
  m <- str_match(txt, paste0("^\\s*", key, ":\\s*(.*)$"))
  v <- m[!is.na(m[, 2]), 2]
  if (length(v)) trimws(v[1]) else NA_character_
}

parse_overrides <- function(s) {
  if (is.na(s) || !nzchar(trimws(s))) return(list())
  kv <- str_split_fixed(trimws(str_split(s, ";")[[1]]), "=", 2)
  kv <- kv[nzchar(kv[, 1]), , drop = FALSE]
  setNames(as.list(as.numeric(kv[, 2])), trimws(kv[, 1]))
}

resolve_param_map <- function(cols) {
  cand <- setdiff(cols, c("Drug", "workspace"))
  known <- cand[cand %in% names(CompoundParameterID)]
  unknown <- setdiff(cand, known)
  if (length(unknown))
    message("  [skip] columns not in CompoundParameterID: ", paste(unknown, collapse = ", "))
  setNames(lapply(known, function(x) CompoundParameterID[[x]]), known)
}

read_back <- function(tag) {
  tryCatch(as.numeric(Simcyp::GetParameter(Tag = tag, Category = CategoryID$Compound,
                                           SubCategory = CompoundID$Substrate)),
           error = function(e) NA_real_)
}

watch_tag <- function(nm) if (nm %in% names(CompoundParameterID)) CompoundParameterID[[nm]] else NA

# Clearance actually produced by the simulation: Dose / AUC0-inf from the mean profile
# (linear-trapezoid AUC0-48, log-linear extrapolation over the last 5 positive points).
sim_clearance <- function(ps, dose) {
  lab <- rownames(ps)
  i <- which(grepl("mean", lab, ignore.case = TRUE) & !grepl("upper|lower", lab, ignore.case = TRUE))[1]
  if (is.na(i) || is.na(dose)) return(NA_real_)
  t <- suppressWarnings(as.numeric(colnames(ps))); c <- as.numeric(ps[i, ])
  ok <- !is.na(t) & !is.na(c); t <- t[ok]; c <- c[ok]
  if (length(t) < 6) return(NA_real_)
  auc <- sum(diff(t) * (head(c, -1) + tail(c, -1)) / 2)
  pos <- which(c > 0); tail5 <- tail(pos, 5)
  kel <- if (length(tail5) >= 3) -unname(coef(lm(log(c[tail5]) ~ t[tail5]))[2]) else NA_real_
  aucinf <- if (!is.na(kel) && kel > 0) auc + c[length(c)] / kel else auc
  if (aucinf <= 0) NA_real_ else dose / aucinf
}

# How closely does the simulated clearance TRACK a reference? Simcyp's mean profile gives
# CL_sim ~ 0.78 x the entered clearance (mean of a variable population, CV 50%), so a constant
# factor is expected; tracking shows up as a small SD of log10(CL_sim / reference).
track <- function(a, b) {
  ok <- !is.na(a) & !is.na(b) & a > 0 & b > 0
  if (sum(ok) < 3) return(c(factor = NA, sd = NA, n = sum(ok)))
  l <- log10(a[ok] / b[ok]); c(factor = 10^median(l), sd = sd(l), n = sum(ok))
}

# ------------------------------ init ---------------------------------
Simcyp::Initialise(CFG$systemfiles, CFG$version, species = SpeciesID$Human, verbose = FALSE)
message("Simcyp initialised.")

all_design <- list()

# --------------------------- main loop -------------------------------
for (i in seq_len(nrow(runs))) {
  run_id <- runs$run_id[i]
  wdir   <- normalizePath(file.path(ROOT, runs$wksz_dir[i]), winslash = "/", mustWork = TRUE)
  outdir <- file.path(OUT_ROOT, run_id)
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  logf <- file.path(outdir, "Simcyp_models_summary.txt")
  if (file.exists(logf)) invisible(file.remove(logf))

  pt <- read_csv(file.path(ROOT, runs$input_csv[i]), show_col_types = FALSE)
  pt$workspace <- paste0(pt$Drug, ".wksz")
  avail <- list.files(wdir, pattern = "\\.wksz$")
  gone  <- setdiff(pt$workspace, avail)
  if (length(gone)) message(sprintf("  [warn] %d workspace(s) missing in %s: %s",
                                    length(gone), runs$wksz_dir[i], paste(head(gone, 8), collapse = ", ")))
  pt <- pt[pt$workspace %in% avail, , drop = FALSE]
  if (TEST) {
    td <- runs$test_drugs[i]
    td <- if (is.na(td) || !nzchar(trimws(td))) CFG$test_drugs else as.numeric(str_split(td, ";")[[1]])
    pt <- pt[pt$Drug %in% td, , drop = FALSE]
  }

  ov <- parse_overrides(runs$overrides[i])
  for (nm in names(ov)) pt[[nm]] <- ov[[nm]]
  param_map <- resolve_param_map(colnames(pt))

  message(sprintf("\n[%s] %d compounds x %d parameters | templates: %s%s",
                  run_id, nrow(pt), length(param_map), runs$wksz_dir[i],
                  if (length(ov)) paste0(" | overrides: ",
                                         paste(sprintf("%s=%s", names(ov), unlist(ov)), collapse = ", ")) else ""))

  ct_all <- list(); audit <- list(); design <- list(); watch <- list(); sentinel <- list()
  applies_cl <- "CL_MetBas" %in% names(param_map)
  old_wd <- setwd(wdir)                              # bare filenames, as in the original Rmd

  for (k in seq_len(nrow(pt))) {
    row <- pt[k, , drop = FALSE]
    ws  <- row$workspace
    t0  <- Sys.time()

    # 1. load the template ONCE, and log it as the template
    dump <- capture_simcyp_output(invisible(SetWorkspace(ws)))
    cat("===== Workspace:", ws, "===== [TEMPLATE state, before parameter application]\n",
        file = logf, append = TRUE)
    cat(paste0("[", format(t0, "%Y-%m-%d %H:%M:%S"), "]"), file = logf, append = TRUE, sep = "\n")
    cat(dump, file = logf, append = TRUE, sep = "\n")
    cat("\n\n", file = logf, append = TRUE)

    design[[k]] <- data.frame(
      run_id = run_id, Drug = row$Drug,
      Dose = suppressWarnings(as.numeric(field(dump, "Dose"))),
      Dose_scaling = field(dump, "Dose scaling"),
      Route = field(dump, "Route of administration"),
      Infusion_h = suppressWarnings(as.numeric(field(dump, "Infusion Duration"))),
      Template_distribution = field(dump, "Distribution model"),
      Template_elimination  = field(dump, "Elimination"))
    if (is.na(design[[k]]$Dose))
      message(sprintf("  [warn] %s: dose not found in the SetWorkspace output; CL_sim will be NA", ws))

    # 1b. template values that this run does not set, read before anything is applied
    tw <- setNames(lapply(WATCH, function(nm) {
      tg <- watch_tag(nm); if (is.na(tg)) NA_real_ else read_back(tg)
    }), paste0("template_", WATCH))
    watch[[k]] <- data.frame(run_id = run_id, Drug = row$Drug, tw)

    # 2. apply parameters to the loaded workspace
    for (nm in names(param_map)) {
      v <- row[[nm]]
      if (is.null(v) || is.na(v)) next
      SetCompoundParameter(param_map[[nm]], CompoundID$Substrate, as.numeric(v))
    }

    # 3. read every parameter back — this is the record of the APPLIED state
    for (nm in names(param_map)) {
      v <- row[[nm]]
      if (is.null(v) || is.na(v)) next
      got <- read_back(param_map[[nm]])
      want <- as.numeric(v)
      audit[[length(audit) + 1]] <- data.frame(
        run_id = run_id, Drug = row$Drug, parameter = nm, intended = want, applied = got,
        ok = !is.na(got) && abs(got - want) <= 1e-6 * max(abs(got), abs(want), 1e-30))
    }

    # 4. simulate the modified workspace — no reload in between
    dbf <- tempfile(pattern = paste0("Sim_", row$Drug, "_"), fileext = ".db")
    Simulate(database = dbf)
    conn <- dbConnect(SQLite(), dbf)
    ps <- GetProfileStats_DB(ProfileID$Csys, CompoundID$Substrate,
                             Inhibition = FALSE, Upper = 95, Lower = 5, Trial = FALSE, conn)
    dbDisconnect(conn); unlink(dbf)

    ct_all[[k]] <- tibble::rownames_to_column(as.data.frame(ps), "row_label") %>%
      mutate(compound_id = as.character(row$Drug))

    # 5. clearance the simulation actually produced (L/h/kg), next to what could have driven it
    cl_sim <- sim_clearance(as.data.frame(ps), design[[k]]$Dose)
    sentinel[[k]] <- data.frame(
      run_id = run_id, Drug = row$Drug, CL_sim_L_h_kg = cl_sim,
      applied_CL_MetBas_L_h_kg  = if (applies_cl) as.numeric(row[["CL_MetBas"]]) / 70 else NA_real_,
      template_CL_MetBas_L_h_kg = tw$template_CL_MetBas / 70,
      template_CLRbase_L_h_kg   = tw$template_CLRbase / 70,
      applied_ClearanceSwitch   = if ("ClearanceSwitch" %in% names(param_map)) read_back(param_map[["ClearanceSwitch"]]) else NA_real_)

    message(sprintf("  %2d/%d  %-6s dose=%-8s %-14s CLsim=%-8.4g %5.1fs", k, nrow(pt), row$Drug,
                    design[[k]]$Dose, design[[k]]$Route, cl_sim,
                    as.numeric(Sys.time() - t0, units = "secs")))
  }
  setwd(old_wd)

  # ---- outputs in the layout eval supplemental.ipynb expects ----
  ct_wide <- bind_rows(ct_all) %>%
    pivot_longer(-c(row_label, compound_id), names_to = "Time_hr", values_to = "Conc_mgl") %>%
    mutate(Time_hr = suppressWarnings(as.numeric(Time_hr)),
           type = case_when(str_detect(row_label, regex("upper", ignore_case = TRUE)) ~ "upper",
                            str_detect(row_label, regex("lower", ignore_case = TRUE)) ~ "lower",
                            str_detect(row_label, regex("mean",  ignore_case = TRUE)) ~ "mean",
                            TRUE ~ NA_character_)) %>%
    filter(!is.na(Time_hr), !is.na(type)) %>%
    group_by(compound_id, Time_hr, type) %>%
    summarise(Conc_mgl = mean(Conc_mgl, na.rm = TRUE), .groups = "drop") %>%
    pivot_wider(names_from = type, values_from = Conc_mgl) %>%
    arrange(as.numeric(compound_id), Time_hr)

  write_xlsx(ct_wide, file.path(outdir, "ct_wide_all.xlsx"))
  write_csv(pt %>% select(Drug, all_of(names(param_map))), file.path(outdir, "adme_key_inputs.csv"))
  aud <- bind_rows(audit); write_csv(aud, file.path(outdir, "applied_vs_intended.csv"))
  des <- bind_rows(design); write_csv(des, file.path(outdir, "design_from_template.csv"))
  all_design[[run_id]] <- des

  nbad <- sum(!aud$ok)
  message(sprintf("[%s] done: %d compounds, %d timepoints/compound. Read-back mismatches: %d%s",
                  run_id, n_distinct(ct_wide$compound_id),
                  round(nrow(ct_wide) / max(1, n_distinct(ct_wide$compound_id))), nbad,
                  if (nbad) "   <-- see applied_vs_intended.csv" else ""))
  if (nbad) print(aud %>% filter(!ok) %>% count(parameter, name = "n_mismatch"))

  # ---- clearance sentinel: which clearance did the simulation actually follow? ----
  write_csv(bind_rows(watch), file.path(outdir, "template_watch.csv"))
  sen <- bind_rows(sentinel)
  write_csv(sen, file.path(outdir, "clearance_sentinel.csv"))
  tmplR <- sen$template_CL_MetBas_L_h_kg + sen$template_CLRbase_L_h_kg
  refs <- list(`applied CL_MetBas`         = sen$applied_CL_MetBas_L_h_kg,
               `applied CL_MetBas+CLRbase` = sen$applied_CL_MetBas_L_h_kg + sen$template_CLRbase_L_h_kg,
               `template CL_MetBas`        = sen$template_CL_MetBas_L_h_kg,
               `template CL_MetBas+CLRbase`= tmplR)
  tr <- lapply(refs, function(r) track(sen$CL_sim_L_h_kg, r))
  for (nm in names(tr)) if (!is.na(tr[[nm]][["sd"]]))
    message(sprintf("[%s] CL_sim vs %-27s factor %.3f  SD(log10) %.3f  n=%d",
                    run_id, nm, tr[[nm]][["factor"]], tr[[nm]][["sd"]], tr[[nm]][["n"]]))
  sd_t <- suppressWarnings(min(tr[["template CL_MetBas"]][["sd"]], tr[["template CL_MetBas+CLRbase"]][["sd"]], na.rm = TRUE))
  sd_a <- suppressWarnings(min(tr[["applied CL_MetBas"]][["sd"]], tr[["applied CL_MetBas+CLRbase"]][["sd"]], na.rm = TRUE))
  # A CLint run whose clearance tracks the template's RENAL clearance has no hepatic
  # contribution at all: the entered CLint did nothing. Observed 2026-09-19, when
  # WOMC_CLintType1 was not set — SD was 0.054 and 0.125 for two such runs, against 1.03 for a
  # working in vivo arm. The "tracks template CL_MetBas" test below does not catch this.
  if (!applies_cl) {
    tr_clr <- track(sen$CL_sim_L_h_kg, sen$template_CLRbase_L_h_kg)
    if (!is.na(tr_clr[["sd"]])) {
      message(sprintf("[%s] CL_sim vs %-27s factor %.3f  SD(log10) %.3f  n=%d",
                      run_id, "template CLRbase (renal)", tr_clr[["factor"]], tr_clr[["sd"]], tr_clr[["n"]]))
      if (tr_clr[["sd"]] < 0.15)
        message(sprintf("[%s] *** WARNING: simulated clearance is the template's RENAL clearance alone ", run_id),
                "(SD < 0.15) — there is no hepatic contribution, so the entered CLint was not used. ",
                "Check that ClearanceSwitch = 1 AND WOMC_CLintType1 = 2 are both set. Do NOT use these results.")
    }
  }
  if (!applies_cl && is.finite(sd_t) && sd_t < 0.10)
    message(sprintf("[%s] *** WARNING: this run does not set CL_MetBas, yet its simulated clearance tracks the ",
                    run_id), "template's CL_MetBas (SD < 0.10). The elimination switch probably did not take effect. ",
            "Do NOT use these results; run probe_clearance_switch.R.")
  if (applies_cl && is.finite(sd_a) && sd_a > 0.25)
    message(sprintf("[%s] *** WARNING: simulated clearance does not track the applied CL_MetBas (SD > 0.25).", run_id))
}

# ---- cross-run design check: same compound, different dose/route/infusion? ----
dsg <- bind_rows(all_design)
if (length(unique(dsg$run_id)) > 1) {
  clash <- dsg %>% group_by(Drug) %>%
    summarise(n_dose = n_distinct(Dose), n_route = n_distinct(Route),
              n_inf = n_distinct(Infusion_h), doses = paste(sort(unique(Dose)), collapse = " / "),
              .groups = "drop") %>%
    filter(n_dose > 1 | n_route > 1 | n_inf > 1)
  write_csv(clash, file.path(OUT_ROOT, "design_inconsistencies.csv"))
  message(if (nrow(clash)) sprintf("\n[design] %d compound(s) dosed differently across runs -> design_inconsistencies.csv", nrow(clash))
          else "\n[design] dose, route and infusion identical across all runs.")
}

runs %>% mutate(scenario = run_id, ct_file = file.path(OUT_ROOT, run_id, "ct_wide_all.xlsx"),
                test_mode = TEST, produced = format(Sys.time(), "%Y-%m-%d %H:%M:%S")) %>%
  write_csv(file.path(OUT_ROOT, "run_mapping.csv"))
message("\nFinished. Point eval supplemental.ipynb at SIM_BASE = \"", OUT_ROOT, "\"")
