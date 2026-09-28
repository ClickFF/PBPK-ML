#!/usr/bin/env Rscript
# =====================================================================
# Probe 2 — find the setting that actually makes Simcyp use an entered microsomal CLint.
#
#     Rscript probe2_find_clint_route.R
#
# Probe 1 established: ClearanceSwitch = 1 switches elimination away from the entered in vivo
# CL_MetBas, but the value in WOMC_MicrosomeCLintValue1 is ignored — clearance is identical
# with CLint at the S+ value, ten times it, or zero. Something else is eliminating the drug.
#
# The compound parameter list exposes a selector that was never set: WholeOrganRadio (0 in all
# 41 templates), plus per-system CLint/Vmax-Km mode switches. This probe sweeps the plausible
# combinations and reports which one makes clearance respond to CLint.
#
# Stage A  compound 1001 (posaconazole: template CLRbase = 0, so any clearance seen is
#          non-renal and the signal is clean) — ClearanceSwitch x WholeOrganRadio x CLint{0,10x}
# Stage B  for every responsive combination: the other two microsomal slots (Value0, Value2),
#          then confirmation on 52 and 1009.
#
# Nothing is written to the workspaces; every case starts from a freshly loaded template.
# =====================================================================

WKSZ_DIR  <- "wksz_tier1_minimal"
INPUT     <- "re_run_inputs/tables/h1_run0_adme_key_inputs.csv"
LEAD      <- 1001                      # CLRbase = 0 -> cleanest signal
CONFIRM   <- c(52, 1009)
SWITCHES  <- 0:3
RADIOS    <- 0:3
OUT       <- "output_test/probe2_find_clint_route.csv"
SYSFILES  <- "C:\\Program Files\\Simcyp Simulator V24\\Screens\\SystemFiles"

suppressPackageStartupMessages({ library(Simcyp); library(RSQLite); library(readr); library(stringr) })

ROOT <- normalizePath(getwd(), winslash = "/")
inp  <- read_csv(file.path(ROOT, INPUT), show_col_types = FALSE)
Simcyp::Initialise(SYSFILES, 24, species = SpeciesID$Human, verbose = FALSE)

has <- function(nm) nm %in% names(CompoundParameterID)
P   <- function(nm) CompoundParameterID[[nm]]
rb  <- function(nm) if (!has(nm)) NA_real_ else
  tryCatch(as.numeric(Simcyp::GetParameter(Tag = P(nm), Category = CategoryID$Compound,
                                           SubCategory = CompoundID$Substrate)),
           error = function(e) NA_real_)
st  <- function(nm, v) if (has(nm))
  tryCatch({ SetCompoundParameter(P(nm), CompoundID$Substrate, as.numeric(v)); TRUE },
           error = function(e) FALSE) else FALSE

capture_simcyp_output <- function(expr) {
  collected <- character()
  out <- capture.output(withCallingHandlers(
    expr,
    message = function(m) { collected <<- c(collected, conditionMessage(m)); invokeRestart("muffleMessage") },
    warning = function(w) { collected <<- c(collected, conditionMessage(w)); invokeRestart("muffleWarning") }))
  unique(c(collected, out))
}
dose_of <- function(d) { m <- str_match(d, "^\\s*Dose:\\s*(.*)$"); as.numeric(trimws(m[!is.na(m[, 2]), 2][1])) }

sim_cl <- function(dose) {
  dbf <- tempfile(fileext = ".db")
  ok <- tryCatch({ Simulate(database = dbf); TRUE }, error = function(e) FALSE)
  if (!ok) return(NA_real_)
  conn <- dbConnect(SQLite(), dbf)
  ps <- GetProfileStats_DB(ProfileID$Csys, CompoundID$Substrate, Inhibition = FALSE,
                           Upper = 95, Lower = 5, Trial = FALSE, conn)
  dbDisconnect(conn); unlink(dbf)
  i <- which(grepl("mean", rownames(ps), ignore.case = TRUE) &
               !grepl("upper|lower", rownames(ps), ignore.case = TRUE))[1]
  t <- suppressWarnings(as.numeric(colnames(ps))); c <- as.numeric(ps[i, ])
  ok2 <- !is.na(t) & !is.na(c); t <- t[ok2]; c <- c[ok2]
  if (length(t) < 6 || is.na(dose)) return(NA_real_)
  auc <- sum(diff(t) * (head(c, -1) + tail(c, -1)) / 2)
  tl <- tail(which(c > 0), 5)
  kel <- if (length(tl) >= 3) -unname(coef(lm(log(c[tl]) ~ t[tl]))[2]) else NA_real_
  dose / (auc + if (!is.na(kel) && kel > 0) c[length(c)] / kel else 0)
}

res <- list()
one <- function(drug, switch_v, radio_v, slot, clint, tag) {
  ws <- paste0(drug, ".wksz")
  dump <- capture_simcyp_output(invisible(SetWorkspace(ws)))
  dose <- dose_of(dump)
  if (is.na(dose)) stop("could not read the dose of ", ws, " — script problem", call. = FALSE)
  st("ClearanceSwitch", switch_v)
  set_radio <- st("WholeOrganRadio", radio_v)
  st(slot, clint); st("WOMC_MicrosomeFUinc11", 1)
  cl <- sim_cl(dose)
  res[[length(res) + 1]] <<- data.frame(
    stage = tag, Drug = drug, ClearanceSwitch = switch_v, WholeOrganRadio = radio_v,
    slot = slot, CLint_set = clint, radio_settable = set_radio,
    switch_back = rb("ClearanceSwitch"), radio_back = rb("WholeOrganRadio"),
    clint_back = rb(slot), CLRbase = rb("CLRbase"), CL_MetBas = rb("CL_MetBas"),
    dose = dose, CL_sim = cl)
  cl
}

cl_ref <- inp$WOMC_MicrosomeCLintValue1[inp$Drug == LEAD]
old <- setwd(file.path(ROOT, WKSZ_DIR))

message(sprintf("Stage A: compound %d, ClearanceSwitch %s x WholeOrganRadio %s x CLint {0, %.4g}",
                LEAD, paste(SWITCHES, collapse = "/"), paste(RADIOS, collapse = "/"), 10 * cl_ref))
hits <- list()
for (sw in SWITCHES) for (rd in RADIOS) {
  lo <- one(LEAD, sw, rd, "WOMC_MicrosomeCLintValue1", 0, "A")
  hi <- one(LEAD, sw, rd, "WOMC_MicrosomeCLintValue1", 10 * cl_ref, "A")
  resp <- if (is.na(lo) || is.na(hi) || lo <= 0) NA else hi / lo
  message(sprintf("   switch %d radio %d : CL(CLint=0) %-9.4g CL(10x) %-9.4g  response x%s",
                  sw, rd, lo, hi, if (is.na(resp)) "NA" else sprintf("%.2f", resp)))
  if (!is.na(resp) && resp > 1.05) hits[[length(hits) + 1]] <- c(sw, rd)
}

if (length(hits)) {
  message("\nStage B: responsive combination(s) found — testing the other microsomal slots and two more compounds")
  for (h in hits) for (slot in c("WOMC_MicrosomeCLintValue0", "WOMC_MicrosomeCLintValue2")) {
    lo <- one(LEAD, h[1], h[2], slot, 0, "B-slot"); hi <- one(LEAD, h[1], h[2], slot, 10 * cl_ref, "B-slot")
    message(sprintf("   switch %d radio %d slot %-26s response x%s", h[1], h[2], slot,
                    if (is.na(lo) || lo <= 0) "NA" else sprintf("%.2f", hi / lo)))
  }
  for (h in hits) for (d in CONFIRM) {
    cr <- inp$WOMC_MicrosomeCLintValue1[inp$Drug == d]
    lo <- one(d, h[1], h[2], "WOMC_MicrosomeCLintValue1", 0, "B-confirm")
    md <- one(d, h[1], h[2], "WOMC_MicrosomeCLintValue1", cr, "B-confirm")
    hi <- one(d, h[1], h[2], "WOMC_MicrosomeCLintValue1", 10 * cr, "B-confirm")
    message(sprintf("   drug %-5d switch %d radio %d : CL 0 / S+ / 10x = %.4g / %.4g / %.4g",
                    d, h[1], h[2], lo, md, hi))
  }
} else {
  message("\nNo combination of ClearanceSwitch x WholeOrganRadio made clearance respond to CLint.")
}
setwd(old)

res <- do.call(rbind, res)
dir.create(dirname(file.path(ROOT, OUT)), showWarnings = FALSE, recursive = TRUE)
write_csv(res, file.path(ROOT, OUT))
cat("\n================ VERDICT ================\n")
if (length(hits)) {
  cat("Responsive setting(s):\n")
  for (h in hits) cat(sprintf("   ClearanceSwitch = %d, WholeOrganRadio = %d\n", h[1], h[2]))
  cat("Send output_test/probe2_find_clint_route.csv; the h1 tables will be rebuilt with these flags.\n")
} else {
  cat("NONE of the swept settings routes elimination through the entered microsomal CLint.\n")
  cat("The bottom-up arm cannot be built from R with these parameters; send the CSV.\n")
}
