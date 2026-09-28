#!/usr/bin/env Rscript
# =====================================================================
# Probe: does ClearanceSwitch = 1 route elimination through the WOMC CLint?
#
# Run ONCE before the h1r batch (about a minute):
#     Rscript probe_clearance_switch.R
#
# Why a probe rather than a read-back: a read-back proves a value was stored, not
# that the model used it. The v1-derived templates carry CL_MetBas = the OBSERVED
# clearance. If ClearanceSwitch = 1 silently failed, the h1r arm would simulate
# with the observed clearance and look excellent — the failure mode of the
# published v2 arm. The probe perturbs each input and checks that the simulated
# clearance responds only to the ones it should.
#
# Cases, each from a freshly loaded template:
#   A  template as loaded (in vivo CL_MetBas)                       reference
#   B  switch=1, CLint = S+ HLM, fuinc = 1                          the h1r setting
#   C  switch=1, CLint = 10 x S+ HLM                                CLint must drive clearance
#   D  switch=1, CLint = S+ HLM, CL_MetBas = 10 x template          CL_MetBas must be inert
#   E  switch=0, CLint = S+ HLM                                     CLint must be inert under switch 0
#   F  switch=1, CLint = 0                                          non-hepatic baseline under switch 1
# Checks per compound:
#   1 B != A     switching changes elimination           |log10(B/A)| > 0.05
#   2 C >  B     CLint drives clearance                  C/B > 1.05
#   3 D == B     CL_MetBas ignored under switch 1        |log10(D/B)| < 0.01
#   4 E == A     CLint ignored under switch 0            |log10(E/A)| < 0.01
#   5 linear     hepatic part scales with CLint          (C-F)/(B-F) >= half the well-stirred expected increase
#   6 magnitude  hepatic part (B-F) within 3-fold of a well-stirred expectation band
#                built from the workspace's own Fu and B/P (MPPGL 30-45 mg/g, liver 1.4-1.9 kg,
#                QH 80-100 L/h, x0.6-1.1 for Simcyp's in vivo scaling seen in the h2/v1 runs)
# Checks 1-4 test the switch; 5 tests that CLint is used; 6 tests that it is scaled as expected.
# =====================================================================

WKSZ_DIR <- "wksz_tier1_minimal"
INPUT    <- "re_run_inputs/tables/h1_run0_adme_key_inputs.csv"
DRUGS    <- c(52, 1009, 1001)      # low-CLint reference; no enabled route + large CLRbase; dose check
OUT      <- "output_test/probe_clearance_switch.csv"
SYSFILES <- "C:\\Program Files\\Simcyp Simulator V24\\Screens\\SystemFiles"

suppressPackageStartupMessages({ library(Simcyp); library(RSQLite); library(readr); library(stringr) })

ROOT <- normalizePath(getwd(), winslash = "/")
inp  <- read_csv(file.path(ROOT, INPUT), show_col_types = FALSE)
Simcyp::Initialise(SYSFILES, 24, species = SpeciesID$Human, verbose = FALSE)

P  <- function(nm) CompoundParameterID[[nm]]
rb <- function(nm) tryCatch(as.numeric(Simcyp::GetParameter(Tag = P(nm), Category = CategoryID$Compound,
                                                          SubCategory = CompoundID$Substrate)),
                            error = function(e) NA_real_)
set <- function(nm, v) SetCompoundParameter(P(nm), CompoundID$Substrate, as.numeric(v))

# Simcyp prints the SetWorkspace summary on the MESSAGE channel; capture.output() alone
# returns nothing, which made every dose (and so every clearance) NA in the first probe run.
capture_simcyp_output <- function(expr) {
  collected <- character()
  out <- capture.output(withCallingHandlers(
    expr,
    message = function(m) { collected <<- c(collected, conditionMessage(m)); invokeRestart("muffleMessage") },
    warning = function(w) { collected <<- c(collected, conditionMessage(w)); invokeRestart("muffleWarning") }
  ))
  unique(c(collected, out))
}

dose_of <- function(dump) {
  m <- str_match(dump, "^\\s*Dose:\\s*(.*)$"); as.numeric(trimws(m[!is.na(m[, 2]), 2][1]))
}
sim_cl <- function(dose) {
  dbf <- tempfile(fileext = ".db"); Simulate(database = dbf)
  conn <- dbConnect(SQLite(), dbf)
  ps <- GetProfileStats_DB(ProfileID$Csys, CompoundID$Substrate, Inhibition = FALSE,
                           Upper = 95, Lower = 5, Trial = FALSE, conn)
  dbDisconnect(conn); unlink(dbf)
  i <- which(grepl("mean", rownames(ps), ignore.case = TRUE) & !grepl("upper|lower", rownames(ps), ignore.case = TRUE))[1]
  t <- suppressWarnings(as.numeric(colnames(ps))); c <- as.numeric(ps[i, ])
  ok <- !is.na(t) & !is.na(c); t <- t[ok]; c <- c[ok]
  auc <- sum(diff(t) * (head(c, -1) + tail(c, -1)) / 2)
  tl <- tail(which(c > 0), 5); kel <- -unname(coef(lm(log(c[tl]) ~ t[tl]))[2])
  dose / (auc + if (kel > 0) c[length(c)] / kel else 0)
}

res <- list()
old <- setwd(file.path(ROOT, WKSZ_DIR))
for (d in DRUGS) {
  r <- inp[inp$Drug == d, ]
  if (!nrow(r)) { message("skip ", d, ": not in ", INPUT); next }
  ws <- paste0(d, ".wksz")
  cases <- list(
    A = list(),
    B = list(ClearanceSwitch = 1, WOMC_MicrosomeCLintValue1 = r$WOMC_MicrosomeCLintValue1, WOMC_MicrosomeFUinc11 = 1),
    C = list(ClearanceSwitch = 1, WOMC_MicrosomeCLintValue1 = 10 * r$WOMC_MicrosomeCLintValue1, WOMC_MicrosomeFUinc11 = 1),
    D = list(ClearanceSwitch = 1, WOMC_MicrosomeCLintValue1 = r$WOMC_MicrosomeCLintValue1, WOMC_MicrosomeFUinc11 = 1, CL_MetBas = "x10"),
    E = list(ClearanceSwitch = 0, WOMC_MicrosomeCLintValue1 = r$WOMC_MicrosomeCLintValue1, WOMC_MicrosomeFUinc11 = 1),
    F = list(ClearanceSwitch = 1, WOMC_MicrosomeCLintValue1 = 0, WOMC_MicrosomeFUinc11 = 1))
  for (cs in names(cases)) {
    dump <- capture_simcyp_output(invisible(SetWorkspace(ws)))   # fresh template for every case
    dose <- dose_of(dump); tmpl_cl <- rb("CL_MetBas"); tmpl_sw <- rb("ClearanceSwitch")
    if (is.na(dose)) stop(sprintf("Could not read the dose of %s from the SetWorkspace output. ", ws),
                          "This is a script problem, not a model result.", call. = FALSE)
    for (nm in names(cases[[cs]])) {
      v <- cases[[cs]][[nm]]
      set(nm, if (identical(v, "x10")) 10 * tmpl_cl else v)
    }
    cl <- sim_cl(dose)
    res[[length(res) + 1]] <- data.frame(Drug = d, case = cs, template_switch = tmpl_sw,
                                         switch_after = rb("ClearanceSwitch"), CLint = rb("WOMC_MicrosomeCLintValue1"),
                                         CL_MetBas = rb("CL_MetBas"), CLRbase = rb("CLRbase"),
                                         Fu = rb("Fu"), bp = rb("bp"), dose = dose, CL_sim = cl)
    message(sprintf("  %-5s case %s  switch %s->%s  CL_sim %.4g L/h/kg", d, cs, tmpl_sw, rb("ClearanceSwitch"), cl))
  }
}
setwd(old)
res <- do.call(rbind, res)
dir.create(dirname(file.path(ROOT, OUT)), showWarnings = FALSE, recursive = TRUE)
write_csv(res, file.path(ROOT, OUT))

cat("\n================ VERDICT ================\n")
lr <- function(a, b) abs(log10(a / b))
ws_band <- function(clint, fu, bp) {             # well-stirred hepatic blood CL, L/h
  v <- c()
  for (mppgl in c(30, 45)) for (lw in c(1400, 1900)) for (qh in c(80, 100)) {
    h <- clint * mppgl * lw * 60 / 1e6; fb <- fu / bp; v <- c(v, qh * fb * h / (qh + fb * h))
  }
  range(v) * c(0.6, 1.1)
}
ws_ratio <- function(clint, fu, bp, k = 10) {    # expected hepatic(k x CLint) / hepatic(CLint)
  f <- function(ci) { h <- ci * 37.5 * 1650 * 60 / 1e6; fb <- fu / bp; 90 * fb * h / (90 + fb * h) }
  f(k * clint) / f(clint)
}
mech_pass <- TRUE; mag_pass <- TRUE
for (d in unique(res$Drug)) {
  s_ <- res[res$Drug == d, ]; g <- as.list(setNames(s_$CL_sim, s_$case))
  if (anyNA(unlist(g))) {
    cat(sprintf("\nDrug %s   ERROR: clearance could not be computed for case(s) %s - script problem, no verdict.\n",
                d, paste(names(g)[is.na(unlist(g))], collapse = ", ")))
    mech_pass <- FALSE; mag_pass <- FALSE; next
  }
  hB <- (g$B - g$F) * 70; hC <- (g$C - g$F) * 70
  bB <- s_[s_$case == "B", ]
  band <- ws_band(bB$CLint, bB$Fu, bB$bp)
  exp_ratio <- ws_ratio(bB$CLint, bB$Fu, bB$bp)
  chk <- c(`1 switching changes elimination (B != A)` = lr(g$B, g$A) > 0.05,
           `2 CLint drives clearance      (C > B)`    = g$C / g$B > 1.05,
           `3 CL_MetBas inert, switch 1   (D == B)`   = lr(g$D, g$B) < 0.01,
           `4 CLint inert, switch 0       (E == A)`   = lr(g$E, g$A) < 0.01,
           `5 hepatic part scales with CLint (>= half the expected increase)` =
             isTRUE(hB > 0 && hC / hB >= 1 + 0.5 * (exp_ratio - 1)))
  mag <- isTRUE(hB >= band[1] / 3 && hB <= band[2] * 3)
  cat(sprintf("\nDrug %s   CL_sim (L/h/kg)  A=%.4g  B=%.4g  C=%.4g  D=%.4g  E=%.4g  F=%.4g\n",
              d, g$A, g$B, g$C, g$D, g$E, g$F))
  cat(sprintf("   hepatic part (B-F) = %.3g L/h; 10x CLint gave x%.2f (well-stirred expects x%.2f); expectation %.3g - %.3g L/h\n",
              hB, hC / hB, exp_ratio, band[1], band[2]))
  for (k in names(chk)) cat(sprintf("   %s  %s\n", if (isTRUE(chk[[k]])) "PASS" else "FAIL", k))
  cat(sprintf("   %s  6 magnitude within 3-fold of expectation (hepatic/expectation-midpoint = %.2f)\n",
              if (mag) "PASS" else "FAIL", hB / mean(band)))
  mech_pass <- mech_pass && all(chk %in% TRUE); mag_pass <- mag_pass && mag
}
cat(if (mech_pass && mag_pass) "\nALL PASS: ClearanceSwitch = 1 routes elimination through the WOMC CLint at the expected scale. Safe to run the h1 arms.\n"
    else if (mech_pass) "\nMECHANISM OK, MAGNITUDE OFF: the CLint is used but scaled differently from a standard microsomal well-stirred model. Do not run the h1 arms yet. Send output_test/probe_clearance_switch.csv.\n"
    else "\nFAILED: do not run the h1 arms. Send output_test/probe_clearance_switch.csv.\n")
