#!/usr/bin/env Rscript
# =====================================================================
# Find, by difference, which parameters encode "whole-organ metabolic clearance from an
# entered HLM CLint" — the setting that cannot be reached from R with ClearanceSwitch and
# WholeOrganRadio alone (probe 2: 16 combinations, no response to CLint).
#
# WHAT TO DO IN THE SIMCYP GUI FIRST (one compound, a few minutes)
#   1. Open  wksz_tier1_minimal/1001.wksz
#   2. Elimination tab: switch the substrate to whole-organ metabolic clearance from
#      in vitro human liver microsomes, and enter
#          CLint  = 123.456        (this exact value, so it is easy to spot)
#          fu,inc = 1
#      Leave everything else untouched.
#   3. Save As ->  wksz_gui_reference/1001_WOMC.wksz     (create that folder)
#
# THEN
#   Rscript diff_workspace_params.R
#
# It dumps all compound parameters from both workspaces, prints every parameter that differs,
# and writes output_test/workspace_param_diff.csv. Those parameters are exactly what the h1
# tables need to set. Send the CSV.
# =====================================================================

BEFORE <- "wksz_tier1_minimal/1001.wksz"
AFTER  <- "wksz_gui_reference/1001_WOMC.wksz"
MARKER <- 123.456                     # the CLint value entered in the GUI
OUT    <- "output_test/workspace_param_diff.csv"
SYSFILES <- "C:\\Program Files\\Simcyp Simulator V24\\Screens\\SystemFiles"

suppressPackageStartupMessages({ library(Simcyp); library(readr) })

ROOT <- normalizePath(getwd(), winslash = "/")
for (f in c(BEFORE, AFTER))
  if (!file.exists(file.path(ROOT, f)))
    stop("Not found: ", f, "\n  Do the GUI step described at the top of this script first.", call. = FALSE)

Simcyp::Initialise(SYSFILES, 24, species = SpeciesID$Human, verbose = FALSE)

dump_params <- function(path) {
  invisible(capture.output(suppressMessages(SetWorkspace(file.path(ROOT, path)))))
  nm <- names(CompoundParameterID)
  v <- vapply(nm, function(p) {
    x <- tryCatch(Simcyp::GetParameter(Tag = CompoundParameterID[[p]], Category = CategoryID$Compound,
                                       SubCategory = CompoundID$Substrate),
                  error = function(e) NA)
    if (length(x) != 1) x <- paste(as.character(x), collapse = "|")
    as.character(x)
  }, character(1))
  setNames(v, nm)
}

message("dumping ", BEFORE); a <- dump_params(BEFORE)
message("dumping ", AFTER);  b <- dump_params(AFTER)
message(sprintf("%d compound parameters read from each workspace", length(a)))

same <- names(a)[!is.na(a) & !is.na(b) & a == b]
diff <- setdiff(names(a), same)
num <- function(x) suppressWarnings(as.numeric(x))
out <- data.frame(parameter = diff, before = unname(a[diff]), after = unname(b[diff]),
                  stringsAsFactors = FALSE)
out$is_marker <- !is.na(num(out$after)) & abs(num(out$after) - MARKER) < 1e-6

write_csv(out, file.path(ROOT, OUT))
cat(sprintf("\n%d of %d parameters differ\n\n", nrow(out), length(a)))
if (any(out$is_marker))
  cat("The CLint you typed landed in:\n",
      paste(sprintf("   %s = %s\n", out$parameter[out$is_marker], out$after[out$is_marker]), collapse = ""), "\n")
print(out[, c("parameter", "before", "after")], row.names = FALSE, right = FALSE)
cat("\nwritten to ", OUT, "\n", sep = "")
