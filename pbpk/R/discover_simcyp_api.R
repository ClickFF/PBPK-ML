#!/usr/bin/env Rscript
# Print the real Simcyp R API surface for dose / trial-design fields.
# Run ONCE on the Simcyp machine:   Rscript discover_simcyp_api.R > simcyp_api.txt
# Then paste the correct ids into DESIGN_MAP in run_simcyp_batch.R.
suppressPackageStartupMessages(library(Simcyp))
Simcyp::Initialise("C:/Program Files/Simcyp Simulator V24/Screens/SystemFiles",
                   24, species = SpeciesID$Human, verbose = FALSE)

cat("=== exported functions matching set/get/param/dose/design ===\n")
fns <- ls("package:Simcyp")
print(grep("(?i)set|get|param|dose|design|trial|regimen", fns, value = TRUE, perl = TRUE))

cat("\n=== formals of the setters/getters ===\n")
for (f in intersect(c("SetParameter","GetParameter","SetCompoundParameter",
                      "GetCompoundParameter","SetPopulationParameter"), fns)) {
  cat("\n--", f, "\n"); print(args(get(f, asNamespace("Simcyp"))))
}

cat("\n=== every exported *ID lookup list, and its dose/design-looking members ===\n")
for (nm in grep("ID$", fns, value = TRUE)) {
  obj <- tryCatch(get(nm, asNamespace("Simcyp")), error = function(e) NULL)
  if (is.null(obj) || !length(names(obj))) next
  hits <- grep("(?i)dose|inf|regimen|interval|route|admin|design|duration|weight|bodyw",
               names(obj), value = TRUE, perl = TRUE)
  cat(sprintf("\n%-26s  %d members", nm, length(names(obj))))
  if (length(hits)) {
    cat("\n   candidates:\n")
    for (h in hits) cat(sprintf("     %-34s = %s\n", h, paste(obj[[h]], collapse = ",")))
  }
}

cat("\n=== full member list of CategoryID (pick the design category) ===\n")
print(names(CategoryID))

cat("\n=== read back one workspace's dose via each plausible category ===\n")
ws <- list.files("wksz_tier2_full", pattern = "^1001\\.wksz$", full.names = TRUE)
if (length(ws)) {
  invisible(SetWorkspace(ws[1]))
  cat("workspace 1001 loaded; try GetParameter with the candidate tags printed above.\n")
} else cat("1001.wksz not found from the current working directory.\n")
