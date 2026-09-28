# Hybrid ML–PBPK pipeline

This is the simulation half of the study, including the all-predicted workflow reported as
**Figure 7** of the paper (internally `h2_run0` / `PBPK_v4_ML`: DL–ML fu, VDss and CLsys on an S+
physicochemical background).

## What is here

| Path | Contents |
|---|---|
| `R/run_simcyp_batch.R` | The batch driver. Initialises Simcyp V24, applies the per-arm inputs, runs each compound and reads the applied parameters back. |
| `R/runs_config.csv` | The run matrix: one row per arm, naming the input table and the workspace set. |
| `R/probe_*.R`, `R/diff_workspace_params.R`, `R/discover_simcyp_api.R` | The probes that established how the simulator routes clearance. |
| `inputs/` | The entered parameter table for each arm — what was loaded into Simcyp. |
| `applied/<arm>/` | What Simcyp actually used, read back per compound: `adme_key_inputs.csv`, `applied_vs_intended.csv`, `clearance_sentinel.csv`, `template_watch.csv`. |

## Why the read-back files are here

An earlier version of this work reported a bottom-up arm that had not been parameterised as
described: in Simcyp v24 an entered microsomal CLint is ignored unless the intrinsic-clearance type
is also set, and because the pipeline had no read-back step, the question "was this setting applied?"
could not be answered from the outputs. The driver now hard-fails an arm unless both
`ClearanceSwitch = 1` and `WOMC_CLintType1 = 2`, and the applied values are written out per compound.
`applied/<arm>/applied_vs_intended.csv` is the file to check if you want to confirm that a scenario
ran as its label says.

## What is not here, and why

The 41 Simcyp compound workspaces (`.wksz`) are **not** redistributed. They derive from Certara's
Simcyp compound library and are not ours to publish. Everything needed to rebuild them is here: the
entered inputs per arm, the run matrix, and the driver. Re-executing the simulations requires a
licensed **Simcyp Simulator V24** and the Simcyp R package (v24.0.8); the driver expects the
simulator's system files at the default Windows install path and will need that line adjusted for
another installation.

Simulated profiles are evaluated by `si_analysis/analysis/01_evaluate.py`.
