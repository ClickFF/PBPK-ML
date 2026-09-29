# PBPK-ML

This repository contains the code, trained encoders, data and analysis for "Graph-Based Molecular
Embeddings for Hybrid ML–PBPK Modeling and Systematic Evaluation of Clearance Parameterization"
(*Journal of Chemical Information and Modeling*, manuscript ci-2026-02076e).

Graph attention (AttentiveFP) embeddings are combined with RDKit descriptors to predict three human
PK parameters, and those predictions are then supplied to physiologically based pharmacokinetic
simulations in Simcyp and evaluated against clinical concentration–time profiles for 41
intravenously dosed compounds.

- [Abstract](#abstract)
- [Repository contents](#repository-contents)
- [Data availability](#data-availability)
- [Requirements](#requirements)
- [Installation](#installation)
- [Reproducing the results](#reproducing-the-results)
- [Notes on reproducibility](#notes-on-reproducibility)
- [License](#license)
- [Citation](#citation)

## Abstract

Accurate prediction of human pharmacokinetics (PK) remains challenging in early drug discovery when
experimentally determined ADME parameters are unavailable. We developed a graph-based deep learning
encoder–machine learning predictor (DL–ML) framework for human PK parameters and evaluated its
integration into physiologically based pharmacokinetic (PBPK) simulations in Simcyp. On benchmark
datasets, graph attention–derived molecular embeddings combined with RDKit descriptors performed
comparably to descriptor-based ML models for systemic clearance (CLsys), plasma unbound fraction
(Fu), and volume of distribution at steady state (VDss). Paired comparisons against reimplemented
and reproduced descriptor-based controls showed no detectable differences. For 41 compounds, we then
evaluated how predicted PK inputs propagated through a common PBPK framework, including bottom-up
scaled intrinsic clearance (CLint) and predicted CLsys-based clearance parameterizations. Replacing
observed clearance with a predicted value reduced twofold coverage for area under the
concentration–time curve over the observed interval (AUC0–t) from 82.9% to 41.5–53.7%, whereas
non-clearance input substitutions produced comparatively small changes in drug exposure metrics. The
two predicted-clearance workflows were not detectably different on the evaluated accuracy endpoints.
As a practical end-to-end assessment of the structure-to-PBPK with Fu, VDss, and CLsys all predicted
by the DL–ML models, AUC0–t and peak concentration (Cmax) were within twofold for 41% and 59% of
compounds, respectively. Persistent Cmax underprediction with observed inputs was consistently
sensitive to distribution-model structure instead of input-prediction error. Overall, this work
suggested that learned graph representations complemented rather than replaced engineered
descriptors, while clearance-prediction accuracy was a major practical constraint on downstream drug
exposure and PK profiles performance.

![Molecular graph to AttentiveFP embeddings concatenated with RDKit descriptors, to a support-vector
regressor per endpoint, giving predicted fu, CLsys and VDss as inputs to a minimal PBPK model in
Simcyp and a simulated concentration-time profile. Below, AUC0-t within twofold of the clinical
value for 41 compounds: 83% with observed clearance against 42-54% with predicted
clearance.](docs/toc_graphic.png)

## Repository contents

- `ml/` — the graph attention encoder package (`AttentiveFP/`), the training and regressor notebooks,
  the trained checkpoints (`weights/`, with loading notes in `provenance.md`), the PubChem CIDs for
  every data split (`splits/`), and two standalone scripts, `scripts/make_splits.py` and
  `scripts/extract_embeddings.py`.
- `pbpk/` — the R driver that runs Simcyp, the parameters entered for each scenario (`inputs/`), and
  the values read back from the simulator afterwards to confirm which inputs were active
  (`applied/`).
- `si_analysis/` — the pipeline that produces the Supporting Information tables and the main-text
  figures; `si_analysis/README.md` maps each figure to the script that builds it.
- `data/` — Data S1 (modelling datasets, split assignments and predictions), Data S2 (PBPK inputs as
  entered and as read back, per compound and scenario), Data S3 (the 41-compound analysis table, 92
  columns, with a dictionary giving the source of each column), and a convenience copy of the
  Supporting Information; the version published by the journal governs.
- `environment/` — pinned specifications for both environments.
- `MANIFEST.csv` — the sha256 of every file in the repository.

## Data availability

The modelling datasets, the PBPK inputs and read-back records, and the compound-level analysis
table are in `data/`. The underlying human PK measurements are from Jia et al. (*J. Med. Chem.*
2025, 68, 7737–7750); physicochemical and ADME inputs were generated with ADMET Predictor v11
(Simulations Plus), and template parameters come from the Simcyp compound library. Those three
sources are third-party and are attributed rather than relicensed.

The Simcyp compound workspaces are **not** redistributed. Re-running the simulations requires a
licensed Simcyp Simulator V24 and the Simcyp R package; `pbpk/inputs/` records every entered
parameter so the scenarios can be rebuilt. See `pbpk/README.md`.

## Requirements

ML training and embedding extraction, Python 3.8.18:

- torch 2.1.2 (built against CUDA 12.1)
- rdkit 2022.03.5
- numpy 1.22.1, pandas 2.0.3, scipy 1.7.3
- scikit-learn 1.3.2, matplotlib 3.4.3, seaborn 0.13.1

Analysis and figures, Python 3.12.1:

- pandas 3.0.5, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, openpyxl 3.1.5

## Installation

rdkit comes from conda-forge, so conda is the route to prefer:

```bash
conda env create -f environment/environment-ml.yml
conda activate atfp_env
```

The analysis pipeline is independent of the ML environment and needs only pip:

```bash
pip install -r environment/requirements-analysis.txt
```

## Reproducing the results

`GETTING_STARTED.md` covers each stage in order. In short:

```bash
python ml/scripts/make_splits.py            # split identifier files
python ml/scripts/extract_embeddings.py     # embeddings from the released checkpoints
python si_analysis/analysis/01_evaluate.py  # then 02 .. 05, 06, 09
```

Training an encoder from scratch needs a CUDA GPU; extracting embeddings from the released
checkpoints does not. Re-running the PBPK simulations needs Simcyp.

## Notes on reproducibility

The checkpoints in `ml/weights/` are whole-model pickles and need `AttentiveFP/` on the import path.
No fitted scikit-learn regressor is shipped, so the scikit-learn version matters only if you refit
the regressors.

The endpoint encoders were trained once on Training Set #1 and reused without retraining, so Test
Set #2 is not encoder-naive: 81 of its 110 compounds, including 32 of the 41 carried forward to
PBPK, were seen with their endpoint labels during encoder training. The regressors themselves never
saw Test Set #2. The identifier files in `ml/splits/` make this independently checkable, and the
exposure is tabulated in the Supporting Information.

Figures and tables regenerate exactly from the released data. Retraining will not reproduce the
published checkpoints bit-for-bit, since training used early stopping on a held-out fold and a GPU
backend, but the reported conclusions do not depend on a particular run.

## License

Code is BSD-3-Clause (`LICENSE`); data files generated by the authors are CC BY 4.0
(`LICENSE-DATA`), which also lists the third-party sources that grant does not cover.

## Citation

    L. Cai, A. Badkul, J. Zhai, M. McWilliams, L. Xie, J. Wang
    "Graph-Based Molecular Embeddings for Hybrid ML-PBPK Modeling and Systematic Evaluation
    of Clearance Parameterization"
    Journal of Chemical Information and Modeling (manuscript ci-2026-02076e)

The DOI will be added here on publication. `CITATION.cff` carries the same reference in machine-readable form.
