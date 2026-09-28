# Getting started

Two environments, because the training half and the analysis half were run on different stacks.
Both are pinned in `environment/`.

## 1. Reproduce the analysis, tables and figures

This is the lighter path and needs no GPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r environment/requirements-analysis.txt

python ml/scripts/make_splits.py            # writes ml/splits/, prints a check against Table 1
python si_analysis/analysis/01_evaluate.py  # then 02_paired.py, 03_decomposition.py, ...
python si_analysis/analysis/09_si_tables.py
```

`make_splits.py` should end with `all counts reproduce Table 1 of the paper`. If it does not, the
copy of Data S1 in `data/` is not the one the paper was built from.

## 2. Re-extract the graph embeddings

Needs the ML environment. A GPU makes it faster but is not required.

```bash
pip install -r environment/requirements-ml.txt

python ml/scripts/extract_embeddings.py \
    --checkpoint ml/weights/fu_attentivefp.pt \
    --smiles <a CSV with an identifier column and a SMILES column> \
    --task Fu
```

Run it from the repository root: the checkpoints are whole-model pickles and need `AttentiveFP/` on
the import path. Per-endpoint embedding widths and loading notes are in `ml/weights/provenance.md`.

## 3. Retrain the encoders

`ml/notebooks/DL_train_lgCL.ipynb` is the documented reference workflow — start there. It keeps the
stored outputs of the published run, so the reported numbers can be read without a GPU.
`DL_train_lgFu.ipynb` and `DL_train_lgVD.ipynb` are the working notebooks for the other two
endpoints and have their outputs cleared. Each notebook writes checkpoints to its own
`saved_*_mods/` directory and reports the selected epoch as `best_mod`. Training was done on
CUDA 12.1.

`ml/notebooks/embML_CL_SVR.ipynb` is the regressor stage: merged embeddings plus RDKit descriptors
into a support-vector regressor. Its stored outputs are also those of the published run.

## 4. Re-run the PBPK simulations

Requires a licensed Simcyp Simulator V24 and the Simcyp R package. See `pbpk/README.md` — the
compound workspaces are not redistributed, but the entered inputs, the run matrix and the driver
are all here.

## Where the data lives

`data/` holds Data S1 (modelling datasets, split assignments and predictions), Data S2 (PBPK inputs
as entered and as read back) and Data S3 (the 41-compound curated analysis table as a `.csv`, with
its data dictionary). The Data S1 and Data S2 workbooks each carry a `changelog` sheet recording the
corrections applied to them.

The merged embedding + RDKit feature matrices are not in Data S1; they are in `ml/data_prep/` as
`*Embeddings_RDKIT_S1.csv`.
