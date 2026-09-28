# Environments

Two environments were used, and they are different.

| | file | used for |
|---|---|---|
| ML training and embedding extraction | `environment-ml.yml` (primary), `requirements-ml.txt` | the AttentiveFP encoders, embedding extraction, the SVR regressors |
| Analysis and figures | `requirements-analysis.txt` | everything under `si_analysis/`, which produces the SI tables and the main-text figures |

Each file lists the packages the published code imports, at the versions used. The ML versions were
read from the archived training environment; the analysis versions are those that produced the
published figures and tables.

`rdkit` is installed from conda-forge, so `environment-ml.yml` is the route to prefer:

```bash
conda env create -f environment/environment-ml.yml
conda activate atfp_env
```

## Notes

The encoders were trained on CUDA 12.1. `ml/scripts/extract_embeddings.py` runs on CPU when no GPU
is present, so the embeddings can be regenerated without one. No pickled scikit-learn model is
shipped, so the scikit-learn version matters only if you refit the regressors.

The PBPK half needs a licensed Simcyp Simulator V24 and is not covered by these files; see
`pbpk/README.md`.
