# ML_Models

This repository collects the notebook workflows, local code dependencies, and result artifacts used for the PKDL3 modeling work.

## Shared code

- `AttentiveFP/`: local Python package required by the Jupyter notebooks
- `config.py`: central training and path configuration
- `utils.py`: shared helper functions for training/evaluation flow
- `requirements.txt`: Python dependency list
- `README.md`: notes on what to keep with the notebook workflow

## Main notebook cases

- `DL_train_lgCL.ipynb`: the CL deep-learning training notebook that imports `AttentiveFP`
- `embML_CL_SVR.ipynb`: embedding + SVR workflow for lgCL evaluation

## Data files in this repo

- `data_prep/df_feature_5620.csv`: main input table used by training notebook
- `data_prep/lgCL_Embeddings_RDKIT_S1.csv`
- `data_prep/lgFu_Embeddings_RDKIT_S1.csv`
- `data_prep/lgVD_Embeddings_RDKIT_S1.csv`

Local cache file (not required for clean sharing/reproduction):

- `data_prep/df_feature_5620.pickle.pickle`

## Notes

- Keep generated checkpoints, checkpoint caches, and other large intermediates out of Git unless they are part of the deliverable.
- Use `.gitignore` to avoid accidental commits of notebook caches, Python bytecode, and temporary artifacts.
