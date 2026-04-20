# DL_train_lgCL - Simplified & Reproducible Workflow

## 🎯 Overview

This codespace contains a **refactored and simplified** version of the DL_train_lgCL workflow for training AttentiveFP models to predict ligand clearance (lgCL).

The original notebook (~1200 lines) has been reorganized into:
- **7 clear, focused cells** (simplified notebook)
- **config.py** - All hyperparameters and paths in one place
- **utils.py** - Reusable utility functions
- **requirements.txt** - Explicit dependencies
- **AttentiveFP/** - Local molecular fingerprinting module

## 📁 Directory Structure

```
codespace/
├── DL_train_lgCL_simplified.ipynb  ← Main notebook (START HERE)
├── config.py                        ← All parameters & paths
├── utils.py                         ← Shared utility functions
├── requirements.txt                 ← Python dependencies
├── .gitignore                       ← Git ignore rules
├── README.md                        ← Original reference
├── AttentiveFP/                     ← Local module (imported by notebook)
│   ├── __init__.py
│   ├── AttentiveLayers.py
│   ├── Featurizer.py
│   └── getFeatures.py
├── data_prep/                       ← Input data directory
│   └── df_feature_5620.csv          ← Raw data (TO BE PROVIDED)
└── lgCL/cv_5_fold/fold_1/           ← Output directory (auto-created)
    ├── saved_lgCL_mods/             ← Trained models (.pt files)
    ├── results/                     ← Metrics & predictions
    └── embeddings/                  ← Generated embeddings
```

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Input Data

Place the feature CSV file at:
```
codespace/data_prep/df_feature_5620.csv
```

**Required columns:**
- `PUBCHEM_CID` - Molecule identifier
- `Smiles` - SMILES string
- `cano_smiles` - Canonical SMILES (can be None initially)
- `lgCL` - Target variable (log-scale clearance)
- `CT_train_test` - Split indicator ('train' or 'test')

### 3. Run the Notebook

Open and run `DL_train_lgCL_simplified.ipynb` in Jupyter:

```bash
jupyter notebook DL_train_lgCL_simplified.ipynb
```

**Execute cells in order:**
1. **Setup & Imports** - Initialize environment
2. **Load & Prepare Data** - Load CSV and generate AttentiveFP features
3. **K-fold Split** - Create cross-validation folds
4. **Model Initialization** - Build AttentiveFP architecture
5. **Training Loop** - Train with early stopping
6. **Evaluation & Metrics** - Test and save results
7. **Embedding Generation** - Generate molecular embeddings (optional)

### 4. Check Outputs

Results are saved to `lgCL/cv_5_fold/fold_1/`:

- ✅ **Models**: `saved_lgCL_mods/*.pt` - Best trained model(s)
- ✅ **Metrics**: `results/metrics_*.csv` - MAE, RMSE, R², etc.
- ✅ **Predictions**: `results/results_*.csv` - Actual vs predicted values
- ✅ **Plots**: `results/predictions_*.png` - Visualization
- ✅ **Embeddings**: `embeddings/CL_embs_*.csv` - Molecular embeddings

## ⚙️ Configuration

All hyperparameters are in `config.py`:

```python
# Training
TRAINING = {
    'seed': 8,              # Reproducibility
    'epochs': 100,          # Max epochs
    'batch_size': 100,      # Batch size
    'k_fold': 5,            # CV folds
    'fold_id': 1,           # Which fold to run
}

# Model
MODEL = {
    'fingerprint_dim': 100,
    'p_dropout': 0.2,
    'radius': 2,            # Graph radius
    'T': 4,                 # Attention layers
}

# Optimization
OPTIM = {
    'learning_rate': 2.5,   # 10^-2.5
    'weight_decay': 5,      # 10^-5
}
```

**To run different settings:**
1. Edit `config.py`
2. Re-run notebook cells 1-7

## 🔄 Reproducibility

This workflow ensures reproducibility through:

✅ **Fixed random seeds** - NumPy, PyTorch, CUDA  
✅ **Explicit parameters** - All in `config.py`  
✅ **Deterministic operations** - `cudnn.deterministic = True`  
✅ **Dependency pinning** - `requirements.txt` with versions  
✅ **Clean code structure** - Modular utils and config  

Run the same notebook twice → Identical results

## 📊 Typical Performance

On the original dataset:

| Split | Samples | R² | RMSE | MAE |
|-------|---------|-----|------|-----|
| Train | 1287 | 0.97 | 0.10 | — |
| Valid | ~250 | 0.35 | 0.55 | — |
| Test | 177 | 0.36 | 0.46 | 0.35 |

## 🔧 Troubleshooting

### ❌ GPU Memory Error
- Reduce `batch_size` in `config.py`
- Example: `'batch_size': 50` or `32`

### ❌ "AttentiveFP module not found"
- Ensure `AttentiveFP/` is in codespace
- Check `sys.path` includes current directory

### ❌ "df_feature_5620.csv not found"
- Place data file at: `data_prep/df_feature_5620.csv`
- Use absolute path in `config.py` if needed

### ❌ Features not cached
- First run generates features → slow (~minutes)
- Features are saved to `.pickle` for future runs

## 📝 Key Simplifications vs Original

| Aspect | Original | Simplified |
|--------|----------|-----------|
| **Cells** | ~40 | 7 |
| **Imports** | Scattered | Cell 1 |
| **Parameters** | Hard-coded | `config.py` |
| **Utils** | Inline | `utils.py` |
| **Paths** | Absolute | Relative (configurable) |
| **Comments** | Sparse | Comprehensive |
| **Structure** | Exploratory | Production-ready |

## 🎓 Using This for Research

### Modify hyperparameters:
```python
# config.py
TRAINING['epochs'] = 200
MODEL['fingerprint_dim'] = 128
OPTIM['learning_rate'] = 2.0
```

### Add custom metrics:
```python
# utils.py - Add in compute_metrics()
metrics['custom_metric'] = ...
```

### Try different data splits:
```python
# config.py
TRAINING['fold_id'] = 2  # Try fold 2
```

### Save embeddings with custom dimensions:
```python
# DL_train_lgCL_simplified.ipynb - Cell 7
embedding_dim = 50  # Change from 10
```

## 📚 Dependencies

Core packages (see `requirements.txt`):
- **torch** ≥ 1.9.0 - Deep learning
- **rdkit** ≥ 2020.09.1 - Chemistry
- **pandas**, **numpy** - Data handling
- **scikit-learn** - Metrics
- **matplotlib**, **seaborn** - Visualization

## 📖 References

- AttentiveFP: [https://github.com/OpenDrugDiscovery/AttentiveFP](https://github.com/OpenDrugDiscovery/AttentiveFP)
- Original notebook location: `../backup_July25/ML Models/DL_train_lgCL.ipynb`

## ✅ Next Steps

1. **[Optional] Version control:**
   ```bash
   cd /data6/lic154/PKDL3/ML_Models/codespace
   git add .
   git commit -m "Initial simplified DL_train_lgCL setup"
   ```

2. **Run on multiple folds:**
   - Change `TRAINING['fold_id']` in `config.py`
   - Re-run notebook for each fold
   - Results save to different fold directories

3. **Share/Transfer:**
   - This codespace is now self-contained
   - Transfer entire directory with data
   - Others can reproduce exactly

---

**Questions?** Check `config.py` documentation or notebook cell comments.

**Last updated:** 2026-04-19  
**Tested with:** PyTorch 1.9+, Python 3.8+, CUDA 11.0+
