# Encoder checkpoint provenance

One AttentiveFP encoder per endpoint, each the checkpoint selected by the training run as
`best_mod` (the epoch with the best validation score, fold 1 of the 5-fold split).

| File | Endpoint | Original filename | Size |
|---|---|---|---|
| `fu_attentivefp.pt` | fraction unbound | `df_feature_5620_Sat_Apr__5_21-26-10_2025_83.pt` | 3.3 MB |
| `CLsys_attentivefp.pt` | systemic clearance | `df_feature_5620_Sun_Apr__6_20-32-40_2025_54.pt` | 881 KB |
| `VDss_attentivefp.pt` | steady-state volume | `df_feature_5620_Sat_Apr__5_21-26-12_2025_18.pt` | 881 KB |

## Loading them

These are whole-model `torch.save` pickles, not state dicts, so the `AttentiveFP` package must be
importable when you load one:

```python
import sys, torch
sys.path.insert(0, ".")            # repository root, so `AttentiveFP` resolves
model = torch.load("ml/weights/fu_attentivefp.pt", map_location="cpu", weights_only=False)
```

`weights_only=False` is required: `torch.load` defaults to `True` from torch 2.6, which refuses
pickled model objects. The environment they were produced in is pinned in
`environment/requirements-ml.txt` (torch 2.1.2+cu121, Python 3.8.18).

## Embedding width

The published feature matrices in `ml/data_prep/` use a different embedding width per endpoint,
chosen during model selection:

| Endpoint | Width |
|---|---|
| fu | 100 |
| CL<sub>sys</sub> | 50 |
| VD<sub>ss</sub> | 10 |

`ml/scripts/extract_embeddings.py` defaults to these. Pass `--n-embeddings` to override.
