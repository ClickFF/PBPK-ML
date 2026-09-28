#!/usr/bin/env python3
"""Extract graph embeddings from a trained AttentiveFP encoder.

This is the `feature_save` routine of the training notebooks, lifted out so the embeddings can be
regenerated without opening a notebook. The encoder's regression head is replaced by a linear layer
of the requested width and the molecule-level activations are written out, which is how the
published embedding matrices in `ml/data_prep/` were produced.

The selected width differs per endpoint, and these are the defaults:

    fu 100    CLsys 50    VDss 10

Example, from the repository root:

    python ml/scripts/extract_embeddings.py \
        --checkpoint ml/weights/fu_attentivefp.pt \
        --smiles data/compounds.csv --task Fu

Notes
-----
The checkpoints are whole-model `torch.save` pickles, not state dicts, so `AttentiveFP/` must be
importable — run from the repository root, or set PYTHONPATH. The original notebooks pinned the
tensors to CUDA; this script runs on CUDA when it is available and falls back to CPU, which makes
the embeddings reproducible on a machine without a GPU.

Requires torch, pandas, numpy and rdkit; see environment/requirements-ml.txt for the pinned
versions the published embeddings were produced with.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from AttentiveFP.getFeatures import get_smiles_array, save_smiles_dicts   # noqa: E402

DEFAULT_WIDTH = {"Fu": 100, "CL": 50, "VD": 10}


def setup_seed(seed: int = 0) -> None:
    """Match the seeding of the training notebooks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def fingerprint_dim_of(model) -> int:
    """The encoder's molecule-representation width, read off the existing head."""
    head = getattr(model, "output")
    layer = head[0] if isinstance(head, nn.Sequential) else head
    return int(layer.in_features)


def generate_embeddings(model, dataset, batch_size, feature_dicts, device, smiles_col):
    n_batches = len(dataset) // batch_size + (0 if len(dataset) % batch_size == 0 else 1)
    long = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long, device=device)   # noqa: E731
    flt = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float, device=device)   # noqa: E731
    out = []
    for i in range(n_batches):
        batch = dataset.iloc[i * batch_size:(i + 1) * batch_size, :]
        if batch.empty:
            continue
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, _ = get_smiles_array(
            batch[smiles_col].values, feature_dicts)
        with torch.no_grad():
            _, mol = model(flt(x_atom), flt(x_bonds), long(x_atom_index), long(x_bond_index),
                           flt(x_mask))
        out.append(pd.DataFrame(mol.cpu().numpy(), index=batch.iloc[:, 0]))
    return pd.concat(out) if out else pd.DataFrame()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path, help="trained encoder .pt")
    p.add_argument("--smiles", required=True, type=Path,
                   help="CSV whose first column is the identifier and which has a SMILES column")
    p.add_argument("--task", required=True, choices=sorted(DEFAULT_WIDTH),
                   help="endpoint tag used to name the output columns")
    p.add_argument("--n-embeddings", type=int, default=None,
                   help="embedding width (default: the published width for this endpoint)")
    p.add_argument("--smiles-col", default="cano_smiles")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)

    width = a.n_embeddings if a.n_embeddings is not None else DEFAULT_WIDTH[a.task]
    out = a.out or ROOT / "ml" / "embeddings" / ("%s_embs_%d.csv" % (a.task, width))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed()

    df = pd.read_csv(a.smiles)
    if a.smiles_col not in df.columns:
        raise SystemExit("column %r not in %s; columns are %s"
                         % (a.smiles_col, a.smiles, list(df.columns)[:12]))

    model = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model.to(device).eval()
    model.output = nn.Sequential(
        nn.Linear(in_features=fingerprint_dim_of(model), out_features=width, bias=True)).to(device)

    feature_dicts = save_smiles_dicts(df[a.smiles_col].tolist(), str(a.smiles))
    emb = generate_embeddings(model, df, a.batch_size, feature_dicts, device, a.smiles_col)
    emb.columns = ["%s_%d" % (a.task, i) for i in range(1, emb.shape[1] + 1)]
    emb.reset_index(inplace=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    emb.to_csv(out, index=False)
    print("device %s | %d molecules x %d embeddings -> %s" % (device, len(emb), width, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
