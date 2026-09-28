"""
Utility functions for DL_train_lgCL workflow.
Consolidates repeated code and provides reusable components.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split


# ============================================================================
# SETUP AND REPRODUCIBILITY
# ============================================================================

def setup_seed(seed=8):
    """Set random seeds for reproducibility across numpy, torch, and CUDA."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(output_dirs):
    """Create all required output directories."""
    for key, path in output_dirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"✓ Directory ready: {path}")


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_raw_data(csv_path, tag_col='tag', task_col='lgCL'):
    """Load raw CSV data with SMILES and property values."""
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded data from {csv_path}: shape {df.shape}")
    return df


def filter_valid_molecules(df, feature_dicts):
    """
    Filter molecules that have valid features in the dictionary.
    Returns filtered dataframe and list of invalid molecules.
    """
    remained_df = df[df['cano_smiles'].isin(feature_dicts['smiles_to_atom_mask'].keys())]
    uncovered_df = df.drop(remained_df.index)
    print(f"✓ Valid molecules: {remained_df.shape[0]}")
    print(f"  Invalid molecules: {uncovered_df.shape[0]}")
    return remained_df, uncovered_df


def split_k_fold_data(df, task_col, tag_col, k=5, seed=42):
    """
    Split data into k-fold cross-validation sets.
    
    Returns:
        dict: {fold_id: (train_df, cv_df)} for each fold
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    
    training_all = df[df[tag_col] == 'train'].copy()
    test_df = df[df[tag_col] == 'test'].copy()
    
    fold_data = {}
    for fold_id, (train_idx, cv_idx) in enumerate(kf.split(training_all), 1):
        train_fold = training_all.iloc[train_idx].copy()
        cv_fold = training_all.iloc[cv_idx].copy()
        fold_data[fold_id] = (train_fold, cv_fold, test_df)
    
    print(f"✓ Created {k}-fold split")
    return fold_data


# ============================================================================
# METRICS COMPUTATION (Consolidated)
# ============================================================================

def compute_metrics(y_true, y_pred, threshold_fe=[2, 3, 4, 5]):
    """
    Compute comprehensive evaluation metrics.
    
    Returns:
        dict: All metrics {MAE, MSE, R2, RMSE, GMFE, FE<n for each n}
    """
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
    }
    
    # Geometric Mean Fold Error (GMFE)
    epsilon = 1e-10
    y_true_lin = np.maximum(10 ** y_true, epsilon)
    y_pred_lin = np.maximum(10 ** y_pred, epsilon)
    ratios = np.maximum(y_pred_lin / y_true_lin, y_true_lin / y_pred_lin)
    metrics['GMFE'] = np.exp(np.mean(np.log(ratios)))
    
    # Fold error frequencies
    for threshold in threshold_fe:
        below_threshold = np.sum(ratios <= threshold)
        metrics[f'FE<{threshold}'] = below_threshold / len(ratios)
    
    return metrics


def compute_accuracy(y_true, y_pred, threshold=0.1):
    """Compute accuracy within threshold of log units."""
    ae = np.abs(y_pred - y_true)
    return np.sum(ae <= threshold) / len(y_pred)


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_epoch(model, dataset, optimizer, loss_fn, batch_size, feature_dicts, feature_dicts_full):
    """
    Execute one training epoch.
    
    Args:
        model: AttentiveFP model
        dataset: DataFrame with columns [PUBCHEM_CID, task, cano_smiles]
        optimizer: torch optimizer
        loss_fn: loss function
        batch_size: batch size
        feature_dicts: feature dictionary for molecule encoding
        feature_dicts_full: full feature dictionary (if needed)
    """
    model.train()
    losses = []
    
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    
    for i in range(0, len(dataset), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_df = dataset.iloc[batch_indices]
        
        # TODO: Assuming get_smiles_array is imported from AttentiveFP
        # x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, _ = \
        #     get_smiles_array(batch_df['cano_smiles'].values, feature_dicts)
        
        # Forward pass
        # atoms_pred, mol_pred = model(...)
        # loss = loss_fn(mol_pred, torch.Tensor(y_batch).view(-1, 1))
        
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # losses.append(loss.item())
        pass
    
    return np.mean(losses) if losses else 0.0


def evaluate(model, dataset, batch_size, feature_dicts, feature_dicts_full, task_col):
    """
    Evaluate model on dataset.
    
    Returns:
        tuple: (y_true, y_pred, metrics_dict)
    """
    model.eval()
    
    all_actuals = []
    all_preds = []
    
    with torch.no_grad():
        indices = np.arange(len(dataset))
        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_df = dataset.iloc[batch_indices]
            
            # TODO: Forward pass similar to train_epoch
            # all_actuals.extend(...)
            # all_preds.extend(...)
            pass
    
    y_true = np.array(all_actuals)
    y_pred = np.array(all_preds)
    metrics = compute_metrics(y_true, y_pred)
    
    return y_true, y_pred, metrics


# ============================================================================
# MODEL SAVING AND LOADING
# ============================================================================

def save_best_model(model, save_path, metrics, epoch):
    """Save model checkpoint with metadata."""
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, save_path)
    print(f"✓ Model saved: {save_path}")


def load_model_checkpoint(model, checkpoint_path):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    print(f"✓ Model loaded: {checkpoint_path}")
    return model, checkpoint


# ============================================================================
# RESULT SAVING
# ============================================================================

def save_predictions(y_true, y_pred, dataset_df, output_path, task_name):
    """Save predictions to CSV with molecular IDs."""
    results_df = pd.DataFrame({
        'PUBCHEM_CID': dataset_df['PUBCHEM_CID'].values,
        'Actual': y_true,
        'Predicted': y_pred,
        'Dataset': task_name,
    })
    results_df.to_csv(output_path, index=False)
    print(f"✓ Results saved: {output_path}")
    return results_df


def save_metrics(metrics_dict, output_path, dataset_names):
    """Save evaluation metrics."""
    metrics_list = []
    for dataset_name in dataset_names:
        row = {'Dataset': dataset_name}
        row.update(metrics_dict.get(dataset_name, {}))
        metrics_list.append(row)
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(output_path, index=False)
    print(f"✓ Metrics saved: {output_path}")
    return metrics_df


# ============================================================================
# EMBEDDINGS GENERATION
# ============================================================================

def generate_embeddings(model, dataset, output_dim, batch_size, feature_dicts):
    """
    Generate molecular embeddings using the model.
    
    Returns:
        pd.DataFrame: Embeddings with original molecule IDs
    """
    model.eval()
    all_embeddings = []
    all_ids = []
    
    with torch.no_grad():
        indices = np.arange(len(dataset))
        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_df = dataset.iloc[batch_indices]
            
            # TODO: Forward pass to get embeddings
            # embeddings = model.get_embeddings(...)  # depends on model implementation
            # all_embeddings.extend(embeddings.cpu().numpy())
            # all_ids.extend(batch_df['PUBCHEM_CID'].values)
            pass
    
    embeddings_array = np.array(all_embeddings)
    column_names = [f'emb_{i+1}' for i in range(output_dim)]
    
    embeddings_df = pd.DataFrame(embeddings_array, columns=column_names)
    embeddings_df.insert(0, 'PUBCHEM_CID', all_ids)
    
    return embeddings_df


# ============================================================================
# VISUALIZATION (Placeholder)
# ============================================================================

def plot_training_history(history, output_path):
    """Plot training history (loss, metrics over epochs)."""
    # TODO: Implement using matplotlib
    pass


def plot_predictions(y_true, y_pred, output_path, title='Predictions vs Actual'):
    """Plot predicted vs actual values."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    
    ax.scatter(y_true, y_pred, alpha=0.6, s=30)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual (log scale)')
    ax.set_ylabel('Predicted (log scale)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Plot saved: {output_path}")
