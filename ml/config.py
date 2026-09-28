"""
Configuration for DL_train_lgCL workflow
"""

import os
from datetime import datetime

# ============================================================================
# TASK PARAMETERS
# ============================================================================
TASK_NAME = 'lgCL'
TAG = 'CL_train_test'  # Split Tag for Benchmarking
# TAG = 'CT_train_test'  # Split Tag for Downstream C-T Profiles Prediction   
# ============================================================================
# TRAINING PARAMETERS
# ============================================================================
TRAINING = {
    'seed': 8,
    'cuda_benchmark': True,
    'fold_id': 1,
    'k_fold': 5,
    'batch_size': 100,
    'epochs': 100,       # Early stopping may terminate earlier
    'random_seed': 0,
    'frac_train': 0.8,
}

# ============================================================================
# MODEL ARCHITECTURE PARAMETERS
# ============================================================================
MODEL = {
    'fingerprint_dim': 100,
    'p_dropout': 0.2,
    'output_units_num': 1,  # Regression output
    'radius': 2,
    'T': 4,  # Attention layers
}

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================
OPTIM = {
    'learning_rate': 2.5,      # Will be used as 10^-learning_rate
    'weight_decay': 5,         # Will be used as 10^-weight_decay
    'loss_fn': 'MSE',
}

# ============================================================================
# EARLY STOPPING CRITERIA
# ============================================================================
EARLY_STOPPING = {
    'patience': 50,            # Epochs without improvement
    'min_delta': 1e-4,         # Minimal improvement threshold
    'rmse_threshold': 0.65,    # RMSE cutoff for model saving
}

# ============================================================================
# FILE PATHS (relative to codespace root)
# ============================================================================
# Input data
INPUT_DATA = 'data_prep/df_feature_5620.csv'
FEATURE_CACHE = 'data_prep/df_feature_5620.pickle'

# Output layout options:
# - 'nested': task/cv_k_fold/fold_n/... (original)
# - 'compact': outputs/task_k{k}_f{fold}/... (shallower)
OUTPUT_PATHS = {
    'layout': 'compact',
    'root': '.',
}

# Output structure
def get_output_dirs(task_name=TASK_NAME, k=TRAINING['k_fold'], fold_id=TRAINING['fold_id']):
    """
    Generate output directory paths based on task and fold.
    Returns a dict with all output locations.
    """
    layout = OUTPUT_PATHS.get('layout', 'nested')
    root = OUTPUT_PATHS.get('root', '.')

    if layout == 'compact':
        train_dir = os.path.join(root, 'outputs', task_name)
        fold_dir = os.path.join(train_dir, f'k{k}_f{fold_id}')
    else:
        train_dir = os.path.join(root, task_name, f'cv_{k}_fold')
        fold_dir = os.path.join(train_dir, f'fold_{fold_id}')

    return {
        'train_dir': train_dir,
        'fold_dir': fold_dir,
        'model_dir': os.path.join(fold_dir, f'saved_{task_name}_mods'),
        'result_dir': os.path.join(fold_dir, 'results'),
        'emb_dir': os.path.join(fold_dir, 'embeddings'),
    }

# ============================================================================
# EVALUATION METRICS
# ============================================================================
METRICS = {
    'threshold_fe': [2, 3, 4, 5],  # Fold error thresholds for reporting
    'compute_gmfe': True,
}

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
TIMESTAMP = datetime.now().strftime('%a_%b__%d_%H-%M-%S_%Y')

if __name__ == '__main__':
    # Print config for verification
    print('TRAINING PARAMS:', TRAINING)
    print('MODEL PARAMS:', MODEL)
    print('OPTIM PARAMS:', OPTIM)
    print('OUTPUT PATHS:', OUTPUT_PATHS)
    print('OUTPUT DIRS:', get_output_dirs())
