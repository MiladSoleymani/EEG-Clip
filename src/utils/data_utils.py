"""
Data utilities for creating train/val/test splits and k-fold cross-validation.
"""

from typing import Dict, List, Tuple
from sklearn.model_selection import KFold
import numpy as np


def create_data_splits(
    config: dict,
    data_prefix: str
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create train/val/test splits based on configuration.

    Args:
        config: Configuration dictionary
        data_prefix: Prefix path for data files

    Returns:
        train_files: List of training file paths
        val_files: List of validation file paths
        test_files: List of test file paths
    """
    subjects = config['data']['subjects']
    val_config = config['validation']

    if val_config['strategy'] == 'train_test_split':
        return _create_train_test_split(subjects, val_config, data_prefix)
    else:
        raise ValueError(
            f"Use create_k_fold_splits() for strategy: {val_config['strategy']}"
        )


def _create_train_test_split(
    subjects: Dict[str, List[str]],
    val_config: dict,
    data_prefix: str
) -> Tuple[List[str], List[str], List[str]]:
    """Create train/val/test split"""
    split_config = val_config['train_test_split']

    test_subjects = split_config.get('test_subjects', [])
    val_subjects = split_config.get('val_subjects', [])

    # Collect files
    train_files = []
    val_files = []
    test_files = []

    for subject_name, subject_files in subjects.items():
        # Add data prefix to files
        full_paths = [f"{data_prefix}{f}" for f in subject_files]

        if subject_name in test_subjects:
            test_files.extend(full_paths)
        elif subject_name in val_subjects:
            val_files.extend(full_paths)
        else:
            train_files.extend(full_paths)

    # If no validation subjects specified, split from training
    if len(val_files) == 0 and len(train_files) > 0:
        val_split = split_config.get('val_split', 0.2)
        n_val = int(len(train_files) * val_split)

        # Shuffle and split
        np.random.shuffle(train_files)
        val_files = train_files[:n_val]
        train_files = train_files[n_val:]

    print("\n" + "="*80)
    print("DATA SPLIT")
    print("="*80)
    print(f"Training files:   {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files:       {len(test_files)}")
    print("="*80 + "\n")

    return train_files, val_files, test_files


def create_k_fold_splits(
    config: dict,
    data_prefix: str,
    fold_idx: int = 0
) -> Tuple[List[str], List[str]]:
    """
    Create k-fold cross-validation splits.

    Args:
        config: Configuration dictionary
        data_prefix: Prefix path for data files
        fold_idx: Which fold to use (0 to n_folds-1)

    Returns:
        train_files: List of training file paths
        val_files: List of validation file paths
    """
    subjects = config['data']['subjects']
    kfold_config = config['validation']['k_fold']

    n_folds = kfold_config['n_folds']
    fold_strategy = kfold_config['fold_strategy']

    if fold_strategy == 'subject_wise':
        return _create_subject_wise_kfold(
            subjects, data_prefix, n_folds, fold_idx
        )
    elif fold_strategy == 'trial_wise':
        return _create_trial_wise_kfold(
            subjects, data_prefix, n_folds, fold_idx
        )
    else:
        raise ValueError(f"Unknown fold strategy: {fold_strategy}")


def _create_subject_wise_kfold(
    subjects: Dict[str, List[str]],
    data_prefix: str,
    n_folds: int,
    fold_idx: int
) -> Tuple[List[str], List[str]]:
    """Create subject-wise k-fold splits"""
    subject_names = list(subjects.keys())
    n_subjects = len(subject_names)

    if n_folds > n_subjects:
        raise ValueError(
            f"n_folds ({n_folds}) cannot be larger than "
            f"number of subjects ({n_subjects})"
        )

    # Create k-fold splitter
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Get train/val subject indices for this fold
    splits = list(kf.split(subject_names))
    train_indices, val_indices = splits[fold_idx]

    # Get subject names
    train_subject_names = [subject_names[i] for i in train_indices]
    val_subject_names = [subject_names[i] for i in val_indices]

    # Collect files
    train_files = []
    val_files = []

    for subject_name in train_subject_names:
        files = [f"{data_prefix}{f}" for f in subjects[subject_name]]
        train_files.extend(files)

    for subject_name in val_subject_names:
        files = [f"{data_prefix}{f}" for f in subjects[subject_name]]
        val_files.extend(files)

    print("\n" + "="*80)
    print(f"K-FOLD SPLIT (Fold {fold_idx+1}/{n_folds}) - Subject-wise")
    print("="*80)
    print(f"Training subjects:   {train_subject_names}")
    print(f"Validation subjects: {val_subject_names}")
    print(f"Training files:      {len(train_files)}")
    print(f"Validation files:    {len(val_files)}")
    print("="*80 + "\n")

    return train_files, val_files


def _create_trial_wise_kfold(
    subjects: Dict[str, List[str]],
    data_prefix: str,
    n_folds: int,
    fold_idx: int
) -> Tuple[List[str], List[str]]:
    """Create trial-wise k-fold splits"""
    # Collect all files
    all_files = []
    for subject_files in subjects.values():
        files = [f"{data_prefix}{f}" for f in subject_files]
        all_files.extend(files)

    # Create k-fold splitter
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Get train/val file indices for this fold
    splits = list(kf.split(all_files))
    train_indices, val_indices = splits[fold_idx]

    # Get files
    train_files = [all_files[i] for i in train_indices]
    val_files = [all_files[i] for i in val_indices]

    print("\n" + "="*80)
    print(f"K-FOLD SPLIT (Fold {fold_idx+1}/{n_folds}) - Trial-wise")
    print("="*80)
    print(f"Training files:   {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print("="*80 + "\n")

    return train_files, val_files


def run_all_folds(
    config: dict,
    train_fn,
    data_prefix: str
) -> List[Dict]:
    """
    Run training for all folds in k-fold cross-validation.

    Args:
        config: Configuration dictionary
        train_fn: Training function that takes (config, train_files, val_files, fold_idx)
        data_prefix: Prefix path for data files

    Returns:
        results: List of results for each fold
    """
    kfold_config = config['validation']['k_fold']
    n_folds = kfold_config['n_folds']

    results = []

    print("\n" + "="*80)
    print(f"RUNNING {n_folds}-FOLD CROSS-VALIDATION")
    print("="*80 + "\n")

    for fold_idx in range(n_folds):
        print(f"\n{'='*80}")
        print(f"TRAINING FOLD {fold_idx+1}/{n_folds}")
        print(f"{'='*80}\n")

        # Create splits for this fold
        train_files, val_files = create_k_fold_splits(
            config, data_prefix, fold_idx
        )

        # Train model
        fold_results = train_fn(config, train_files, val_files, fold_idx)
        results.append(fold_results)

    # Print summary
    print("\n" + "="*80)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("="*80)

    if results and 'val_acc' in results[0]:
        accs = [r['val_acc'] for r in results]
        print(f"\nValidation Accuracies:")
        for i, acc in enumerate(accs):
            print(f"  Fold {i+1}: {acc*100:.2f}%")
        print(f"\nMean ± Std: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")

    print("="*80 + "\n")

    return results
