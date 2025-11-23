"""
Main training script for EEG-CLIP.

Supports:
- 4-class fixed templates and batch-wise contrastive learning
- Train-test split and k-fold cross-validation
- Configurable via YAML files

Usage:
    # Train with default config
    python train.py

    # Train with custom config
    python train.py --config config/experiments/4class_fixed.yaml

    # Train specific fold in k-fold
    python train.py --config config/experiments/kfold_4class.yaml --fold 0

    # Run all folds
    python train.py --config config/experiments/kfold_4class.yaml --all-folds
"""

import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    Callback
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

from src.utils.config import load_config, merge_configs, save_config
from src.utils.data_utils import create_data_splits, create_k_fold_splits, run_all_folds
from src.data import MultiSubjectECoGDataset
from src.training import EEGCLIPTrainer4Class, EEGCLIPTrainerBatch
from src.evaluation import Evaluator
from src.utils.visualization import plot_confusion_matrix, plot_embeddings_tsne, plot_training_curves


class MetricsHistoryCallback(Callback):
    """Callback to collect training and validation metrics history"""

    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of training epoch"""
        # Get the logged metrics
        metrics = trainer.logged_metrics

        # Extract train loss (prioritize epoch-level)
        if 'train_loss_epoch' in metrics:
            self.train_loss.append(float(metrics['train_loss_epoch']))
        elif 'train_loss' in metrics:
            self.train_loss.append(float(metrics['train_loss']))

        # Extract train accuracy (prioritize unified 'train_acc')
        if 'train_acc_epoch' in metrics:
            self.train_acc.append(float(metrics['train_acc_epoch']))
        elif 'train_acc' in metrics:
            self.train_acc.append(float(metrics['train_acc']))
        elif 'train_acc_retrieval' in metrics:
            self.train_acc.append(float(metrics['train_acc_retrieval']))
        elif 'train_acc_eeg' in metrics:
            self.train_acc.append(float(metrics['train_acc_eeg']))

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of validation epoch"""
        # Get the logged metrics
        metrics = trainer.logged_metrics

        # Extract validation loss
        if 'val_loss' in metrics:
            self.val_loss.append(float(metrics['val_loss']))

        # Extract validation accuracy (prioritize unified 'val_acc')
        if 'val_acc' in metrics:
            self.val_acc.append(float(metrics['val_acc']))
        elif 'val_acc_class' in metrics:
            self.val_acc.append(float(metrics['val_acc_class']))
        elif 'val_acc_eeg' in metrics:
            self.val_acc.append(float(metrics['val_acc_eeg']))


def setup_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)


def create_dataloaders(train_files, val_files, config):
    """Create train and validation dataloaders"""
    # Create datasets
    train_dataset = MultiSubjectECoGDataset(
        file_list=train_files,
        freq_band=tuple(config['data']['freq_band']),
        time_range=tuple(config['data']['time_range']),
        remove_bad=config['data']['remove_bad'],
        scale=config['data']['scale']
    )

    val_dataset = MultiSubjectECoGDataset(
        file_list=val_files,
        freq_band=tuple(config['data']['freq_band']),
        time_range=tuple(config['data']['time_range']),
        remove_bad=config['data']['remove_bad'],
        scale=config['data']['scale']
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['experiment']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['experiment']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader


def create_trainer(config, fold_idx=None):
    """Create PyTorch Lightning trainer with callbacks"""
    # Experiment name
    exp_name = config['experiment']['name']
    if fold_idx is not None:
        exp_name = f"{exp_name}_fold{fold_idx}"

    # Callbacks
    callbacks = []

    # Metrics history callback
    metrics_callback = MetricsHistoryCallback()
    callbacks.append(metrics_callback)

    # Model checkpoint
    checkpoint_dir = os.path.join(config['logging']['checkpoint_dir'], exp_name)

    # Create filename based on monitored metric
    monitor_metric = config['logging']['monitor']
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'epoch={{epoch:02d}}-{monitor_metric}={{{monitor_metric}:.4f}}',
        monitor=monitor_metric,
        mode=config['logging']['mode'],
        save_top_k=config['logging']['save_top_k'],
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if config['training']['early_stopping']['enabled']:
        early_stop_callback = EarlyStopping(
            monitor=config['training']['early_stopping']['monitor'],
            patience=config['training']['early_stopping']['patience'],
            mode=config['training']['early_stopping']['mode'],
            verbose=True
        )
        callbacks.append(early_stop_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # Logger
    if config['logging']['wandb']['enabled']:
        logger = WandbLogger(
            project=config['logging']['wandb']['project'],
            entity=config['logging']['wandb']['entity'],
            name=exp_name
        )
    else:
        log_dir = os.path.join(config['logging']['log_dir'], exp_name)
        logger = TensorBoardLogger(save_dir=log_dir, name='')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['experiment']['device'],
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
    )

    return trainer, checkpoint_callback, metrics_callback


def create_model(config):
    """Create model based on config"""
    if config['model']['type'] == '4class_fixed':
        return EEGCLIPTrainer4Class(config)
    elif config['model']['type'] == 'batch_contrastive':
        return EEGCLIPTrainerBatch(config)
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")


def train_single_fold(config, train_files, val_files, fold_idx=None):
    """Train a single model (one fold or full train-test split)"""
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(train_files, val_files, config)

    # Create model
    print("\nInitializing model...")
    model = create_model(config)

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {total_params - trainable_params:,}")
    print("="*80 + "\n")

    # Create trainer
    print("Setting up trainer...")
    trainer, checkpoint_callback, metrics_callback = create_trainer(config, fold_idx)

    # Train
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")

    trainer.fit(model, train_loader, val_loader)

    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"\nLoading best model from: {best_model_path}")
        model = model.__class__.load_from_checkpoint(best_model_path, config=config)
    else:
        print("\nUsing final model (no checkpoint saved)")

    # Evaluate on validation set
    print("\n" + "="*80)
    print("EVALUATING ON VALIDATION SET")
    print("="*80)

    model.eval()
    model.to(config['experiment']['device'] if config['experiment']['device'] != 'auto' else 'cuda')

    # Run evaluation
    exp_name = config['experiment']['name']
    if fold_idx is not None:
        exp_name = f"{exp_name}_fold{fold_idx}"

    eval_dir = os.path.join(config['logging']['log_dir'], exp_name, 'evaluation')
    evaluator = Evaluator(
        model=model,
        dataloader=val_loader,
        class_names=config['data']['class_names'],
        device=config['experiment']['device'] if config['experiment']['device'] != 'auto' else 'cuda',
        save_dir=eval_dir
    )

    results = evaluator.evaluate()

    # Visualizations
    if config['evaluation']['visualize_embeddings']:
        print("\nGenerating visualizations...")

        # Confusion matrix
        cm = results['confusion_matrix']
        plot_confusion_matrix(
            confusion_matrix=cm,
            class_names=config['data']['class_names'],
            save_path=os.path.join(eval_dir, 'confusion_matrix.png')
        )

        # t-SNE embeddings
        eeg_embeddings, labels = evaluator.get_embeddings()

        # Get text embeddings
        if hasattr(model, 'get_text_embeddings'):
            if config['model']['type'] == '4class_fixed':
                text_embeddings = model.get_text_embeddings().cpu().numpy()
            else:
                from src.data import CaptionGenerator, ClassTemplates
                template_style = config['captions']['template_style']
                if template_style in ['simple', 'action_focused', 'motor_imagery', 'neural', 'descriptive']:
                    generator = ClassTemplates(template_style)
                    class_templates = generator.get_all_captions()
                else:
                    generator = CaptionGenerator(template_style)
                    class_templates = generator.get_first_variants()
                text_embeddings = model.get_text_embeddings(class_templates).cpu().numpy()
        else:
            text_embeddings = None

        plot_embeddings_tsne(
            eeg_embeddings=eeg_embeddings,
            labels=labels,
            class_names=config['data']['class_names'],
            text_embeddings=text_embeddings,
            save_path=os.path.join(eval_dir, 'embeddings_tsne.png'),
            num_samples=config['evaluation']['tsne_samples']
        )

    # Plot training curves if enabled
    if config['evaluation']['visualize_training_curves']:
        print("\nGenerating training curves...")

        # Check if we have sufficient metrics
        has_loss = len(metrics_callback.train_loss) > 0 and len(metrics_callback.val_loss) > 0
        has_acc = len(metrics_callback.train_acc) > 0 and len(metrics_callback.val_acc) > 0

        if has_loss and has_acc:
            # Ensure all lists are the same length
            min_len = min(
                len(metrics_callback.train_loss),
                len(metrics_callback.val_loss),
                len(metrics_callback.train_acc),
                len(metrics_callback.val_acc)
            )

            if min_len > 0:
                plot_training_curves(
                    train_losses=metrics_callback.train_loss[:min_len],
                    val_losses=metrics_callback.val_loss[:min_len],
                    train_accs=metrics_callback.train_acc[:min_len],
                    val_accs=metrics_callback.val_acc[:min_len],
                    save_path=os.path.join(eval_dir, 'training_curves.png')
                )
            else:
                print("  Warning: No metrics collected for plotting training curves")
        else:
            missing = []
            if not has_loss:
                missing.append("loss metrics")
            if not has_acc:
                missing.append("accuracy metrics")
            print(f"  Warning: Missing {', '.join(missing)} - skipping training curves plot")

    return {
        'val_acc': results['accuracy'],
        'results': results,
        'best_model_path': best_model_path
    }


def main(args):
    """Main training function"""
    # Load config
    print("="*80)
    print("EEG-CLIP TRAINING")
    print("="*80)

    # Load base config
    base_config = load_config('config/base_config.yaml')

    # Load experiment config if provided
    if args.config:
        exp_config = load_config(args.config)
        config = merge_configs(base_config, exp_config)
    else:
        config = base_config

    # Set seed
    setup_seed(config['experiment']['seed'])

    # Disable tokenizer parallelism for notebooks/multi-process
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Get data prefix
    data_prefix = config['data']['data_prefix']

    # Determine training strategy
    val_strategy = config['validation']['strategy']

    if val_strategy == 'train_test_split':
        # Single train-test split
        print("\nUsing train-test split strategy")
        train_files, val_files, test_files = create_data_splits(config, data_prefix)

        # Train model
        results = train_single_fold(config, train_files, val_files)

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Final validation accuracy: {results['val_acc']*100:.2f}%")
        print(f"Best model saved to: {results['best_model_path']}")
        print("="*80)

    elif val_strategy == 'k_fold':
        # K-fold cross-validation
        print("\nUsing k-fold cross-validation strategy")

        kfold_config = config['validation']['k_fold']
        current_fold = kfold_config.get('current_fold', -1)

        if args.all_folds or current_fold == -1:
            # Run all folds
            results = run_all_folds(
                config=config,
                train_fn=train_single_fold,
                data_prefix=data_prefix
            )
        else:
            # Run specific fold
            if args.fold is not None:
                fold_idx = args.fold
            else:
                fold_idx = current_fold

            print(f"\nTraining fold {fold_idx}")
            train_files, val_files = create_k_fold_splits(config, data_prefix, fold_idx)
            results = train_single_fold(config, train_files, val_files, fold_idx)

            print("\n" + "="*80)
            print("TRAINING COMPLETE")
            print("="*80)
            print(f"Fold {fold_idx} validation accuracy: {results['val_acc']*100:.2f}%")
            print(f"Best model saved to: {results['best_model_path']}")
            print("="*80)

    else:
        raise ValueError(f"Unknown validation strategy: {val_strategy}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EEG-CLIP model')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to experiment config file (defaults to base_config.yaml)'
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=None,
        help='Specific fold to run in k-fold cross-validation'
    )
    parser.add_argument(
        '--all-folds',
        action='store_true',
        help='Run all folds in k-fold cross-validation'
    )

    args = parser.parse_args()
    main(args)
