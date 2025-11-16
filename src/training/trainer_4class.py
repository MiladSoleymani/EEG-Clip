"""
PyTorch Lightning trainer for 4-class fixed EEG-CLIP.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from ..models import EEG_CLIP_4Class
from ..data import ClassTemplates


class EEGCLIPTrainer4Class(pl.LightningModule):
    """
    PyTorch Lightning trainer for EEG-CLIP with 4 fixed class templates.

    Args:
        config: Configuration dictionary containing all hyperparameters
    """

    def __init__(self, config: dict):
        super().__init__()

        self.save_hyperparameters(config)
        self.config = config

        # Create class templates
        template_style = config['captions']['template_style']
        class_templates_obj = ClassTemplates(template_style=template_style)
        class_templates = class_templates_obj.get_all_captions()

        # Initialize model
        self.model = EEG_CLIP_4Class(
            num_classes=config['data']['num_classes'],
            num_electrodes=config['model']['num_electrodes'],
            chunk_size=config['model']['chunk_size'],
            embedding_dim=config['model']['embedding_dim'],
            temperature=config['model']['temperature'],
            text_model=config['model']['text_model'],
            class_templates=class_templates,
            atcnet_config=config['model'].get('atcnet', None)
        )

        # Loss function: Simple cross-entropy
        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics for training
        num_classes = config['data']['num_classes']
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

        # Per-class metrics
        self.val_precision = torchmetrics.Precision(
            task='multiclass', num_classes=num_classes, average='none'
        )
        self.val_recall = torchmetrics.Recall(
            task='multiclass', num_classes=num_classes, average='none'
        )
        self.val_f1 = torchmetrics.F1Score(
            task='multiclass', num_classes=num_classes, average='none'
        )

        # Confusion matrix
        self.val_confusion = torchmetrics.ConfusionMatrix(
            task='multiclass', num_classes=num_classes
        )
        self.test_confusion = torchmetrics.ConfusionMatrix(
            task='multiclass', num_classes=num_classes
        )

    def forward(self, eeg_data):
        return self.model(eeg_data)

    def training_step(self, batch, batch_idx):
        eeg_data, labels = batch

        # Forward pass: get logits for 4 classes
        logits = self(eeg_data)  # (batch_size, 4)

        # Calculate loss
        loss = self.loss_fn(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)

        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        eeg_data, labels = batch

        # Forward pass
        logits = self(eeg_data)

        # Calculate loss
        loss = self.loss_fn(logits, labels)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, labels)

        # Update per-class metrics
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_f1.update(preds, labels)
        self.val_confusion.update(preds, labels)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        eeg_data, labels = batch

        # Forward pass
        logits = self(eeg_data)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, labels)
        self.test_confusion.update(preds, labels)

        # Log metrics
        self.log('test_acc', acc, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        """Log per-class metrics at end of validation epoch"""
        # Compute per-class metrics
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        f1 = self.val_f1.compute()

        # Log per-class metrics
        class_names = self.config['data']['class_names']
        for i, class_name in enumerate(class_names):
            self.log(f'val_precision_{class_name}', precision[i], sync_dist=True)
            self.log(f'val_recall_{class_name}', recall[i], sync_dist=True)
            self.log(f'val_f1_{class_name}', f1[i], sync_dist=True)

        # Log average metrics
        self.log('val_precision_avg', precision.mean(), prog_bar=True, sync_dist=True)
        self.log('val_recall_avg', recall.mean(), sync_dist=True)
        self.log('val_f1_avg', f1.mean(), prog_bar=True, sync_dist=True)

        # Reset metrics
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # Only optimize EEG encoder + temperature
        optimizer = torch.optim.AdamW(
            [
                {'params': self.model.eeg_encoder.parameters()},
                {'params': [self.model.temperature]}
            ],
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        # Learning rate scheduler
        scheduler_config = self.config['training']['scheduler']
        scheduler_type = scheduler_config['type']

        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config['cosine']['T_max'],
                eta_min=scheduler_config['cosine']['eta_min']
            )
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config['step']['step_size'],
                gamma=scheduler_config['step']['gamma']
            )
        elif scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config['reduce_on_plateau']['mode'],
                factor=scheduler_config['reduce_on_plateau']['factor'],
                patience=scheduler_config['reduce_on_plateau']['patience'],
                min_lr=scheduler_config['reduce_on_plateau']['min_lr']
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': 'val_loss' if scheduler_type == 'reduce_on_plateau' else None
            }
        }

    def get_embeddings(self, eeg_data):
        """Get EEG embeddings for visualization/analysis"""
        return self.model.get_eeg_embeddings(eeg_data)

    def get_text_embeddings(self):
        """Get text embeddings for visualization/analysis"""
        return self.model.get_text_embeddings()
