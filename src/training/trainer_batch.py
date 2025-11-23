"""
PyTorch Lightning trainer for batch-wise contrastive EEG-CLIP.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from ..models import EEG_CLIP_Batch
from ..data import CaptionGenerator


class EEGCLIPTrainerBatch(pl.LightningModule):
    """
    PyTorch Lightning trainer for EEG-CLIP with batch-wise contrastive learning.

    Args:
        config: Configuration dictionary containing all hyperparameters
    """

    def __init__(self, config: dict):
        super().__init__()

        self.save_hyperparameters(config)
        self.config = config

        # Create caption generator
        template_style = config['captions']['template_style']
        self.caption_generator = CaptionGenerator(template_style=template_style)

        # Initialize model
        self.model = EEG_CLIP_Batch(
            num_electrodes=config['model']['num_electrodes'],
            chunk_size=config['model']['chunk_size'],
            embedding_dim=config['model']['embedding_dim'],
            temperature=config['model']['temperature'],
            text_model=config['model']['text_model'],
            atcnet_config=config['model'].get('atcnet', None)
        )

        # Loss function: Cross-entropy for contrastive learning
        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics for retrieval accuracy (not used for logging, just internal tracking)
        self.train_acc_retrieval = torchmetrics.Accuracy(task='multiclass', num_classes=32)
        self.val_acc_retrieval = torchmetrics.Accuracy(task='multiclass', num_classes=32)

        # Metrics for classification accuracy (4 classes)
        num_classes = config['data']['num_classes']
        self.train_acc_class = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc_class = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc_class = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

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

    def forward(self, eeg_data, text_captions):
        return self.model(eeg_data, text_captions)

    def training_step(self, batch, batch_idx):
        eeg_data, labels = batch
        batch_size = eeg_data.size(0)

        # Generate captions for this batch
        text_captions = self.caption_generator.generate_batch_captions(labels.cpu().numpy())

        # Forward pass: get similarity matrices
        logits_eeg_to_text, logits_text_to_eeg = self(eeg_data, text_captions)

        # Create labels: diagonal indices (each sample matches itself)
        targets = torch.arange(batch_size, device=eeg_data.device)

        # Symmetric contrastive loss
        loss_eeg = self.loss_fn(logits_eeg_to_text, targets)  # EEG → Text
        loss_text = self.loss_fn(logits_text_to_eeg, targets)  # Text → EEG
        loss = (loss_eeg + loss_text) / 2

        # Calculate retrieval accuracy
        preds_eeg = torch.argmax(logits_eeg_to_text, dim=1)
        preds_text = torch.argmax(logits_text_to_eeg, dim=1)

        acc_eeg = (preds_eeg == targets).float().mean()
        acc_text = (preds_text == targets).float().mean()
        acc_retrieval = (acc_eeg + acc_text) / 2

        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc_retrieval, prog_bar=True, on_step=True, on_epoch=True)  # Unified metric
        self.log('train_acc_retrieval', acc_retrieval, on_step=False, on_epoch=True)
        self.log('train_acc_eeg', acc_eeg, on_step=False, on_epoch=True)
        self.log('train_acc_text', acc_text, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        eeg_data, labels = batch
        batch_size = eeg_data.size(0)

        # Generate captions
        text_captions = self.caption_generator.generate_batch_captions(labels.cpu().numpy())

        # Forward pass
        logits_eeg_to_text, logits_text_to_eeg = self(eeg_data, text_captions)

        # Retrieval loss and accuracy
        targets = torch.arange(batch_size, device=eeg_data.device)
        loss_eeg = self.loss_fn(logits_eeg_to_text, targets)
        loss_text = self.loss_fn(logits_text_to_eeg, targets)
        loss = (loss_eeg + loss_text) / 2

        preds_eeg = torch.argmax(logits_eeg_to_text, dim=1)
        preds_text = torch.argmax(logits_text_to_eeg, dim=1)

        acc_eeg = (preds_eeg == targets).float().mean()
        acc_text = (preds_text == targets).float().mean()
        acc_retrieval = (acc_eeg + acc_text) / 2

        # Classification accuracy: which class has highest similarity?
        # Get embeddings for all 4 class templates
        class_templates = self.caption_generator.get_first_variants()
        eeg_emb = self.model.get_eeg_embeddings(eeg_data)
        class_emb = self.model.get_text_embeddings(class_templates)
        class_logits = torch.matmul(eeg_emb, class_emb.t())
        class_preds = torch.argmax(class_logits, dim=1)

        acc_class = self.val_acc_class(class_preds, labels)

        # Update per-class metrics
        self.val_precision.update(class_preds, labels)
        self.val_recall.update(class_preds, labels)
        self.val_f1.update(class_preds, labels)
        self.val_confusion.update(class_preds, labels)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val_acc', acc_class, prog_bar=True, on_epoch=True, sync_dist=True)  # Unified metric (classification)
        self.log('val_acc_retrieval', acc_retrieval, on_epoch=True, sync_dist=True)
        self.log('val_acc_eeg', acc_eeg, on_epoch=True, sync_dist=True)
        self.log('val_acc_text', acc_text, on_epoch=True, sync_dist=True)
        self.log('val_acc_class', acc_class, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        eeg_data, labels = batch

        # Classification using class templates
        class_templates = self.caption_generator.get_first_variants()
        eeg_emb = self.model.get_eeg_embeddings(eeg_data)
        class_emb = self.model.get_text_embeddings(class_templates)
        class_logits = torch.matmul(eeg_emb, class_emb.t())
        class_preds = torch.argmax(class_logits, dim=1)

        acc = self.test_acc_class(class_preds, labels)
        self.test_confusion.update(class_preds, labels)

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

    def get_text_embeddings(self, text_captions):
        """Get text embeddings for visualization/analysis"""
        return self.model.get_text_embeddings(text_captions)
