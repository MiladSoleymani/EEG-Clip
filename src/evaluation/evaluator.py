"""
Comprehensive evaluation for EEG-CLIP models.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Optional
import json
from pathlib import Path


class Evaluator:
    """
    Comprehensive evaluator for EEG-CLIP models.

    Computes various metrics and generates evaluation reports.
    """

    def __init__(
        self,
        model,
        dataloader,
        class_names: List[str],
        device: str = 'cuda',
        save_dir: Optional[str] = None
    ):
        """
        Args:
            model: Trained PyTorch Lightning model
            dataloader: DataLoader for evaluation
            class_names: List of class names
            device: Device to run evaluation on
            save_dir: Directory to save results (optional)
        """
        self.model = model
        self.dataloader = dataloader
        self.class_names = class_names
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else None

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        self.model.to(device)

        # Storage for predictions and embeddings
        self.all_preds = []
        self.all_labels = []
        self.all_eeg_embeddings = []
        self.all_logits = []

    def evaluate(self) -> Dict:
        """
        Run full evaluation.

        Returns:
            results: Dictionary containing all evaluation metrics
        """
        print("\n" + "="*80)
        print("RUNNING EVALUATION")
        print("="*80)

        # Collect predictions
        self._collect_predictions()

        # Compute metrics
        results = self._compute_metrics()

        # Print results
        self._print_results(results)

        # Save results if save_dir is provided
        if self.save_dir:
            self._save_results(results)

        return results

    def _collect_predictions(self):
        """Collect predictions and embeddings from all batches"""
        print("\nCollecting predictions...")

        # Check if dataloader is empty
        if len(self.dataloader) == 0:
            raise ValueError(
                "Dataloader is empty! No data to evaluate. "
                "Please check that your validation dataset contains samples."
            )

        with torch.no_grad():
            for batch_idx, (eeg_data, labels) in enumerate(self.dataloader):
                eeg_data = eeg_data.to(self.device)
                labels = labels.to(self.device)

                # Get logits
                if hasattr(self.model, 'model'):
                    # For PyTorch Lightning modules
                    if hasattr(self.model.model, 'class_embeddings'):
                        # 4-class fixed model
                        logits = self.model(eeg_data)
                    else:
                        # Batch contrastive model - need class templates
                        from ..data import CaptionGenerator
                        template_style = self.model.config['captions']['template_style']
                        if template_style in ['simple', 'action_focused', 'descriptive']:
                            from ..data import ClassTemplates
                            generator = ClassTemplates(template_style)
                            class_templates = generator.get_all_captions()
                        else:
                            generator = CaptionGenerator(template_style)
                            class_templates = generator.get_first_variants()

                        eeg_emb = self.model.get_embeddings(eeg_data)
                        text_emb = self.model.get_text_embeddings(class_templates)
                        logits = torch.matmul(eeg_emb, text_emb.t())
                else:
                    # Direct model
                    logits = self.model(eeg_data)

                # Get predictions
                preds = torch.argmax(logits, dim=1)

                # Get embeddings
                if hasattr(self.model, 'get_embeddings'):
                    embeddings = self.model.get_embeddings(eeg_data)
                elif hasattr(self.model, 'model'):
                    embeddings = self.model.model.get_eeg_embeddings(eeg_data)
                else:
                    embeddings = self.model.get_eeg_embeddings(eeg_data)

                # Store results
                self.all_preds.append(preds.cpu())
                self.all_labels.append(labels.cpu())
                self.all_logits.append(logits.cpu())
                self.all_eeg_embeddings.append(embeddings.cpu())

                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(self.dataloader)} batches")

        # Check if any predictions were collected
        if len(self.all_preds) == 0:
            raise ValueError(
                "No predictions were collected! The dataloader loop did not execute. "
                "Please check your dataset and dataloader configuration."
            )

        # Concatenate all results
        self.all_preds = torch.cat(self.all_preds).numpy()
        self.all_labels = torch.cat(self.all_labels).numpy()
        self.all_logits = torch.cat(self.all_logits).numpy()
        self.all_eeg_embeddings = torch.cat(self.all_eeg_embeddings).numpy()

        print(f"\n✓ Collected predictions for {len(self.all_preds)} samples")

    def _compute_metrics(self) -> Dict:
        """Compute all evaluation metrics"""
        print("\nComputing metrics...")

        results = {}

        # Overall metrics
        results['accuracy'] = accuracy_score(self.all_labels, self.all_preds)
        results['precision_macro'] = precision_score(
            self.all_labels, self.all_preds, average='macro', zero_division=0
        )
        results['recall_macro'] = recall_score(
            self.all_labels, self.all_preds, average='macro', zero_division=0
        )
        results['f1_macro'] = f1_score(
            self.all_labels, self.all_preds, average='macro', zero_division=0
        )

        # Per-class metrics
        precision_per_class = precision_score(
            self.all_labels, self.all_preds, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            self.all_labels, self.all_preds, average=None, zero_division=0
        )
        f1_per_class = f1_score(
            self.all_labels, self.all_preds, average=None, zero_division=0
        )

        results['per_class'] = {}
        for i, class_name in enumerate(self.class_names):
            results['per_class'][class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i])
            }

        # Confusion matrix
        cm = confusion_matrix(self.all_labels, self.all_preds)
        results['confusion_matrix'] = cm.tolist()

        # Classification report
        report = classification_report(
            self.all_labels,
            self.all_preds,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        results['classification_report'] = report

        return results

    def _print_results(self, results: Dict):
        """Print evaluation results"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)

        # Overall metrics
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {results['accuracy']*100:.2f}%")
        print(f"  Precision: {results['precision_macro']*100:.2f}%")
        print(f"  Recall:    {results['recall_macro']*100:.2f}%")
        print(f"  F1 Score:  {results['f1_macro']*100:.2f}%")

        # Per-class metrics
        print(f"\nPer-Class Metrics:")
        for class_name, metrics in results['per_class'].items():
            print(f"\n  {class_name}:")
            print(f"    Precision: {metrics['precision']*100:.2f}%")
            print(f"    Recall:    {metrics['recall']*100:.2f}%")
            print(f"    F1 Score:  {metrics['f1']*100:.2f}%")

        # Confusion matrix
        print(f"\nConfusion Matrix:")
        cm = np.array(results['confusion_matrix'])
        print(f"  {'':15s}", end='')
        for name in self.class_names:
            print(f"{name:15s}", end='')
        print()
        for i, row_name in enumerate(self.class_names):
            print(f"  {row_name:15s}", end='')
            for val in cm[i]:
                print(f"{val:15d}", end='')
            print()

        print("="*80)

    def _save_results(self, results: Dict):
        """Save evaluation results to files"""
        print(f"\nSaving results to {self.save_dir}...")

        # Save metrics as JSON
        metrics_file = self.save_dir / 'evaluation_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  ✓ Saved metrics to {metrics_file}")

        # Save predictions
        predictions_file = self.save_dir / 'predictions.npz'
        np.savez(
            predictions_file,
            predictions=self.all_preds,
            labels=self.all_labels,
            logits=self.all_logits
        )
        print(f"  ✓ Saved predictions to {predictions_file}")

        # Save embeddings
        embeddings_file = self.save_dir / 'embeddings.npz'
        np.savez(
            embeddings_file,
            eeg_embeddings=self.all_eeg_embeddings,
            labels=self.all_labels
        )
        print(f"  ✓ Saved embeddings to {embeddings_file}")

    def get_embeddings(self) -> tuple:
        """
        Get computed embeddings.

        Returns:
            eeg_embeddings: EEG embeddings
            labels: Corresponding labels
        """
        return self.all_eeg_embeddings, self.all_labels

    def get_predictions(self) -> tuple:
        """
        Get predictions and labels.

        Returns:
            predictions: Predicted labels
            labels: True labels
            logits: Model logits
        """
        return self.all_preds, self.all_labels, self.all_logits
