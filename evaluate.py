"""
Evaluation script for trained EEG-CLIP models.

Usage:
    # Evaluate a trained model
    python evaluate.py --checkpoint path/to/checkpoint.ckpt --config config/experiments/4class_fixed.yaml

    # Evaluate with custom data
    python evaluate.py --checkpoint path/to/checkpoint.ckpt --config config.yaml --data-files file1.fif file2.fif
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, merge_configs
from src.data import MultiSubjectECoGDataset
from src.training import EEGCLIPTrainer4Class, EEGCLIPTrainerBatch
from src.evaluation import Evaluator
from src.utils.visualization import plot_confusion_matrix, plot_embeddings_tsne


def main(args):
    """Main evaluation function"""
    print("="*80)
    print("EEG-CLIP EVALUATION")
    print("="*80)

    # Load config
    base_config = load_config('config/base_config.yaml')

    if args.config:
        exp_config = load_config(args.config)
        config = merge_configs(base_config, exp_config)
    else:
        config = base_config

    # Disable tokenizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Determine data files
    if args.data_files:
        # Use provided data files
        data_files = args.data_files
        print(f"\nUsing provided data files: {data_files}")
    else:
        # Use test set from config
        from src.utils.data_utils import create_data_splits
        data_prefix = config['data']['data_prefix']
        _, _, data_files = create_data_splits(config, data_prefix)

        if len(data_files) == 0:
            print("\nNo test files found in config, using validation files")
            _, data_files, _ = create_data_splits(config, data_prefix)

        print(f"\nUsing test/validation files from config: {len(data_files)} files")

    # Create dataset and dataloader
    print("\nCreating dataset...")
    dataset = MultiSubjectECoGDataset(
        file_list=data_files,
        freq_band=tuple(config['data']['freq_band']),
        time_range=tuple(config['data']['time_range']),
        remove_bad=config['data']['remove_bad'],
        scale=config['data']['scale']
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['experiment']['num_workers'],
        pin_memory=True
    )

    # Load model
    print(f"\nLoading model from checkpoint: {args.checkpoint}")

    if config['model']['type'] == '4class_fixed':
        model = EEGCLIPTrainer4Class.load_from_checkpoint(
            args.checkpoint,
            config=config
        )
    elif config['model']['type'] == 'batch_contrastive':
        model = EEGCLIPTrainerBatch.load_from_checkpoint(
            args.checkpoint,
            config=config
        )
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")

    # Set device
    device = config['experiment']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    model.to(device)

    print(f"✓ Model loaded and moved to {device}")

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        output_dir = os.path.join('evaluation_results', checkpoint_name)

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    # Run evaluation
    print("\n" + "="*80)
    print("RUNNING EVALUATION")
    print("="*80)

    evaluator = Evaluator(
        model=model,
        dataloader=dataloader,
        class_names=config['data']['class_names'],
        device=device,
        save_dir=output_dir
    )

    results = evaluator.evaluate()

    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Confusion matrix
    print("\nPlotting confusion matrix...")
    cm = results['confusion_matrix']
    plot_confusion_matrix(
        confusion_matrix=cm,
        class_names=config['data']['class_names'],
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )

    # t-SNE embeddings
    if args.plot_embeddings:
        print("\nPlotting t-SNE embeddings...")
        eeg_embeddings, labels = evaluator.get_embeddings()

        # Get text embeddings
        if config['model']['type'] == '4class_fixed':
            text_embeddings = model.get_text_embeddings().cpu().numpy()
        else:
            from src.data import CaptionGenerator, ClassTemplates
            template_style = config['captions']['template_style']
            if template_style in ['simple', 'action_focused', 'descriptive']:
                generator = ClassTemplates(template_style)
                class_templates = generator.get_all_captions()
            else:
                generator = CaptionGenerator(template_style)
                class_templates = generator.get_first_variants()
            text_embeddings = model.get_text_embeddings(class_templates).cpu().numpy()

        plot_embeddings_tsne(
            eeg_embeddings=eeg_embeddings,
            labels=labels,
            class_names=config['data']['class_names'],
            text_embeddings=text_embeddings,
            save_path=os.path.join(output_dir, 'embeddings_tsne.png'),
            num_samples=args.tsne_samples
        )

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Overall Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Macro F1 Score:   {results['f1_macro']*100:.2f}%")
    print(f"\nResults saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained EEG-CLIP model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.ckpt file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (defaults to base_config.yaml)'
    )
    parser.add_argument(
        '--data-files',
        type=str,
        nargs='+',
        default=None,
        help='List of data files to evaluate on (optional, uses config if not provided)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save evaluation results (optional)'
    )
    parser.add_argument(
        '--plot-embeddings',
        action='store_true',
        default=True,
        help='Generate t-SNE embedding plots'
    )
    parser.add_argument(
        '--tsne-samples',
        type=int,
        default=500,
        help='Number of samples to use for t-SNE visualization'
    )

    args = parser.parse_args()
    main(args)
