"""
Visualization utilities for EEG-CLIP.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
from typing import Optional, List


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True,
    figsize: tuple = (10, 8)
):
    """
    Plot confusion matrix.

    Args:
        confusion_matrix: Confusion matrix array or list
        class_names: List of class names
        save_path: Path to save figure (optional)
        normalize: Whether to normalize the confusion matrix
        figsize: Figure size
    """
    # Convert to numpy array if it's a list
    if isinstance(confusion_matrix, list):
        cm_original = np.array(confusion_matrix)
    else:
        cm_original = confusion_matrix.copy()

    cm = cm_original.copy()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Set ticks
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True Label',
        xlabel='Predicted Label'
    )

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                text = f'{cm[i, j]:.2f}\n({int(cm_original[i, j])})'
            else:
                text = f'{int(cm[i, j])}'

            ax.text(j, i, text,
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=10)

    title = 'Confusion Matrix' + (' (Normalized)' if normalize else '')
    ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved confusion matrix to {save_path}")

    plt.show()
    plt.close()


def plot_embeddings_tsne(
    eeg_embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    text_embeddings: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    num_samples: int = 500,
    perplexity: int = 30,
    random_state: int = 42,
    figsize: tuple = (12, 10)
):
    """
    Visualize embeddings using t-SNE.

    Args:
        eeg_embeddings: EEG embeddings array (n_samples, embedding_dim)
        labels: Labels array (n_samples,)
        class_names: List of class names
        text_embeddings: Optional text embeddings (n_classes, embedding_dim)
        save_path: Path to save figure (optional)
        num_samples: Maximum number of samples to plot
        perplexity: t-SNE perplexity parameter
        random_state: Random state for reproducibility
        figsize: Figure size
    """
    # Subsample if needed
    if len(eeg_embeddings) > num_samples:
        indices = np.random.RandomState(random_state).choice(
            len(eeg_embeddings), num_samples, replace=False
        )
        eeg_embeddings = eeg_embeddings[indices]
        labels = labels[indices]

    # Combine EEG and text embeddings
    if text_embeddings is not None:
        all_embeddings = np.vstack([eeg_embeddings, text_embeddings])
    else:
        all_embeddings = eeg_embeddings

    # Apply t-SNE
    print(f"\nApplying t-SNE to {len(all_embeddings)} embeddings...")
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=min(perplexity, len(all_embeddings) - 1)
    )
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Split back
    eeg_2d = embeddings_2d[:len(eeg_embeddings)]
    if text_embeddings is not None:
        text_2d = embeddings_2d[len(eeg_embeddings):]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    # Plot EEG embeddings
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        mask = labels == i
        if mask.sum() > 0:
            ax.scatter(
                eeg_2d[mask, 0],
                eeg_2d[mask, 1],
                c=[color],
                label=f'EEG: {class_name}',
                alpha=0.6,
                s=30,
                edgecolors='none'
            )

    # Plot text embeddings as stars
    if text_embeddings is not None:
        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            ax.scatter(
                text_2d[i, 0],
                text_2d[i, 1],
                c=[color],
                marker='*',
                s=800,
                edgecolors='black',
                linewidths=2,
                label=f'Text: {class_name}',
                zorder=10
            )

    ax.set_title('EEG and Text Embeddings (t-SNE)', fontsize=16, pad=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved t-SNE visualization to {save_path}")

    plt.show()
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5)
):
    """
    Plot training curves.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves (convert to percentages)
    train_accs_pct = [acc * 100 for acc in train_accs]
    val_accs_pct = [acc * 100 for acc in val_accs]

    ax2.plot(epochs, train_accs_pct, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs_pct, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved training curves to {save_path}")

    plt.show()
    plt.close()
