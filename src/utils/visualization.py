"""
Utilitaires de visualisation pour l'analyse et l'évaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch


# Labels des émotions
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Plot l'historique d'entraînement (loss et accuracy).

    Args:
        history: Dictionnaire avec 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Chemin pour sauvegarder la figure
        figsize: Taille de la figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot F1-Score
    if 'val_f1' in history:
        axes[2].plot(epochs, history['val_f1'], 'g-', label='Val F1', linewidth=2)
        axes[2].set_title('Validation F1-Score', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1-Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")

    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot la matrice de confusion.

    Args:
        cm: Matrice de confusion
        class_names: Noms des classes
        normalize: Si True, normalise par ligne (%)
        save_path: Chemin de sauvegarde
        figsize: Taille de la figure
    """
    if class_names is None:
        class_names = EMOTION_LABELS

    # Normalize
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'}
    )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to {save_path}")

    plt.show()


def plot_class_distribution(
    class_counts: Dict[str, int],
    title: str = "Class Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot la distribution des classes.

    Args:
        class_counts: Dictionnaire {classe: count}
        title: Titre du plot
        save_path: Chemin de sauvegarde
        figsize: Taille de la figure
    """
    emotions = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=figsize)

    # Bar plot
    bars = plt.bar(emotions, counts, color='steelblue', alpha=0.8, edgecolor='black')

    # Ajouter valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Distribution plot saved to {save_path}")

    plt.show()


def visualize_predictions(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    num_samples: int = 16,
    save_path: Optional[str] = None,
    denormalize: bool = True
):
    """
    Visualise des prédictions du modèle.

    Args:
        images: Tensor d'images (B, C, H, W)
        true_labels: Labels vrais
        pred_labels: Labels prédits
        num_samples: Nombre d'échantillons à afficher
        save_path: Chemin de sauvegarde
        denormalize: Dénormaliser les images
    """
    num_samples = min(num_samples, len(images))

    # Grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for idx in range(num_samples):
        img = images[idx]
        true_label = true_labels[idx].item()
        pred_label = pred_labels[idx].item()

        # Dénormaliser si nécessaire
        if denormalize:
            img = img * 0.5 + 0.5  # Inverse de normalize (mean=0.5, std=0.5)

        # Convert to numpy
        if img.shape[0] == 1:
            # Grayscale
            img_np = img.squeeze(0).cpu().numpy()
            cmap = 'gray'
        else:
            # RGB
            img_np = img.permute(1, 2, 0).cpu().numpy()
            cmap = None

        # Plot
        axes[idx].imshow(img_np, cmap=cmap)
        axes[idx].axis('off')

        # Title with true and predicted labels
        true_emotion = EMOTION_LABELS[true_label]
        pred_emotion = EMOTION_LABELS[pred_label]

        correct = true_label == pred_label
        color = 'green' if correct else 'red'

        axes[idx].set_title(
            f'True: {true_emotion}\nPred: {pred_emotion}',
            fontsize=10,
            color=color,
            fontweight='bold'
        )

    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Predictions visualization saved to {save_path}")

    plt.show()


def plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None
):
    """
    Plot les courbes d'apprentissage (train vs val loss).

    Utile pour détecter overfitting/underfitting.

    Args:
        train_losses: Losses d'entraînement
        val_losses: Losses de validation
        save_path: Chemin de sauvegarde
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    # Annotations
    min_val_loss = min(val_losses)
    min_val_epoch = val_losses.index(min_val_loss) + 1

    plt.plot(min_val_epoch, min_val_loss, 'r*', markersize=15)
    plt.annotate(
        f'Best: {min_val_loss:.4f}\n(Epoch {min_val_epoch})',
        xy=(min_val_epoch, min_val_loss),
        xytext=(10, 10),
        textcoords='offset points',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        arrowprops=dict(arrowstyle='->')
    )

    plt.title('Learning Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Learning curves saved to {save_path}")

    plt.show()


def plot_per_class_accuracy(
    per_class_acc: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot l'accuracy par classe.

    Args:
        per_class_acc: Dict {emotion: accuracy}
        save_path: Chemin de sauvegarde
        figsize: Taille de la figure
    """
    emotions = list(per_class_acc.keys())
    accuracies = [per_class_acc[e] * 100 for e in emotions]  # En %

    plt.figure(figsize=figsize)

    bars = plt.bar(emotions, accuracies, color='coral', alpha=0.8, edgecolor='black')

    # Ligne moyenne
    mean_acc = np.mean(accuracies)
    plt.axhline(y=mean_acc, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.1f}%')

    # Valeurs sur barres
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Per-class accuracy saved to {save_path}")

    plt.show()


if __name__ == '__main__':
    # Test visualizations
    print("Testing visualization utilities...")

    # Test confusion matrix
    cm = np.random.randint(0, 100, (7, 7))
    plot_confusion_matrix(cm, normalize=True)

    # Test class distribution
    class_counts = {
        'angry': 4953,
        'disgust': 547,
        'fear': 5121,
        'happy': 8989,
        'sad': 6077,
        'surprise': 4002,
        'neutral': 6198
    }
    plot_class_distribution(class_counts)

    print("✅ Visualization tests complete")
