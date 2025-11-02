"""
Métriques pour l'évaluation des modèles de reconnaissance d'émotions.

Implémente:
- Accuracy (globale et par classe)
- F1-score (macro, weighted)
- Confusion matrix
- Precision, Recall
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


# Labels des émotions
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Calcule l'accuracy.

    Args:
        predictions: Prédictions du modèle (logits ou probabilities)
        targets: Labels vrais

    Returns:
        Accuracy (0-1)
    """
    # Si predictions sont des probabilities/logits, prendre argmax
    if predictions.dim() > 1:
        predictions = torch.argmax(predictions, dim=1)

    correct = (predictions == targets).sum().item()
    total = targets.size(0)

    return correct / total if total > 0 else 0.0


def compute_f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    average: str = 'macro'
) -> float:
    """
    Calcule le F1-score.

    Args:
        predictions: Prédictions
        targets: Labels vrais
        average: 'macro', 'weighted', ou 'micro'

    Returns:
        F1-score
    """
    if predictions.dim() > 1:
        predictions = torch.argmax(predictions, dim=1)

    # Convert to numpy
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    return f1_score(targets_np, preds_np, average=average, zero_division=0)


def compute_precision_recall(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    average: str = 'macro'
) -> Tuple[float, float]:
    """
    Calcule precision et recall.

    Args:
        predictions: Prédictions
        targets: Labels vrais
        average: 'macro', 'weighted', ou 'micro'

    Returns:
        (precision, recall)
    """
    if predictions.dim() > 1:
        predictions = torch.argmax(predictions, dim=1)

    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    precision = precision_score(
        targets_np, preds_np, average=average, zero_division=0
    )
    recall = recall_score(
        targets_np, preds_np, average=average, zero_division=0
    )

    return precision, recall


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 7
) -> np.ndarray:
    """
    Calcule la matrice de confusion.

    Args:
        predictions: Prédictions
        targets: Labels vrais
        num_classes: Nombre de classes

    Returns:
        Confusion matrix (num_classes x num_classes)
    """
    if predictions.dim() > 1:
        predictions = torch.argmax(predictions, dim=1)

    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    cm = confusion_matrix(
        targets_np,
        preds_np,
        labels=list(range(num_classes))
    )

    return cm


class MetricsCalculator:
    """
    Calculateur de métriques pour l'entraînement.

    Accumule les prédictions et targets pendant une epoch,
    puis calcule toutes les métriques à la fin.
    """

    def __init__(self, num_classes: int = 7):
        """
        Args:
            num_classes: Nombre de classes
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset les accumulateurs."""
        self.all_predictions = []
        self.all_targets = []
        self.all_losses = []

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: Optional[float] = None
    ):
        """
        Ajoute un batch aux accumulateurs.

        Args:
            predictions: Prédictions du modèle
            targets: Labels vrais
            loss: Loss du batch (optionnel)
        """
        # Détacher et déplacer sur CPU
        if predictions.dim() > 1:
            preds = torch.argmax(predictions, dim=1).detach().cpu()
        else:
            preds = predictions.detach().cpu()

        targets = targets.detach().cpu()

        self.all_predictions.append(preds)
        self.all_targets.append(targets)

        if loss is not None:
            self.all_losses.append(loss)

    def compute(self) -> Dict[str, float]:
        """
        Calcule toutes les métriques accumulées.

        Returns:
            Dictionnaire de métriques
        """
        if len(self.all_predictions) == 0:
            return {}

        # Concatener tous les batches
        predictions = torch.cat(self.all_predictions)
        targets = torch.cat(self.all_targets)

        # Calculer métriques
        metrics = {}

        # Loss moyenne
        if len(self.all_losses) > 0:
            metrics['loss'] = np.mean(self.all_losses)

        # Accuracy
        metrics['accuracy'] = compute_accuracy(predictions, targets)

        # F1-score
        metrics['f1_macro'] = compute_f1_score(predictions, targets, average='macro')
        metrics['f1_weighted'] = compute_f1_score(predictions, targets, average='weighted')

        # Precision & Recall
        precision, recall = compute_precision_recall(
            predictions, targets, average='macro'
        )
        metrics['precision'] = precision
        metrics['recall'] = recall

        # Per-class accuracy
        per_class_acc = self._compute_per_class_accuracy(predictions, targets)
        for emotion_idx, acc in enumerate(per_class_acc):
            emotion_name = EMOTION_LABELS[emotion_idx]
            metrics[f'acc_{emotion_name}'] = acc

        return metrics

    def _compute_per_class_accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> List[float]:
        """Calcule l'accuracy par classe."""
        accuracies = []

        for class_idx in range(self.num_classes):
            # Masque pour cette classe
            mask = targets == class_idx

            if mask.sum() == 0:
                # Pas d'exemples pour cette classe
                accuracies.append(0.0)
                continue

            # Accuracy pour cette classe
            class_preds = predictions[mask]
            class_targets = targets[mask]

            acc = (class_preds == class_targets).float().mean().item()
            accuracies.append(acc)

        return accuracies

    def get_confusion_matrix(self) -> np.ndarray:
        """Retourne la matrice de confusion."""
        predictions = torch.cat(self.all_predictions)
        targets = torch.cat(self.all_targets)

        return compute_confusion_matrix(predictions, targets, self.num_classes)

    def get_classification_report(self) -> str:
        """Retourne un rapport de classification détaillé."""
        predictions = torch.cat(self.all_predictions)
        targets = torch.cat(self.all_targets)

        preds_np = predictions.numpy()
        targets_np = targets.numpy()

        report = classification_report(
            targets_np,
            preds_np,
            target_names=EMOTION_LABELS,
            zero_division=0
        )

        return report


class AverageMeter:
    """
    Compteur de moyenne mobile.

    Utile pour suivre loss, accuracy, etc. pendant l'entraînement.
    """

    def __init__(self, name: str = ""):
        """
        Args:
            name: Nom de la métrique
        """
        self.name = name
        self.reset()

    def reset(self):
        """Reset le compteur."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Met à jour le compteur.

        Args:
            val: Valeur à ajouter
            n: Nombre d'éléments (pour moyenne pondérée)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f} (current: {self.val:.4f})"


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Affiche les métriques de manière formatée.

    Args:
        metrics: Dictionnaire de métriques
        prefix: Préfixe (ex: "Train" ou "Val")
    """
    print(f"\n{prefix} Metrics:")
    print("-" * 50)

    # Metrics principales
    main_metrics = ['loss', 'accuracy', 'f1_macro', 'f1_weighted', 'precision', 'recall']

    for key in main_metrics:
        if key in metrics:
            print(f"  {key:15s}: {metrics[key]:.4f}")

    # Per-class accuracies
    print(f"\n  Per-class Accuracy:")
    for emotion in EMOTION_LABELS:
        key = f'acc_{emotion}'
        if key in metrics:
            print(f"    {emotion:10s}: {metrics[key]:.4f}")

    print("-" * 50 + "\n")


if __name__ == '__main__':
    # Test des métriques
    print("Testing metrics...")

    # Simuler prédictions et targets
    torch.manual_seed(42)

    predictions = torch.rand(100, 7)  # 100 samples, 7 classes
    targets = torch.randint(0, 7, (100,))

    # Test compute_accuracy
    acc = compute_accuracy(predictions, targets)
    print(f"Accuracy: {acc:.4f}")

    # Test compute_f1_score
    f1 = compute_f1_score(predictions, targets)
    print(f"F1-score: {f1:.4f}")

    # Test MetricsCalculator
    calculator = MetricsCalculator()
    calculator.update(predictions, targets, loss=0.5)

    metrics = calculator.compute()
    print_metrics(metrics, prefix="Test")

    # Test confusion matrix
    cm = calculator.get_confusion_matrix()
    print("Confusion Matrix:")
    print(cm)

    # Classification report
    print("\nClassification Report:")
    print(calculator.get_classification_report())
