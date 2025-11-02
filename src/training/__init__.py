"""
Module d'entraînement pour les modèles de reconnaissance d'émotions.

Ce module contient:
- Trainer avec support Mixed Precision (AMD ROCm)
- Métriques (accuracy, F1-score, confusion matrix)
- Callbacks (early stopping, checkpointing, LR scheduling)
"""

from .metrics import MetricsCalculator, compute_accuracy, compute_f1_score
from .callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback
from .trainer import EmotionTrainer

__all__ = [
    'MetricsCalculator',
    'compute_accuracy',
    'compute_f1_score',
    'EarlyStopping',
    'ModelCheckpoint',
    'LRSchedulerCallback',
    'EmotionTrainer',
]

__version__ = '1.0.0'
