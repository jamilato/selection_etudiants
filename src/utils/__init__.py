"""
Module utilitaire pour le projet de reconnaissance d'émotions.

Contient:
- Configuration (chargement YAML)
- Visualisation (plots, confusion matrix)
- Helpers divers
"""

from .config import load_config, load_all_configs, save_config
from .visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_class_distribution,
    visualize_predictions,
    plot_learning_curves
)

__all__ = [
    'load_config',
    'load_all_configs',
    'save_config',
    'plot_training_history',
    'plot_confusion_matrix',
    'plot_class_distribution',
    'visualize_predictions',
    'plot_learning_curves',
]

__version__ = '1.0.0'
