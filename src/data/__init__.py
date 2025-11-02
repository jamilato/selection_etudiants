"""
Module de gestion des données pour la reconnaissance d'émotions faciales.

Ce module contient:
- Datasets personnalisés pour FER2013 et RAF-DB
- Transformations et augmentations de données
- Création de DataLoaders optimisés
"""

from .datasets import FER2013Dataset, RAFDBDataset, EmotionDataset
from .transforms import get_train_transforms, get_val_transforms, get_test_transforms
from .loaders import create_dataloaders, create_train_val_loaders

__all__ = [
    'FER2013Dataset',
    'RAFDBDataset',
    'EmotionDataset',
    'get_train_transforms',
    'get_val_transforms',
    'get_test_transforms',
    'create_dataloaders',
    'create_train_val_loaders',
]

__version__ = '1.0.0'
