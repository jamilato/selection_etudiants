"""
Transformations et augmentations de données pour la reconnaissance d'émotions.

Basé sur les meilleures pratiques 2025 pour FER2013:
- Augmentation: flip, rotation, brightness, contrast
- Normalisation standard ImageNet
- Support images grayscale et RGB
"""

import torch
import torchvision.transforms as transforms
from typing import Tuple, Optional


def get_train_transforms(
    img_size: Tuple[int, int] = (48, 48),
    grayscale: bool = True,
    augment: bool = True
) -> transforms.Compose:
    """
    Transformations pour l'ensemble d'entraînement.

    Args:
        img_size: Taille de l'image de sortie (height, width)
        grayscale: Si True, convertit en niveaux de gris
        augment: Si True, applique l'augmentation de données

    Returns:
        Composition de transformations PyTorch

    Bonnes pratiques:
    - Horizontal flip (p=0.5) pour symétrie faciale
    - Rotation légère (±15°) pour variations de pose
    - ColorJitter pour variations d'éclairage
    - Normalisation cohérente
    """
    transform_list = []

    # Conversion en grayscale si nécessaire
    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    # Resize à la taille cible
    transform_list.append(transforms.Resize(img_size))

    # Augmentation de données (uniquement pour train)
    if augment:
        # Flip horizontal (50% chance)
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        # Rotation aléatoire (±15 degrés)
        transform_list.append(transforms.RandomRotation(degrees=15))

        # Variations de luminosité et contraste
        if not grayscale:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.2,  # ±20% luminosité
                    contrast=0.2,    # ±20% contraste
                    saturation=0.1,  # ±10% saturation
                    hue=0.05         # ±5% teinte
                )
            )
        else:
            # Pour grayscale, uniquement brightness et contrast
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2
                )
            )

        # Affine transformation légère
        transform_list.append(
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # ±10% translation
                scale=(0.9, 1.1),       # 90-110% scale
                shear=5                 # ±5° shear
            )
        )

        # Random erasing (simulate occlusions)
        # Appliqué après ToTensor

    # Conversion en Tensor
    transform_list.append(transforms.ToTensor())

    # Normalisation
    if grayscale:
        # Pour images grayscale (1 canal)
        transform_list.append(
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )
        )
    else:
        # Pour images RGB (3 canaux) - stats ImageNet
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )

    # Random erasing (après normalisation)
    if augment:
        transform_list.append(
            transforms.RandomErasing(
                p=0.1,           # 10% chance
                scale=(0.02, 0.1),  # 2-10% de l'image
                ratio=(0.3, 3.3)
            )
        )

    return transforms.Compose(transform_list)


def get_val_transforms(
    img_size: Tuple[int, int] = (48, 48),
    grayscale: bool = True
) -> transforms.Compose:
    """
    Transformations pour l'ensemble de validation.

    Args:
        img_size: Taille de l'image de sortie
        grayscale: Si True, convertit en niveaux de gris

    Returns:
        Composition de transformations PyTorch

    Note:
    - Pas d'augmentation pour validation/test
    - Seulement resize + normalisation
    """
    transform_list = []

    # Conversion en grayscale si nécessaire
    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    # Resize
    transform_list.append(transforms.Resize(img_size))

    # Conversion en Tensor
    transform_list.append(transforms.ToTensor())

    # Normalisation (mêmes stats que train)
    if grayscale:
        transform_list.append(
            transforms.Normalize(mean=[0.5], std=[0.5])
        )
    else:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )

    return transforms.Compose(transform_list)


def get_test_transforms(
    img_size: Tuple[int, int] = (48, 48),
    grayscale: bool = True
) -> transforms.Compose:
    """
    Transformations pour l'ensemble de test.

    Identique à get_val_transforms() pour cohérence.
    """
    return get_val_transforms(img_size, grayscale)


def get_tta_transforms(
    img_size: Tuple[int, int] = (48, 48),
    grayscale: bool = True,
    n_augmentations: int = 5
) -> list:
    """
    Test-Time Augmentation (TTA) transforms.

    Applique plusieurs transformations à la même image au test
    pour améliorer la robustesse des prédictions.

    Args:
        img_size: Taille de l'image
        grayscale: Si grayscale
        n_augmentations: Nombre de versions augmentées

    Returns:
        Liste de transformations pour TTA
    """
    base_transforms = []

    # Transform de base (sans augmentation)
    base_transforms.append(get_test_transforms(img_size, grayscale))

    # Ajout de variations
    for _ in range(n_augmentations - 1):
        transform_list = []

        if grayscale:
            transform_list.append(transforms.Grayscale(num_output_channels=1))

        transform_list.append(transforms.Resize(img_size))

        # Variations légères
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor()
        ])

        if grayscale:
            transform_list.append(
                transforms.Normalize(mean=[0.5], std=[0.5])
            )
        else:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )

        base_transforms.append(transforms.Compose(transform_list))

    return base_transforms


# Transformations pour des résolutions spécifiques
def get_transforms_for_model(
    model_name: str,
    split: str = 'train'
) -> transforms.Compose:
    """
    Récupère les transformations adaptées à un modèle spécifique.

    Args:
        model_name: Nom du modèle ('emotionnet', 'resnet', 'efficientnet', etc.)
        split: 'train', 'val', ou 'test'

    Returns:
        Transformations appropriées
    """
    # Résolutions par modèle
    resolutions = {
        'emotionnet': (48, 48),
        'emotionnet_nano': (48, 48),
        'resnet18': (224, 224),
        'resnet34': (224, 224),
        'resnet50': (224, 224),
        'efficientnet_b0': (224, 224),
        'efficientnet_b7': (600, 600),
        'vgg16': (224, 224),
    }

    # Grayscale par défaut sauf pour certains modèles
    grayscale_models = {'emotionnet', 'emotionnet_nano'}

    img_size = resolutions.get(model_name.lower(), (48, 48))
    grayscale = model_name.lower() in grayscale_models

    if split == 'train':
        return get_train_transforms(img_size, grayscale, augment=True)
    elif split in ['val', 'validation']:
        return get_val_transforms(img_size, grayscale)
    else:  # test
        return get_test_transforms(img_size, grayscale)


def denormalize(tensor: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    """
    Dénormalise un tensor pour visualisation.

    Args:
        tensor: Tensor normalisé (C, H, W)
        mean: Moyennes utilisées pour normalisation
        std: Écarts-types utilisés

    Returns:
        Tensor dénormalisé
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    return tensor * std + mean


# Constantes pour dénormalisation
GRAYSCALE_MEAN = [0.5]
GRAYSCALE_STD = [0.5]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
