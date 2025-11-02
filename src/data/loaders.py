"""
Création de DataLoaders optimisés pour l'entraînement.

Bonnes pratiques 2025:
- num_workers pour parallélisation
- pin_memory pour accélération GPU
- WeightedRandomSampler pour déséquilibre de classes
- Batch size adaptatif selon VRAM
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from typing import Tuple, Optional, Dict
import numpy as np

from .datasets import FER2013Dataset, RAFDBDataset, EmotionDataset, EMOTION_LABELS
from .transforms import get_train_transforms, get_val_transforms


def create_dataloaders(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_dataset: Optional[torch.utils.data.Dataset] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampler: bool = True
) -> Dict[str, DataLoader]:
    """
    Crée les DataLoaders pour train, val, et test.

    Args:
        train_dataset: Dataset d'entraînement
        val_dataset: Dataset de validation
        test_dataset: Dataset de test (optionnel)
        batch_size: Taille des batchs
        num_workers: Nombre de workers pour chargement parallèle
        pin_memory: Si True, pin memory pour accélération GPU
        use_weighted_sampler: Si True, utilise WeightedRandomSampler

    Returns:
        Dictionnaire {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}

    Bonnes pratiques:
    - num_workers=4-8 pour CPU avec 8+ cores
    - pin_memory=True si GPU disponible
    - WeightedRandomSampler pour gérer déséquilibre de classes
    - shuffle=True pour train, False pour val/test
    """
    dataloaders = {}

    # Sampler pour train (si weighted)
    train_sampler = None
    shuffle_train = True

    if use_weighted_sampler and hasattr(train_dataset, 'get_class_weights'):
        # Créer WeightedRandomSampler
        class_weights = train_dataset.get_class_weights()

        # Poids pour chaque échantillon
        sample_weights = []
        for _, label in train_dataset.samples:
            sample_weights.append(class_weights[label])

        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True  # Avec remplacement pour équilibrer
        )

        shuffle_train = False  # Pas de shuffle si sampler est utilisé

    # Train DataLoader
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop dernier batch incomplet (pour batch norm)
        persistent_workers=num_workers > 0  # Garde workers en vie
    )

    # Validation DataLoader
    dataloaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Pas de shuffle pour validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    # Test DataLoader (si fourni)
    if test_dataset is not None:
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )

    return dataloaders


def create_train_val_loaders(
    dataset_type: str = 'fer2013',
    data_dir: str = 'data',
    batch_size: int = 64,
    num_workers: int = 4,
    val_split: float = 0.15,
    img_size: Tuple[int, int] = (48, 48),
    grayscale: bool = True,
    use_weighted_sampler: bool = True,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Fonction helper pour créer train/val loaders rapidement.

    Args:
        dataset_type: 'fer2013', 'rafdb', ou 'custom'
        data_dir: Chemin racine des données
        batch_size: Taille des batchs
        num_workers: Nombre de workers
        val_split: Proportion de validation (si pas de val split existant)
        img_size: Taille des images
        grayscale: Si grayscale
        use_weighted_sampler: Utiliser weighted sampling
        pin_memory: Pin memory pour GPU

    Returns:
        (train_loader, val_loader)
    """
    # Transformations
    train_transform = get_train_transforms(img_size, grayscale, augment=True)
    val_transform = get_val_transforms(img_size, grayscale)

    # Créer datasets
    if dataset_type.lower() == 'fer2013':
        train_dataset = FER2013Dataset(
            root_dir=f"{data_dir}/fer2013",
            split='train',
            transform=train_transform
        )

        # Check si val split existe
        try:
            val_dataset = FER2013Dataset(
                root_dir=f"{data_dir}/fer2013",
                split='val',
                transform=val_transform
            )
        except (FileNotFoundError, ValueError):
            # Créer val split depuis train
            print(f"Creating validation split ({val_split:.0%}) from training data...")
            train_size = int((1 - val_split) * len(train_dataset))
            val_size = len(train_dataset) - train_size

            train_dataset, val_dataset = random_split(
                train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            # Appliquer val transform au val split
            # Note: avec random_split, les transforms sont déjà appliqués

    elif dataset_type.lower() == 'rafdb':
        train_dataset = RAFDBDataset(
            root_dir=f"{data_dir}/rafdb",
            split='train',
            transform=train_transform
        )

        try:
            val_dataset = RAFDBDataset(
                root_dir=f"{data_dir}/rafdb",
                split='test',  # RAF-DB utilise 'test' comme val
                transform=val_transform
            )
        except FileNotFoundError:
            # Split depuis train
            train_size = int((1 - val_split) * len(train_dataset))
            val_size = len(train_dataset) - train_size

            train_dataset, val_dataset = random_split(
                train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Créer DataLoaders
    loaders = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        use_weighted_sampler=use_weighted_sampler
    )

    return loaders['train'], loaders['val']


def get_optimal_num_workers() -> int:
    """
    Détermine le nombre optimal de workers selon le CPU.

    Returns:
        Nombre recommandé de workers
    """
    import os

    # Nombre de CPUs
    num_cpus = os.cpu_count() or 4

    # Règle empirique: num_workers = min(num_cpus - 1, 8)
    # Garder 1 CPU pour le main process
    # Max 8 pour éviter overhead
    optimal = min(num_cpus - 1, 8)

    return max(optimal, 0)  # Au moins 0


def get_optimal_batch_size(
    model: torch.nn.Module,
    img_size: Tuple[int, int],
    device: torch.device,
    max_vram_gb: float = 20.0
) -> int:
    """
    Estime le batch size optimal selon la VRAM disponible.

    Args:
        model: Modèle PyTorch
        img_size: Taille des images
        device: Device (cuda ou cpu)
        max_vram_gb: VRAM maximale disponible (GB)

    Returns:
        Batch size recommandé

    Note:
    Ceci est une estimation. Il faut tester pour trouver l'optimal.
    """
    if device.type == 'cpu':
        return 32  # Batch size raisonnable pour CPU

    # Estimation VRAM par image (très approximatif)
    # Dépend de l'architecture du modèle
    H, W = img_size
    channels = 1  # Grayscale par défaut

    # Mémoire par image (input + gradients + activations)
    # Approximation: ~10x la taille de l'input
    bytes_per_image = H * W * channels * 4 * 10  # 4 bytes par float32, factor 10

    # Mémoire disponible (GB -> bytes, garder 20% de marge)
    available_bytes = max_vram_gb * 1e9 * 0.8

    # Batch size estimé
    estimated_batch = int(available_bytes / bytes_per_image)

    # Limites raisonnables
    return min(max(estimated_batch, 16), 256)


def test_dataloader(
    dataloader: DataLoader,
    num_batches: int = 3
):
    """
    Teste un DataLoader et affiche des informations.

    Args:
        dataloader: DataLoader à tester
        num_batches: Nombre de batches à tester
    """
    print(f"\n{'='*60}")
    print("DataLoader Test")
    print(f"{'='*60}")
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Num batches: {len(dataloader)}")
    print(f"Num workers: {dataloader.num_workers}")
    print(f"Pin memory: {dataloader.pin_memory}")

    # Test quelques batches
    print(f"\nTesting {num_batches} batches...")

    for i, (images, labels) in enumerate(dataloader):
        if i >= num_batches:
            break

        print(f"\nBatch {i+1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Image dtype: {images.dtype}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Labels: {labels[:10].tolist()}")  # Premier 10

        # Distribution des labels dans ce batch
        unique, counts = torch.unique(labels, return_counts=True)
        print(f"  Label distribution in batch:")
        for label_idx, count in zip(unique.tolist(), counts.tolist()):
            emotion = EMOTION_LABELS[label_idx]
            print(f"    {emotion}: {count}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Test des DataLoaders
    print("Testing DataLoader creation...")

    try:
        train_loader, val_loader = create_train_val_loaders(
            dataset_type='fer2013',
            data_dir='data',
            batch_size=32,
            num_workers=0,  # 0 pour test rapide
            grayscale=True
        )

        print(f"\n✅ Train loader created: {len(train_loader)} batches")
        print(f"✅ Val loader created: {len(val_loader)} batches")

        # Test train loader
        test_dataloader(train_loader, num_batches=2)

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure FER2013 data is in data/fer2013/")
