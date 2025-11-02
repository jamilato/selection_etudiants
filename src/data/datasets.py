"""
Datasets personnalisés pour la reconnaissance d'émotions faciales.

Implémente les Dataset PyTorch pour:
- FER2013
- RAF-DB
- Dataset générique d'émotions

Bonnes pratiques 2025:
- Héritage de torch.utils.data.Dataset
- Implémentation __len__ et __getitem__
- Cache optionnel pour accélérer le chargement
- Support multi-format (CSV, dossiers, etc.)
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import Optional, Callable, Tuple, List
from pathlib import Path

import torch
from torch.utils.data import Dataset


# Mapping des émotions (standard FER2013)
EMOTION_LABELS = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

EMOTION_TO_IDX = {v: k for k, v in EMOTION_LABELS.items()}


class FER2013Dataset(Dataset):
    """
    Dataset pour FER2013.

    Structure attendue:
    ```
    data/fer2013/
        train/
            angry/
                img1.jpg
                img2.jpg
            happy/
                ...
        val/
            ...
        test/
            ...
    ```

    Ou format CSV:
    ```
    emotion,pixels
    0,128 129 130 ...
    3,50 51 52 ...
    ```

    Attributes:
        root_dir: Chemin racine du dataset
        split: 'train', 'val', ou 'test'
        transform: Transformations à appliquer
        from_csv: Si True, charge depuis CSV
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        from_csv: bool = False,
        csv_path: Optional[str] = None,
        cache: bool = False
    ):
        """
        Args:
            root_dir: Chemin vers data/fer2013/
            split: 'train', 'val', ou 'test'
            transform: Transformations PyTorch
            from_csv: Si True, charge depuis CSV
            csv_path: Chemin vers le CSV (si from_csv=True)
            cache: Si True, met en cache les images (consomme RAM)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.from_csv = from_csv
        self.cache = cache

        self.samples = []  # Liste de (image_path, label)
        self._cache_data = {}  # Cache des images

        if from_csv:
            self._load_from_csv(csv_path or self.root_dir / f"{split}.csv")
        else:
            self._load_from_folders()

    def _load_from_folders(self):
        """Charge les images depuis la structure de dossiers."""
        split_dir = self.root_dir / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"Directory not found: {split_dir}")

        # Parcourir chaque dossier d'émotion
        for emotion_name, emotion_idx in EMOTION_TO_IDX.items():
            emotion_dir = split_dir / emotion_name

            if not emotion_dir.exists():
                print(f"Warning: {emotion_dir} not found, skipping...")
                continue

            # Lister toutes les images
            for img_path in emotion_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), emotion_idx))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {split_dir}")

        print(f"Loaded {len(self.samples)} images for {self.split} split")

    def _load_from_csv(self, csv_path: Path):
        """Charge depuis format CSV FER2013."""
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Le CSV FER2013 a format: emotion,pixels
        # pixels = string "0 1 2 ... 2303" (48x48 = 2304 pixels)
        for idx, row in df.iterrows():
            emotion = int(row['emotion'])
            pixels = np.array(row['pixels'].split(), dtype=np.uint8)
            pixels = pixels.reshape(48, 48)  # 48x48 image

            # Stocker (pixels_array, label)
            self.samples.append((pixels, emotion))

        print(f"Loaded {len(self.samples)} samples from CSV for {self.split}")

    def __len__(self) -> int:
        """Retourne le nombre d'échantillons."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Récupère un échantillon.

        Args:
            idx: Index de l'échantillon

        Returns:
            (image_tensor, label)
        """
        # Check cache
        if self.cache and idx in self._cache_data:
            image, label = self._cache_data[idx]
        else:
            if self.from_csv:
                # pixels est déjà un numpy array
                pixels, label = self.samples[idx]
                image = Image.fromarray(pixels, mode='L')  # Grayscale
            else:
                # Charger depuis fichier
                img_path, label = self.samples[idx]
                image = Image.open(img_path).convert('RGB')

            # Cache si activé
            if self.cache:
                self._cache_data[idx] = (image, label)

        # Appliquer transformations
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> dict:
        """Retourne la distribution des classes."""
        distribution = {i: 0 for i in range(7)}

        for _, label in self.samples:
            distribution[label] += 1

        return distribution

    def get_class_weights(self) -> torch.Tensor:
        """
        Calcule les poids de classes pour gérer le déséquilibre.

        Utile pour WeightedRandomSampler ou loss function.

        Returns:
            Tensor de poids par classe
        """
        distribution = self.get_class_distribution()
        total = sum(distribution.values())

        # Poids inversement proportionnel à la fréquence
        weights = torch.tensor([
            total / (distribution[i] * 7) if distribution[i] > 0 else 0
            for i in range(7)
        ], dtype=torch.float32)

        return weights


class RAFDBDataset(Dataset):
    """
    Dataset pour RAF-DB (Real-world Affective Faces Database).

    Structure attendue:
    ```
    data/rafdb/
        train/
            angry/
            happy/
            ...
        test/
            ...
    ```

    RAF-DB a les mêmes 7 émotions que FER2013.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        cache: bool = False
    ):
        """
        Args:
            root_dir: Chemin vers data/rafdb/
            split: 'train' ou 'test'
            transform: Transformations
            cache: Cache des images
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.cache = cache

        self.samples = []
        self._cache_data = {}

        self._load_dataset()

    def _load_dataset(self):
        """Charge le dataset depuis la structure de dossiers."""
        split_dir = self.root_dir / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"Directory not found: {split_dir}")

        # Parcourir les dossiers d'émotions
        for emotion_name, emotion_idx in EMOTION_TO_IDX.items():
            emotion_dir = split_dir / emotion_name

            if not emotion_dir.exists():
                print(f"Warning: {emotion_dir} not found, skipping...")
                continue

            for img_path in emotion_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), emotion_idx))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {split_dir}")

        print(f"RAF-DB {self.split}: Loaded {len(self.samples)} images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Check cache
        if self.cache and idx in self._cache_data:
            image, label = self._cache_data[idx]
        else:
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')

            if self.cache:
                self._cache_data[idx] = (image, label)

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> dict:
        distribution = {i: 0 for i in range(7)}
        for _, label in self.samples:
            distribution[label] += 1
        return distribution


class EmotionDataset(Dataset):
    """
    Dataset générique pour n'importe quel dataset d'émotions.

    Utilisation:
    - Studients faces
    - Custom dataset
    - Etc.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None,
        emotion_mapping: Optional[dict] = None
    ):
        """
        Args:
            image_paths: Liste des chemins d'images
            labels: Liste des labels correspondants
            transform: Transformations
            emotion_mapping: Mapping custom des émotions
        """
        assert len(image_paths) == len(labels), \
            "Number of images and labels must match"

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.emotion_mapping = emotion_mapping or EMOTION_LABELS

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    @classmethod
    def from_folder(
        cls,
        root_dir: str,
        transform: Optional[Callable] = None
    ) -> 'EmotionDataset':
        """
        Crée un dataset depuis une structure de dossiers.

        Structure:
        ```
        root_dir/
            emotion1/
                img1.jpg
            emotion2/
                img2.jpg
        ```
        """
        root_path = Path(root_dir)
        image_paths = []
        labels = []

        for emotion_name, emotion_idx in EMOTION_TO_IDX.items():
            emotion_dir = root_path / emotion_name

            if not emotion_dir.exists():
                continue

            for img_path in emotion_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_paths.append(str(img_path))
                    labels.append(emotion_idx)

        return cls(image_paths, labels, transform)


def create_subset_dataset(
    dataset: Dataset,
    indices: List[int]
) -> Dataset:
    """
    Crée un sous-ensemble d'un dataset.

    Utile pour créer train/val splits.

    Args:
        dataset: Dataset source
        indices: Indices à inclure

    Returns:
        Subset du dataset
    """
    from torch.utils.data import Subset
    return Subset(dataset, indices)


def print_dataset_info(dataset: Dataset, name: str = "Dataset"):
    """
    Affiche des informations sur le dataset.

    Args:
        dataset: Dataset à analyser
        name: Nom du dataset
    """
    print(f"\n{'='*60}")
    print(f"{name} Information")
    print(f"{'='*60}")
    print(f"Total samples: {len(dataset)}")

    if hasattr(dataset, 'get_class_distribution'):
        dist = dataset.get_class_distribution()
        print(f"\nClass distribution:")
        for emotion_idx, count in dist.items():
            emotion_name = EMOTION_LABELS[emotion_idx]
            percentage = 100 * count / len(dataset)
            print(f"  {emotion_name:10s}: {count:5d} ({percentage:5.2f}%)")

    print(f"{'='*60}\n")
