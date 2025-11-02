"""
Module de prétraitement des visages pour le modèle de classification d'émotions.

Ce module prend les visages détectés et les prépare pour l'inférence:
- Redimensionnement à la taille du modèle (48x48 par défaut)
- Conversion en niveaux de gris si nécessaire
- Normalisation des pixels
- Conversion en tenseur PyTorch

Author: Projet IA Identification Étudiants
Date: 2025-11-02
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Union, Optional, Dict
import logging


class FacePreprocessor:
    """
    Préprocesseur de visages pour la classification d'émotions.

    Attributes:
        target_size (Tuple[int, int]): Taille cible (height, width)
        grayscale (bool): Convertir en niveaux de gris
        normalize (bool): Normaliser les pixels à [0, 1]
        mean (List[float]): Moyenne pour normalisation
        std (List[float]): Écart-type pour normalisation
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (48, 48),
        grayscale: bool = True,
        normalize: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        device: str = "cuda"
    ):
        """
        Initialise le préprocesseur.

        Args:
            target_size: Taille de sortie (height, width)
            grayscale: Si True, convertir en niveaux de gris
            normalize: Si True, normaliser les pixels
            mean: Moyenne pour normalisation (None = ImageNet ou [0.5])
            std: Écart-type pour normalisation (None = ImageNet ou [0.5])
            device: Device PyTorch ('cuda' ou 'cpu')
        """
        self.target_size = target_size
        self.grayscale = grayscale
        self.normalize = normalize
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Valeurs de normalisation par défaut
        if self.grayscale:
            # Pour grayscale: moyenne et std simples
            self.mean = mean if mean is not None else [0.5]
            self.std = std if std is not None else [0.5]
        else:
            # Pour RGB: utiliser ImageNet stats
            self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
            self.std = std if std is not None else [0.229, 0.224, 0.225]

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"FacePreprocessor initialized: size={target_size}, "
            f"grayscale={grayscale}, normalize={normalize}"
        )

    def preprocess_single(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Prétraite une seule image de visage.

        Args:
            face_image: Image BGR (format OpenCV)

        Returns:
            Tenseur PyTorch de forme [1, C, H, W]
        """
        # Convertir en grayscale si nécessaire
        if self.grayscale:
            if len(face_image.shape) == 3:
                face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_image
            processed = face_gray
        else:
            # Convertir BGR -> RGB pour modèles RGB
            if len(face_image.shape) == 3:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            processed = face_rgb

        # Redimensionner
        processed = cv2.resize(
            processed,
            (self.target_size[1], self.target_size[0]),  # width, height pour cv2
            interpolation=cv2.INTER_AREA
        )

        # Convertir en float et normaliser à [0, 1]
        processed = processed.astype(np.float32) / 255.0

        # Ajouter dimension de canal si grayscale
        if self.grayscale and len(processed.shape) == 2:
            processed = np.expand_dims(processed, axis=0)  # [H, W] -> [1, H, W]
        else:
            # Pour RGB: [H, W, C] -> [C, H, W]
            processed = np.transpose(processed, (2, 0, 1))

        # Convertir en tenseur
        tensor = torch.from_numpy(processed).float()

        # Normalisation avec mean/std
        if self.normalize:
            mean_tensor = torch.tensor(self.mean).view(-1, 1, 1)
            std_tensor = torch.tensor(self.std).view(-1, 1, 1)
            tensor = (tensor - mean_tensor) / std_tensor

        # Ajouter dimension batch: [C, H, W] -> [1, C, H, W]
        tensor = tensor.unsqueeze(0)

        return tensor

    def preprocess_batch(self, face_images: List[np.ndarray]) -> torch.Tensor:
        """
        Prétraite un batch d'images de visages.

        Args:
            face_images: Liste d'images BGR (format OpenCV)

        Returns:
            Tenseur PyTorch de forme [B, C, H, W]
        """
        if not face_images:
            # Retourner un tenseur vide avec la bonne forme
            channels = 1 if self.grayscale else 3
            return torch.empty(
                (0, channels, self.target_size[0], self.target_size[1]),
                dtype=torch.float32
            )

        # Prétraiter chaque image
        tensors = [self.preprocess_single(img) for img in face_images]

        # Concaténer en un batch
        batch_tensor = torch.cat(tensors, dim=0)

        return batch_tensor

    def preprocess_with_detection(
        self,
        image: np.ndarray,
        detections: List[Dict[str, any]],
        margin: float = 0.2
    ) -> Tuple[torch.Tensor, List[Dict[str, any]]]:
        """
        Prétraite les visages à partir des détections sur l'image complète.

        Args:
            image: Image source BGR complète
            detections: Liste de détections (format de FaceDetector.detect())
            margin: Marge autour des bboxes (0.2 = 20%)

        Returns:
            Tuple de:
                - Tenseur batch [B, C, H, W]
                - Liste des détections correspondantes
        """
        h_img, w_img = image.shape[:2]
        face_crops = []
        valid_detections = []

        for detection in detections:
            x, y, w, h = detection['bbox']

            # Ajouter une marge
            margin_w = int(w * margin)
            margin_h = int(h * margin)

            x1 = max(0, x - margin_w)
            y1 = max(0, y - margin_h)
            x2 = min(w_img, x + w + margin_w)
            y2 = min(h_img, y + h + margin_h)

            # Extraire et ajouter si valide
            if x2 > x1 and y2 > y1:
                face_crop = image[y1:y2, x1:x2]
                face_crops.append(face_crop)
                valid_detections.append(detection)

        # Prétraiter le batch
        batch_tensor = self.preprocess_batch(face_crops)

        return batch_tensor, valid_detections

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Dénormalise un tenseur pour visualisation.

        Args:
            tensor: Tenseur normalisé [B, C, H, W] ou [C, H, W]

        Returns:
            Tenseur dénormalisé avec pixels dans [0, 255]
        """
        # Clone pour ne pas modifier l'original
        denorm = tensor.clone()

        if self.normalize:
            # Inverser la normalisation
            mean_tensor = torch.tensor(self.mean).view(-1, 1, 1)
            std_tensor = torch.tensor(self.std).view(-1, 1, 1)

            if denorm.dim() == 4:  # Batch
                mean_tensor = mean_tensor.unsqueeze(0)
                std_tensor = std_tensor.unsqueeze(0)

            denorm = denorm * std_tensor + mean_tensor

        # Clipper à [0, 1] et convertir à [0, 255]
        denorm = torch.clamp(denorm, 0, 1) * 255.0

        return denorm

    def tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convertit un tenseur en image OpenCV (BGR).

        Args:
            tensor: Tenseur [C, H, W] ou [1, C, H, W]

        Returns:
            Image numpy BGR uint8
        """
        # Supprimer dimension batch si présente
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # Dénormaliser
        denorm = self.denormalize(tensor)

        # Convertir en numpy
        img_np = denorm.cpu().numpy()

        # [C, H, W] -> [H, W, C]
        if img_np.shape[0] in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))

        # Grayscale: [H, W, 1] -> [H, W]
        if img_np.shape[2] == 1:
            img_np = img_np.squeeze(2)
            # Convertir grayscale -> BGR pour affichage
            img_np = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            # RGB -> BGR
            img_np = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)

        return img_np

    def augment_face(
        self,
        face_image: np.ndarray,
        rotation_range: int = 10,
        brightness_range: float = 0.2,
        flip_horizontal: bool = False
    ) -> np.ndarray:
        """
        Applique des augmentations légères pour améliorer la robustesse.
        Utile pour le test-time augmentation (TTA).

        Args:
            face_image: Image de visage
            rotation_range: Rotation max en degrés
            brightness_range: Variation de luminosité (0.0-1.0)
            flip_horizontal: Autoriser le flip horizontal

        Returns:
            Image augmentée
        """
        augmented = face_image.copy()
        h, w = augmented.shape[:2]

        # Rotation aléatoire
        if rotation_range > 0:
            angle = np.random.uniform(-rotation_range, rotation_range)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(augmented, M, (w, h))

        # Ajustement de luminosité
        if brightness_range > 0:
            factor = np.random.uniform(1 - brightness_range, 1 + brightness_range)
            augmented = np.clip(augmented * factor, 0, 255).astype(np.uint8)

        # Flip horizontal
        if flip_horizontal and np.random.random() > 0.5:
            augmented = cv2.flip(augmented, 1)

        return augmented

    def __repr__(self) -> str:
        return (
            f"FacePreprocessor(target_size={self.target_size}, "
            f"grayscale={self.grayscale}, normalize={self.normalize})"
        )


def create_face_preprocessor(config: Dict) -> FacePreprocessor:
    """
    Factory function pour créer un FacePreprocessor depuis la configuration.

    Args:
        config: Dictionnaire de configuration (chargé depuis config.yaml)

    Returns:
        Instance de FacePreprocessor configurée

    Example:
        >>> from src.utils.config import load_config
        >>> config = load_config('configs/config.yaml')
        >>> preprocessor = create_face_preprocessor(config)
    """
    model_config = config.get('emotion_model', {})
    device_config = config.get('device', {})

    # Récupérer la taille d'entrée
    input_size = model_config.get('input_size', [48, 48])
    if isinstance(input_size, list):
        target_size = tuple(input_size)
    else:
        target_size = (input_size, input_size)

    return FacePreprocessor(
        target_size=target_size,
        grayscale=model_config.get('grayscale', True),
        normalize=True,
        device=device_config.get('type', 'cuda')
    )


if __name__ == "__main__":
    """Test du FacePreprocessor."""
    import sys

    # Configuration de logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test avec image factice
    print("Testing FacePreprocessor...")

    # Créer une image de test (100x100 RGB)
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Test grayscale preprocessor
    preprocessor_gray = FacePreprocessor(target_size=(48, 48), grayscale=True)
    tensor_gray = preprocessor_gray.preprocess_single(test_image)
    print(f"Grayscale output shape: {tensor_gray.shape}")  # [1, 1, 48, 48]

    # Test RGB preprocessor
    preprocessor_rgb = FacePreprocessor(target_size=(48, 48), grayscale=False)
    tensor_rgb = preprocessor_rgb.preprocess_single(test_image)
    print(f"RGB output shape: {tensor_rgb.shape}")  # [1, 3, 48, 48]

    # Test batch processing
    test_batch = [test_image, test_image, test_image]
    batch_tensor = preprocessor_gray.preprocess_batch(test_batch)
    print(f"Batch output shape: {batch_tensor.shape}")  # [3, 1, 48, 48]

    print("\n✓ FacePreprocessor module loaded successfully!")
    print(f"Grayscale: {preprocessor_gray}")
    print(f"RGB: {preprocessor_rgb}")
