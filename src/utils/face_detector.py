"""
Module de détection de visages pour le système d'identification d'étudiants.

Supporte deux méthodes de détection:
- MTCNN (Multi-task Cascaded Convolutional Networks) - Méthode principale, précise
- Haar Cascade - Fallback rapide, moins précis

Author: Projet IA Identification Étudiants
Date: 2025-11-02
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import logging

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    logging.warning("facenet_pytorch not available. MTCNN detection disabled.")


class FaceDetector:
    """
    Détecteur de visages avec support MTCNN et Haar Cascade.

    Attributes:
        method (str): Méthode de détection ('mtcnn' ou 'haar_cascade')
        min_face_size (int): Taille minimale du visage en pixels
        confidence_threshold (float): Seuil de confiance pour MTCNN (0.0-1.0)
        device (torch.device): Device pour MTCNN (CPU ou CUDA)
    """

    def __init__(
        self,
        method: str = "mtcnn",
        min_face_size: int = 40,
        confidence_threshold: float = 0.9,
        haar_cascade_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialise le détecteur de visages.

        Args:
            method: Méthode de détection ('mtcnn' ou 'haar_cascade')
            min_face_size: Taille minimale du visage détecté
            confidence_threshold: Seuil de confiance MTCNN (0.0-1.0)
            haar_cascade_path: Chemin vers le fichier XML Haar Cascade
            device: Device PyTorch ('cuda' ou 'cpu')
        """
        self.method = method.lower()
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.logger = logging.getLogger(__name__)

        # Initialiser le détecteur selon la méthode
        if self.method == "mtcnn":
            if not MTCNN_AVAILABLE:
                self.logger.warning("MTCNN not available, falling back to Haar Cascade")
                self.method = "haar_cascade"
            else:
                self._init_mtcnn()

        if self.method == "haar_cascade":
            self._init_haar_cascade(haar_cascade_path)

        self.logger.info(f"FaceDetector initialized with method: {self.method}")

    def _init_mtcnn(self):
        """Initialise le détecteur MTCNN."""
        try:
            self.mtcnn = MTCNN(
                image_size=160,
                min_face_size=self.min_face_size,
                thresholds=[0.6, 0.7, self.confidence_threshold],
                post_process=False,
                device=self.device,
                keep_all=True,  # Détecter tous les visages
                selection_method=None
            )
            self.logger.info(f"MTCNN initialized on device: {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to initialize MTCNN: {e}")
            self.logger.warning("Falling back to Haar Cascade")
            self.method = "haar_cascade"

    def _init_haar_cascade(self, cascade_path: Optional[str] = None):
        """
        Initialise le détecteur Haar Cascade.

        Args:
            cascade_path: Chemin vers le fichier XML Haar Cascade
        """
        if cascade_path and Path(cascade_path).exists():
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            # Utiliser le cascade par défaut d'OpenCV
            cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.haar_cascade = cv2.CascadeClassifier(cascade_file)

        if self.haar_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")

        self.logger.info("Haar Cascade initialized")

    def detect(self, image: np.ndarray) -> List[Dict[str, any]]:
        """
        Détecte les visages dans une image.

        Args:
            image: Image BGR (format OpenCV)

        Returns:
            Liste de dictionnaires contenant:
                - 'bbox': [x, y, width, height] en pixels
                - 'confidence': Score de confiance (0.0-1.0)
                - 'landmarks': Points caractéristiques du visage (5 points pour MTCNN)
        """
        if self.method == "mtcnn":
            return self._detect_mtcnn(image)
        else:
            return self._detect_haar(image)

    def _detect_mtcnn(self, image: np.ndarray) -> List[Dict[str, any]]:
        """
        Détecte les visages avec MTCNN.

        Args:
            image: Image BGR (OpenCV format)

        Returns:
            Liste de détections avec bbox, confidence, landmarks
        """
        # Convertir BGR -> RGB pour MTCNN
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            # Détecter les visages
            boxes, probs, landmarks = self.mtcnn.detect(image_rgb, landmarks=True)

            detections = []

            if boxes is not None:
                for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                    # Filtrer par confiance
                    if prob < self.confidence_threshold:
                        continue

                    # Convertir box format [x1, y1, x2, y2] -> [x, y, w, h]
                    x1, y1, x2, y2 = box
                    x, y = int(x1), int(y1)
                    w, h = int(x2 - x1), int(y2 - y1)

                    # Filtrer par taille minimale
                    if w < self.min_face_size or h < self.min_face_size:
                        continue

                    # Créer la détection
                    detection = {
                        'bbox': [x, y, w, h],
                        'confidence': float(prob),
                        'landmarks': landmark.tolist() if landmark is not None else None
                    }

                    detections.append(detection)

            return detections

        except Exception as e:
            self.logger.error(f"MTCNN detection failed: {e}")
            return []

    def _detect_haar(self, image: np.ndarray) -> List[Dict[str, any]]:
        """
        Détecte les visages avec Haar Cascade.

        Args:
            image: Image BGR (OpenCV format)

        Returns:
            Liste de détections avec bbox et confidence estimée
        """
        # Convertir en niveaux de gris pour Haar Cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Égalisation d'histogramme pour améliorer la détection
        gray = cv2.equalizeHist(gray)

        # Détecter les visages
        faces = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        detections = []

        for (x, y, w, h) in faces:
            # Haar Cascade ne fournit pas de score de confiance
            # On utilise une confiance fixe de 0.95
            detection = {
                'bbox': [int(x), int(y), int(w), int(h)],
                'confidence': 0.95,
                'landmarks': None  # Pas de landmarks avec Haar
            }
            detections.append(detection)

        return detections

    def detect_largest(self, image: np.ndarray) -> Optional[Dict[str, any]]:
        """
        Détecte uniquement le visage le plus grand dans l'image.
        Utile pour le mode single-person ou pour sélectionner le visage principal.

        Args:
            image: Image BGR (OpenCV format)

        Returns:
            Détection du plus grand visage, ou None si aucun visage détecté
        """
        detections = self.detect(image)

        if not detections:
            return None

        # Trouver le visage avec la plus grande bbox
        largest = max(detections, key=lambda d: d['bbox'][2] * d['bbox'][3])
        return largest

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, any]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_confidence: bool = True,
        show_landmarks: bool = False
    ) -> np.ndarray:
        """
        Dessine les détections sur l'image.

        Args:
            image: Image BGR à annoter
            detections: Liste de détections de detect()
            color: Couleur BGR de la boîte
            thickness: Épaisseur de la boîte
            show_confidence: Afficher le score de confiance
            show_landmarks: Dessiner les landmarks (si disponibles)

        Returns:
            Image annotée
        """
        result = image.copy()

        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            landmarks = detection.get('landmarks')

            # Dessiner la boîte englobante
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)

            # Afficher la confiance
            if show_confidence:
                text = f"{confidence:.2f}"
                cv2.putText(
                    result,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )

            # Dessiner les landmarks (yeux, nez, bouche)
            if show_landmarks and landmarks is not None:
                for (lx, ly) in landmarks:
                    cv2.circle(result, (int(lx), int(ly)), 2, (0, 0, 255), -1)

        return result

    def get_face_crops(
        self,
        image: np.ndarray,
        detections: List[Dict[str, any]],
        margin: float = 0.2
    ) -> List[np.ndarray]:
        """
        Extrait les régions de visages détectés avec une marge.

        Args:
            image: Image source BGR
            detections: Liste de détections
            margin: Marge autour du visage (0.2 = 20% de chaque côté)

        Returns:
            Liste d'images de visages cropées
        """
        h_img, w_img = image.shape[:2]
        crops = []

        for detection in detections:
            x, y, w, h = detection['bbox']

            # Ajouter une marge
            margin_w = int(w * margin)
            margin_h = int(h * margin)

            x1 = max(0, x - margin_w)
            y1 = max(0, y - margin_h)
            x2 = min(w_img, x + w + margin_w)
            y2 = min(h_img, y + h + margin_h)

            # Extraire la région
            face_crop = image[y1:y2, x1:x2]
            crops.append(face_crop)

        return crops

    def __repr__(self) -> str:
        return (
            f"FaceDetector(method={self.method}, "
            f"min_face_size={self.min_face_size}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"device={self.device})"
        )


def create_face_detector(config: Dict) -> FaceDetector:
    """
    Factory function pour créer un FaceDetector depuis la configuration.

    Args:
        config: Dictionnaire de configuration (chargé depuis config.yaml)

    Returns:
        Instance de FaceDetector configurée

    Example:
        >>> from src.utils.config import load_config
        >>> config = load_config('configs/config.yaml')
        >>> detector = create_face_detector(config)
    """
    face_config = config.get('face_detection', {})
    device_config = config.get('device', {})

    return FaceDetector(
        method=face_config.get('method', 'mtcnn'),
        min_face_size=face_config.get('min_face_size', 40),
        confidence_threshold=face_config.get('confidence_threshold', 0.9),
        haar_cascade_path=face_config.get('haar_cascade_path'),
        device=device_config.get('type', 'cuda')
    )


if __name__ == "__main__":
    """Test du FaceDetector avec une image de test."""
    import sys

    # Configuration de logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test avec MTCNN
    print("Testing MTCNN FaceDetector...")
    detector_mtcnn = FaceDetector(method="mtcnn", device="cpu")

    # Test avec Haar Cascade
    print("\nTesting Haar Cascade FaceDetector...")
    detector_haar = FaceDetector(method="haar_cascade")

    print("\n✓ FaceDetector module loaded successfully!")
    print(f"MTCNN: {detector_mtcnn}")
    print(f"Haar: {detector_haar}")
