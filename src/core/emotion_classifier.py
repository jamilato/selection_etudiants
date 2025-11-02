"""
Module de classification d'émotions en temps réel.

Ce module encapsule le modèle EmotionNet pour l'inférence optimisée:
- Chargement de modèles entraînés ou TorchScript
- Inférence batch avec optimisations GPU
- Support mixed precision (FP16)
- Gestion de modèles factices pour tests

Author: Projet IA Identification Étudiants
Date: 2025-11-02
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import logging

from src.models.emotion_net import EmotionNetNano, create_emotion_model


class EmotionClassifier:
    """
    Classificateur d'émotions en temps réel.

    Attributes:
        model: Modèle PyTorch pour la classification
        num_classes (int): Nombre de classes d'émotions
        class_names (List[str]): Noms des émotions
        device: Device PyTorch (CPU ou CUDA)
        use_fp16 (bool): Utiliser mixed precision FP16
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = 7,
        class_names: Optional[List[str]] = None,
        device: str = "cuda",
        use_fp16: bool = False,
        input_channels: int = 1
    ):
        """
        Initialise le classificateur d'émotions.

        Args:
            model_path: Chemin vers le checkpoint du modèle (.pt ou .pth)
            num_classes: Nombre de classes d'émotions
            class_names: Noms des émotions (ordre correspond aux indices)
            device: Device ('cuda' ou 'cpu')
            use_fp16: Utiliser FP16 pour accélération
            input_channels: Nombre de canaux d'entrée (1=grayscale, 3=RGB)
        """
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.input_channels = input_channels

        # Noms des émotions par défaut (FER2013/RAF-DB standard)
        if class_names is None:
            self.class_names = [
                "angry", "disgust", "fear", "happy",
                "sad", "surprise", "neutral"
            ]
        else:
            self.class_names = class_names

        self.logger = logging.getLogger(__name__)

        # Charger le modèle
        self.model = self._load_model(model_path)

        # Mettre en mode évaluation
        self.model.eval()

        # Déplacer sur le device
        self.model.to(self.device)

        # Convertir en FP16 si demandé
        if self.use_fp16:
            self.model.half()

        self.logger.info(
            f"EmotionClassifier initialized on {self.device} "
            f"(FP16={self.use_fp16})"
        )

    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """
        Charge le modèle depuis un checkpoint ou crée un modèle factice.

        Args:
            model_path: Chemin vers le checkpoint

        Returns:
            Modèle PyTorch
        """
        if model_path is None or not Path(model_path).exists():
            self.logger.warning(
                f"Model path '{model_path}' not found. "
                "Creating dummy model for testing."
            )
            return self._create_dummy_model()

        try:
            # Tenter de charger comme TorchScript
            if model_path.endswith('.torchscript') or model_path.endswith('.pt'):
                try:
                    model = torch.jit.load(model_path, map_location='cpu')
                    self.logger.info(f"Loaded TorchScript model from {model_path}")
                    return model
                except Exception as e:
                    self.logger.debug(f"Not a TorchScript model: {e}")

            # Charger checkpoint standard PyTorch
            checkpoint = torch.load(model_path, map_location='cpu')

            # Créer le modèle
            model = EmotionNetNano(
                num_classes=self.num_classes,
                input_channels=self.input_channels
            )

            # Charger les poids
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    self.logger.info(
                        f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})"
                    )
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)

            self.logger.info(f"Model loaded successfully from {model_path}")
            return model

        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            self.logger.warning("Creating dummy model for testing.")
            return self._create_dummy_model()

    def _create_dummy_model(self) -> nn.Module:
        """
        Crée un modèle factice pour les tests sans modèle entraîné.

        Returns:
            Modèle EmotionNetNano avec poids aléatoires
        """
        model = EmotionNetNano(
            num_classes=self.num_classes,
            input_channels=self.input_channels
        )
        self.logger.info("Created dummy EmotionNetNano model (random weights)")
        return model

    @torch.no_grad()
    def predict(
        self,
        face_tensor: torch.Tensor,
        return_probabilities: bool = True
    ) -> Union[List[int], Tuple[List[int], List[np.ndarray]]]:
        """
        Prédit les émotions pour un batch de visages.

        Args:
            face_tensor: Tenseur [B, C, H, W] ou [C, H, W]
            return_probabilities: Si True, retourne aussi les probabilités

        Returns:
            Si return_probabilities=False:
                Liste des indices de classes prédites
            Si return_probabilities=True:
                Tuple de (indices, probabilités)
                - indices: List[int]
                - probabilités: List[np.ndarray] de forme [num_classes]
        """
        # Ajouter dimension batch si nécessaire
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)

        # Déplacer sur le device
        face_tensor = face_tensor.to(self.device)

        # Convertir en FP16 si nécessaire
        if self.use_fp16:
            face_tensor = face_tensor.half()

        # Inférence
        logits = self.model(face_tensor)

        # Calculer probabilités avec softmax
        probabilities = torch.softmax(logits, dim=1)

        # Prédictions (indices des classes)
        predictions = torch.argmax(probabilities, dim=1)

        # Convertir en listes Python
        predictions_list = predictions.cpu().tolist()
        probabilities_list = probabilities.cpu().numpy()

        if return_probabilities:
            return predictions_list, probabilities_list
        else:
            return predictions_list

    def predict_single(
        self,
        face_tensor: torch.Tensor
    ) -> Tuple[int, np.ndarray, str]:
        """
        Prédit l'émotion pour un seul visage.

        Args:
            face_tensor: Tenseur [1, C, H, W] ou [C, H, W]

        Returns:
            Tuple de (class_idx, probabilities, class_name)
                - class_idx: Index de la classe prédite
                - probabilities: Array numpy [num_classes]
                - class_name: Nom de l'émotion
        """
        predictions, probabilities = self.predict(face_tensor, return_probabilities=True)

        class_idx = predictions[0]
        class_probs = probabilities[0]
        class_name = self.class_names[class_idx]

        return class_idx, class_probs, class_name

    def predict_batch_with_names(
        self,
        face_tensors: torch.Tensor
    ) -> List[Dict[str, any]]:
        """
        Prédit les émotions pour un batch et retourne des dictionnaires détaillés.

        Args:
            face_tensors: Tenseur [B, C, H, W]

        Returns:
            Liste de dictionnaires contenant:
                - 'class_idx': Index de la classe
                - 'class_name': Nom de l'émotion
                - 'confidence': Probabilité de la classe prédite
                - 'probabilities': Dict {emotion: probability}
        """
        predictions, probabilities = self.predict(face_tensors, return_probabilities=True)

        results = []

        for pred_idx, probs in zip(predictions, probabilities):
            result = {
                'class_idx': pred_idx,
                'class_name': self.class_names[pred_idx],
                'confidence': float(probs[pred_idx]),
                'probabilities': {
                    name: float(prob)
                    for name, prob in zip(self.class_names, probs)
                }
            }
            results.append(result)

        return results

    def get_top_k_emotions(
        self,
        face_tensor: torch.Tensor,
        k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Retourne les k émotions les plus probables.

        Args:
            face_tensor: Tenseur [C, H, W] ou [1, C, H, W]
            k: Nombre d'émotions à retourner

        Returns:
            Liste de tuples (emotion_name, probability) triée par probabilité décroissante
        """
        _, probabilities, _ = self.predict_single(face_tensor)

        # Trier par probabilité décroissante
        top_k_indices = np.argsort(probabilities)[::-1][:k]

        top_k_emotions = [
            (self.class_names[idx], float(probabilities[idx]))
            for idx in top_k_indices
        ]

        return top_k_emotions

    def benchmark(
        self,
        input_size: Tuple[int, int] = (48, 48),
        batch_size: int = 32,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark la vitesse d'inférence du modèle.

        Args:
            input_size: Taille de l'entrée (H, W)
            batch_size: Taille du batch
            num_iterations: Nombre d'itérations

        Returns:
            Dictionnaire de métriques:
                - 'fps': Images par seconde
                - 'latency_ms': Latence moyenne en ms
                - 'throughput': Images/seconde pour le batch
        """
        import time

        # Créer un batch factice
        dummy_input = torch.randn(
            batch_size, self.input_channels, input_size[0], input_size[1]
        ).to(self.device)

        if self.use_fp16:
            dummy_input = dummy_input.half()

        # Warmup
        for _ in range(10):
            _ = self.model(dummy_input)

        # Synchroniser GPU si CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()

        for _ in range(num_iterations):
            _ = self.model(dummy_input)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        end_time = time.time()

        total_time = end_time - start_time
        total_images = batch_size * num_iterations

        fps = total_images / total_time
        latency_ms = (total_time / num_iterations) * 1000
        throughput = batch_size / (total_time / num_iterations)

        results = {
            'fps': fps,
            'latency_ms': latency_ms,
            'throughput': throughput,
            'device': str(self.device),
            'fp16': self.use_fp16,
            'batch_size': batch_size
        }

        self.logger.info(
            f"Benchmark: {fps:.1f} FPS, {latency_ms:.2f}ms latency, "
            f"{throughput:.1f} images/sec throughput"
        )

        return results

    def export_torchscript(self, output_path: str, input_size: Tuple[int, int] = (48, 48)):
        """
        Exporte le modèle en TorchScript pour déploiement optimisé.

        Args:
            output_path: Chemin de sortie (.pt)
            input_size: Taille de l'entrée (H, W)
        """
        # Créer un exemple d'entrée
        example_input = torch.randn(
            1, self.input_channels, input_size[0], input_size[1]
        ).to(self.device)

        if self.use_fp16:
            example_input = example_input.half()

        # Tracer le modèle
        traced_model = torch.jit.trace(self.model, example_input)

        # Sauvegarder
        traced_model.save(output_path)
        self.logger.info(f"Model exported to TorchScript: {output_path}")

    def __repr__(self) -> str:
        return (
            f"EmotionClassifier(num_classes={self.num_classes}, "
            f"device={self.device}, fp16={self.use_fp16})"
        )


def create_emotion_classifier(config: Dict) -> EmotionClassifier:
    """
    Factory function pour créer un EmotionClassifier depuis la configuration.

    Args:
        config: Dictionnaire de configuration (chargé depuis config.yaml)

    Returns:
        Instance de EmotionClassifier configurée

    Example:
        >>> from src.utils.config import load_config
        >>> config = load_config('configs/config.yaml')
        >>> classifier = create_emotion_classifier(config)
    """
    model_config = config.get('emotion_model', {})
    device_config = config.get('device', {})
    emotion_config = config.get('emotions', {})

    return EmotionClassifier(
        model_path=model_config.get('weights_path'),
        num_classes=model_config.get('num_classes', 7),
        class_names=emotion_config.get('labels'),
        device=device_config.get('type', 'cuda'),
        use_fp16=device_config.get('mixed_precision', False),
        input_channels=1 if model_config.get('grayscale', True) else 3
    )


if __name__ == "__main__":
    """Test du EmotionClassifier."""
    import sys

    # Configuration de logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Testing EmotionClassifier...")

    # Créer un classificateur factice (sans modèle entraîné)
    classifier = EmotionClassifier(
        model_path=None,  # Utilisera un modèle factice
        num_classes=7,
        device='cpu'
    )

    # Test avec un tenseur factice
    test_tensor = torch.randn(1, 1, 48, 48)

    # Prédiction simple
    class_idx, probs, class_name = classifier.predict_single(test_tensor)
    print(f"\nPrediction: {class_name} (confidence: {probs[class_idx]:.2f})")

    # Top-3 émotions
    top_3 = classifier.get_top_k_emotions(test_tensor, k=3)
    print(f"\nTop-3 emotions:")
    for emotion, prob in top_3:
        print(f"  {emotion}: {prob:.2f}")

    # Benchmark
    print("\nRunning benchmark...")
    metrics = classifier.benchmark(batch_size=8, num_iterations=50)
    print(f"FPS: {metrics['fps']:.1f}")
    print(f"Latency: {metrics['latency_ms']:.2f}ms")

    print("\n✓ EmotionClassifier module loaded successfully!")
    print(classifier)
