"""
Système intégré d'identification d'étudiants avec analyse d'émotions.

Ce module coordonne tous les composants du système:
- Détection de visages (FaceDetector)
- Prétraitement (FacePreprocessor)
- Classification d'émotions (EmotionClassifier)
- Identification d'étudiants (StudentIdentifier)

Features:
- Traitement temps réel webcam
- Traitement vidéo
- Traitement image unique
- Visualisation des résultats
- Logging des détections
- Sélection automatique d'étudiant

Author: Projet IA Identification Étudiants
Date: 2025-11-02
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from collections import deque
import csv

from src.utils.face_detector import FaceDetector
from src.utils.preprocessor import FacePreprocessor
from src.core.emotion_classifier import EmotionClassifier
from src.core.student_identifier import StudentIdentifier


class EmotionRecognitionSystem:
    """
    Système complet d'identification d'étudiants avec reconnaissance d'émotions.

    Attributes:
        face_detector: Détecteur de visages
        preprocessor: Préprocesseur de visages
        emotion_classifier: Classificateur d'émotions
        student_identifier: Identificateur d'étudiants
        config: Configuration du système
    """

    def __init__(
        self,
        face_detector: FaceDetector,
        preprocessor: FacePreprocessor,
        emotion_classifier: EmotionClassifier,
        student_identifier: Optional[StudentIdentifier] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialise le système complet.

        Args:
            face_detector: Instance de FaceDetector
            preprocessor: Instance de FacePreprocessor
            emotion_classifier: Instance de EmotionClassifier
            student_identifier: Instance de StudentIdentifier (optionnel)
            config: Configuration du système
        """
        self.face_detector = face_detector
        self.preprocessor = preprocessor
        self.emotion_classifier = emotion_classifier
        self.student_identifier = student_identifier

        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.show_fps = self.config.get('realtime', {}).get('show_fps', True)
        self.show_confidence = self.config.get('realtime', {}).get('show_confidence', True)
        self.skip_frames = self.config.get('realtime', {}).get('skip_frames', 0)

        # Visualisation
        viz_config = self.config.get('visualization', {})
        self.window_name = viz_config.get('window_name', 'Emotion Recognition System')
        self.box_color = tuple(viz_config.get('box_color', [0, 255, 0]))
        self.box_thickness = viz_config.get('box_thickness', 2)
        self.font_scale = viz_config.get('font_scale', 0.5)
        self.font_thickness = viz_config.get('font_thickness', 2)

        # Couleurs par émotion (BGR)
        emotions_config = self.config.get('emotions', {})
        self.emotion_colors = emotions_config.get('colors', {})

        # FPS tracking
        self.fps_buffer = deque(maxlen=30)
        self.frame_count = 0

        # Logging détections
        self.log_detections = self.config.get('logging', {}).get('log_detections', False)
        self.detection_log_path = self.config.get('logging', {}).get('detection_log_path')

        if self.log_detections and self.detection_log_path:
            self._init_detection_log()

        self.logger.info("EmotionRecognitionSystem initialized successfully")

    def _init_detection_log(self):
        """Initialise le fichier de log des détections."""
        log_path = Path(self.detection_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Créer le fichier CSV avec headers si nécessaire
        if not log_path.exists():
            with open(log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'student_id', 'student_name',
                    'emotion', 'emotion_confidence',
                    'face_x', 'face_y', 'face_w', 'face_h',
                    'face_confidence'
                ])

    def _log_detection(
        self,
        student_id: Optional[str],
        student_name: Optional[str],
        emotion: str,
        emotion_confidence: float,
        bbox: List[int],
        face_confidence: float
    ):
        """
        Enregistre une détection dans le fichier log.

        Args:
            student_id: ID de l'étudiant (ou None)
            student_name: Nom de l'étudiant (ou None)
            emotion: Émotion détectée
            emotion_confidence: Confiance de l'émotion
            bbox: Boîte englobante [x, y, w, h]
            face_confidence: Confiance de la détection de visage
        """
        if not self.log_detections or not self.detection_log_path:
            return

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        with open(self.detection_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                student_id or 'unknown',
                student_name or 'Unknown',
                emotion,
                f"{emotion_confidence:.4f}",
                bbox[0], bbox[1], bbox[2], bbox[3],
                f"{face_confidence:.4f}"
            ])

    def process_frame(
        self,
        frame: np.ndarray,
        detect_students: bool = True
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Traite une seule frame (détection + classification + identification).

        Args:
            frame: Image BGR (OpenCV format)
            detect_students: Si True, identifie aussi les étudiants

        Returns:
            Tuple de (frame annotée, liste de détections)
            Chaque détection contient:
                - 'bbox': [x, y, w, h]
                - 'face_confidence': Confiance détection visage
                - 'emotion': Nom de l'émotion
                - 'emotion_confidence': Confiance émotion
                - 'emotion_probs': Dict {emotion: probability}
                - 'student_id': ID étudiant (si identify=True)
                - 'student_name': Nom étudiant (si identify=True)
                - 'student_similarity': Similarité (si identify=True)
        """
        start_time = time.time()

        # 1. Détecter les visages
        face_detections = self.face_detector.detect(frame)

        if not face_detections:
            # Pas de visages détectés
            return frame, []

        # 2. Extraire et prétraiter les visages
        face_tensor, valid_detections = self.preprocessor.preprocess_with_detection(
            frame, face_detections
        )

        if face_tensor.shape[0] == 0:
            return frame, []

        # 3. Classifier les émotions
        emotion_results = self.emotion_classifier.predict_batch_with_names(face_tensor)

        # 4. Combiner les résultats
        results = []

        for detection, emotion_result in zip(valid_detections, emotion_results):
            result = {
                'bbox': detection['bbox'],
                'face_confidence': detection['confidence'],
                'emotion': emotion_result['class_name'],
                'emotion_confidence': emotion_result['confidence'],
                'emotion_probs': emotion_result['probabilities']
            }

            # 5. Identifier l'étudiant si demandé
            if detect_students and self.student_identifier is not None:
                x, y, w, h = detection['bbox']
                face_crop = frame[y:y+h, x:x+w]

                student_result = self.student_identifier.identify_student(face_crop)

                if student_result:
                    result['student_id'] = student_result['student_id']
                    result['student_name'] = student_result['name']
                    result['student_similarity'] = student_result['similarity']
                else:
                    result['student_id'] = None
                    result['student_name'] = 'Unknown'
                    result['student_similarity'] = 0.0

            results.append(result)

            # Log la détection
            self._log_detection(
                student_id=result.get('student_id'),
                student_name=result.get('student_name'),
                emotion=result['emotion'],
                emotion_confidence=result['emotion_confidence'],
                bbox=result['bbox'],
                face_confidence=result['face_confidence']
            )

        # 6. Annoter la frame
        annotated_frame = self._annotate_frame(frame, results)

        # 7. Calculer FPS
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        self.fps_buffer.append(fps)

        # Afficher FPS
        if self.show_fps:
            avg_fps = np.mean(self.fps_buffer) if self.fps_buffer else 0
            cv2.putText(
                annotated_frame,
                f"FPS: {avg_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        return annotated_frame, results

    def _annotate_frame(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """
        Annote la frame avec les résultats (boîtes, émotions, noms).

        Args:
            frame: Image source
            results: Liste de résultats de process_frame()

        Returns:
            Frame annotée
        """
        annotated = frame.copy()

        for result in results:
            x, y, w, h = result['bbox']
            emotion = result['emotion']
            confidence = result['emotion_confidence']

            # Couleur selon l'émotion
            color = self.emotion_colors.get(emotion, self.box_color)
            if isinstance(color, list):
                color = tuple(color)

            # Dessiner la boîte
            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                color,
                self.box_thickness
            )

            # Préparer le texte
            texts = []

            # Nom de l'étudiant
            if 'student_name' in result:
                student_name = result['student_name']
                texts.append(student_name)

            # Émotion
            if self.show_confidence:
                texts.append(f"{emotion}: {confidence:.2f}")
            else:
                texts.append(emotion)

            # Afficher les textes
            y_offset = y - 10

            for text in texts:
                # Fond noir pour meilleure lisibilité
                (text_width, text_height), _ = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    self.font_thickness
                )

                cv2.rectangle(
                    annotated,
                    (x, y_offset - text_height - 5),
                    (x + text_width, y_offset),
                    (0, 0, 0),
                    -1
                )

                # Texte
                cv2.putText(
                    annotated,
                    text,
                    (x, y_offset - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    color,
                    self.font_thickness
                )

                y_offset -= (text_height + 10)

        return annotated

    def run_webcam(self, camera_id: int = 0, window_name: Optional[str] = None):
        """
        Lance le traitement en temps réel avec webcam.

        Args:
            camera_id: ID de la caméra (0 par défaut)
            window_name: Nom de la fenêtre (optionnel)

        Controls:
            - 'q': Quitter
            - 's': Prendre une capture d'écran
            - 'p': Pause/Resume
        """
        window = window_name or self.window_name

        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            self.logger.error(f"Failed to open camera {camera_id}")
            return

        self.logger.info(f"Starting webcam capture (camera {camera_id})")
        self.logger.info("Press 'q' to quit, 's' for screenshot, 'p' to pause")

        paused = False
        frame_idx = 0

        try:
            while True:
                if not paused:
                    ret, frame = cap.read()

                    if not ret:
                        self.logger.error("Failed to read frame")
                        break

                    # Skip frames si configuré
                    if self.skip_frames > 0 and frame_idx % (self.skip_frames + 1) != 0:
                        frame_idx += 1
                        continue

                    # Traiter la frame
                    annotated_frame, results = self.process_frame(frame)

                    # Afficher
                    cv2.imshow(window, annotated_frame)

                    frame_idx += 1
                else:
                    # Mode pause
                    cv2.imshow(window, annotated_frame)

                # Gestion des touches
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Screenshot
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    self.logger.info(f"Screenshot saved: {filename}")
                elif key == ord('p'):
                    paused = not paused
                    self.logger.info(f"{'Paused' if paused else 'Resumed'}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Webcam capture stopped")

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_window: bool = True
    ):
        """
        Traite une vidéo complète.

        Args:
            video_path: Chemin de la vidéo d'entrée
            output_path: Chemin de la vidéo de sortie (optionnel)
            show_window: Afficher la fenêtre pendant le traitement
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return

        # Infos vidéo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(
            f"Processing video: {video_path} "
            f"({width}x{height}, {fps} FPS, {total_frames} frames)"
        )

        # Writer si output demandé
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Traiter
                annotated_frame, results = self.process_frame(frame)

                # Écrire
                if writer:
                    writer.write(annotated_frame)

                # Afficher
                if show_window:
                    cv2.imshow(self.window_name, annotated_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_idx += 1

                if frame_idx % 100 == 0:
                    self.logger.info(f"Processed {frame_idx}/{total_frames} frames")

        finally:
            cap.release()
            if writer:
                writer.release()
            if show_window:
                cv2.destroyAllWindows()

            self.logger.info(f"Video processing complete: {frame_idx} frames")

            if output_path:
                self.logger.info(f"Output saved: {output_path}")

    def process_image(self, image_path: str, output_path: Optional[str] = None) -> List[Dict]:
        """
        Traite une image unique.

        Args:
            image_path: Chemin de l'image
            output_path: Chemin de sortie (optionnel)

        Returns:
            Liste de détections
        """
        image = cv2.imread(image_path)

        if image is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return []

        annotated_frame, results = self.process_frame(image)

        if output_path:
            cv2.imwrite(output_path, annotated_frame)
            self.logger.info(f"Output saved: {output_path}")

        return results

    def select_student(
        self,
        frame: np.ndarray,
        criteria: str = "most_engaged"
    ) -> Optional[Dict]:
        """
        Sélectionne un étudiant selon des critères.

        Args:
            frame: Frame à analyser
            criteria: Critère de sélection
                - "most_engaged": Plus engagé (happy, surprise)
                - "least_engaged": Moins engagé (sad, bored, neutral)
                - "random": Aléatoire
                - "confidence": Plus haute confiance

        Returns:
            Dictionnaire de la détection de l'étudiant sélectionné, ou None
        """
        _, results = self.process_frame(frame)

        if not results:
            return None

        if criteria == "random":
            import random
            return random.choice(results)

        elif criteria == "confidence":
            return max(results, key=lambda r: r['emotion_confidence'])

        elif criteria == "most_engaged":
            engaged_emotions = {'happy': 3, 'surprise': 2, 'neutral': 1}

            def engagement_score(result):
                emotion = result['emotion']
                base_score = engaged_emotions.get(emotion, 0)
                confidence = result['emotion_confidence']
                return base_score * confidence

            return max(results, key=engagement_score)

        elif criteria == "least_engaged":
            disengaged_emotions = {'sad': 3, 'fear': 2, 'neutral': 1, 'angry': 2}

            def disengagement_score(result):
                emotion = result['emotion']
                base_score = disengaged_emotions.get(emotion, 0)
                confidence = result['emotion_confidence']
                return base_score * confidence

            return max(results, key=disengagement_score)

        else:
            self.logger.warning(f"Unknown criteria: {criteria}")
            return results[0] if results else None

    def __repr__(self) -> str:
        return (
            f"EmotionRecognitionSystem(\n"
            f"  detector={self.face_detector},\n"
            f"  preprocessor={self.preprocessor},\n"
            f"  classifier={self.emotion_classifier},\n"
            f"  identifier={self.student_identifier}\n"
            f")"
        )


def create_system_from_config(config: Dict) -> EmotionRecognitionSystem:
    """
    Factory function pour créer le système complet depuis la configuration.

    Args:
        config: Configuration complète (config.yaml)

    Returns:
        Instance de EmotionRecognitionSystem

    Example:
        >>> from src.utils.config import load_config
        >>> config = load_config('configs/config.yaml')
        >>> system = create_system_from_config(config)
        >>> system.run_webcam()
    """
    from src.utils.face_detector import create_face_detector
    from src.utils.preprocessor import create_face_preprocessor
    from src.core.emotion_classifier import create_emotion_classifier
    from src.core.student_identifier import create_student_identifier

    # Créer les composants
    face_detector = create_face_detector(config)
    preprocessor = create_face_preprocessor(config)
    emotion_classifier = create_emotion_classifier(config)

    # Student identifier (optionnel)
    student_identifier = None
    if 'student_recognition' in config:
        try:
            student_identifier = create_student_identifier(config)
        except Exception as e:
            logging.warning(f"Failed to create student identifier: {e}")

    # Créer le système
    system = EmotionRecognitionSystem(
        face_detector=face_detector,
        preprocessor=preprocessor,
        emotion_classifier=emotion_classifier,
        student_identifier=student_identifier,
        config=config
    )

    return system


if __name__ == "__main__":
    """Test du système."""
    import sys
    from src.utils.config import load_config

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Testing EmotionRecognitionSystem...")

    # Charger la configuration
    config = load_config('configs/config.yaml')

    # Créer le système
    system = create_system_from_config(config)

    print(f"\n{system}")
    print("\n✓ EmotionRecognitionSystem module loaded successfully!")
