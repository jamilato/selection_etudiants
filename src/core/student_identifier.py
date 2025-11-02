"""
Module d'identification d'étudiants par reconnaissance faciale.

Ce module utilise DeepFace pour extraire des embeddings de visages
et identifier les étudiants par comparaison de similarité.

Features:
- Extraction d'embeddings avec DeepFace (Facenet, VGG-Face, etc.)
- Base de données d'embeddings d'étudiants
- Identification par similarité cosinus
- Gestion du seuil de confiance
- Enregistrement de nouveaux étudiants

Author: Projet IA Identification Étudiants
Date: 2025-11-02
"""

import numpy as np
import cv2
import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from collections import defaultdict

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logging.warning("DeepFace not available. Face recognition disabled.")


class StudentIdentifier:
    """
    Identificateur d'étudiants par reconnaissance faciale.

    Attributes:
        model_name (str): Modèle DeepFace ('Facenet', 'VGG-Face', 'ArcFace', etc.)
        similarity_threshold (float): Seuil de similarité cosinus (0.0-1.0)
        embeddings_db (Dict): Base de données {student_id: embeddings_list}
        student_metadata (Dict): Métadonnées {student_id: {name, class, etc.}}
    """

    def __init__(
        self,
        model_name: str = "Facenet",
        similarity_threshold: float = 0.6,
        embeddings_db_path: Optional[str] = None,
        detector_backend: str = "skip"  # Skip car on a déjà détecté les visages
    ):
        """
        Initialise l'identificateur d'étudiants.

        Args:
            model_name: Modèle DeepFace ('Facenet', 'VGG-Face', 'ArcFace', 'Dlib', 'OpenFace')
            similarity_threshold: Seuil de similarité cosinus (0.6 recommandé)
            embeddings_db_path: Chemin vers la DB d'embeddings sauvegardée (.pkl)
            detector_backend: Backend de détection (on utilise 'skip' car déjà fait)
        """
        if not DEEPFACE_AVAILABLE:
            raise ImportError(
                "DeepFace is not installed. "
                "Install with: pip install deepface"
            )

        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.detector_backend = detector_backend

        # Base de données d'embeddings
        self.embeddings_db = {}  # {student_id: [embedding1, embedding2, ...]}
        self.student_metadata = {}  # {student_id: {name, class, ...}}

        self.logger = logging.getLogger(__name__)

        # Charger la DB si elle existe
        if embeddings_db_path and Path(embeddings_db_path).exists():
            self.load_database(embeddings_db_path)
        else:
            self.logger.info("Starting with empty embeddings database")

        self.logger.info(
            f"StudentIdentifier initialized with model: {model_name}, "
            f"threshold: {similarity_threshold}"
        )

    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrait l'embedding d'un visage avec DeepFace.

        Args:
            face_image: Image du visage (BGR ou RGB)

        Returns:
            Embedding numpy array, ou None si échec
        """
        try:
            # DeepFace attend BGR (OpenCV format)
            if len(face_image.shape) == 2:
                # Grayscale -> BGR
                face_bgr = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
            else:
                face_bgr = face_image

            # Extraire l'embedding
            embedding_objs = DeepFace.represent(
                img_path=face_bgr,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False  # Ne pas échouer si pas de visage détecté
            )

            if embedding_objs and len(embedding_objs) > 0:
                embedding = np.array(embedding_objs[0]["embedding"])
                return embedding
            else:
                self.logger.warning("No embedding extracted from face")
                return None

        except Exception as e:
            self.logger.error(f"Failed to extract embedding: {e}")
            return None

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calcule la similarité cosinus entre deux embeddings.

        Args:
            embedding1: Premier embedding
            embedding2: Deuxième embedding

        Returns:
            Similarité cosinus (0.0-1.0), plus haut = plus similaire
        """
        # Normaliser les vecteurs
        emb1_norm = embedding1 / np.linalg.norm(embedding1)
        emb2_norm = embedding2 / np.linalg.norm(embedding2)

        # Produit scalaire = cosinus de l'angle
        similarity = np.dot(emb1_norm, emb2_norm)

        # Convertir de [-1, 1] à [0, 1]
        similarity = (similarity + 1) / 2

        return float(similarity)

    def identify_student(
        self,
        face_image: np.ndarray
    ) -> Optional[Dict[str, any]]:
        """
        Identifie un étudiant à partir de son visage.

        Args:
            face_image: Image du visage (BGR)

        Returns:
            Dictionnaire contenant:
                - 'student_id': ID de l'étudiant
                - 'name': Nom de l'étudiant
                - 'similarity': Score de similarité
                - 'metadata': Métadonnées de l'étudiant
            Ou None si aucune correspondance
        """
        if not self.embeddings_db:
            self.logger.warning("Embeddings database is empty")
            return None

        # Extraire l'embedding du visage
        query_embedding = self.extract_embedding(face_image)

        if query_embedding is None:
            return None

        # Comparer avec tous les étudiants
        best_match = None
        best_similarity = -1

        for student_id, embeddings_list in self.embeddings_db.items():
            # Calculer la similarité moyenne avec tous les embeddings de cet étudiant
            similarities = [
                self.cosine_similarity(query_embedding, stored_emb)
                for stored_emb in embeddings_list
            ]

            avg_similarity = np.mean(similarities)

            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match = student_id

        # Vérifier le seuil
        if best_similarity >= self.similarity_threshold:
            result = {
                'student_id': best_match,
                'name': self.student_metadata.get(best_match, {}).get('name', 'Unknown'),
                'similarity': best_similarity,
                'metadata': self.student_metadata.get(best_match, {})
            }
            return result
        else:
            self.logger.debug(
                f"Best similarity {best_similarity:.3f} below threshold "
                f"{self.similarity_threshold}"
            )
            return None

    def register_student(
        self,
        student_id: str,
        face_images: List[np.ndarray],
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Enregistre un nouvel étudiant dans la base de données.

        Args:
            student_id: ID unique de l'étudiant
            face_images: Liste d'images du visage (recommandé: 10-50)
            metadata: Métadonnées (nom, classe, etc.)

        Returns:
            True si succès, False sinon
        """
        if student_id in self.embeddings_db:
            self.logger.warning(f"Student {student_id} already exists. Use update_student().")
            return False

        # Extraire les embeddings
        embeddings = []
        for face_img in face_images:
            emb = self.extract_embedding(face_img)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            self.logger.error(f"Failed to extract embeddings for student {student_id}")
            return False

        # Ajouter à la DB
        self.embeddings_db[student_id] = embeddings
        self.student_metadata[student_id] = metadata or {}

        self.logger.info(
            f"Registered student {student_id} with {len(embeddings)} embeddings"
        )

        return True

    def update_student_embeddings(
        self,
        student_id: str,
        face_images: List[np.ndarray],
        replace: bool = False
    ) -> bool:
        """
        Met à jour les embeddings d'un étudiant existant.

        Args:
            student_id: ID de l'étudiant
            face_images: Nouvelles images du visage
            replace: Si True, remplace tous les embeddings. Si False, ajoute.

        Returns:
            True si succès, False sinon
        """
        if student_id not in self.embeddings_db:
            self.logger.error(f"Student {student_id} not found. Use register_student().")
            return False

        # Extraire les nouveaux embeddings
        new_embeddings = []
        for face_img in face_images:
            emb = self.extract_embedding(face_img)
            if emb is not None:
                new_embeddings.append(emb)

        if not new_embeddings:
            self.logger.error("Failed to extract new embeddings")
            return False

        # Mettre à jour
        if replace:
            self.embeddings_db[student_id] = new_embeddings
        else:
            self.embeddings_db[student_id].extend(new_embeddings)

        self.logger.info(
            f"Updated student {student_id}: {len(new_embeddings)} new embeddings "
            f"(total: {len(self.embeddings_db[student_id])})"
        )

        return True

    def remove_student(self, student_id: str) -> bool:
        """
        Supprime un étudiant de la base de données.

        Args:
            student_id: ID de l'étudiant

        Returns:
            True si succès, False si étudiant introuvable
        """
        if student_id not in self.embeddings_db:
            self.logger.error(f"Student {student_id} not found")
            return False

        del self.embeddings_db[student_id]
        if student_id in self.student_metadata:
            del self.student_metadata[student_id]

        self.logger.info(f"Removed student {student_id}")
        return True

    def get_all_students(self) -> List[Dict[str, any]]:
        """
        Retourne la liste de tous les étudiants enregistrés.

        Returns:
            Liste de dictionnaires avec student_id, name, num_embeddings, metadata
        """
        students = []

        for student_id in self.embeddings_db.keys():
            student_info = {
                'student_id': student_id,
                'name': self.student_metadata.get(student_id, {}).get('name', 'Unknown'),
                'num_embeddings': len(self.embeddings_db[student_id]),
                'metadata': self.student_metadata.get(student_id, {})
            }
            students.append(student_info)

        return students

    def save_database(self, output_path: str):
        """
        Sauvegarde la base de données d'embeddings.

        Args:
            output_path: Chemin de sortie (.pkl)
        """
        data = {
            'embeddings_db': self.embeddings_db,
            'student_metadata': self.student_metadata,
            'model_name': self.model_name,
            'similarity_threshold': self.similarity_threshold
        }

        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        self.logger.info(f"Database saved to {output_path}")

    def load_database(self, input_path: str):
        """
        Charge une base de données d'embeddings.

        Args:
            input_path: Chemin du fichier (.pkl)
        """
        try:
            with open(input_path, 'rb') as f:
                data = pickle.load(f)

            self.embeddings_db = data.get('embeddings_db', {})
            self.student_metadata = data.get('student_metadata', {})

            # Charger les paramètres si disponibles
            if 'model_name' in data:
                self.model_name = data['model_name']
            if 'similarity_threshold' in data:
                self.similarity_threshold = data['similarity_threshold']

            num_students = len(self.embeddings_db)
            total_embeddings = sum(len(embs) for embs in self.embeddings_db.values())

            self.logger.info(
                f"Database loaded: {num_students} students, "
                f"{total_embeddings} total embeddings"
            )

        except Exception as e:
            self.logger.error(f"Failed to load database from {input_path}: {e}")
            raise

    def build_database_from_folder(
        self,
        students_folder: str,
        save_path: Optional[str] = None
    ):
        """
        Construit la base de données depuis un dossier de photos d'étudiants.

        Structure attendue:
        students_folder/
            student_001/
                photo1.jpg
                photo2.jpg
                ...
                metadata.txt (optionnel: "Name: John Doe\\nClass: A1")
            student_002/
                ...

        Args:
            students_folder: Chemin du dossier principal
            save_path: Chemin pour sauvegarder la DB (optionnel)
        """
        students_path = Path(students_folder)

        if not students_path.exists():
            self.logger.error(f"Folder {students_folder} does not exist")
            return

        # Parcourir les sous-dossiers
        for student_folder in students_path.iterdir():
            if not student_folder.is_dir():
                continue

            student_id = student_folder.name

            # Charger les métadonnées si disponibles
            metadata_file = student_folder / "metadata.txt"
            metadata = {}

            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.strip().split(':', 1)
                            metadata[key.strip().lower()] = value.strip()

            # Charger toutes les images
            face_images = []
            for img_path in student_folder.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        face_images.append(img)

            if not face_images:
                self.logger.warning(f"No images found for student {student_id}")
                continue

            # Enregistrer l'étudiant
            success = self.register_student(student_id, face_images, metadata)

            if success:
                self.logger.info(
                    f"Registered {student_id} with {len(face_images)} images"
                )

        # Sauvegarder si demandé
        if save_path:
            self.save_database(save_path)

        self.logger.info(
            f"Database built: {len(self.embeddings_db)} students registered"
        )

    def __repr__(self) -> str:
        num_students = len(self.embeddings_db)
        total_embeddings = sum(len(embs) for embs in self.embeddings_db.values())

        return (
            f"StudentIdentifier(model={self.model_name}, "
            f"threshold={self.similarity_threshold}, "
            f"students={num_students}, embeddings={total_embeddings})"
        )


def create_student_identifier(config: Dict) -> StudentIdentifier:
    """
    Factory function pour créer un StudentIdentifier depuis la configuration.

    Args:
        config: Dictionnaire de configuration (chargé depuis config.yaml)

    Returns:
        Instance de StudentIdentifier configurée

    Example:
        >>> from src.utils.config import load_config
        >>> config = load_config('configs/config.yaml')
        >>> identifier = create_student_identifier(config)
    """
    recognition_config = config.get('student_recognition', {})

    return StudentIdentifier(
        model_name=recognition_config.get('model_name', 'Facenet'),
        similarity_threshold=recognition_config.get('similarity_threshold', 0.6),
        embeddings_db_path=recognition_config.get('embeddings_db_path')
    )


if __name__ == "__main__":
    """Test du StudentIdentifier."""
    import sys

    # Configuration de logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not DEEPFACE_AVAILABLE:
        print("DeepFace not available. Skipping test.")
        sys.exit(0)

    print("Testing StudentIdentifier...")

    # Créer l'identificateur
    identifier = StudentIdentifier(model_name="Facenet", similarity_threshold=0.6)

    # Test avec des images factices
    dummy_face1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    dummy_face2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Note: Avec des images aléatoires, les embeddings ne seront pas significatifs
    # Pour un vrai test, utilisez de vraies photos de visages

    print(f"\n{identifier}")
    print("\n✓ StudentIdentifier module loaded successfully!")
    print("Note: For real testing, use actual face images.")
