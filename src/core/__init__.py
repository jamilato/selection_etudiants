"""
Module core pour le système d'identification d'étudiants.

Ce module contient les composants principaux du système:
- EmotionClassifier: Classification d'émotions en temps réel
- StudentIdentifier: Reconnaissance et identification d'étudiants
- System: Système intégré complet
"""

from .emotion_classifier import EmotionClassifier, create_emotion_classifier

__all__ = [
    'EmotionClassifier',
    'create_emotion_classifier',
]
