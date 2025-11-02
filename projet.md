Projet : Système d’Intelligence Artificielle pour la Sélection Dynamique d’Étudiants en Classe
Objectif du projet
L’objectif de ce projet est de concevoir une application d’intelligence artificielle interactive capable d’observer une classe en temps réel, d’analyser les visages des étudiants à partir d’un flux vidéo, et de choisir automatiquement un étudiant pour répondre à une question. L’idée est d’utiliser la vision par ordinateur et l’analyse émotionnelle pour rendre les cours plus participatifs et dynamiques.
Principe de fonctionnement
L’application capte le flux vidéo d’une caméra placée dans la salle de classe. L’IA détecte et reconnaît les visages des étudiants présents, analyse leurs émotions ou leur niveau d’attention (sourire, concentration, regard), puis sélectionne automatiquement un étudiant selon une logique définie (au hasard ou en fonction de l’attention). Le visage de l’étudiant choisi est ensuite affiché ou annoncé.
Architecture technique
1. Acquisition vidéo
Utilisation d’une webcam pour capter le flux en direct via la bibliothèque OpenCV. Ce module permet d’afficher la classe en temps réel et de traiter les images image par image.
2. Détection des visages
Utilisation de MediaPipe ou face_recognition pour identifier les visages dans le flux vidéo. Chaque visage détecté est encadré et enregistré temporairement pour analyse.
3. Analyse émotionnelle et attentionnelle
Grâce à la bibliothèque DeepFace, le système analyse les émotions dominantes (joie, surprise, fatigue, etc.). Ces informations permettent d’évaluer le niveau d’attention ou d’implication de chaque étudiant.
4. Sélection intelligente de l’étudiant
L’IA peut choisir un étudiant au hasard ou celui qui semble le plus attentif ou le plus distrait, selon l’objectif pédagogique. Une logique simple ou un petit modèle d’apprentissage automatique peut être ajouté.
5. Affichage du résultat
Le visage de l’étudiant choisi est mis en surbrillance dans la vidéo. Le système peut aussi annoncer le prénom de l’étudiant via une synthèse vocale.
Outils et technologies utilisés
Composant	Outil / Bibliothèque	Fonction
Vision par ordinateur	OpenCV	Capture et traitement des images
Détection de visages	MediaPipe / face_recognition	Localisation des visages
Analyse émotionnelle	DeepFace	Détection des émotions et de l’attention
Interface utilisateur	Streamlit ou Tkinter	Visualisation et interaction
Langage de programmation	Python 3.10+	Développement du système
Améliorations futures
- Ajout d’une base de données pour enregistrer les participations des étudiants.
- Intégration d’une reconnaissance faciale nominative.
- Génération d’un rapport d’attention global pour l’enseignant.
- Utilisation d’une voix synthétique pour appeler directement l’étudiant choisi.
Aspects éthiques et confidentialité
Comme le projet implique l’analyse d’images de personnes, il est essentiel d’obtenir le consentement explicite des étudiants filmés, d’éviter toute conservation non autorisée des données et d’anonymiser les visages si nécessaire. Le projet sera réalisé dans un cadre académique et expérimental, uniquement à des fins pédagogiques.
Conclusion
Ce projet combine vision par ordinateur, analyse émotionnelle et interaction en temps réel pour créer une expérience d’enseignement plus interactive. Il illustre parfaitement l’application concrète de l’intelligence artificielle dans le domaine de l’éducation.
