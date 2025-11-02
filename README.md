# 🎓 Système d'Identification d'Étudiants avec Analyse d'Émotions

Système de reconnaissance faciale en temps réel combinant l'identification d'étudiants et l'analyse de leurs états émotionnels, optimisé pour AMD Radeon 7900 XT.

## 📋 Table des Matières

- [Caractéristiques](#caractéristiques)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [Licence](#licence)

## ✨ Caractéristiques

- ✅ Reconnaissance faciale en temps réel (>25 FPS)
- ✅ Détection de 7 émotions de base
- ✅ Identification d'étudiants
- ✅ Optimisé pour AMD Radeon 7900 XT avec ROCm
- ✅ Interface de visualisation en direct
- ✅ Logging et statistiques

## 🔧 Prérequis

### Matériel
- **GPU** : AMD Radeon RX 7900 XT (20 GB VRAM)
- **RAM** : 16 GB minimum, 64 GB recommandé
- **Stockage** : 50 GB disponibles (pour datasets et modèles)

### Logiciel
- **OS** : Ubuntu 22.04 LTS (recommandé) ou Windows 11
- **Python** : 3.10+
- **ROCm** : 5.7 ou supérieur
- **Webcam** : Caméra compatible (résolution 720p minimum)

## 📦 Installation

### 1. Installer ROCm (Ubuntu)

```bash
# Ajouter les dépôts AMD
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_5.7.50700-1_all.deb
sudo dpkg -i amdgpu-install_5.7.50700-1_all.deb
sudo apt update

# Installer ROCm
sudo amdgpu-install --usecase=rocm

# Ajouter utilisateur au groupe render
sudo usermod -a -G render,video $LOGNAME

# Redémarrer
sudo reboot
```

### 2. Vérifier l'installation ROCm

```bash
rocm-smi
rocminfo | grep "Name:"
```

Vous devriez voir votre AMD Radeon RX 7900 XT.

### 3. Créer l'environnement Python

```bash
# Cloner le projet
cd ~/Downloads
cd "Projet IA identification étudiant"

# Créer environnement virtuel
python3.10 -m venv venv_emotion
source venv_emotion/bin/activate  # Linux
# ou
venv_emotion\Scripts\activate  # Windows

# Mettre à jour pip
pip install --upgrade pip
```

### 4. Installer PyTorch avec ROCm

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

### 5. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 6. Tester l'installation GPU

```bash
python scripts/test_gpu.py
```

Sortie attendue :
```
PyTorch version: 2.x.x
CUDA available: True
Device count: 1
Device name: AMD Radeon RX 7900 XT
Device memory: 20.00 GB
✅ GPU détecté et fonctionnel !
```

## 🚀 Utilisation

### Démarrage Rapide

```bash
# Activer l'environnement
source venv_emotion/bin/activate

# Lancer le système en temps réel
python main.py --mode realtime

# Ou avec configuration personnalisée
python main.py --config configs/config.yaml
```

### Modes d'Utilisation

#### 1. Mode Temps Réel (Webcam)
```bash
python main.py --mode realtime
```

#### 2. Mode Traitement Vidéo
```bash
python main.py --mode video --input path/to/video.mp4
```

#### 3. Mode Image Unique
```bash
python main.py --mode image --input path/to/image.jpg
```

#### 4. Mode Entraînement
```bash
python scripts/train.py --config configs/train_config.yaml
```

### Options de Ligne de Commande

```bash
python main.py --help

Options:
  --mode {realtime,video,image}  Mode d'exécution
  --config PATH                  Fichier de configuration
  --model PATH                   Chemin vers le modèle
  --device {cuda,cpu}            Device à utiliser
  --fps-target INT               FPS cible (défaut: 30)
  --show-fps                     Afficher FPS en temps réel
  --save-output PATH             Sauvegarder la sortie
```

## 🏗️ Architecture du Projet

```
Projet IA identification étudiant/
├── README.md                  # Ce fichier
├── projet.md                  # Documentation détaillée du projet
├── plan.md                    # Roadmap de développement
├── requirements.txt           # Dépendances Python
├── main.py                    # Point d'entrée principal
├── setup.py                   # Installation du package
│
├── configs/                   # Fichiers de configuration
│   ├── config.yaml           # Configuration principale
│   ├── train_config.yaml     # Configuration entraînement
│   └── model_config.yaml     # Configuration modèle
│
├── src/                       # Code source
│   ├── __init__.py
│   ├── models/               # Architectures de modèles
│   │   ├── __init__.py
│   │   ├── emotion_net.py    # EmotionNet Nano
│   │   ├── efficient_net.py  # EfficientNet wrapper
│   │   └── face_embedding.py # Modèles d'embedding
│   │
│   ├── data/                 # Gestion des données
│   │   ├── __init__.py
│   │   ├── datasets.py       # PyTorch Datasets
│   │   ├── transforms.py     # Augmentations
│   │   └── loaders.py        # DataLoaders
│   │
│   ├── utils/                # Utilitaires
│   │   ├── __init__.py
│   │   ├── face_detector.py  # MTCNN/Haar Cascade
│   │   ├── preprocessor.py   # Prétraitement
│   │   ├── visualizer.py     # Visualisation
│   │   └── logger.py         # Logging
│   │
│   ├── core/                 # Logique métier
│   │   ├── __init__.py
│   │   ├── emotion_classifier.py
│   │   ├── student_identifier.py
│   │   └── system.py         # Système intégré
│   │
│   └── train/                # Scripts d'entraînement
│       ├── __init__.py
│       ├── trainer.py
│       └── evaluator.py
│
├── scripts/                   # Scripts utilitaires
│   ├── test_gpu.py           # Test GPU
│   ├── download_datasets.py  # Téléchargement datasets
│   ├── train.py              # Entraînement
│   ├── evaluate.py           # Évaluation
│   └── benchmark.py          # Benchmarking
│
├── notebooks/                 # Jupyter notebooks
│   ├── 01_EDA.ipynb          # Analyse exploratoire
│   ├── 02_Training.ipynb     # Entraînement
│   └── 03_Evaluation.ipynb   # Évaluation
│
├── data/                      # Données (non versionné)
│   ├── fer2013/
│   ├── rafdb/
│   └── students/
│
├── models/                    # Modèles entraînés (non versionné)
│   ├── emotion_net_nano.pt
│   ├── emotion_net_scripted.pt
│   └── student_embeddings.pkl
│
└── logs/                      # Logs et résultats (non versionné)
    ├── tensorboard/
    ├── checkpoints/
    └── results/
```

## 📊 Datasets

### Téléchargement Automatique

```bash
python scripts/download_datasets.py --dataset fer2013
python scripts/download_datasets.py --dataset rafdb
```

### Datasets Supportés

1. **FER2013** : ~35,000 images, 7 émotions
2. **RAF-DB** : ~30,000 images haute qualité
3. **CK+** : ~593 séquences vidéo (conditions lab)

Voir `projet.md` pour plus de détails sur les datasets.

## 🎯 Performance

### Métriques Cibles

| Métrique | Cible | État |
|----------|-------|------|
| FPS | >30 | ⏳ |
| Latence | <33ms | ⏳ |
| Précision Émotions | >70% | ⏳ |
| Précision Identification | >95% | ⏳ |

### Benchmarking

```bash
python scripts/benchmark.py --iterations 1000
```

## 🔬 Développement

### Tests

```bash
# Installer dépendances de test
pip install pytest pytest-cov

# Lancer les tests
pytest tests/

# Avec couverture
pytest --cov=src tests/
```

### Entraîner un Nouveau Modèle

```bash
python scripts/train.py \
    --model emotion_net_nano \
    --dataset fer2013 \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.001
```

### Évaluation

```bash
python scripts/evaluate.py \
    --model models/emotion_net_nano.pt \
    --dataset data/fer2013/test
```

## 📝 Documentation

- **`projet.md`** : Documentation technique complète
- **`plan.md`** : Roadmap de développement en 6 phases
- **Notebooks** : Tutoriels interactifs dans `notebooks/`

## 🤝 Contribution

Ce projet est développé dans un cadre académique. Pour toute suggestion :

1. Créer une issue
2. Proposer une pull request
3. Contacter l'équipe

## ⚖️ Considérations Éthiques

⚠️ **Important** : Ce système traite des données biométriques sensibles.

- Obtenir le consentement explicite avant collecte
- Respecter le RGPD et législations locales
- Chiffrer les données stockées
- Limiter la rétention des données
- Auditer les biais algorithmiques

Voir `projet.md` section "Considérations Éthiques" pour plus de détails.

## 📄 Licence

[À définir selon le contexte académique/commercial]

## 🙏 Remerciements

- AMD pour le support ROCm sur Radeon 7900 XT
- Équipe PyTorch pour l'intégration ROCm
- Communauté DeepFace
- Créateurs des datasets FER2013, RAF-DB, CK+

## 📞 Support

Pour toute question :
- Consulter `projet.md` et `plan.md`
- Ouvrir une issue GitHub
- Contacter l'équipe du projet

---

**Version** : 1.0
**Dernière mise à jour** : 2025-10-25
**Optimisé pour** : AMD Radeon RX 7900 XT avec ROCm 5.7+
