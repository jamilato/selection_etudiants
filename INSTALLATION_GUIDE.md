# Guide d'Installation - WSL2 Ubuntu

Guide complet pour installer et configurer le système d'identification d'étudiants avec IA sur WSL2.

## Table des Matières

1. [Prérequis](#prérequis)
2. [Installation Rapide](#installation-rapide)
3. [Installation Manuelle](#installation-manuelle)
4. [Configuration GPU AMD](#configuration-gpu-amd)
5. [Vérification](#vérification)
6. [Résolution des Problèmes](#résolution-des-problèmes)

---

## Prérequis

### Windows

- **Windows 10** version 2004+ ou **Windows 11**
- **WSL2** installé et configuré
- **Ubuntu 22.04** (recommandé) ou 20.04
- Au moins **20 GB** d'espace disque libre
- **8 GB RAM** minimum (16 GB recommandé)

### Pour GPU AMD (optionnel)

- **AMD Radeon RX 7900 XT** ou compatible
- **ROCm 5.7+** compatible avec votre GPU

### Vérifier WSL2

```powershell
# Dans PowerShell (Windows)
wsl --version
wsl --list --verbose
```

Si WSL n'est pas installé, suivez : https://learn.microsoft.com/en-us/windows/wsl/install

---

## Installation Rapide

### Option 1 : Installation Automatique (Recommandé)

```bash
# 1. Cloner ou copier le projet dans WSL
cd ~
# (Le projet devrait déjà être accessible depuis Windows)

# 2. Naviguer vers le projet
cd "/mnt/c/Users/MNB/Downloads/Projet IA identification étudiant"

# 3. Rendre le script exécutable
chmod +x install_wsl.sh

# 4a. Installation CPU uniquement
bash install_wsl.sh --cpu-only

# 4b. Installation avec support GPU AMD (ROCm)
bash install_wsl.sh --with-rocm

# 4c. Installation auto-détection
bash install_wsl.sh
```

Le script va :
- ✅ Mettre à jour le système
- ✅ Installer Python 3.10+ et pip
- ✅ Installer toutes les dépendances
- ✅ Installer PyTorch (CPU ou GPU)
- ✅ Installer ROCm (si --with-rocm)
- ✅ Configurer l'environnement
- ✅ Créer les dossiers nécessaires
- ✅ Vérifier l'installation

**Durée estimée :** 15-30 minutes

---

## Installation Manuelle

Si vous préférez installer manuellement ou si le script automatique échoue.

### Étape 1 : Mise à jour du système

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### Étape 2 : Installer Python et outils

```bash
sudo apt-get install -y \
    python3 python3-pip python3-dev python3-venv \
    build-essential cmake git wget curl unzip
```

### Étape 3 : Installer les bibliothèques système

```bash
# Pour OpenCV
sudo apt-get install -y \
    libopencv-dev libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev libgomp1

# Pour vidéo
sudo apt-get install -y \
    ffmpeg libavcodec-dev libavformat-dev \
    libswscale-dev libv4l-dev

# Pour calcul scientifique
sudo apt-get install -y \
    libatlas-base-dev gfortran
```

### Étape 4 : Créer un environnement virtuel (recommandé)

```bash
cd "/mnt/c/Users/MNB/Downloads/Projet IA identification étudiant"

# Créer l'environnement
python3 -m venv venv

# Activer
source venv/bin/activate
```

### Étape 5 : Installer PyTorch

**Pour CPU uniquement :**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Pour GPU AMD avec ROCm :**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

**Auto-détection (recommandé) :**
```bash
pip install torch torchvision torchaudio
```

### Étape 6 : Installer les dépendances Python

```bash
pip install -r requirements.txt
```

Si `requirements.txt` manque des packages :
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
pip install opencv-python opencv-contrib-python
pip install Pillow albumentations imgaug
pip install deepface facenet-pytorch mtcnn
pip install tensorboard onnx onnxruntime
pip install kaggle pyyaml tqdm
```

### Étape 7 : Créer les dossiers

```bash
mkdir -p data/fer2013 data/rafdb data/students
mkdir -p models logs checkpoints
```

### Étape 8 : Configurer Kaggle API

```bash
# Créer le dossier
mkdir -p ~/.kaggle

# Copier le fichier kaggle.json depuis Windows
cp /mnt/c/Users/MNB/Downloads/kaggle.json ~/.kaggle/

# Ou créer manuellement
cat > ~/.kaggle/kaggle.json << 'EOF'
{"username":"nasserson","key":"a0711bbd0d7b8d8323ffc79aab6afef5"}
EOF

# Définir les permissions
chmod 600 ~/.kaggle/kaggle.json
```

---

## Configuration GPU AMD

### Installation de ROCm

```bash
# Ajouter le dépôt ROCm
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7/ ubuntu main' | \
    sudo tee /etc/apt/sources.list.d/rocm.list

# Installer ROCm
sudo apt-get update
sudo apt-get install -y rocm-dkms rocm-libs

# Ajouter l'utilisateur aux groupes
sudo usermod -a -G video $LOGNAME
sudo usermod -a -G render $LOGNAME
```

### Configuration .wslconfig (Windows)

Créez `C:\Users\MNB\.wslconfig` :

```ini
[wsl2]
memory=16GB
processors=8
swap=8GB
localhostForwarding=true

[experimental]
autoMemoryReclaim=gradual
sparseVhd=true
```

### Variables d'environnement

Ajoutez à `~/.bashrc` :

```bash
# ROCm Environment
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

Puis :
```bash
source ~/.bashrc
```

### ⚠️ Redémarrage Requis

Après installation de ROCm, **redémarrez WSL** :

```powershell
# Dans PowerShell (Windows)
wsl --shutdown
# Puis rouvrez WSL
```

---

## Vérification

### Script de Vérification Automatique

```bash
chmod +x verify_installation.sh
bash verify_installation.sh
```

Ce script vérifie :
- ✅ Python et pip
- ✅ Toutes les bibliothèques Python
- ✅ PyTorch et GPU
- ✅ OpenCV
- ✅ Modules du projet
- ✅ Configuration Kaggle
- ✅ Structure des dossiers
- ✅ ROCm (si installé)

### Tests Manuels

**Test Python :**
```bash
python3 --version  # Devrait être >= 3.8
pip3 --version
```

**Test PyTorch :**
```python
python3 << EOF
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF
```

**Test GPU avec script fourni :**
```bash
python scripts/test_gpu.py
```

**Test OpenCV :**
```python
python3 << EOF
import cv2
print(f"OpenCV: {cv2.__version__}")
EOF
```

**Test des modules du projet :**
```bash
python3 << EOF
from src.models.emotion_net import EmotionNetNano
from src.utils.config import load_config
print("Modules OK!")
EOF
```

---

## Résolution des Problèmes

### Problème : Python version < 3.8

**Solution :**
```bash
# Ajouter deadsnakes PPA
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update

# Installer Python 3.10
sudo apt-get install python3.10 python3.10-venv python3.10-dev
```

### Problème : Erreur d'import OpenCV

**Solution :**
```bash
# Installer les dépendances manquantes
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

# Réinstaller OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python opencv-contrib-python --no-cache-dir
```

### Problème : GPU AMD non détecté

**Solutions :**

1. **Vérifier ROCm :**
```bash
rocm-smi
/opt/rocm/bin/rocminfo
```

2. **Vérifier variable d'environnement :**
```bash
echo $HSA_OVERRIDE_GFX_VERSION  # Devrait afficher 11.0.0
```

3. **Réinstaller PyTorch ROCm :**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

4. **Vérifier .wslconfig :**
Assurez-vous que `C:\Users\MNB\.wslconfig` existe et contient les bonnes valeurs.

### Problème : Kaggle API 403 Forbidden

**Solution :**

1. Allez sur la page de la compétition : https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
2. Cliquez sur "Join Competition" et acceptez les règles
3. Ou utilisez le dataset public :
```bash
kaggle datasets download -d msambare/fer2013
```

### Problème : Mémoire insuffisante

**Solution :**

Modifiez `.wslconfig` pour allouer plus de RAM :
```ini
[wsl2]
memory=16GB  # Augmenter selon votre RAM
```

Puis redémarrez WSL :
```powershell
wsl --shutdown
```

### Problème : Erreur "UnicodeEncodeError" dans les scripts

**Solution :**

C'est un problème d'encodage Windows. Les scripts fonctionnent malgré l'erreur. Pour éviter l'erreur :

```bash
# Définir l'encodage UTF-8
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Ajouter à ~/.bashrc pour permanence
echo "export PYTHONIOENCODING=utf-8" >> ~/.bashrc
```

### Problème : Lenteur du système

**Solutions :**

1. **Utiliser un SSD :** Placez le projet sur un SSD plutôt qu'un HDD
2. **Optimiser WSL :** Ajustez `.wslconfig`
3. **Fermer les applications inutiles**
4. **Utiliser CPU-only si pas de GPU :**
```bash
python scripts/train.py --device cpu
```

---

## Commandes Utiles

### Gestion WSL (depuis PowerShell Windows)

```powershell
# Lister les distributions
wsl --list --verbose

# Arrêter WSL
wsl --shutdown

# Démarrer une distribution
wsl -d Ubuntu-22.04

# Définir distribution par défaut
wsl --set-default Ubuntu-22.04
```

### Gestion de l'environnement virtuel

```bash
# Activer
source venv/bin/activate

# Désactiver
deactivate

# Vérifier packages installés
pip list

# Mettre à jour tous les packages
pip list --outdated
pip install --upgrade <package>
```

### Alias utiles (déjà configurés par install_wsl.sh)

```bash
# Activer l'environnement
activate-venv

# Entraîner le modèle
train

# Test GPU
test-gpu

# Mode temps réel
run-realtime

# Mode vidéo
run-video video.mp4

# Mode image
run-image image.jpg
```

---

## Prochaines Étapes

Après installation réussie :

1. **Télécharger le dataset :**
```bash
python scripts/download_datasets.py
```

2. **Préparer les données :**
```bash
python scripts/prepare_data.py --all
```

3. **Entraîner le modèle :**
```bash
python scripts/train.py
```

4. **Tester le système :**
```bash
python main.py --mode realtime
```

---

## Ressources Supplémentaires

- **README.md** : Vue d'ensemble du projet
- **QUICKSTART.md** : Guide de démarrage rapide
- **INSTALLATION_ROCM.md** : Guide détaillé ROCm
- **INSTALLATION_WSL2_ROCM.md** : Guide WSL2 + ROCm
- **Documentation PyTorch ROCm** : https://pytorch.org/docs/stable/notes/hip.html
- **Documentation WSL** : https://learn.microsoft.com/en-us/windows/wsl/

---

## Support

Si vous rencontrez des problèmes :

1. Exécutez `bash verify_installation.sh` pour diagnostiquer
2. Consultez les logs dans `installation_*.log`
3. Vérifiez les issues GitHub du projet
4. Consultez la documentation officielle des bibliothèques

**Bon courage avec votre projet ! 🚀**
