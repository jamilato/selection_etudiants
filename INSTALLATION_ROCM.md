# 🔧 Guide d'Installation ROCm pour AMD Radeon 7900 XT

Guide complet pour installer et configurer ROCm + PyTorch sur AMD Radeon RX 7900 XT pour le deep learning.

## 📋 Table des Matières

- [Prérequis](#prérequis)
- [Installation Ubuntu](#installation-ubuntu)
- [Installation Windows](#installation-windows)
- [Vérification](#vérification)
- [Dépannage](#dépannage)
- [Optimisations](#optimisations)

---

## 🔍 Prérequis

### Matériel Requis
- **GPU** : AMD Radeon RX 7900 XT (RDNA 3, 20 GB VRAM)
- **RAM** : 16 GB minimum, 64 GB recommandé
- **Stockage** : 50 GB d'espace libre

### Systèmes d'Exploitation Supportés
- ✅ **Ubuntu 22.04 LTS** (RECOMMANDÉ - support stable)
- ✅ **Ubuntu 20.04 LTS** (supporté)
- ⚠️ **Windows 11** (support preview, moins stable)
- ❌ **Windows 10** (non supporté officiellement)

---

## 🐧 Installation Ubuntu (RECOMMANDÉ)

### Étape 1 : Préparation du Système

```bash
# Mettre à jour le système
sudo apt update && sudo apt upgrade -y

# Installer dépendances de base
sudo apt install -y \
    build-essential \
    git \
    wget \
    curl \
    python3.10 \
    python3.10-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0
```

### Étape 2 : Installation du Driver AMD et ROCm

#### Option A : Installation Automatique (Recommandé)

```bash
# Télécharger l'installeur AMD
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_5.7.50700-1_all.deb

# Installer le package
sudo dpkg -i amdgpu-install_5.7.50700-1_all.deb

# Mettre à jour les dépôts
sudo apt update

# Installer ROCm complet
sudo amdgpu-install --usecase=rocm -y
```

#### Option B : Installation Manuelle

```bash
# Ajouter les clés GPG
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -

# Ajouter le dépôt ROCm
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7 ubuntu main' | \
    sudo tee /etc/apt/sources.list.d/rocm.list

# Installer ROCm
sudo apt update
sudo apt install rocm-hip-sdk rocm-libs -y
```

### Étape 3 : Configuration des Permissions

```bash
# Ajouter l'utilisateur aux groupes render et video
sudo usermod -a -G render,video $LOGNAME

# Vérifier l'appartenance aux groupes
groups $LOGNAME
```

**⚠️ IMPORTANT** : Déconnectez-vous et reconnectez-vous (ou redémarrez) pour que les changements de groupe prennent effet.

```bash
sudo reboot
```

### Étape 4 : Vérification de l'Installation ROCm

Après redémarrage :

```bash
# Vérifier ROCm
rocm-smi

# Vérifier les informations GPU
rocminfo | grep -A 5 "Name:"

# Vérifier la version ROCm
/opt/rocm/bin/rocm-smi --showdriverversion
```

**Sortie attendue pour rocm-smi :**
```
========================= ROCm System Management Interface =========================
=========================== GPU0 : AMD Radeon RX 7900 XT ===========================
GPU[0]  : Temperature: 45.0°C
GPU[0]  : GPU use (%): 0
GPU[0]  : Memory use: 0% (0MB / 20480MB)
```

### Étape 5 : Installation de Python et Environnement Virtuel

```bash
# Vérifier version Python
python3.10 --version

# Créer environnement virtuel
cd ~/path/to/project
python3.10 -m venv venv_emotion

# Activer l'environnement
source venv_emotion/bin/activate

# Mettre à jour pip
pip install --upgrade pip
```

### Étape 6 : Installation de PyTorch avec Support ROCm

```bash
# Installer PyTorch avec ROCm 5.7
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

**Versions spécifiques** (optionnel, si vous avez besoin d'une version précise) :
```bash
# PyTorch 2.1.0 avec ROCm 5.7 (exemple)
pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/rocm5.7
```

### Étape 7 : Vérification de PyTorch

Créer un script de test `test_pytorch_rocm.py` :

```python
import torch

print("=" * 60)
print("Test PyTorch + ROCm")
print("=" * 60)

# Version PyTorch
print(f"\nPyTorch version: {torch.__version__}")

# Vérifier CUDA (ROCm apparaît comme CUDA)
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Test calcul GPU
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("\n✅ Test de calcul GPU réussi!")
else:
    print("\n❌ GPU non détecté!")
```

Exécuter :
```bash
python test_pytorch_rocm.py
```

**Sortie attendue :**
```
============================================================
Test PyTorch + ROCm
============================================================

PyTorch version: 2.x.x+rocm5.7
CUDA available: True
CUDA device count: 1
CUDA device name: AMD Radeon RX 7900 XT
CUDA device memory: 20.00 GB

✅ Test de calcul GPU réussi!
```

### Étape 8 : Installation des Dépendances du Projet

```bash
# Dans le répertoire du projet
cd ~/path/to/projet

# Activer environnement si nécessaire
source venv_emotion/bin/activate

# Installer dépendances
pip install -r requirements.txt
```

---

## 🪟 Installation Windows (Preview)

### Prérequis Windows

- Windows 11 (version 22H2 ou plus récente)
- AMD Radeon RX 7900 XT avec drivers récents
- Python 3.10 ou 3.11

### Étape 1 : Installer le Driver AMD

1. Télécharger **AMD Software: Adrenalin Edition** depuis [AMD.com](https://www.amd.com/en/support)
2. Installer le driver complet
3. Redémarrer

### Étape 2 : Installer PyTorch avec DirectML (Alternative ROCm)

ROCm sur Windows est en preview. Alternative recommandée : **DirectML**

```powershell
# Créer environnement virtuel
python -m venv venv_emotion
venv_emotion\Scripts\activate

# Installer PyTorch avec DirectML
pip install torch-directml
pip install torchvision torchaudio
```

### Étape 3 : Test DirectML

```python
import torch
import torch_directml

dml = torch_directml.device()
print(f"DirectML device: {dml}")

x = torch.randn(1000, 1000).to(dml)
y = torch.randn(1000, 1000).to(dml)
z = torch.matmul(x, y)
print("✅ DirectML fonctionnel!")
```

### PyTorch ROCm sur Windows (Experimental)

⚠️ **Attention** : Support experimental, bugs possibles

```powershell
# Installer ROCm pour Windows (preview)
# Télécharger depuis: https://github.com/RadeonOpenCompute/ROCm/releases

# Installer PyTorch preview
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

---

## ✅ Vérification Complète

### Script de Vérification Complet

Utiliser le script fourni dans le projet :

```bash
python scripts/test_gpu.py
```

Ce script vérifie :
- ✅ Version PyTorch
- ✅ Détection GPU
- ✅ Mémoire VRAM
- ✅ Calculs matriciels
- ✅ Mixed Precision (FP16)

### Benchmarking Performance

```bash
# Tester performance du GPU
python -c "
import torch
import time

device = torch.device('cuda')
size = 8192

a = torch.randn(size, size).to(device)
b = torch.randn(size, size).to(device)

torch.cuda.synchronize()
start = time.time()

c = torch.matmul(a, b)
torch.cuda.synchronize()

elapsed = time.time() - start
tflops = (2 * size**3) / (elapsed * 1e12)

print(f'Temps: {elapsed:.4f}s')
print(f'Performance: {tflops:.2f} TFLOPS')
"
```

**Performance attendue AMD 7900 XT** : 30-50 TFLOPS (FP32)

---

## 🔧 Dépannage

### Problème : GPU non détecté par PyTorch

**Causes possibles :**
1. ROCm mal installé
2. Utilisateur pas dans les groupes `render` et `video`
3. Version incompatible PyTorch/ROCm

**Solutions :**

```bash
# 1. Vérifier ROCm
rocm-smi

# 2. Vérifier groupes
groups $LOGNAME
# Si render/video absents:
sudo usermod -a -G render,video $LOGNAME
# Puis se déconnecter/reconnecter

# 3. Réinstaller PyTorch
pip uninstall torch torchvision torchaudio
pip cache purge
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# 4. Vérifier variables d'environnement
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # Pour RDNA 3
python test_pytorch_rocm.py
```

### Problème : Erreur "HIP out of memory"

**Solution :**
```python
# Réduire batch size
batch_size = 32  # au lieu de 64

# Activer gradient checkpointing
torch.utils.checkpoint.checkpoint(...)

# Libérer cache
torch.cuda.empty_cache()
```

### Problème : Performance Lente

**Optimisations :**

```python
# 1. Activer TF32 (si supporté)
torch.backends.cudnn.allow_tf32 = True

# 2. Activer benchmark
torch.backends.cudnn.benchmark = True

# 3. Utiliser Mixed Precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
```

### Problème : Driver Conflicts

Si vous avez des problèmes après mise à jour driver :

```bash
# Purger anciens packages
sudo apt purge rocm-* amdgpu-*
sudo apt autoremove
sudo apt clean

# Réinstaller proprement
sudo amdgpu-install --usecase=rocm -y
sudo reboot
```

---

## ⚡ Optimisations Avancées

### 1. Variables d'Environnement ROCm

Ajouter à `~/.bashrc` :

```bash
# ROCm paths
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# Pour RDNA 3 (7900 XT)
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Optimisations
export HSA_ENABLE_SDMA=0
export GPU_MAX_HW_QUEUES=4
```

Recharger :
```bash
source ~/.bashrc
```

### 2. Tuning GPU avec rocm-smi

```bash
# Augmenter limite de puissance (si refroidissement adéquat)
sudo rocm-smi --setpoweroverdrive 10  # +10%

# Définir profil performance
sudo rocm-smi --setperflevel high

# Monitorer
watch -n 1 rocm-smi
```

### 3. Configuration PyTorch Optimale

```python
import torch

# Configuration pour AMD 7900 XT
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

# Mixed Precision
torch.set_float32_matmul_precision('high')

# Batch size optimal (à ajuster)
batch_size = 64  # Maximiser utilisation 20 GB VRAM
```

### 4. Monitoring en Temps Réel

```bash
# Terminal 1 : Monitoring GPU
watch -n 0.5 rocm-smi

# Terminal 2 : Monitoring système
htop

# Alternative : radeontop
sudo apt install radeontop
sudo radeontop
```

---

## 📚 Ressources Utiles

### Documentation Officielle
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm](https://pytorch.org/get-started/locally/)
- [AMD GPUs for AI](https://www.amd.com/en/graphics/servers-radeon-instinct-mi)

### Communauté
- [ROCm GitHub](https://github.com/RadeonOpenCompute/ROCm)
- [PyTorch Forums - AMD](https://discuss.pytorch.org/)
- [r/ROCM](https://www.reddit.com/r/ROCm/)

### Tutoriels
- [Getting Started with ROCm](https://github.com/RadeonOpenCompute/ROCm)
- [PyTorch AMD Tutorial](https://pytorch.org/blog/amd-extends-support-for-pt-ml/)

---

## 🎯 Checklist Finale

Avant de commencer le projet, vérifiez :

- [ ] ROCm installé (`rocm-smi` fonctionne)
- [ ] GPU détecté (`rocminfo` montre 7900 XT)
- [ ] Utilisateur dans groupes render/video
- [ ] PyTorch installé avec ROCm
- [ ] GPU détecté par PyTorch (`torch.cuda.is_available() == True`)
- [ ] Test de calcul GPU réussi
- [ ] Mixed Precision fonctionne
- [ ] Dépendances du projet installées
- [ ] Script `test_gpu.py` passe tous les tests

Si tous les points sont ✅, vous êtes prêt !

---

**Dernière mise à jour** : 2025-10-25
**Version ROCm** : 5.7+
**GPU testé** : AMD Radeon RX 7900 XT
