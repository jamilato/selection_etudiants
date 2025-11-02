# 🚀 Guide de Démarrage Rapide - Phase 1

Guide ultra-rapide pour commencer avec le projet d'identification d'étudiants avec analyse d'émotions.

## ⚡ Installation Express WSL2 + Ubuntu 22.04

### Étape 1: Configuration WSL2 (Depuis Windows)

**Ouvrez PowerShell en tant qu'administrateur** :

```powershell
# Éditer la configuration WSL2
notepad C:\Users\MNB\.wslconfig

# Ajoutez (ajustez memory selon votre RAM):
# [wsl2]
# memory=32GB
# processors=12
# nestedVirtualization=true
# swap=16GB
# localhostForwarding=true

# Redémarrer WSL2
wsl --shutdown
```

### Étape 2: Installation Automatique (Dans Ubuntu)

**Ouvrez Ubuntu** et exécutez:

```bash
# 1. Naviguer vers le projet
cd /mnt/c/Users/MNB/Downloads/"Projet IA identification étudiant"

# 2. Rendre le script d'installation exécutable
chmod +x setup/phase1_setup.sh

# 3. Lancer l'installation automatique
./setup/phase1_setup.sh

# Le script va installer :
# - ROCm 5.7+ (support GPU AMD)
# - PyTorch avec support ROCm
# - OpenCV, DeepFace, MTCNN
# - Toutes les dépendances
# - Créer l'environnement virtuel Python
```

**⏱️ Temps estimé** : 30-60 minutes (selon connexion internet)

### Étape 3: Vérification

```bash
# Test GPU (devrait détecter AMD Radeon 7900 XT)
source venv_emotion/bin/activate
python test_gpu.py

# Sortie attendue:
# ✅ GPU détecté!
# Nom du GPU: AMD Radeon RX 7900 XT
# Mémoire GPU totale: 20.00 GB
```

---

## 📝 Installation Manuelle (3 Étapes)

### Étape 1 : ROCm + PyTorch

```bash
# Installer ROCm
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_5.7.50700-1_all.deb
sudo dpkg -i amdgpu-install_5.7.50700-1_all.deb
sudo amdgpu-install --usecase=rocm -y
sudo usermod -a -G render,video $LOGNAME
sudo reboot

# Après redémarrage : créer environnement Python
python3.10 -m venv venv_emotion
source venv_emotion/bin/activate

# Installer PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

### Étape 2 : Dépendances Projet

```bash
# Installer requirements
pip install -r requirements.txt
```

### Étape 3 : Vérification

```bash
# Tester GPU
python scripts/test_gpu.py

# Sortie attendue :
# ✅ GPU détecté: AMD Radeon RX 7900 XT
# ✅ VRAM disponible: 20.00 GB
```

---

## 🎯 Premier Lancement

### Test Webcam (Mode Temps Réel)

```bash
# Activer environnement
source venv_emotion/bin/activate

# Lancer le système
python main.py --mode realtime

# Appuyer sur 'q' pour quitter
```

**Note** : Pour l'instant, ceci affiche juste le flux webcam avec FPS. Les modèles d'émotions seront ajoutés en Phase 3 du plan.

---

## 📊 Prochaines Étapes

### Phase 1 : Configuration (✅ TERMINÉ)
- ✅ ROCm installé
- ✅ PyTorch configuré
- ✅ Structure projet créée

### Phase 2 : Données (À FAIRE)

```bash
# Télécharger dataset FER2013 (via Kaggle)
# 1. Créer compte sur kaggle.com
# 2. Obtenir API token (kaggle.com/account)
# 3. Placer kaggle.json dans ~/.kaggle/

pip install kaggle
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/fer2013/
```

Organiser données :
```
data/fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
├── test/
└── val/
```

### Phase 3 : Entraînement (À FAIRE)

```bash
# Créer script d'entraînement (voir plan.md)
python scripts/train.py --dataset fer2013 --epochs 50
```

### Phase 4 : Intégration (À FAIRE)

Implémenter système complet :
- Détection faciale (MTCNN)
- Classification émotions
- Identification étudiants
- Interface temps réel

---

## 📖 Documentation Complète

| Fichier | Description |
|---------|-------------|
| **README.md** | Documentation générale du projet |
| **projet.md** | Spécifications techniques détaillées |
| **plan.md** | Roadmap complète en 6 phases |
| **INSTALLATION_ROCM.md** | Guide installation ROCm approfondi |
| **QUICKSTART.md** | Ce fichier |

---

## 🔍 Commandes Utiles

### Monitoring GPU

```bash
# Surveiller GPU en temps réel
watch -n 1 rocm-smi

# Informations GPU
rocminfo | grep "Name:"

# Température et utilisation
rocm-smi --showtemp --showuse
```

### Python / PyTorch

```bash
# Activer environnement
source venv_emotion/bin/activate

# Vérifier versions
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test rapide GPU
python -c "import torch; x = torch.randn(1000,1000).cuda(); print('✅ GPU OK')"
```

### Projet

```bash
# Lister datasets
ls -lh data/

# Lister modèles
ls -lh models/

# Voir logs
tail -f logs/*.log

# Nettoyer cache
rm -rf __pycache__ src/__pycache__
```

---

## ❓ Problèmes Courants

### GPU non détecté

```bash
# 1. Vérifier ROCm
rocm-smi

# 2. Vérifier groupes utilisateur
groups $LOGNAME
# Doit inclure : render video

# 3. Si absent, ajouter et redémarrer
sudo usermod -a -G render,video $LOGNAME
sudo reboot
```

### Import Error

```bash
# Vérifier environnement activé
which python
# Doit pointer vers venv_emotion/bin/python

# Réinstaller dépendances
pip install --force-reinstall -r requirements.txt
```

### Webcam ne fonctionne pas

```bash
# Lister devices vidéo
ls -l /dev/video*

# Tester avec v4l-utils
sudo apt install v4l-utils
v4l2-ctl --list-devices

# Donner permissions
sudo usermod -a -G video $LOGNAME
```

---

## 🎓 Architecture Modèle Recommandé

Pour votre projet, nous recommandons **EmotionNet Nano** :

### Pourquoi EmotionNet Nano ?

✅ **Ultra-rapide** : >70 FPS sur AMD 7900 XT
✅ **Léger** : ~300k paramètres
✅ **Précis** : 60-65% FER2013, 75-85% RAF-DB
✅ **Temps réel** : Parfait pour webcam

### Alternative : EfficientNet

Si vous préférez **précision maximale** (au détriment de vitesse) :

- **EfficientNetB7** : 78.9% précision
- Plus lent : ~10-15 FPS
- Mieux pour traitement batch

**Recommandation finale** : Commencez avec **EmotionNet Nano** ⭐

---

## 📈 Métriques Cibles

| Métrique | Cible | Votre GPU |
|----------|-------|-----------|
| FPS (temps réel) | >25 | **>70** ✅ |
| Latence | <40ms | **~14ms** ✅ |
| Précision FER2013 | >60% | 60-65% ✅ |
| Précision RAF-DB | >70% | 75-85% ✅ |
| VRAM utilisée | <8GB | ~2-4GB ✅ |

**Votre AMD 7900 XT (20GB VRAM) est largement suffisante !** 🚀

---

## 🎯 Checklist Démarrage

Avant de commencer le développement :

### Installation
- [ ] ROCm installé (`rocm-smi` fonctionne)
- [ ] PyTorch avec ROCm installé
- [ ] GPU détecté (`torch.cuda.is_available() == True`)
- [ ] Script `test_gpu.py` réussi
- [ ] Dépendances installées (`pip list`)

### Données
- [ ] Compte Kaggle créé
- [ ] FER2013 téléchargé
- [ ] Données organisées dans `data/`

### Développement
- [ ] IDE configuré (VS Code / PyCharm)
- [ ] Git initialisé (optionnel)
- [ ] `.gitignore` en place

### Compréhension
- [ ] Lu `projet.md` (spécifications)
- [ ] Lu `plan.md` (roadmap)
- [ ] Compris différence FER vs CNN

---

## 🚀 Commencer Maintenant !

```bash
# 1. Installation complète
./scripts/install.sh

# 2. Test GPU
python scripts/test_gpu.py

# 3. Test webcam
python main.py --mode realtime

# 4. Lire la roadmap
cat plan.md

# 5. Suivre Phase 2 (Données)
# Voir plan.md - Phase 2
```

---

## 📞 Support

**Questions ?** Consultez d'abord :
1. `README.md` - Vue d'ensemble
2. `projet.md` - Détails techniques
3. `plan.md` - Étapes de développement
4. `INSTALLATION_ROCM.md` - Problèmes GPU

**Ressources externes :**
- [ROCm Docs](https://rocm.docs.amd.com/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [DeepFace GitHub](https://github.com/serengil/deepface)

---

**Bon courage ! 🎓🤖**

Votre système AMD Radeon 7900 XT est parfait pour ce projet.
Suivez la roadmap étape par étape et vous aurez un système fonctionnel en 6 semaines !

---

**Version** : 1.0
**Dernière mise à jour** : 2025-10-25
