# Guide Rapide : ROCm 6.4 sur WSL2 pour RX 7900 XT

Guide condensé pour installer et utiliser ROCm sur WSL2 avec AMD Radeon RX 7900 XT.

---

## ⚡ Installation en 5 Minutes

### 📋 Prérequis Obligatoires

**Sur Windows :**
1. **Windows 11** (requis)
2. **Driver AMD Adrenalin 24.6.1+** installé
   - Télécharger : https://www.amd.com/en/support
   - Choisir : Radeon RX 7900 XT → Windows 11
3. **WSL2** avec **Ubuntu 22.04** ou **24.04**

**Vérification rapide (PowerShell) :**
```powershell
# Vérifier Windows
winver  # Doit afficher Windows 11

# Vérifier WSL2
wsl --version
wsl --list --verbose  # Ubuntu doit être VERSION 2

# Vérifier driver AMD
Get-WmiObject Win32_VideoController | Select-Object Name, DriverVersion
```

### 🔧 Configuration .wslconfig

**Créer `C:\Users\MNB\.wslconfig` :**

```ini
[wsl2]
memory=16GB
processors=8
swap=8GB
guiApplications=true
nestedVirtualization=true

[experimental]
autoMemoryReclaim=gradual
sparseVhd=true
```

**Puis redémarrer WSL :**
```powershell
wsl --shutdown
```

### 🚀 Installation Automatique

```bash
# 1. Démarrer WSL
wsl -d Ubuntu-22.04

# 2. Naviguer vers le projet
cd "/mnt/c/Users/MNB/Downloads/Projet IA identification étudiant"

# 3. Exécuter le script
bash install_wsl_rocm.sh
```

**Durée estimée :** 10-15 minutes

---

## 🧪 Vérification Post-Installation

### Test 1 : ROCm Info

```bash
# Vérifier l'installation ROCm
rocminfo | grep gfx1100

# Devrait afficher :
#   Name:                    gfx1100
#   Marketing Name:          AMD Radeon RX 7900 XT
```

### Test 2 : ROCm SMI

```bash
rocm-smi

# Devrait afficher :
# ======================= ROCm System Management Interface =======================
# GPU  Temp   AvgPwr  SCLK    MCLK     Fan  Perf  PwrCap  VRAM%  GPU%
# 0    30.0c  15.0W   800Mhz  1000Mhz  0%   auto  355.0W  0%     0%
```

### Test 3 : PyTorch + GPU

```bash
python3 << EOF
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test calcul
x = torch.randn(1000, 1000).cuda()
y = torch.matmul(x, x)
print("✓ GPU Test OK!")
EOF
```

### Test 4 : Script du Projet

```bash
python scripts/test_gpu.py
```

---

## ❌ Si GPU Non Détecté

### Solution 1 : Redémarrer WSL

```powershell
# Dans PowerShell (Windows)
wsl --shutdown

# Attendre 10 secondes, puis rouvrir
wsl -d Ubuntu-22.04
```

Puis retester :
```bash
rocminfo | grep gfx1100
rocm-smi
```

### Solution 2 : Vérifier Variables d'Environnement

```bash
echo $HSA_OVERRIDE_GFX_VERSION  # Doit afficher: 11.0.0
echo $PATH | grep rocm           # Doit contenir /opt/rocm/bin
```

Si vide, recharger :
```bash
source ~/.bashrc
```

### Solution 3 : Vérifier Groupes Utilisateur

```bash
groups

# Doit contenir: render video
```

Si manquant, se reconnecter :
```bash
exit  # Quitter WSL
wsl -d Ubuntu-22.04  # Rouvrir
```

### Solution 4 : Réinstaller ROCm

```bash
# Nettoyer
sudo amdgpu-install --uninstall

# Réinstaller
sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms
```

---

## 🔍 Diagnostic Rapide

### Commandes de Debug

```bash
# 1. Version ROCm
/opt/rocm/bin/rocminfo --version

# 2. Liste des agents
rocminfo | grep -A5 "Agent"

# 3. Vérifier HSA
echo $HSA_OVERRIDE_GFX_VERSION

# 4. Packages installés
dpkg -l | grep rocm

# 5. Logs système
dmesg | grep -i amd
dmesg | grep -i rocm
```

### Problèmes Courants

| Symptôme | Cause | Solution |
|----------|-------|----------|
| `rocminfo` ne détecte pas le GPU | WSL pas redémarré | `wsl --shutdown` |
| `HSA_OVERRIDE_GFX_VERSION` vide | .bashrc pas chargé | `source ~/.bashrc` |
| `torch.cuda.is_available()` = False | PyTorch CPU installé | Réinstaller PyTorch ROCm |
| Erreur "no such file or directory" pour `/opt/rocm` | ROCm pas installé | Relancer `install_wsl_rocm.sh` |

---

## 🎯 Utilisation avec le Projet

### Entraînement sur GPU

```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Entraîner avec GPU
python scripts/train.py

# Le GPU sera utilisé automatiquement si détecté
```

### Modes d'Exécution

```bash
# Mode temps réel (webcam)
python main.py --mode realtime

# Mode vidéo
python main.py --mode video --input video.mp4

# Mode image
python main.py --mode image --input image.jpg
```

### Vérifier Utilisation GPU Pendant Entraînement

**Terminal 1 :**
```bash
python scripts/train.py
```

**Terminal 2 (nouveau terminal WSL) :**
```bash
watch -n 1 rocm-smi
```

Vous verrez l'utilisation GPU en temps réel (température, puissance, utilisation VRAM).

---

## 📊 Performances Attendues

### RX 7900 XT + ROCm 6.4 sur WSL2

**Entraînement EmotionNet Nano :**
- **FPS** : 500-800 images/sec (batch 64)
- **VRAM** : ~4-6 GB utilisés
- **Temps entraînement** : ~2-3 heures (100 epochs, FER2013)

**Inférence Temps Réel :**
- **FPS** : 150-200 FPS (avec preprocessing)
- **Latence** : ~5-8ms par frame
- **VRAM** : ~2-3 GB

**Comparaison CPU vs GPU :**
- **CPU (i7-12700K)** : ~20 images/sec → 24-48h entraînement
- **GPU (RX 7900 XT)** : ~600 images/sec → 2-3h entraînement
- **Speedup** : ~30x plus rapide

---

## 🔗 Ressources Officielles

- **AMD ROCm Docs** : https://rocm.docs.amd.com/
- **ROCm WSL Guide** : https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html
- **PyTorch ROCm** : https://pytorch.org/get-started/locally/
- **AMD Support** : https://www.amd.com/en/support

---

## 📞 Support

### Si Problèmes Persistent

1. **Vérifier les logs d'installation :**
   ```bash
   cat rocm_install_*.log
   ```

2. **Exécuter le script de vérification :**
   ```bash
   bash verify_installation.sh
   ```

3. **Consulter les issues GitHub :**
   - ROCm : https://github.com/ROCm/ROCm/issues
   - PyTorch : https://github.com/pytorch/pytorch/issues

4. **Forum AMD :**
   - https://community.amd.com/

---

## ⚡ Checklist Complète

- [ ] Windows 11 installé
- [ ] Driver AMD Adrenalin 24.6.1+ installé sur Windows
- [ ] WSL2 avec Ubuntu 22.04 ou 24.04
- [ ] .wslconfig configuré
- [ ] WSL redémarré après .wslconfig
- [ ] Script `install_wsl_rocm.sh` exécuté
- [ ] WSL redémarré après installation ROCm
- [ ] `rocminfo | grep gfx1100` détecte le GPU
- [ ] `rocm-smi` affiche le GPU
- [ ] `torch.cuda.is_available()` retourne `True`
- [ ] Test GPU réussi
- [ ] Prêt à entraîner !

---

## 🚀 Prochaines Étapes

Une fois ROCm fonctionnel :

1. **Télécharger les données** (si pas fait) :
   ```bash
   python scripts/download_datasets.py
   ```

2. **Préparer les données** :
   ```bash
   python scripts/prepare_data.py --all
   ```

3. **Entraîner le modèle** :
   ```bash
   python scripts/train.py
   ```

4. **Tester le système** :
   ```bash
   python main.py --mode realtime
   ```

**Bon entraînement ! 🎉**
