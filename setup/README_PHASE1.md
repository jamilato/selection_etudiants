# Phase 1: Configuration de l'Environnement

## Guide d'Installation - WSL2 + Ubuntu 22.04 + AMD Radeon 7900 XT

---

## 📋 Prérequis

- ✅ Windows 11 ou Windows 10 (build 21H2+)
- ✅ WSL2 installé avec Ubuntu 22.04
- ✅ Python 3.10+ installé dans Ubuntu
- ✅ Drivers AMD installés sur Windows
- ✅ Au moins 50 GB d'espace disque libre
- ✅ Connexion Internet stable

---

## 🚀 Instructions d'Installation

### Étape 0: Configuration WSL2 (IMPORTANT!)

Avant de commencer, optimisez WSL2 pour votre GPU:

1. **Depuis Windows**, ouvrez le fichier `.wslconfig`:
   ```powershell
   notepad C:\Users\MNB\.wslconfig
   ```

2. Copiez le contenu du fichier `wslconfig_template.txt` et ajustez selon votre RAM

3. **Redémarrez WSL2** depuis PowerShell (Windows):
   ```powershell
   wsl --shutdown
   ```

4. Relancez Ubuntu

### Étape 1: Ouvrir Ubuntu dans WSL2

1. Ouvrez **Windows Terminal** ou **Ubuntu** depuis le menu Démarrer
2. Vous devriez voir un terminal Ubuntu

### Étape 2: Naviguer vers le dossier du projet

```bash
# Depuis WSL, naviguez vers le dossier Windows
cd /mnt/c/Users/MNB/Downloads/"Projet IA identification étudiant"
```

### Étape 3: Rendre le script exécutable

```bash
chmod +x setup/phase1_setup.sh
```

### Étape 4: Exécuter le script d'installation

```bash
./setup/phase1_setup.sh
```

Le script va:
- ✅ Vérifier votre environnement Ubuntu
- ✅ Mettre à jour le système
- ✅ Installer les outils de base
- ⚠️  **Demander si vous voulez installer ROCm** (recommandé: OUI)
- ✅ Créer l'environnement virtuel Python
- ⚠️  **Demander si vous voulez PyTorch avec ROCm** (recommandé: OUI)
- ✅ Installer toutes les bibliothèques
- ✅ Tester la configuration GPU

**Temps d'installation estimé**: 30-60 minutes (selon vitesse Internet)

---

## ⚠️ IMPORTANT: Support GPU dans WSL2

### Limitations connues

Le support ROCm dans WSL2 est **expérimental** et peut présenter des limitations:

- ✅ **Fonctionne bien**: Entraînement PyTorch, inférence, calculs de base
- ⚠️  **Peut être limité**: Certaines fonctionnalités avancées ROCm
- ❌ **Ne fonctionne pas**: Certains outils de profilage GPU

### Si le GPU n'est pas détecté

Si le script de test indique que le GPU n'est pas disponible:

1. **Vérifiez les drivers Windows**:
   - Ouvrez le **Gestionnaire de périphériques**
   - Vérifiez que "AMD Radeon RX 7900 XT" apparaît sans erreur

2. **Vérifiez le support GPU WSL2**:
   ```powershell
   # Depuis PowerShell Windows
   wsl --update
   wsl --version
   ```

3. **Redémarrez complètement**:
   ```powershell
   # Depuis PowerShell Windows
   wsl --shutdown
   # Attendez 10 secondes, puis relancez Ubuntu
   ```

4. **Consultez les logs ROCm**:
   ```bash
   # Depuis Ubuntu WSL
   rocm-smi  # Devrait montrer votre GPU
   rocminfo  # Informations détaillées
   ```

### Alternative: Mode CPU

Si le GPU ne fonctionne pas dans WSL2, vous pouvez:
- ✅ Continuer en mode CPU pour le développement
- ✅ Utiliser un dual-boot Ubuntu pour l'entraînement final
- ✅ Utiliser Google Colab avec GPU cloud pour l'entraînement

---

## 🔍 Vérification de l'Installation

Après l'installation, vérifiez que tout fonctionne:

### 1. Activer l'environnement virtuel

```bash
source venv_emotion/bin/activate
```

Vous devriez voir `(venv_emotion)` avant votre prompt.

### 2. Vérifier PyTorch

```bash
python test_gpu.py
```

**Résultat attendu**:
```
======================================================================
Test de Configuration GPU - PyTorch
======================================================================

PyTorch version: 2.x.x+rocm5.7

CUDA/ROCm disponible: True
Nombre de GPU détectés: 1

✅ GPU détecté!

Nom du GPU: AMD Radeon RX 7900 XT
Mémoire GPU totale: 20.00 GB

Test de calcul sur GPU...
✅ Calcul matriciel sur GPU réussi!
```

### 3. Vérifier les bibliothèques

```bash
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import deepface; print('DeepFace: OK')"
python -c "import mtcnn; print('MTCNN: OK')"
```

---

## 📦 Fichiers Créés

Après l'installation, vous aurez:

```
Projet IA identification étudiant/
├── setup/
│   ├── phase1_setup.sh          ← Script d'installation
│   ├── README_PHASE1.md         ← Ce fichier
│   └── wslconfig_template.txt   ← Config WSL2
├── venv_emotion/                ← Environnement virtuel Python
├── requirements.txt             ← Liste des dépendances
├── test_gpu.py                  ← Script de test GPU
├── projet.md                    ← Documentation projet
└── plan.md                      ← Roadmap complète
```

---

## 🛠️ Commandes Utiles

### Activer l'environnement virtuel
```bash
source venv_emotion/bin/activate
```

### Désactiver l'environnement virtuel
```bash
deactivate
```

### Voir les paquets installés
```bash
pip list
```

### Mettre à jour un paquet
```bash
pip install --upgrade nom_du_paquet
```

### Redémarrer WSL2 (depuis Windows PowerShell)
```powershell
wsl --shutdown
```

### Vérifier la RAM/CPU allouée à WSL
```bash
free -h    # RAM
nproc      # CPUs
```

---

## ❓ Dépannage

### Problème: "Permission denied"
```bash
chmod +x setup/phase1_setup.sh
```

### Problème: "apt-get: command not found"
Assurez-vous d'être dans Ubuntu WSL, pas dans PowerShell Windows.

### Problème: "pip: command not found"
```bash
sudo apt install python3-pip
```

### Problème: ROCm installation échoue
C'est normal sur WSL2. Vous pouvez:
1. Continuer sans ROCm (mode CPU)
2. Utiliser Ubuntu natif (dual-boot)

### Problème: GPU non détecté mais ROCm installé
1. Redémarrez WSL: `wsl --shutdown` depuis Windows
2. Vérifiez `.wslconfig` contient `nestedVirtualization=true`
3. Mettez à jour WSL: `wsl --update`

---

## 📚 Ressources

- [Documentation WSL2](https://learn.microsoft.com/fr-fr/windows/wsl/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm](https://pytorch.org/get-started/locally/)
- [DeepFace GitHub](https://github.com/serengil/deepface)

---

## ✅ Critères de Réussite Phase 1

Avant de passer à la Phase 2, vérifiez:

- ✅ Ubuntu 22.04 fonctionne dans WSL2
- ✅ Python 3.10+ disponible
- ✅ Environnement virtuel `venv_emotion` créé
- ✅ PyTorch installé (avec ou sans ROCm)
- ✅ Toutes les bibliothèques installées (requirements.txt)
- ✅ Script `test_gpu.py` s'exécute sans erreur
- ✅ (Optionnel) GPU détecté par PyTorch

**Si tous les critères sont ✅, vous êtes prêt pour la Phase 2!**

---

## ➡️ Prochaine Étape

Une fois la Phase 1 terminée, passez à:
**Phase 2: Préparation des Données**

Voir `plan.md` (lignes 102-210) pour les détails.

---

**Dernière mise à jour**: 2025-10-25
**Temps estimé Phase 1**: 3-5 heures
