# 🎯 Phase 1 - Instructions Complètes

## ✅ Ce qui a été préparé pour vous

J'ai créé tous les scripts et documents nécessaires pour la Phase 1 du projet :

### 📁 Nouveaux Fichiers Créés

1. **`setup/phase1_setup.sh`** - Script d'installation automatique complet pour WSL2
2. **`setup/verify_installation.sh`** - Script de vérification de l'installation
3. **`setup/README_PHASE1.md`** - Guide détaillé avec dépannage
4. **`setup/wslconfig_template.txt`** - Configuration optimale WSL2
5. **`QUICKSTART.md`** (mis à jour) - Guide de démarrage rapide pour WSL2
6. **`INDEX.md`** (mis à jour) - Index complet de tous les fichiers

---

## 🚀 Prochaines Étapes - À FAIRE MAINTENANT

### Étape 1: Configurer WSL2 (5 minutes)

**Sur Windows, ouvrez PowerShell en tant qu'administrateur** et exécutez:

```powershell
# Éditer la configuration WSL2
notepad C:\Users\MNB\.wslconfig
```

**Copiez ce contenu** (vous pouvez aussi ouvrir `setup/wslconfig_template.txt` dans Notepad et copier):

```ini
[wsl2]
memory=32GB
processors=12
nestedVirtualization=true
swap=16GB
swapFile=C:\\Users\\MNB\\AppData\\Local\\Temp\\wsl-swap.vhdx
localhostForwarding=true
```

**Note**: Ajustez `memory` selon votre RAM totale (recommandé: 50-75% de votre RAM)

**Sauvegardez le fichier**, puis redémarrez WSL2:

```powershell
wsl --shutdown
```

**Attendez 10 secondes**, puis relancez Ubuntu.

---

### Étape 2: Lancer l'Installation Automatique (30-60 minutes)

**Ouvrez Ubuntu** (depuis le menu Démarrer ou Windows Terminal).

**Naviguez vers le projet**:

```bash
cd /mnt/c/Users/MNB/Downloads/"Projet IA identification étudiant"
```

**Vérifiez que vous êtes au bon endroit**:

```bash
ls -la
# Vous devriez voir: plan.md, projet.md, setup/, etc.
```

**Rendez le script exécutable**:

```bash
chmod +x setup/phase1_setup.sh
```

**Lancez l'installation**:

```bash
./setup/phase1_setup.sh
```

### ⚠️ Important pendant l'installation

Le script vous posera **2 questions** :

1. **"Voulez-vous continuer avec l'installation de ROCm?"**
   - **Répondez**: `o` (Oui) ✅
   - ROCm est nécessaire pour utiliser votre GPU AMD

2. **"Installer PyTorch avec support ROCm?"**
   - **Répondez**: `o` (Oui) ✅
   - Cela installera PyTorch optimisé pour votre AMD 7900 XT

**Temps d'installation**: 30-60 minutes selon votre connexion Internet

**Pendant ce temps**, vous pouvez:
- ☕ Prendre un café
- 📖 Lire `projet.md` pour comprendre les détails techniques
- 📋 Consulter `plan.md` pour voir les 6 phases du projet

---

### Étape 3: Vérifier l'Installation (5 minutes)

Une fois l'installation terminée, le script exécutera automatiquement un test GPU.

**Résultats attendus**:

#### ✅ Cas 1: GPU détecté (IDÉAL)

```
✅ GPU détecté!
Nom du GPU: AMD Radeon RX 7900 XT
Mémoire GPU totale: 20.00 GB
✅ Calcul matriciel sur GPU réussi!
```

👉 **Parfait! Vous êtes prêt pour la Phase 2!**

#### ⚠️ Cas 2: GPU NON détecté (NORMAL sur WSL2)

```
⚠️ Aucun GPU détecté - PyTorch s'exécutera en mode CPU
```

**Ne paniquez pas!** C'est normal sur WSL2 car le support ROCm est expérimental.

**Essayez ceci**:

1. **Redémarrez complètement WSL**:
   ```powershell
   # Depuis Windows PowerShell
   wsl --shutdown
   ```
   Attendez 10 secondes, relancez Ubuntu

2. **Retestez**:
   ```bash
   source venv_emotion/bin/activate
   python test_gpu.py
   ```

3. **Si toujours pas de GPU**, vous avez **2 options**:

   **Option A** (Recommandée pour le moment):
   - ✅ Continuez en mode CPU pour développer
   - ✅ Le développement et les tests fonctionneront
   - ⚠️ L'entraînement sera plus lent (mais possible)

   **Option B** (Pour plus tard):
   - Installer Ubuntu en dual-boot pour l'entraînement final
   - Support ROCm natif = performances maximales

---

### Étape 4: Vérification Manuelle (Optionnelle)

Pour une vérification complète, exécutez:

```bash
chmod +x setup/verify_installation.sh
./setup/verify_installation.sh
```

Ce script vérifiera:
- ✅ Version Ubuntu
- ✅ Outils de base installés
- ✅ ROCm (si disponible)
- ✅ Environnement virtuel Python
- ✅ Toutes les bibliothèques
- ✅ PyTorch et GPU

---

## 📋 Checklist de Réussite Phase 1

Avant de passer à la Phase 2, vérifiez:

- [ ] WSL2 configuré avec `.wslconfig`
- [ ] Script `phase1_setup.sh` exécuté sans erreur critique
- [ ] Environnement virtuel `venv_emotion` créé
- [ ] PyTorch installé (vérifier avec `pip list | grep torch`)
- [ ] OpenCV installé (vérifier avec `pip list | grep opencv`)
- [ ] DeepFace installé (vérifier avec `pip list | grep deepface`)
- [ ] GPU détecté OU accepté de travailler en mode CPU
- [ ] Fichier `test_gpu.py` créé et fonctionne
- [ ] Fichier `requirements.txt` créé

**Si tous les critères sont cochés ✅, félicitations!**

---

## 📖 Documentation Complète

Pour plus de détails, consultez:

| Fichier | Contenu |
|---------|---------|
| **`setup/README_PHASE1.md`** | Guide complet Phase 1 avec dépannage détaillé |
| **`QUICKSTART.md`** | Guide de démarrage rapide |
| **`plan.md`** | Roadmap complète (6 phases) |
| **`projet.md`** | Spécifications techniques |
| **`INDEX.md`** | Index de tous les fichiers |

---

## 🔧 Commandes Utiles

### Activer l'environnement virtuel (à faire CHAQUE fois)

```bash
cd /mnt/c/Users/MNB/Downloads/"Projet IA identification étudiant"
source venv_emotion/bin/activate
```

Vous verrez `(venv_emotion)` devant votre prompt.

### Tester PyTorch

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'GPU: {torch.cuda.is_available()}')"
```

### Tester GPU (script complet)

```bash
python test_gpu.py
```

### Voir les bibliothèques installées

```bash
pip list
```

### Désactiver l'environnement virtuel

```bash
deactivate
```

---

## 🆘 Dépannage Rapide

### Problème: "Permission denied" sur le script

**Solution**:
```bash
chmod +x setup/phase1_setup.sh
```

### Problème: GPU non détecté après installation

**Solution 1**: Redémarrage WSL
```powershell
# Depuis Windows PowerShell
wsl --shutdown
# Attendre 10 secondes
# Relancer Ubuntu
```

**Solution 2**: Vérifier ROCm
```bash
rocm-smi  # Devrait afficher votre GPU
```

**Solution 3**: Continuer en mode CPU
Le projet fonctionnera, l'entraînement sera juste plus lent.

### Problème: Erreur d'installation ROCm

C'est **normal sur WSL2**. Le support ROCm sur WSL2 est expérimental.

**Solution**: Installez PyTorch sans ROCm (version CPU):
```bash
pip install torch torchvision torchaudio
```

---

## ➡️ Après la Phase 1: Phase 2

Une fois la Phase 1 terminée, vous passerez à:

**Phase 2: Préparation des Données** (Semaine 2)

Tâches principales:
1. Télécharger FER2013 (dataset d'émotions)
2. Télécharger RAF-DB (dataset amélioré)
3. Organiser les données
4. Analyse exploratoire (EDA)
5. Créer DataLoaders PyTorch

Consultez `plan.md` lignes 102-210 pour les détails.

---

## 🎯 Résumé - 3 Commandes Essentielles

Pour installer tout automatiquement:

```bash
# 1. Naviguer vers le projet
cd /mnt/c/Users/MNB/Downloads/"Projet IA identification étudiant"

# 2. Lancer l'installation
chmod +x setup/phase1_setup.sh && ./setup/phase1_setup.sh

# 3. Vérifier (optionnel)
chmod +x setup/verify_installation.sh && ./setup/verify_installation.sh
```

---

## ✅ Vous êtes Prêt!

Tous les outils et scripts sont en place. Suivez simplement les étapes ci-dessus.

**Temps total estimé Phase 1**: 1-2 heures (incluant téléchargements)

**Bonne chance! 🚀**

N'hésitez pas à consulter la documentation si vous rencontrez des problèmes.

---

**Créé le**: 2025-10-25
**Pour**: Projet IA Identification Étudiants + Analyse Émotions
**GPU**: AMD Radeon RX 7900 XT (20 GB VRAM)
**OS**: WSL2 + Ubuntu 22.04
