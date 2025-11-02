# 🚀 Quick Start - Entraînement du Modèle

## Guide ultra-rapide pour démarrer l'entraînement

---

## ⚡ En 5 Commandes

```bash
# 1. Activer environnement (depuis WSL Ubuntu)
cd /mnt/c/Users/MNB/Downloads/"Projet IA identification étudiant"
source venv_emotion/bin/activate

# 2. Télécharger FER2013
python scripts/download_datasets.py --dataset fer2013

# 3. Préparer les données
python scripts/prepare_data.py --dataset data/fer2013 --all

# 4. Lancer l'entraînement
python scripts/train.py

# 5. Monitorer avec TensorBoard (terminal séparé)
tensorboard --logdir logs/tensorboard
```

**C'est tout! L'entraînement démarre automatiquement. ⏱️ Temps: ~2-3 heures**

---

## 📋 Prérequis

✅ Phase 1 complétée (voir `PHASE1_INSTRUCTIONS.md`)
✅ Environnement virtuel `venv_emotion` créé
✅ PyTorch avec ROCm installé (ou CPU)
✅ Compte Kaggle configuré (pour FER2013)

---

## 🔧 Configuration Rapide Kaggle

Si première fois avec Kaggle:

```bash
# 1. Installer Kaggle API
pip install kaggle

# 2. Créer ~/.kaggle/kaggle.json
# Aller sur kaggle.com → Account → Create New API Token
# Télécharger kaggle.json et placer dans ~/.kaggle/

# 3. Set permissions (Linux/WSL)
chmod 600 ~/.kaggle/kaggle.json
```

---

## 📊 Pendant l'Entraînement

### Ce qui se passe automatiquement:

✅ **Epoch 1-100** - Training avec augmentation
✅ **Early Stopping** - Arrête si pas d'amélioration (patience=15)
✅ **Model Checkpoint** - Sauvegarde meilleur modèle
✅ **TensorBoard** - Logs en temps réel
✅ **LR Scheduling** - Ajuste learning rate
✅ **Mixed Precision** - FP16 sur GPU AMD

### Fichiers générés:

```
checkpoints/
├── best_model_val_loss.pt       ← Meilleur modèle
└── checkpoint_epoch*.pt          ← Checkpoints intermédiaires

logs/
├── tensorboard/                  ← Logs TensorBoard
│   └── 20251025_*/
├── training.log                  ← Log texte
└── training_history.png          ← Graphique final

models/
└── emotion_model_scripted.pt     ← Modèle TorchScript (export)
```

---

## 📈 Monitorer l'Entraînement

### Option 1: Terminal

Les métriques s'affichent à chaque epoch:

```
Epoch 10/100 - Time: 85.23s
======================================================================
Train Metrics:
--------------------------------------------------
  loss           : 1.2345
  accuracy       : 0.6234
  f1_macro       : 0.6012
...
```

### Option 2: TensorBoard

```bash
# Dans un terminal séparé
tensorboard --logdir logs/tensorboard

# Ouvrir navigateur: http://localhost:6006
```

Vous verrez:
- 📉 Loss curves (train/val)
- 📈 Accuracy curves
- 🎯 F1-score
- 🔄 Learning rate

---

## ⚙️ Personnaliser l'Entraînement

### Modifier hyperparamètres (YAML)

Éditez `configs/train_config.yaml`:

```yaml
training:
  epochs: 50           # Réduire epochs

optimizer:
  adamw:
    lr: 0.0005         # Changer learning rate

callbacks:
  early_stopping:
    patience: 10       # Patience early stopping
```

### Override via CLI

```bash
# Changer epochs
python scripts/train.py --epochs 50

# Changer batch size
python scripts/train.py --batch-size 128

# Changer learning rate
python scripts/train.py --lr 0.0001

# Combiner plusieurs
python scripts/train.py --epochs 50 --batch-size 128 --lr 0.0001
```

---

## 🔄 Resume Training

Si l'entraînement s'arrête:

```bash
# Reprendre depuis le dernier checkpoint
python scripts/train.py --resume checkpoints/best_model_val_loss.pt
```

---

## 🎯 Métriques Attendues

### Après ~10 epochs:
- Train Loss: ~1.5
- Val Loss: ~1.6
- Val Accuracy: ~55-60%

### Après ~50 epochs:
- Train Loss: ~0.8
- Val Loss: ~1.0
- Val Accuracy: ~65-70%

### Après ~100 epochs (si pas early stopping):
- Train Loss: ~0.5
- Val Loss: ~0.9
- Val Accuracy: ~70-75%

---

## ⚠️ Problèmes Courants

### GPU non détecté

```python
# Dans le code, ça utilise automatiquement CPU si GPU absent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Impact**: Entraînement plus lent (~10x) mais fonctionne.

**Solution permanente**: Voir `PHASE1_INSTRUCTIONS.md` pour setup GPU.

### Out of Memory (CUDA OOM)

Réduire batch size dans `configs/data_config.yaml`:

```yaml
dataloader:
  batch_size: 32  # Au lieu de 64
```

Ou en CLI:
```bash
python scripts/train.py --batch-size 32
```

### Download échoue (Kaggle)

Vérifier:
1. Kaggle API installée: `pip list | grep kaggle`
2. Credentials configurées: `ls -la ~/.kaggle/kaggle.json`
3. Permissions OK: `chmod 600 ~/.kaggle/kaggle.json`

**Alternative**: Télécharger manuellement depuis [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

### num_workers error (WSL2)

Si erreur avec DataLoader workers:

```yaml
# configs/data_config.yaml
dataloader:
  num_workers: 0  # Désactiver multi-processing
```

---

## 📊 Résultats Attendus

### Fichiers générés après entraînement complet:

```
checkpoints/best_model_val_loss.pt       2.5 MB
logs/training_history.png                 120 KB
logs/tensorboard/20251025_*/events.*      5 MB
models/emotion_model_scripted.pt          2.8 MB
```

### Métriques finales (typiques):

```
Best validation loss: 0.8542
Best validation accuracy: 0.7123
Best validation F1: 0.6987

Per-class Accuracy:
  angry    : 0.6234
  disgust  : 0.5123  ← Difficile (peu d'exemples)
  fear     : 0.6789
  happy    : 0.8456  ← Plus facile
  sad      : 0.7012
  surprise : 0.7345
  neutral  : 0.7234
```

---

## 🎓 Après l'Entraînement

### 1. Analyser les résultats

```bash
# Voir graphiques
open logs/training_history.png

# TensorBoard
tensorboard --logdir logs/tensorboard
```

### 2. Tester le modèle

```python
# Dans Python ou Jupyter
import torch
from src.models.emotion_net import EmotionNetNano

# Charger modèle
model = EmotionNetNano(num_classes=7)
checkpoint = torch.load('checkpoints/best_model_val_loss.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Tester sur une image
# ... (voir notebooks pour exemples complets)
```

### 3. Fine-tuning sur RAF-DB (optionnel)

```bash
# 1. Télécharger RAF-DB (manuel)
# 2. Préparer
python scripts/prepare_data.py --dataset data/rafdb --all

# 3. Fine-tuner
# Modifier configs/train_config.yaml:
#   fine_tuning.enabled = true
#   fine_tuning.checkpoint_path = "checkpoints/best_model_val_loss.pt"

python scripts/train.py --config configs/train_config.yaml
```

---

## 💡 Tips & Tricks

### Accélérer l'entraînement

1. **Augmenter batch size** (si VRAM suffisante)
   ```bash
   python scripts/train.py --batch-size 128
   ```

2. **Utiliser mixed precision** (déjà activé par défaut)
   ```yaml
   training:
     use_mixed_precision: true  # FP16
   ```

3. **Augmenter num_workers**
   ```yaml
   dataloader:
     num_workers: 8  # Si CPU puissant
   ```

### Améliorer accuracy

1. **Plus d'epochs**
   ```bash
   python scripts/train.py --epochs 150
   ```

2. **Learning rate plus petit**
   ```bash
   python scripts/train.py --lr 0.0005
   ```

3. **Augmentation plus forte**
   Modifier `configs/data_config.yaml` → augmentation

4. **Essayer autre modèle**
   ```yaml
   # configs/train_config.yaml
   model:
     name: "resnet18"  # Au lieu de emotionnet_nano
     pretrained: true
   ```

---

## 🚀 En Résumé

```bash
# Installation une fois
./setup/phase1_setup.sh

# À chaque entraînement
source venv_emotion/bin/activate
python scripts/download_datasets.py --dataset fer2013  # Une fois
python scripts/prepare_data.py --dataset data/fer2013 --all  # Une fois
python scripts/train.py  # Entraînement
```

**C'est tout! Le système fait le reste automatiquement. 🎉**

---

**Temps total**:
- Setup: ~1h (une fois)
- Download: ~10 min
- Prepare: ~5 min
- Training: ~2-3h (AMD 7900 XT) ou ~24h (CPU)

**Prêt? `python scripts/train.py` et c'est parti! 🚀**
