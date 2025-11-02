#!/bin/bash
# Script d'installation pour Ubuntu 22.04 avec AMD Radeon 7900 XT
# Système d'Identification d'Étudiants avec Analyse d'Émotions

set -e  # Arrêter en cas d'erreur

echo "=========================================="
echo "Installation du Projet IA"
echo "Système d'Identification avec Émotions"
echo "=========================================="
echo ""

# Couleurs pour output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Vérifier si on est sur Ubuntu
if [ ! -f /etc/os-release ]; then
    echo -e "${RED}❌ Ce script est conçu pour Ubuntu 22.04${NC}"
    exit 1
fi

source /etc/os-release
if [ "$VERSION_ID" != "22.04" ]; then
    echo -e "${YELLOW}⚠️  Version Ubuntu détectée: $VERSION_ID${NC}"
    echo -e "${YELLOW}   Ce script est optimisé pour Ubuntu 22.04${NC}"
    read -p "Continuer quand même? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}✅ Ubuntu $VERSION_ID détecté${NC}"
echo ""

# Fonction pour vérifier une commande
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✅ $1 trouvé${NC}"
        return 0
    else
        echo -e "${RED}❌ $1 non trouvé${NC}"
        return 1
    fi
}

# Étape 1: Mise à jour du système
echo "=========================================="
echo "Étape 1: Mise à jour du système"
echo "=========================================="
sudo apt update
sudo apt upgrade -y
echo -e "${GREEN}✅ Système mis à jour${NC}"
echo ""

# Étape 2: Installer dépendances de base
echo "=========================================="
echo "Étape 2: Installation des dépendances"
echo "=========================================="
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
echo -e "${GREEN}✅ Dépendances installées${NC}"
echo ""

# Étape 3: Installer ROCm
echo "=========================================="
echo "Étape 3: Installation de ROCm"
echo "=========================================="

if check_command rocm-smi; then
    echo -e "${YELLOW}ROCm déjà installé${NC}"
    rocm-smi
else
    echo "Installation de ROCm 5.7..."

    # Télécharger et installer le package
    wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_5.7.50700-1_all.deb
    sudo dpkg -i amdgpu-install_5.7.50700-1_all.deb
    sudo apt update

    # Installer ROCm
    sudo amdgpu-install --usecase=rocm -y

    # Ajouter utilisateur aux groupes
    sudo usermod -a -G render,video $LOGNAME

    echo -e "${GREEN}✅ ROCm installé${NC}"
    echo -e "${YELLOW}⚠️  REDÉMARRAGE REQUIS pour finaliser l'installation ROCm${NC}"
    echo -e "${YELLOW}   Après redémarrage, relancez ce script pour continuer${NC}"

    read -p "Redémarrer maintenant? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo reboot
    else
        echo "Veuillez redémarrer manuellement puis relancer ce script"
        exit 0
    fi
fi
echo ""

# Étape 4: Créer environnement virtuel Python
echo "=========================================="
echo "Étape 4: Environnement Python"
echo "=========================================="

VENV_DIR="venv_emotion"

if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Environnement virtuel existant trouvé${NC}"
    read -p "Recréer? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf $VENV_DIR
        python3.10 -m venv $VENV_DIR
    fi
else
    python3.10 -m venv $VENV_DIR
fi

echo -e "${GREEN}✅ Environnement virtuel créé: $VENV_DIR${NC}"
echo ""

# Activer l'environnement
source $VENV_DIR/bin/activate

# Mettre à jour pip
pip install --upgrade pip

echo -e "${GREEN}✅ pip mis à jour${NC}"
echo ""

# Étape 5: Installer PyTorch avec ROCm
echo "=========================================="
echo "Étape 5: Installation PyTorch + ROCm"
echo "=========================================="

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

echo -e "${GREEN}✅ PyTorch installé avec support ROCm${NC}"
echo ""

# Tester PyTorch
echo "Test de PyTorch..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo -e "${GREEN}✅ GPU détecté par PyTorch${NC}"
else
    echo -e "${RED}❌ GPU non détecté par PyTorch${NC}"
    echo -e "${YELLOW}Vérifiez votre installation ROCm avec: rocm-smi${NC}"
fi
echo ""

# Étape 6: Installer les dépendances du projet
echo "=========================================="
echo "Étape 6: Dépendances du projet"
echo "=========================================="

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Dépendances installées${NC}"
else
    echo -e "${RED}❌ requirements.txt non trouvé${NC}"
    echo "Assurez-vous d'être dans le répertoire du projet"
    exit 1
fi
echo ""

# Étape 7: Créer les répertoires nécessaires
echo "=========================================="
echo "Étape 7: Structure des répertoires"
echo "=========================================="

mkdir -p data/{fer2013,rafdb,students}
mkdir -p models
mkdir -p logs/{tensorboard,checkpoints,results,screenshots}
mkdir -p notebooks

echo -e "${GREEN}✅ Répertoires créés${NC}"
echo ""

# Étape 8: Test final
echo "=========================================="
echo "Étape 8: Tests finaux"
echo "=========================================="

echo "Test du GPU..."
python scripts/test_gpu.py

echo ""
echo -e "${GREEN}✅ Installation terminée avec succès!${NC}"
echo ""
echo "=========================================="
echo "Prochaines étapes:"
echo "=========================================="
echo "1. Activer l'environnement:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo "2. Télécharger les datasets:"
echo "   python scripts/download_datasets.py"
echo ""
echo "3. Tester le système:"
echo "   python main.py --mode realtime"
echo ""
echo "Consultez README.md pour plus d'informations"
echo "=========================================="
