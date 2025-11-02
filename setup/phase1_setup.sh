#!/bin/bash

################################################################################
# Phase 1: Configuration de l'Environnement
# Script d'installation pour WSL2 + Ubuntu 22.04 + AMD Radeon 7900 XT
################################################################################

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}   Phase 1: Configuration de l'Environnement${NC}"
echo -e "${BLUE}   Projet IA - Identification Étudiants + Analyse Émotions${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""

# Function to print section headers
print_section() {
    echo ""
    echo -e "${GREEN}>>> $1${NC}"
    echo ""
}

# Function to print warnings
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to print errors
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

################################################################################
# Étape 1: Vérification de l'environnement
################################################################################
print_section "Étape 1/8: Vérification de l'environnement WSL2 et Ubuntu"

# Check Ubuntu version
echo "Vérification de la version d'Ubuntu..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "OS: $NAME $VERSION"
    if [[ "$VERSION_ID" != "22.04" ]]; then
        print_warning "Ubuntu 22.04 LTS est recommandé. Vous avez: $VERSION_ID"
    else
        print_success "Ubuntu 22.04 LTS détecté"
    fi
else
    print_error "Impossible de déterminer la version du système"
    exit 1
fi

# Check if running in WSL
if grep -qi microsoft /proc/version; then
    print_success "Exécution dans WSL détectée"
else
    print_warning "WSL non détecté. Ce script est optimisé pour WSL2."
fi

# Check Python version
echo "Vérification de Python..."
if command -v python3.10 &> /dev/null; then
    PYTHON_VERSION=$(python3.10 --version)
    print_success "Python détecté: $PYTHON_VERSION"
else
    print_error "Python 3.10 non trouvé"
    exit 1
fi

################################################################################
# Étape 2: Mise à jour du système
################################################################################
print_section "Étape 2/8: Mise à jour du système Ubuntu"

echo "Mise à jour de la liste des paquets..."
sudo apt update

echo "Mise à niveau des paquets existants..."
sudo apt upgrade -y

print_success "Système mis à jour"

################################################################################
# Étape 3: Installation des outils de base
################################################################################
print_section "Étape 3/8: Installation des outils de base"

echo "Installation de build-essential, git, wget, curl..."
sudo apt install -y build-essential git wget curl

echo "Installation des dépendances Python..."
sudo apt install -y python3.10-venv python3-pip python3.10-dev

print_success "Outils de base installés"

################################################################################
# Étape 4: Installation de ROCm 5.7+
################################################################################
print_section "Étape 4/8: Installation de ROCm 5.7+"

print_warning "ATTENTION: Le support ROCm sur WSL2 est limité comparé à Linux natif"
print_warning "Certaines fonctionnalités GPU peuvent ne pas fonctionner comme attendu"
echo ""

read -p "Voulez-vous continuer avec l'installation de ROCm? (o/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Oo]$ ]]; then
    echo "Installation des dépendances ROCm..."
    sudo apt install -y libnuma-dev

    echo "Ajout du dépôt AMD ROCm..."
    # Add ROCm repository
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -

    echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7/ ubuntu main" | \
        sudo tee /etc/apt/sources.list.d/rocm.list

    echo "Mise à jour de la liste des paquets..."
    sudo apt update

    echo "Installation de ROCm (cela peut prendre plusieurs minutes)..."
    sudo apt install -y rocm-hip-sdk rocm-libs miopen-hip

    # Add user to video and render groups
    echo "Ajout de l'utilisateur aux groupes video et render..."
    sudo usermod -a -G video $USER
    sudo usermod -a -G render $USER

    # Set environment variables
    echo "Configuration des variables d'environnement..."
    cat >> ~/.bashrc << 'EOF'

# ROCm Environment Variables
export PATH=/opt/rocm/bin:/opt/rocm/opencl/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/opencl/lib:$LD_LIBRARY_PATH
export HIP_PATH=/opt/rocm
EOF

    # Source the updated bashrc
    source ~/.bashrc

    print_success "ROCm installé"
    print_warning "Vous devrez peut-être redémarrer WSL pour que les changements de groupe prennent effet"
    print_warning "Commande: wsl --shutdown (depuis Windows PowerShell)"
else
    print_warning "Installation de ROCm ignorée"
    print_warning "PyTorch s'exécutera en mode CPU uniquement"
fi

################################################################################
# Étape 5: Création de l'environnement virtuel Python
################################################################################
print_section "Étape 5/8: Création de l'environnement virtuel Python"

# Navigate to project directory
cd "$(dirname "$0")/.."

echo "Création de l'environnement virtuel 'venv_emotion'..."
python3.10 -m venv venv_emotion

echo "Activation de l'environnement virtuel..."
source venv_emotion/bin/activate

echo "Mise à jour de pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

print_success "Environnement virtuel créé et activé"

################################################################################
# Étape 6: Installation de PyTorch avec ROCm
################################################################################
print_section "Étape 6/8: Installation de PyTorch avec support ROCm"

read -p "Installer PyTorch avec support ROCm? (o/N) Si non, version CPU sera installée " -n 1 -r
echo
if [[ $REPLY =~ ^[Oo]$ ]]; then
    echo "Installation de PyTorch avec ROCm 5.7..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
    print_success "PyTorch avec ROCm installé"
else
    echo "Installation de PyTorch version CPU..."
    pip3 install torch torchvision torchaudio
    print_success "PyTorch CPU installé"
fi

################################################################################
# Étape 7: Installation des bibliothèques requises
################################################################################
print_section "Étape 7/8: Installation des bibliothèques requises"

echo "Installation d'OpenCV..."
pip install opencv-python opencv-contrib-python

echo "Installation de DeepFace..."
pip install deepface

echo "Installation de MTCNN..."
pip install mtcnn

echo "Installation des bibliothèques de visualisation..."
pip install pillow numpy pandas matplotlib seaborn

echo "Installation des outils ML..."
pip install scikit-learn

echo "Installation des utilitaires..."
pip install tqdm tensorboard

echo "Génération du fichier requirements.txt..."
pip freeze > requirements.txt

print_success "Toutes les bibliothèques installées"

################################################################################
# Étape 8: Tests de validation
################################################################################
print_section "Étape 8/8: Tests de validation"

echo "Création du script de test GPU..."
cat > test_gpu.py << 'EOFPYTHON'
#!/usr/bin/env python3
"""
Script de test pour vérifier la configuration GPU PyTorch
"""
import torch
import sys

print("=" * 70)
print("Test de Configuration GPU - PyTorch")
print("=" * 70)
print()

# PyTorch version
print(f"PyTorch version: {torch.__version__}")
print()

# CUDA/ROCm availability
print(f"CUDA/ROCm disponible: {torch.cuda.is_available()}")
print(f"Nombre de GPU détectés: {torch.cuda.device_count()}")
print()

if torch.cuda.is_available():
    print("✅ GPU détecté!")
    print()
    print(f"Nom du GPU: {torch.cuda.get_device_name(0)}")

    # Memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Mémoire GPU totale: {total_memory:.2f} GB")

    # Test simple computation
    print()
    print("Test de calcul sur GPU...")
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✅ Calcul matriciel sur GPU réussi!")
    except Exception as e:
        print(f"❌ Erreur lors du calcul GPU: {e}")
        sys.exit(1)
else:
    print("⚠️  Aucun GPU détecté - PyTorch s'exécutera en mode CPU")
    print("   Ceci est normal si ROCm n'est pas installé ou si WSL2 ne supporte pas le GPU")

print()
print("=" * 70)
EOFPYTHON

chmod +x test_gpu.py

echo "Exécution du test GPU..."
python test_gpu.py

print_success "Tests de validation terminés"

################################################################################
# Finalisation
################################################################################
echo ""
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${GREEN}✅ Phase 1 terminée avec succès!${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""
echo "Prochaines étapes:"
echo "  1. Vérifiez les résultats du test GPU ci-dessus"
echo "  2. Si le GPU n'est pas détecté et que vous souhaitez utiliser ROCm,"
echo "     vous devrez peut-être:"
echo "     - Redémarrer WSL: wsl --shutdown (depuis Windows)"
echo "     - Vérifier les drivers AMD sur Windows"
echo "     - Consulter: https://rocm.docs.amd.com/"
echo "  3. Pour activer l'environnement virtuel à l'avenir:"
echo "     source venv_emotion/bin/activate"
echo "  4. Passez à la Phase 2: Préparation des Données"
echo ""
echo "Fichiers créés:"
echo "  - venv_emotion/ (environnement virtuel Python)"
echo "  - requirements.txt (liste des dépendances)"
echo "  - test_gpu.py (script de test)"
echo ""
