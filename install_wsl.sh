#!/bin/bash
################################################################################
# Script d'installation complète pour WSL2 + Ubuntu
# Projet : Système d'Identification d'Étudiants avec IA
#
# Ce script installe et configure :
# - Python 3.10+ et pip
# - Toutes les dépendances Python (PyTorch, OpenCV, DeepFace, etc.)
# - ROCm pour AMD Radeon 7900 XT (optionnel)
# - Configuration de l'environnement
#
# Usage:
#   bash install_wsl.sh [--with-rocm] [--cpu-only]
#
# Options:
#   --with-rocm   : Installer ROCm pour support GPU AMD (recommandé)
#   --cpu-only    : Installation CPU uniquement (pas de GPU)
#   --skip-system : Skip system packages (si déjà installés)
################################################################################

set -e  # Arrêter en cas d'erreur

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Flags
INSTALL_ROCM=false
CPU_ONLY=false
SKIP_SYSTEM=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --with-rocm)
            INSTALL_ROCM=true
            shift
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --skip-system)
            SKIP_SYSTEM=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--with-rocm] [--cpu-only] [--skip-system]"
            echo ""
            echo "Options:"
            echo "  --with-rocm    Install ROCm for AMD GPU support"
            echo "  --cpu-only     CPU-only installation"
            echo "  --skip-system  Skip system packages installation"
            exit 0
            ;;
    esac
done

# Fonctions utilitaires
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_command() {
    if command -v $1 &> /dev/null; then
        print_success "$1 est installé"
        return 0
    else
        print_error "$1 n'est pas installé"
        return 1
    fi
}

################################################################################
# BANNIÈRE
################################################################################

clear
echo "======================================================================"
echo "  Installation du Système d'Identification d'Étudiants avec IA"
echo "  Pour WSL2 + Ubuntu 22.04"
echo "======================================================================"
echo ""
echo "Configuration détectée:"
echo "  - GPU Support: $([ "$INSTALL_ROCM" = true ] && echo "ROCm AMD" || echo "$([ "$CPU_ONLY" = true ] && echo "CPU only" || echo "Auto-detect")")"
echo "  - OS: $(uname -s)"
echo "  - Architecture: $(uname -m)"
echo ""

# Demander confirmation
read -p "Continuer l'installation? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation annulée."
    exit 1
fi

################################################################################
# 1. MISE À JOUR DU SYSTÈME
################################################################################

if [ "$SKIP_SYSTEM" = false ]; then
    print_header "1. Mise à jour du système"

    sudo apt-get update -qq
    sudo apt-get upgrade -y -qq

    print_success "Système mis à jour"
fi

################################################################################
# 2. INSTALLATION DES PAQUETS SYSTÈME
################################################################################

if [ "$SKIP_SYSTEM" = false ]; then
    print_header "2. Installation des paquets système"

    print_info "Installation de Python et outils de développement..."
    sudo apt-get install -y -qq \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        unzip

    print_info "Installation des bibliothèques pour OpenCV..."
    sudo apt-get install -y -qq \
        libopencv-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0

    print_info "Installation des outils supplémentaires..."
    sudo apt-get install -y -qq \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libatlas-base-dev \
        gfortran

    print_success "Paquets système installés"
fi

################################################################################
# 3. VÉRIFICATION PYTHON
################################################################################

print_header "3. Vérification de Python"

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

echo "Version Python: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python 3.8+ requis. Version actuelle: $PYTHON_VERSION"
    exit 1
fi

print_success "Python $PYTHON_VERSION OK (>= 3.8 requis)"

# Mise à jour pip
print_info "Mise à jour de pip..."
python3 -m pip install --upgrade pip setuptools wheel -q

################################################################################
# 4. CRÉATION ENVIRONNEMENT VIRTUEL (OPTIONNEL)
################################################################################

print_header "4. Environnement virtuel"

read -p "Créer un environnement virtuel? (recommandé) (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -d "venv" ]; then
        print_info "Création de l'environnement virtuel..."
        python3 -m venv venv
        print_success "Environnement virtuel créé"
    else
        print_warning "Environnement virtuel existe déjà"
    fi

    print_info "Activation de l'environnement virtuel..."
    source venv/bin/activate
    print_success "Environnement virtuel activé"
else
    print_warning "Installation dans l'environnement système"
fi

################################################################################
# 5. INSTALLATION PYTORCH
################################################################################

print_header "5. Installation de PyTorch"

if [ "$INSTALL_ROCM" = true ]; then
    print_info "Installation PyTorch avec support ROCm..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
    print_success "PyTorch ROCm installé"
elif [ "$CPU_ONLY" = true ]; then
    print_info "Installation PyTorch CPU only..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    print_success "PyTorch CPU installé"
else
    print_info "Installation PyTorch (auto-détection)..."
    pip3 install torch torchvision torchaudio
    print_success "PyTorch installé"
fi

################################################################################
# 6. INSTALLATION DES DÉPENDANCES PYTHON
################################################################################

print_header "6. Installation des dépendances Python"

if [ -f "requirements.txt" ]; then
    print_info "Installation depuis requirements.txt..."
    pip3 install -r requirements.txt -q
    print_success "Dépendances installées"
else
    print_warning "requirements.txt non trouvé, installation manuelle..."

    print_info "Installation des bibliothèques de base..."
    pip3 install numpy pandas scipy scikit-learn matplotlib seaborn tqdm pyyaml -q

    print_info "Installation OpenCV..."
    pip3 install opencv-python opencv-contrib-python -q

    print_info "Installation des outils de vision..."
    pip3 install Pillow albumentations imgaug -q

    print_info "Installation DeepFace..."
    pip3 install deepface -q

    print_info "Installation MTCNN..."
    pip3 install facenet-pytorch mtcnn -q

    print_info "Installation des outils ML..."
    pip3 install tensorboard onnx onnxruntime -q

    print_info "Installation Kaggle API..."
    pip3 install kaggle -q

    print_success "Toutes les dépendances installées"
fi

################################################################################
# 7. INSTALLATION ROCM (SI DEMANDÉ)
################################################################################

if [ "$INSTALL_ROCM" = true ]; then
    print_header "7. Installation ROCm"

    print_info "Ajout du dépôt ROCm..."
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
    echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

    print_info "Mise à jour et installation ROCm..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq rocm-dkms rocm-libs

    print_info "Configuration des permissions..."
    sudo usermod -a -G video $LOGNAME
    sudo usermod -a -G render $LOGNAME

    print_success "ROCm installé"
    print_warning "REDÉMARRAGE REQUIS pour activer ROCm"
fi

################################################################################
# 8. CONFIGURATION KAGGLE
################################################################################

print_header "8. Configuration Kaggle API"

if [ ! -d "$HOME/.kaggle" ]; then
    mkdir -p $HOME/.kaggle
    print_info "Dossier .kaggle créé"
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    print_warning "Fichier kaggle.json non trouvé"
    print_info "Pour configurer Kaggle API:"
    echo "  1. Allez sur https://www.kaggle.com/settings/account"
    echo "  2. Créez un nouveau token API"
    echo "  3. Copiez le fichier kaggle.json vers ~/.kaggle/"
    echo "  4. Exécutez: chmod 600 ~/.kaggle/kaggle.json"
else
    chmod 600 $HOME/.kaggle/kaggle.json
    print_success "Kaggle API configuré"
fi

################################################################################
# 9. VÉRIFICATION DES INSTALLATIONS
################################################################################

print_header "9. Vérification des installations"

print_info "Vérification Python..."
python3 --version
check_command python3

print_info "Vérification pip..."
pip3 --version
check_command pip3

print_info "Vérification des bibliothèques Python..."
python3 -c "
import sys
packages = [
    ('torch', 'PyTorch'),
    ('cv2', 'OpenCV'),
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('sklearn', 'scikit-learn'),
    ('PIL', 'Pillow'),
    ('yaml', 'PyYAML'),
    ('tqdm', 'tqdm'),
    ('kaggle', 'Kaggle API')
]

print('')
for pkg, name in packages:
    try:
        __import__(pkg)
        print(f'✓ {name:20s} OK')
    except ImportError:
        print(f'✗ {name:20s} MANQUANT')
"

################################################################################
# 10. TEST GPU
################################################################################

print_header "10. Test GPU"

python3 << 'EOF'
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Nombre de GPUs: {torch.cuda.device_count()}")
    print(f"GPU actuel: {torch.cuda.current_device()}")
    print(f"Nom du GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Mode CPU - Pas de GPU détecté")
EOF

################################################################################
# 11. CRÉATION DES DOSSIERS
################################################################################

print_header "11. Création des dossiers du projet"

mkdir -p data/fer2013
mkdir -p data/rafdb
mkdir -p data/students
mkdir -p models
mkdir -p logs
mkdir -p checkpoints

print_success "Dossiers créés"

################################################################################
# 12. CONFIGURATION .BASHRC
################################################################################

print_header "12. Configuration de l'environnement"

# Ajouter variables d'environnement si ROCm
if [ "$INSTALL_ROCM" = true ]; then
    if ! grep -q "ROCM" ~/.bashrc; then
        echo "" >> ~/.bashrc
        echo "# ROCm Environment" >> ~/.bashrc
        echo "export HSA_OVERRIDE_GFX_VERSION=11.0.0" >> ~/.bashrc
        echo "export PATH=/opt/rocm/bin:\$PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=/opt/rocm/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
        print_success "Variables ROCm ajoutées à .bashrc"
    fi
fi

# Ajouter alias utiles
if ! grep -q "# IA Student Project" ~/.bashrc; then
    cat >> ~/.bashrc << 'ALIASES'

# IA Student Project Aliases
alias activate-venv='source venv/bin/activate'
alias train='python scripts/train.py'
alias test-gpu='python scripts/test_gpu.py'
alias run-realtime='python main.py --mode realtime'
alias run-video='python main.py --mode video --input'
alias run-image='python main.py --mode image --input'
ALIASES
    print_success "Aliases ajoutés à .bashrc"
fi

################################################################################
# 13. TESTS FINAUX
################################################################################

print_header "13. Tests finaux"

print_info "Test d'import des modules du projet..."
if [ -d "src" ]; then
    python3 << 'EOF'
import sys
sys.path.insert(0, 'src')

try:
    from src.utils.config import load_config
    print("✓ Module config OK")
except Exception as e:
    print(f"✗ Module config: {e}")

try:
    from src.models.emotion_net import EmotionNetNano
    print("✓ Module emotion_net OK")
except Exception as e:
    print(f"✗ Module emotion_net: {e}")

try:
    from src.data.datasets import EmotionDataset
    print("✓ Module datasets OK")
except Exception as e:
    print(f"✗ Module datasets: {e}")
EOF
else
    print_warning "Dossier src/ non trouvé - Tests ignorés"
fi

################################################################################
# RÉSUMÉ ET PROCHAINES ÉTAPES
################################################################################

print_header "INSTALLATION TERMINÉE !"

echo ""
echo "======================================================================"
echo "  Résumé de l'installation"
echo "======================================================================"
echo ""
echo "✓ Système mis à jour"
echo "✓ Python $(python3 --version | cut -d' ' -f2) installé"
echo "✓ PyTorch installé $([ "$INSTALL_ROCM" = true ] && echo "(ROCm)" || echo "(CPU/Auto)")"
echo "✓ Toutes les dépendances installées"
echo "✓ Dossiers du projet créés"
echo "✓ Configuration de l'environnement"
echo ""
echo "======================================================================"
echo "  PROCHAINES ÉTAPES"
echo "======================================================================"
echo ""
echo "1. Si environnement virtuel créé, activez-le:"
echo "   source venv/bin/activate"
echo ""
echo "2. Configurez Kaggle API (si pas encore fait):"
echo "   - Copiez kaggle.json vers ~/.kaggle/"
echo "   - chmod 600 ~/.kaggle/kaggle.json"
echo ""
echo "3. Téléchargez le dataset FER2013:"
echo "   python scripts/download_datasets.py"
echo ""
echo "4. Préparez les données:"
echo "   python scripts/prepare_data.py --all"
echo ""
echo "5. Entraînez le modèle:"
echo "   python scripts/train.py"
echo ""
echo "6. Testez le système:"
echo "   python main.py --mode realtime"
echo ""
if [ "$INSTALL_ROCM" = true ]; then
    echo "⚠️  IMPORTANT: Redémarrez votre système pour activer ROCm"
    echo ""
fi
echo "======================================================================"
echo ""
echo "Pour plus d'informations, consultez:"
echo "  - README.md"
echo "  - QUICKSTART.md"
echo "  - docs/"
echo ""
echo "Bonne chance avec votre projet ! 🚀"
echo ""

# Sauvegarder un log de l'installation
LOG_FILE="installation_$(date +%Y%m%d_%H%M%S).log"
{
    echo "Installation Log - $(date)"
    echo "===================="
    echo "Python: $(python3 --version)"
    echo "Pip: $(pip3 --version)"
    echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
    echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
    echo "ROCm: $INSTALL_ROCM"
    echo "CPU Only: $CPU_ONLY"
} > "$LOG_FILE"

print_success "Log sauvegardé dans: $LOG_FILE"
