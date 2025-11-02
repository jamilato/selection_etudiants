#!/bin/bash
################################################################################
# Script d'installation ROCm 6.4 pour WSL2 Ubuntu 22.04/24.04
# Optimisé pour AMD Radeon RX 7900 XT
#
# Prérequis :
# - Windows 11
# - AMD Adrenalin Driver 24.6.1+ installé sur Windows
# - WSL2 avec Ubuntu 22.04 ou 24.04
# - .wslconfig configuré
#
# Usage: bash install_wsl_rocm.sh
################################################################################

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ $1${NC}"; }

################################################################################
# BANNIÈRE
################################################################################

clear
echo "======================================================================"
echo "  Installation ROCm 6.4 pour WSL2"
echo "  AMD Radeon RX 7900 XT - Ubuntu 22.04/24.04"
echo "======================================================================"
echo ""

# Vérifier Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)
print_info "Ubuntu version: $UBUNTU_VERSION"

if [[ "$UBUNTU_VERSION" != "22.04" && "$UBUNTU_VERSION" != "24.04" ]]; then
    print_error "Ubuntu $UBUNTU_VERSION non supporté. Utilisez 22.04 ou 24.04"
    exit 1
fi

# Confirmation
read -p "Continuer l'installation ROCm pour WSL2? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation annulée."
    exit 1
fi

################################################################################
# 1. NETTOYAGE
################################################################################

print_header "1. Nettoyage des installations précédentes"

# Supprimer anciennes installations
sudo rm -f /etc/apt/sources.list.d/rocm.list
sudo rm -f /etc/apt/sources.list.d/amdgpu.list

# Nettoyer cache
sudo apt-get clean
sudo apt-get update

print_success "Nettoyage terminé"

################################################################################
# 2. INSTALLATION DES PRÉREQUIS
################################################################################

print_header "2. Installation des prérequis"

sudo apt-get update
sudo apt-get install -y \
    wget \
    gnupg2 \
    software-properties-common \
    build-essential \
    cmake \
    git

print_success "Prérequis installés"

################################################################################
# 3. TÉLÉCHARGEMENT ET INSTALLATION AMDGPU-INSTALL
################################################################################

print_header "3. Installation du paquet amdgpu-install"

# Déterminer le codename Ubuntu
if [ "$UBUNTU_VERSION" = "22.04" ]; then
    UBUNTU_CODENAME="jammy"
elif [ "$UBUNTU_VERSION" = "24.04" ]; then
    UBUNTU_CODENAME="noble"
fi

print_info "Codename Ubuntu: $UBUNTU_CODENAME"

# Télécharger amdgpu-install
cd /tmp
AMDGPU_INSTALL_URL="https://repo.radeon.com/amdgpu-install/latest/ubuntu/${UBUNTU_CODENAME}/amdgpu-install_6.4.60402-1_all.deb"

print_info "Téléchargement depuis: $AMDGPU_INSTALL_URL"
wget -q --show-progress "$AMDGPU_INSTALL_URL"

# Installer le paquet
print_info "Installation du paquet amdgpu-install..."
sudo apt-get install -y ./amdgpu-install_6.4.60402-1_all.deb

print_success "Paquet amdgpu-install installé"

################################################################################
# 4. INSTALLATION ROCM POUR WSL2
################################################################################

print_header "4. Installation ROCm 6.4 pour WSL2"

print_warning "Cette étape peut prendre 10-15 minutes..."
print_info "Installation des paquets ROCm pour WSL2..."

# COMMANDE SPÉCIALE POUR WSL2 : --usecase=wsl,rocm --no-dkms
sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms

print_success "ROCm 6.4 installé"

################################################################################
# 5. CONFIGURATION DES PERMISSIONS
################################################################################

print_header "5. Configuration des permissions"

# Ajouter l'utilisateur aux groupes render et video
sudo usermod -a -G render $LOGNAME
sudo usermod -a -G video $LOGNAME

print_success "Utilisateur ajouté aux groupes render et video"
print_warning "Vous devrez vous reconnecter pour que les groupes prennent effet"

################################################################################
# 6. CONFIGURATION VARIABLES D'ENVIRONNEMENT
################################################################################

print_header "6. Configuration des variables d'environnement"

# Ajouter les variables à .bashrc
if ! grep -q "HSA_OVERRIDE_GFX_VERSION" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# ROCm Environment for RX 7900 XT
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
EOF
    print_success "Variables d'environnement ajoutées à ~/.bashrc"
else
    print_info "Variables d'environnement déjà configurées"
fi

# Sourcer pour cette session
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

################################################################################
# 7. VÉRIFICATION ROCM
################################################################################

print_header "7. Vérification de l'installation ROCm"

# Vérifier rocminfo
if command -v rocminfo &> /dev/null; then
    print_success "rocminfo installé"

    print_info "Test de détection GPU..."
    if rocminfo | grep -q "gfx1100"; then
        print_success "GPU RX 7900 XT détecté (gfx1100)"
    else
        print_warning "GPU pas encore détecté - Redémarrage WSL requis"
    fi
else
    print_error "rocminfo non trouvé"
fi

# Vérifier rocm-smi
if command -v rocm-smi &> /dev/null; then
    print_success "rocm-smi installé"
else
    print_warning "rocm-smi non trouvé"
fi

################################################################################
# 8. INSTALLATION PYTHON ET DÉPENDANCES
################################################################################

print_header "8. Installation Python et pip"

sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev

print_success "Python installé"

################################################################################
# 9. INSTALLATION PYTORCH ROCM
################################################################################

print_header "9. Installation PyTorch avec support ROCm"

print_info "Installation de PyTorch ROCm 6.2 (compatible ROCm 6.4)..."

# Créer un environnement virtuel si dans le projet
if [ -f "requirements.txt" ]; then
    print_info "Projet détecté, création d'un environnement virtuel..."

    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Environnement virtuel créé"
    fi

    source venv/bin/activate
    print_success "Environnement virtuel activé"
fi

# Installer PyTorch
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

print_success "PyTorch ROCm installé"

################################################################################
# 10. TEST PYTORCH + GPU
################################################################################

print_header "10. Test PyTorch + GPU"

python3 << 'EOF'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"ROCm disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ GPU détecté: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Test calcul simple
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.matmul(x, x)
        print("✓ Test calcul GPU: OK")
        sys.exit(0)
    except Exception as e:
        print(f"✗ Test calcul GPU échoué: {e}")
        sys.exit(1)
else:
    print("⚠ GPU non détecté - Redémarrage WSL requis")
    print("  Exécutez depuis PowerShell: wsl --shutdown")
    sys.exit(1)
EOF

PYTORCH_TEST=$?

################################################################################
# 11. INSTALLATION DÉPENDANCES PROJET
################################################################################

if [ -f "requirements.txt" ]; then
    print_header "11. Installation des dépendances du projet"

    print_info "Installation depuis requirements.txt..."
    pip3 install -r requirements.txt

    print_success "Dépendances installées"
fi

################################################################################
# RÉSUMÉ
################################################################################

print_header "INSTALLATION TERMINÉE !"

echo ""
echo "======================================================================"
echo "  Résumé"
echo "======================================================================"
echo ""
echo "✓ ROCm 6.4 installé pour WSL2"
echo "✓ PyTorch avec support ROCm installé"
echo "✓ Variables d'environnement configurées"
echo ""

if [ $PYTORCH_TEST -eq 0 ]; then
    echo "✓ GPU AMD RX 7900 XT fonctionnel !"
    echo ""
    echo "Le système est prêt à utiliser le GPU."
else
    echo "⚠ GPU pas encore détecté"
    echo ""
    echo "REDÉMARRAGE WSL REQUIS:"
    echo "  1. Fermez ce terminal WSL"
    echo "  2. Dans PowerShell (Windows): wsl --shutdown"
    echo "  3. Rouvrez WSL: wsl -d Ubuntu-22.04"
    echo "  4. Testez: rocm-smi"
fi

echo ""
echo "======================================================================"
echo "  PROCHAINES ÉTAPES"
echo "======================================================================"
echo ""
echo "1. Redémarrer WSL (si GPU pas détecté):"
echo "   Dans PowerShell: wsl --shutdown"
echo ""
echo "2. Tester ROCm:"
echo "   rocminfo | grep gfx1100"
echo "   rocm-smi"
echo ""
echo "3. Tester PyTorch:"
echo "   python3 -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "4. Entraîner votre modèle:"
echo "   python scripts/train.py"
echo ""
echo "======================================================================"
echo ""

# Log
LOG_FILE="rocm_install_$(date +%Y%m%d_%H%M%S).log"
{
    echo "ROCm WSL2 Installation Log - $(date)"
    echo "====================================="
    echo "Ubuntu: $(lsb_release -ds)"
    echo "ROCm: 6.4"
    echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
    echo "GPU Test: $([ $PYTORCH_TEST -eq 0 ] && echo 'OK' || echo 'Pending reboot')"
} > "$LOG_FILE"

print_success "Log sauvegardé: $LOG_FILE"

echo ""
echo "Installation terminée ! 🚀"
echo ""
