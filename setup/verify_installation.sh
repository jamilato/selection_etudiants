#!/bin/bash

################################################################################
# Script de Vérification - Phase 1
# Vérifie que tous les composants sont correctement installés
################################################################################

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS=0
FAIL=0

echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}   Script de Vérification - Phase 1${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""

# Function to check and report
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✅${NC} $2"
        ((PASS++))
        return 0
    else
        echo -e "${RED}❌${NC} $2"
        ((FAIL++))
        return 1
    fi
}

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✅${NC} $2"
        ((PASS++))
        return 0
    else
        echo -e "${RED}❌${NC} $2"
        ((FAIL++))
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✅${NC} $2"
        ((PASS++))
        return 0
    else
        echo -e "${RED}❌${NC} $2"
        ((FAIL++))
        return 1
    fi
}

################################################################################
# 1. Système
################################################################################
echo -e "${BLUE}>>> Système${NC}"

# Ubuntu version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$VERSION_ID" == "22.04" ]]; then
        echo -e "${GREEN}✅${NC} Ubuntu 22.04 LTS"
        ((PASS++))
    else
        echo -e "${YELLOW}⚠️${NC}  Ubuntu $VERSION_ID (22.04 recommandé)"
        ((PASS++))
    fi
else
    echo -e "${RED}❌${NC} Ubuntu non détecté"
    ((FAIL++))
fi

# WSL check
if grep -qi microsoft /proc/version; then
    echo -e "${GREEN}✅${NC} WSL détecté"
    ((PASS++))
else
    echo -e "${YELLOW}⚠️${NC}  WSL non détecté (normal si Ubuntu natif)"
    ((PASS++))
fi

echo ""

################################################################################
# 2. Outils de base
################################################################################
echo -e "${BLUE}>>> Outils de Base${NC}"

check_command "gcc" "gcc (build-essential)"
check_command "git" "git"
check_command "wget" "wget"
check_command "curl" "curl"
check_command "python3.10" "Python 3.10"
check_command "pip3" "pip3"

echo ""

################################################################################
# 3. ROCm (optionnel)
################################################################################
echo -e "${BLUE}>>> ROCm (Support GPU AMD)${NC}"

if check_command "rocm-smi" "rocm-smi"; then
    # Test rocm-smi output
    if rocm-smi &> /dev/null; then
        echo -e "${GREEN}✅${NC} ROCm fonctionnel"
        ((PASS++))
    else
        echo -e "${YELLOW}⚠️${NC}  ROCm installé mais GPU non détecté"
        ((PASS++))
    fi
else
    echo -e "${YELLOW}⚠️${NC}  ROCm non installé (mode CPU uniquement)"
fi

check_command "rocminfo" "rocminfo" || echo -e "${YELLOW}⚠️${NC}  rocminfo non disponible"

# Check environment variables
if [[ "$PATH" == *"/opt/rocm"* ]]; then
    echo -e "${GREEN}✅${NC} Variables d'environnement ROCm configurées"
    ((PASS++))
else
    echo -e "${YELLOW}⚠️${NC}  Variables ROCm non configurées dans PATH"
fi

echo ""

################################################################################
# 4. Environnement Python
################################################################################
echo -e "${BLUE}>>> Environnement Python${NC}"

# Navigate to project directory
cd "$(dirname "$0")/.."

check_dir "venv_emotion" "Environnement virtuel venv_emotion"

if [ -d "venv_emotion" ]; then
    # Activate venv for checking
    source venv_emotion/bin/activate

    # Check Python in venv
    if [[ "$(which python)" == *"venv_emotion"* ]]; then
        echo -e "${GREEN}✅${NC} Python dans venv_emotion"
        ((PASS++))
    else
        echo -e "${RED}❌${NC} Python n'utilise pas venv_emotion"
        ((FAIL++))
    fi

    # Check pip packages
    echo -e "\n${BLUE}>>> Bibliothèques Python${NC}"

    check_package() {
        if pip show $1 &> /dev/null; then
            VERSION=$(pip show $1 | grep Version | cut -d' ' -f2)
            echo -e "${GREEN}✅${NC} $2 ($VERSION)"
            ((PASS++))
        else
            echo -e "${RED}❌${NC} $2"
            ((FAIL++))
        fi
    }

    check_package "torch" "PyTorch"
    check_package "torchvision" "torchvision"
    check_package "opencv-python" "OpenCV"
    check_package "deepface" "DeepFace"
    check_package "mtcnn" "MTCNN"
    check_package "numpy" "NumPy"
    check_package "pandas" "Pandas"
    check_package "matplotlib" "Matplotlib"
    check_package "scikit-learn" "scikit-learn"
    check_package "tqdm" "tqdm"
    check_package "tensorboard" "TensorBoard"

    echo ""

    ################################################################################
    # 5. Test PyTorch GPU
    ################################################################################
    echo -e "${BLUE}>>> Test PyTorch GPU${NC}"

    PYTORCH_TEST=$(python -c "import torch; print(torch.cuda.is_available())" 2>&1)

    if [[ "$PYTORCH_TEST" == "True" ]]; then
        echo -e "${GREEN}✅${NC} PyTorch détecte le GPU"
        ((PASS++))

        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1)
        echo -e "${GREEN}   GPU: $GPU_NAME${NC}"

        GPU_MEM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}')" 2>&1)
        echo -e "${GREEN}   VRAM: ${GPU_MEM} GB${NC}"
    elif [[ "$PYTORCH_TEST" == "False" ]]; then
        echo -e "${YELLOW}⚠️${NC}  PyTorch en mode CPU (GPU non détecté)"
        echo -e "${YELLOW}   Ceci est normal si ROCm n'est pas installé${NC}"
        ((PASS++))
    else
        echo -e "${RED}❌${NC} Erreur lors du test PyTorch"
        echo -e "${RED}   $PYTORCH_TEST${NC}"
        ((FAIL++))
    fi

    deactivate
else
    echo -e "${RED}❌${NC} Environnement virtuel non trouvé"
    echo -e "${RED}   Exécutez d'abord: ./setup/phase1_setup.sh${NC}"
fi

echo ""

################################################################################
# 6. Fichiers du projet
################################################################################
echo -e "${BLUE}>>> Fichiers du Projet${NC}"

check_file "projet.md" "Documentation projet (projet.md)"
check_file "plan.md" "Roadmap (plan.md)"
check_file "setup/phase1_setup.sh" "Script d'installation Phase 1"
check_file "setup/README_PHASE1.md" "Guide Phase 1"

if [ -f "test_gpu.py" ]; then
    echo -e "${GREEN}✅${NC} Script de test GPU (test_gpu.py)"
    ((PASS++))
else
    echo -e "${YELLOW}⚠️${NC}  test_gpu.py non trouvé (sera créé par phase1_setup.sh)"
fi

if [ -f "requirements.txt" ]; then
    echo -e "${GREEN}✅${NC} requirements.txt"
    ((PASS++))
else
    echo -e "${YELLOW}⚠️${NC}  requirements.txt non trouvé (sera créé par phase1_setup.sh)"
fi

echo ""

################################################################################
# Résumé
################################################################################
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}   Résumé de la Vérification${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""

TOTAL=$((PASS + FAIL))

echo -e "Tests réussis:   ${GREEN}$PASS${NC} / $TOTAL"
echo -e "Tests échoués:   ${RED}$FAIL${NC} / $TOTAL"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}🎉 Félicitations! Phase 1 est complète!${NC}"
    echo ""
    echo "Vous êtes prêt pour la Phase 2: Préparation des Données"
    echo "Consultez plan.md (lignes 102-210) pour les prochaines étapes."
    echo ""
    exit 0
elif [ $FAIL -lt 5 ]; then
    echo -e "${YELLOW}⚠️  Phase 1 presque complète${NC}"
    echo ""
    echo "Quelques éléments sont manquants ou nécessitent attention."
    echo "Consultez les ❌ et ⚠️  ci-dessus."
    echo ""
    echo "Si ROCm/GPU n'est pas détecté, vous pouvez:"
    echo "  1. Continuer en mode CPU pour le développement"
    echo "  2. Consulter setup/README_PHASE1.md pour le dépannage"
    echo ""
    exit 1
else
    echo -e "${RED}❌ Phase 1 incomplète${NC}"
    echo ""
    echo "Plusieurs composants sont manquants."
    echo "Veuillez exécuter: ./setup/phase1_setup.sh"
    echo ""
    exit 1
fi
