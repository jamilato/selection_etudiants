#!/bin/bash
################################################################################
# Script de vérification de l'installation
# Projet : Système d'Identification d'Étudiants avec IA
#
# Ce script vérifie que toutes les dépendances sont correctement installées
#
# Usage: bash verify_installation.sh
################################################################################

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

check_pass() {
    echo -e "${GREEN}✓ $1${NC}"
}

check_fail() {
    echo -e "${RED}✗ $1${NC}"
    ((ERRORS++))
}

check_warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
    ((WARNINGS++))
}

check_command() {
    if command -v $1 &> /dev/null; then
        VERSION=$($1 --version 2>&1 | head -n1)
        check_pass "$1 installé ($VERSION)"
        return 0
    else
        check_fail "$1 non trouvé"
        return 1
    fi
}

################################################################################
# BANNIÈRE
################################################################################

clear
echo "======================================================================"
echo "  Vérification de l'Installation"
echo "  Système d'Identification d'Étudiants avec IA"
echo "======================================================================"
echo ""

################################################################################
# 1. SYSTÈME
################################################################################

print_header "1. Informations Système"

echo "OS: $(uname -s)"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"
echo "Distribution: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"

################################################################################
# 2. COMMANDES DE BASE
################################################################################

print_header "2. Commandes de Base"

check_command python3
check_command pip3
check_command git
check_command wget
check_command curl
check_command unzip

################################################################################
# 3. VERSION PYTHON
################################################################################

print_header "3. Python"

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

echo "Version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
    check_pass "Python version >= 3.8"
else
    check_fail "Python version < 3.8 (requis >= 3.8)"
fi

################################################################################
# 4. BIBLIOTHÈQUES PYTHON
################################################################################

print_header "4. Bibliothèques Python"

python3 << 'EOF'
import sys

packages = [
    ('torch', 'PyTorch', True),
    ('torchvision', 'TorchVision', True),
    ('cv2', 'OpenCV', True),
    ('numpy', 'NumPy', True),
    ('pandas', 'Pandas', True),
    ('sklearn', 'scikit-learn', True),
    ('PIL', 'Pillow', True),
    ('yaml', 'PyYAML', True),
    ('tqdm', 'tqdm', True),
    ('matplotlib', 'Matplotlib', True),
    ('albumentations', 'Albumentations', False),
    ('deepface', 'DeepFace', False),
    ('facenet_pytorch', 'Facenet-PyTorch', False),
    ('kaggle', 'Kaggle API', False),
]

errors = 0
warnings = 0

for pkg, name, critical in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'N/A')
        print(f"\033[0;32m✓\033[0m {name:20s} {version}")
    except ImportError:
        if critical:
            print(f"\033[0;31m✗\033[0m {name:20s} MANQUANT (CRITIQUE)")
            errors += 1
        else:
            print(f"\033[1;33m⚠\033[0m {name:20s} MANQUANT (optionnel)")
            warnings += 1

sys.exit(errors)
EOF

if [ $? -ne 0 ]; then
    ((ERRORS+=$?))
fi

################################################################################
# 5. PYTORCH ET GPU
################################################################################

print_header "5. PyTorch et GPU"

python3 << 'EOF'
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Chemin installation: {torch.__file__}")
print("")

cuda_available = torch.cuda.is_available()
print(f"CUDA disponible: {cuda_available}")

if cuda_available:
    print(f"\033[0;32m✓\033[0m GPU détecté")
    print(f"  - Nombre de GPUs: {torch.cuda.device_count()}")
    print(f"  - GPU actuel: {torch.cuda.current_device()}")
    print(f"  - Nom: {torch.cuda.get_device_name(0)}")
    print(f"  - VRAM totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Test simple
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.matmul(x, x)
        print(f"  - Test GPU: OK")
        print(f"\033[0;32m✓\033[0m PyTorch GPU fonctionnel")
    except Exception as e:
        print(f"\033[0;31m✗\033[0m Test GPU échoué: {e}")
else:
    print(f"\033[1;33m⚠\033[0m Mode CPU - Pas de GPU détecté")
    print("  L'entraînement sera beaucoup plus lent sur CPU")
EOF

################################################################################
# 6. OPENCV
################################################################################

print_header "6. OpenCV"

python3 << 'EOF'
import cv2

print(f"OpenCV version: {cv2.__version__}")

# Test webcam
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("\033[0;32m✓\033[0m Webcam accessible")
        cap.release()
    else:
        print("\033[1;33m⚠\033[0m Webcam non accessible (normal sous WSL)")
except:
    print("\033[1;33m⚠\033[0m Test webcam impossible")
EOF

################################################################################
# 7. MODULES DU PROJET
################################################################################

print_header "7. Modules du Projet"

if [ -d "src" ]; then
    python3 << 'EOF'
import sys
sys.path.insert(0, 'src')

modules = [
    ('src.utils.config', 'Config Utils'),
    ('src.utils.face_detector', 'Face Detector'),
    ('src.utils.preprocessor', 'Preprocessor'),
    ('src.models.emotion_net', 'EmotionNet'),
    ('src.data.datasets', 'Datasets'),
    ('src.data.loaders', 'DataLoaders'),
    ('src.training.trainer', 'Trainer'),
    ('src.core.emotion_classifier', 'Emotion Classifier'),
    ('src.core.student_identifier', 'Student Identifier'),
    ('src.core.system', 'System'),
]

for module, name in modules:
    try:
        __import__(module)
        print(f"\033[0;32m✓\033[0m {name:25s} OK")
    except Exception as e:
        print(f"\033[0;31m✗\033[0m {name:25s} Erreur: {e}")
EOF
else
    check_fail "Dossier src/ non trouvé"
fi

################################################################################
# 8. FICHIERS DE CONFIGURATION
################################################################################

print_header "8. Fichiers de Configuration"

FILES=(
    "configs/config.yaml:Configuration principale"
    "configs/data_config.yaml:Configuration données"
    "configs/train_config.yaml:Configuration entraînement"
    "configs/model_config.yaml:Configuration modèles"
    "requirements.txt:Dépendances Python"
    "main.py:Point d'entrée"
)

for item in "${FILES[@]}"; do
    FILE="${item%%:*}"
    DESC="${item##*:}"
    if [ -f "$FILE" ]; then
        check_pass "$DESC ($FILE)"
    else
        check_fail "$DESC manquant ($FILE)"
    fi
done

################################################################################
# 9. STRUCTURE DES DOSSIERS
################################################################################

print_header "9. Structure des Dossiers"

DIRS=(
    "data:Données"
    "models:Modèles entraînés"
    "logs:Logs"
    "src:Code source"
    "scripts:Scripts"
    "configs:Configurations"
)

for item in "${DIRS[@]}"; do
    DIR="${item%%:*}"
    DESC="${item##*:}"
    if [ -d "$DIR" ]; then
        check_pass "$DESC ($DIR/)"
    else
        check_warn "$DESC manquant ($DIR/) - sera créé automatiquement"
    fi
done

################################################################################
# 10. DATASET FER2013
################################################################################

print_header "10. Dataset FER2013"

if [ -d "data/fer2013" ]; then
    check_pass "Dossier data/fer2013/ existe"

    # Compter les images
    if [ -d "data/fer2013/train" ]; then
        TRAIN_COUNT=$(find data/fer2013/train -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
        if [ "$TRAIN_COUNT" -gt 0 ]; then
            check_pass "Train set: $TRAIN_COUNT images"
        else
            check_warn "Train set vide"
        fi
    else
        check_warn "data/fer2013/train/ manquant"
    fi

    if [ -d "data/fer2013/val" ]; then
        VAL_COUNT=$(find data/fer2013/val -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
        if [ "$VAL_COUNT" -gt 0 ]; then
            check_pass "Validation set: $VAL_COUNT images"
        else
            check_warn "Validation set vide"
        fi
    else
        check_warn "data/fer2013/val/ manquant - exécutez prepare_data.py"
    fi

    if [ -d "data/fer2013/test" ]; then
        TEST_COUNT=$(find data/fer2013/test -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
        if [ "$TEST_COUNT" -gt 0 ]; then
            check_pass "Test set: $TEST_COUNT images"
        else
            check_warn "Test set vide"
        fi
    else
        check_warn "data/fer2013/test/ manquant"
    fi
else
    check_warn "Dataset FER2013 non téléchargé"
    echo "    Téléchargez avec: python scripts/download_datasets.py"
fi

################################################################################
# 11. KAGGLE API
################################################################################

print_header "11. Kaggle API"

if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    check_pass "Fichier kaggle.json trouvé"

    # Vérifier permissions
    PERMS=$(stat -c %a $HOME/.kaggle/kaggle.json 2>/dev/null)
    if [ "$PERMS" = "600" ]; then
        check_pass "Permissions correctes (600)"
    else
        check_warn "Permissions incorrectes ($PERMS) - devrait être 600"
        echo "    Exécutez: chmod 600 ~/.kaggle/kaggle.json"
    fi

    # Test Kaggle CLI
    if command -v kaggle &> /dev/null; then
        kaggle --version &>/dev/null
        if [ $? -eq 0 ]; then
            check_pass "Kaggle CLI fonctionnel"
        else
            check_warn "Kaggle CLI non fonctionnel"
        fi
    fi
else
    check_warn "Kaggle API non configuré"
    echo "    Voir: https://github.com/Kaggle/kaggle-api#api-credentials"
fi

################################################################################
# 12. ROCM (SI INSTALLÉ)
################################################################################

print_header "12. ROCm (si installé)"

if [ -d "/opt/rocm" ]; then
    check_pass "ROCm installé"

    if command -v rocm-smi &> /dev/null; then
        check_pass "rocm-smi disponible"
        echo ""
        rocm-smi
    else
        check_warn "rocm-smi non trouvé"
    fi
else
    echo "ROCm non installé (normal si CPU seulement)"
fi

################################################################################
# RÉSUMÉ
################################################################################

print_header "RÉSUMÉ"

echo ""
echo "======================================================================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ INSTALLATION COMPLÈTE ET FONCTIONNELLE !${NC}"
    echo ""
    echo "Toutes les vérifications ont réussi."
    echo "Vous pouvez commencer à utiliser le système."
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ INSTALLATION FONCTIONNELLE AVEC AVERTISSEMENTS${NC}"
    echo ""
    echo "Avertissements: $WARNINGS"
    echo "Les avertissements ne sont pas critiques."
else
    echo -e "${RED}✗ PROBLÈMES DÉTECTÉS${NC}"
    echo ""
    echo "Erreurs: $ERRORS"
    echo "Avertissements: $WARNINGS"
    echo ""
    echo "Veuillez corriger les erreurs avant de continuer."
fi

echo "======================================================================"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo "Prochaines étapes:"
    echo ""

    if [ ! -d "data/fer2013/train" ] || [ $(find data/fer2013/train -type f 2>/dev/null | wc -l) -eq 0 ]; then
        echo "1. Télécharger le dataset:"
        echo "   python scripts/download_datasets.py"
        echo ""
        echo "2. Préparer les données:"
        echo "   python scripts/prepare_data.py --all"
        echo ""
    fi

    echo "3. Entraîner le modèle:"
    echo "   python scripts/train.py"
    echo ""
    echo "4. Tester le système:"
    echo "   python main.py --mode realtime"
    echo ""
fi

echo "======================================================================"
echo ""

exit $ERRORS
