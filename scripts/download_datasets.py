#!/usr/bin/env python3
"""
Script de téléchargement automatique des datasets.

Télécharge:
- FER2013 depuis Kaggle
- RAF-DB depuis source officielle (manuel + script helper)

Prérequis:
- Compte Kaggle
- Kaggle API configurée (~/.kaggle/kaggle.json)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import shutil
import zipfile


def check_kaggle_installed():
    """Vérifie si Kaggle API est installée."""
    try:
        import kaggle
        return True
    except ImportError:
        return False


def install_kaggle():
    """Installe Kaggle API."""
    print("Installing Kaggle API...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
    print("✅ Kaggle API installed")


def check_kaggle_credentials():
    """Vérifie si les credentials Kaggle sont configurées."""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'

    if not kaggle_json.exists():
        return False

    # Vérifier permissions (Unix only)
    if os.name != 'nt':  # Not Windows
        perms = oct(kaggle_json.stat().st_mode)[-3:]
        if perms != '600':
            print(f"⚠️  Warning: kaggle.json permissions are {perms}, should be 600")
            print(f"   Run: chmod 600 {kaggle_json}")

    return True


def setup_kaggle_credentials():
    """Guide l'utilisateur pour configurer Kaggle credentials."""
    print("\n" + "="*70)
    print("Kaggle API Configuration Required")
    print("="*70)
    print("\nTo download FER2013 from Kaggle, you need to:")
    print("\n1. Create a Kaggle account: https://www.kaggle.com/")
    print("2. Go to: https://www.kaggle.com/account")
    print("3. Scroll down to 'API' section")
    print("4. Click 'Create New API Token'")
    print("5. This downloads 'kaggle.json'")
    print("\n6. Place kaggle.json in:")

    kaggle_dir = Path.home() / '.kaggle'
    print(f"   {kaggle_dir}")

    print("\n7. On Linux/Mac, set permissions:")
    print(f"   chmod 600 {kaggle_dir / 'kaggle.json'}")

    print("\n" + "="*70)

    input("\nPress Enter when you've completed the setup...")

    # Create .kaggle directory if it doesn't exist
    kaggle_dir.mkdir(exist_ok=True)

    # Check again
    if check_kaggle_credentials():
        print("✅ Kaggle credentials found!")
        return True
    else:
        print("❌ kaggle.json still not found. Please follow the instructions above.")
        return False


def download_fer2013(data_dir: str = 'data', force: bool = False):
    """
    Télécharge FER2013 depuis Kaggle.

    Args:
        data_dir: Dossier de destination
        force: Si True, retélécharge même si existe
    """
    print("\n" + "="*70)
    print("Downloading FER2013 from Kaggle")
    print("="*70 + "\n")

    data_path = Path(data_dir)
    fer2013_dir = data_path / 'fer2013'

    # Check si déjà téléchargé
    if fer2013_dir.exists() and not force:
        print(f"✅ FER2013 already exists in {fer2013_dir}")
        print("   Use --force to re-download")
        return

    # Check Kaggle API
    if not check_kaggle_installed():
        print("Kaggle API not installed.")
        install_kaggle()

    if not check_kaggle_credentials():
        print("Kaggle credentials not configured.")
        if not setup_kaggle_credentials():
            print("❌ Cannot download without Kaggle credentials")
            return

    # Import kaggle after installation
    import kaggle

    # Créer dossier data
    data_path.mkdir(exist_ok=True)

    # Télécharger FER2013
    print("Downloading FER2013 dataset (may take a few minutes)...")

    try:
        # Dataset: msambare/fer2013 (version organisée par dossiers)
        kaggle.api.dataset_download_files(
            'msambare/fer2013',
            path=str(data_path),
            unzip=True
        )

        print("✅ FER2013 downloaded successfully")

        # Check structure
        expected_files = ['train', 'test']
        downloaded_files = list(data_path.glob('*'))

        print(f"\nDownloaded files:")
        for f in downloaded_files:
            print(f"  - {f.name}")

        # Parfois le dataset est dans un sous-dossier
        # Réorganiser si nécessaire
        if not (data_path / 'train').exists():
            # Chercher le dossier train
            for subdir in data_path.glob('*/train'):
                parent = subdir.parent
                print(f"\nReorganizing files from {parent.name}...")

                # Déplacer vers fer2013/
                fer2013_dir.mkdir(exist_ok=True)
                for item in parent.glob('*'):
                    shutil.move(str(item), str(fer2013_dir / item.name))

                # Supprimer le dossier parent vide
                parent.rmdir()
                break

        print(f"\n✅ FER2013 ready in {fer2013_dir}")

    except Exception as e:
        print(f"❌ Error downloading FER2013: {e}")
        print("\nYou can manually download from:")
        print("https://www.kaggle.com/datasets/msambare/fer2013")


def download_rafdb(data_dir: str = 'data'):
    """
    Instructions pour télécharger RAF-DB.

    RAF-DB nécessite téléchargement manuel depuis le site officiel.

    Args:
        data_dir: Dossier de destination
    """
    print("\n" + "="*70)
    print("RAF-DB (Real-world Affective Faces Database) Download")
    print("="*70 + "\n")

    rafdb_dir = Path(data_dir) / 'rafdb'

    if rafdb_dir.exists():
        print(f"✅ RAF-DB directory exists: {rafdb_dir}")
        print("   If empty, follow instructions below")
    else:
        print(f"RAF-DB will be downloaded to: {rafdb_dir}")

    print("\n⚠️  RAF-DB requires MANUAL download from official website:\n")
    print("1. Visit: http://www.whdeng.cn/raf/model1.html")
    print("   (Or search 'RAF-DB dataset')")
    print("\n2. Request access (may require registration)")
    print("\n3. Download the dataset")
    print("\n4. Extract to:")
    print(f"   {rafdb_dir}")
    print("\n5. Organize as:")
    print("   rafdb/")
    print("     train/")
    print("       angry/")
    print("       happy/")
    print("       ...")
    print("     test/")
    print("       ...")

    print("\n" + "="*70)
    print("Alternative: Use FER2013 only for this project")
    print("RAF-DB is recommended but optional for fine-tuning")
    print("="*70 + "\n")


def verify_dataset_structure(data_dir: str = 'data'):
    """Vérifie la structure des datasets téléchargés."""
    print("\n" + "="*70)
    print("Verifying Dataset Structure")
    print("="*70 + "\n")

    data_path = Path(data_dir)

    # Vérifier FER2013
    fer2013_dir = data_path / 'fer2013'
    if fer2013_dir.exists():
        print("✅ FER2013 found")

        splits = ['train', 'test']
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        for split in splits:
            split_dir = fer2013_dir / split
            if split_dir.exists():
                print(f"  ✅ {split}/ found")

                for emotion in emotions:
                    emotion_dir = split_dir / emotion
                    if emotion_dir.exists():
                        num_images = len(list(emotion_dir.glob('*.jpg'))) + \
                                    len(list(emotion_dir.glob('*.png')))
                        if num_images > 0:
                            print(f"     {emotion:10s}: {num_images:5d} images")
                    else:
                        print(f"     ⚠️  {emotion} folder missing")
            else:
                print(f"  ❌ {split}/ NOT found")
    else:
        print("❌ FER2013 NOT found")

    print()

    # Vérifier RAF-DB
    rafdb_dir = data_path / 'rafdb'
    if rafdb_dir.exists():
        print("✅ RAF-DB directory found")

        splits = ['train', 'test']
        for split in splits:
            split_dir = rafdb_dir / split
            if split_dir.exists():
                num_images = len(list(split_dir.rglob('*.jpg'))) + \
                            len(list(split_dir.rglob('*.png')))
                print(f"  ✅ {split}/: {num_images} images")
            else:
                print(f"  ❌ {split}/ NOT found")
    else:
        print("ℹ️  RAF-DB not found (optional)")

    print("\n" + "="*70 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download datasets for emotion recognition"
    )
    parser.add_argument(
        '--dataset',
        choices=['fer2013', 'rafdb', 'all'],
        default='all',
        help='Which dataset to download'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Data directory (default: data/)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if exists'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify dataset structure'
    )

    args = parser.parse_args()

    if args.verify_only:
        verify_dataset_structure(args.data_dir)
        return

    print("\n" + "="*70)
    print("Dataset Download Script")
    print("="*70 + "\n")

    # Download datasets
    if args.dataset in ['fer2013', 'all']:
        download_fer2013(args.data_dir, args.force)

    if args.dataset in ['rafdb', 'all']:
        download_rafdb(args.data_dir)

    # Verify
    print("\nVerifying downloads...")
    verify_dataset_structure(args.data_dir)

    print("\n" + "="*70)
    print("Download Complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run: python scripts/prepare_data.py")
    print("2. Check: notebooks/01_EDA.ipynb")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
