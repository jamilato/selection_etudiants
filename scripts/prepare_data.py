#!/usr/bin/env python3
"""
Script de préparation et organisation des données.

Fonctionnalités:
- Créer train/val/test splits
- Vérifier et réparer la structure des dossiers
- Générer statistiques sur les datasets
- Détecter et corriger déséquilibres de classes
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import random
import json


# Emotions standard
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def count_images_in_directory(directory: Path) -> Dict[str, int]:
    """
    Compte les images par classe dans un dossier.

    Args:
        directory: Dossier à analyser

    Returns:
        Dictionnaire {emotion: count}
    """
    counts = {}

    for emotion in EMOTIONS:
        emotion_dir = directory / emotion

        if not emotion_dir.exists():
            counts[emotion] = 0
            continue

        # Compter images
        num_images = len([
            f for f in emotion_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])

        counts[emotion] = num_images

    return counts


def create_val_split(
    dataset_dir: Path,
    val_ratio: float = 0.15,
    seed: int = 42
):
    """
    Crée un split validation à partir du dossier train.

    Args:
        dataset_dir: Dossier du dataset (ex: data/fer2013)
        val_ratio: Proportion de validation (0.15 = 15%)
        seed: Random seed pour reproductibilité
    """
    random.seed(seed)

    train_dir = dataset_dir / 'train'
    val_dir = dataset_dir / 'val'

    if not train_dir.exists():
        print(f"❌ Train directory not found: {train_dir}")
        return

    if val_dir.exists():
        print(f"⚠️  Validation directory already exists: {val_dir}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            return
        shutil.rmtree(val_dir)

    print(f"\nCreating validation split ({val_ratio:.0%})...")

    # Créer structure val/
    val_dir.mkdir(exist_ok=True)

    total_moved = 0

    for emotion in EMOTIONS:
        emotion_train_dir = train_dir / emotion
        emotion_val_dir = val_dir / emotion

        if not emotion_train_dir.exists():
            continue

        # Créer dossier val pour cette émotion
        emotion_val_dir.mkdir(exist_ok=True)

        # Lister toutes les images
        images = [
            f for f in emotion_train_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ]

        # Shuffle
        random.shuffle(images)

        # Calculer nombre d'images pour validation
        num_val = int(len(images) * val_ratio)

        # Déplacer vers val
        for img in images[:num_val]:
            dest = emotion_val_dir / img.name
            shutil.move(str(img), str(dest))
            total_moved += 1

        print(f"  {emotion:10s}: {num_val:4d} images moved to validation")

    print(f"\n✅ Created validation split: {total_moved} images")


def analyze_dataset(dataset_dir: Path, split: str = 'train') -> Dict:
    """
    Analyse un dataset et retourne des statistiques.

    Args:
        dataset_dir: Dossier du dataset
        split: 'train', 'val', ou 'test'

    Returns:
        Dictionnaire de statistiques
    """
    split_dir = dataset_dir / split

    if not split_dir.exists():
        return {'error': f'{split_dir} not found'}

    counts = count_images_in_directory(split_dir)

    total = sum(counts.values())

    # Calculer statistiques
    stats = {
        'split': split,
        'total_images': total,
        'num_classes': len([c for c in counts.values() if c > 0]),
        'counts': counts,
        'percentages': {
            emotion: (count / total * 100 if total > 0 else 0)
            for emotion, count in counts.items()
        },
        'class_balance': None
    }

    # Vérifier équilibre
    if total > 0:
        max_count = max(counts.values())
        min_count = min([c for c in counts.values() if c > 0], default=0)

        if min_count > 0:
            imbalance_ratio = max_count / min_count
            stats['class_balance'] = {
                'max_count': max_count,
                'min_count': min_count,
                'imbalance_ratio': imbalance_ratio,
                'balanced': imbalance_ratio < 2.0  # Considéré équilibré si ratio < 2
            }

    return stats


def print_dataset_stats(stats: Dict):
    """Affiche les statistiques d'un dataset."""
    print(f"\n{'='*70}")
    print(f"Dataset Statistics - {stats['split'].upper()} Split")
    print(f"{'='*70}")

    print(f"\nTotal images: {stats['total_images']}")
    print(f"Number of classes: {stats['num_classes']}")

    print(f"\nClass distribution:")
    for emotion in EMOTIONS:
        count = stats['counts'][emotion]
        pct = stats['percentages'][emotion]
        bar_length = int(pct / 2)  # Scale for display
        bar = '█' * bar_length

        print(f"  {emotion:10s}: {count:5d} ({pct:5.1f}%) {bar}")

    if stats['class_balance']:
        balance = stats['class_balance']
        print(f"\nClass balance:")
        print(f"  Max count: {balance['max_count']}")
        print(f"  Min count: {balance['min_count']}")
        print(f"  Imbalance ratio: {balance['imbalance_ratio']:.2f}")

        if balance['balanced']:
            print(f"  Status: ✅ Reasonably balanced")
        else:
            print(f"  Status: ⚠️  Imbalanced (consider WeightedRandomSampler)")

    print(f"{'='*70}\n")


def verify_and_fix_structure(dataset_dir: Path):
    """
    Vérifie et répare la structure des dossiers.

    Args:
        dataset_dir: Dossier du dataset
    """
    print(f"\nVerifying dataset structure: {dataset_dir}")

    splits = ['train', 'val', 'test']

    for split in splits:
        split_dir = dataset_dir / split

        if not split_dir.exists():
            print(f"  ⚠️  {split}/ not found")
            continue

        print(f"  ✅ {split}/ found")

        # Vérifier dossiers d'émotions
        for emotion in EMOTIONS:
            emotion_dir = split_dir / emotion

            if not emotion_dir.exists():
                print(f"     Creating {emotion}/")
                emotion_dir.mkdir(exist_ok=True)


def export_dataset_info(dataset_dir: Path, output_file: str = 'dataset_info.json'):
    """
    Exporte les informations du dataset en JSON.

    Args:
        dataset_dir: Dossier du dataset
        output_file: Fichier de sortie
    """
    info = {
        'dataset_name': dataset_dir.name,
        'dataset_path': str(dataset_dir),
        'splits': {}
    }

    for split in ['train', 'val', 'test']:
        split_dir = dataset_dir / split

        if split_dir.exists():
            stats = analyze_dataset(dataset_dir, split)
            info['splits'][split] = stats

    # Sauvegarder
    output_path = dataset_dir / output_file

    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"✅ Dataset info exported to: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Prepare and organize emotion recognition datasets"
    )
    parser.add_argument(
        '--dataset',
        default='data/fer2013',
        help='Dataset directory (default: data/fer2013)'
    )
    parser.add_argument(
        '--create-val-split',
        action='store_true',
        help='Create validation split from train'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation split ratio (default: 0.15)'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze dataset and show statistics'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify and fix directory structure'
    )
    parser.add_argument(
        '--export-info',
        action='store_true',
        help='Export dataset info to JSON'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all operations'
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)

    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        print(f"\nRun first: python scripts/download_datasets.py")
        return

    print(f"\n{'='*70}")
    print(f"Dataset Preparation - {dataset_dir.name}")
    print(f"{'='*70}")

    # Verify structure
    if args.verify or args.all:
        verify_and_fix_structure(dataset_dir)

    # Create validation split
    if args.create_val_split or args.all:
        create_val_split(dataset_dir, args.val_ratio)

    # Analyze
    if args.analyze or args.all:
        for split in ['train', 'val', 'test']:
            split_dir = dataset_dir / split
            if split_dir.exists():
                stats = analyze_dataset(dataset_dir, split)
                print_dataset_stats(stats)

    # Export info
    if args.export_info or args.all:
        export_dataset_info(dataset_dir)

    # Si aucune option, afficher help
    if not any([
        args.create_val_split,
        args.analyze,
        args.verify,
        args.export_info,
        args.all
    ]):
        print("\nNo operation specified. Use --help for options.")
        print("\nQuick start:")
        print("  python scripts/prepare_data.py --all")

    print(f"\n{'='*70}")
    print("Preparation Complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
