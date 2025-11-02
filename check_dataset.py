#!/usr/bin/env python3
"""Script rapide pour vérifier les statistiques du dataset FER2013"""

import os
from pathlib import Path
from collections import defaultdict

dataset_path = Path("data/fer2013")
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

print("\n" + "=" * 70)
print("STATISTIQUES DU DATASET FER2013")
print("=" * 70)

total_all = 0

for split in ["train", "val", "test"]:
    split_path = dataset_path / split

    if not split_path.exists():
        continue

    print(f"\n{split.upper()}:")
    print("-" * 40)

    split_total = 0

    for emotion in emotions:
        emotion_path = split_path / emotion
        if emotion_path.exists():
            count = len(list(emotion_path.glob("*.png"))) + len(list(emotion_path.glob("*.jpg")))
            print(f"  {emotion:10s}: {count:5d} images")
            split_total += count

    print(f"  {'TOTAL':10s}: {split_total:5d} images")
    total_all += split_total

print("\n" + "=" * 70)
print(f"TOTAL GÉNÉRAL: {total_all:,} images")
print("=" * 70)

print("\n✅ Dataset prêt pour l'entraînement !")
