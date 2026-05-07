#!/usr/bin/env python3
"""
scripts/download_data.py

Downloads FER2013 dataset from Kaggle.

Requirements:
  1. Create a Kaggle account at kaggle.com
  2. Go to Account -> API -> Create New Token -> downloads kaggle.json
  3. Place kaggle.json at ~/.kaggle/kaggle.json
  4. Run: python scripts/download_data.py

Dataset: ~60MB, 35,887 images, 7 emotion classes
"""

import os
import zipfile
import shutil
from pathlib import Path

DATA_DIR = Path("model/data")
FER_DIR = DATA_DIR / "fer2013"
KAGGLE_DATASET = "msambare/fer2013"


def check_kaggle_credentials():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("ERROR: kaggle.json not found at ~/.kaggle/kaggle.json")
        print()
        print("To fix:")
        print("  1. Go to https://www.kaggle.com/account")
        print("  2. Scroll to 'API' -> 'Create New API Token'")
        print("  3. Move downloaded kaggle.json to ~/.kaggle/kaggle.json")
        return False
    return True


def download_fer2013():
    print("Downloading FER2013 from Kaggle...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(DATA_DIR),
            unzip=True,
        )
        print("FER2013 downloaded and extracted")
    except Exception as e:
        print(f"ERROR: Kaggle download failed: {e}")
        print()
        print("Alternative: download manually from:")
        print("  https://www.kaggle.com/datasets/msambare/fer2013")
        print(f"  Extract to: {FER_DIR.absolute()}")
        return False

    return True


def verify_structure():
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    ok = True

    for split in ["train", "test"]:
        split_dir = FER_DIR / split
        if not split_dir.exists():
            print(f"  Missing: {split_dir}")
            ok = False
            continue
        for emotion in emotions:
            emotion_dir = split_dir / emotion
            if not emotion_dir.exists():
                print(f"  Missing: {emotion_dir}")
                ok = False
            else:
                count = len(list(emotion_dir.glob("*.jpg")) + list(emotion_dir.glob("*.png")))
                print(f"  OK  {split}/{emotion}: {count} images")

    return ok


def print_dataset_stats():
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    print("\nDataset statistics:")
    print(f"{'Emotion':<12} {'Train':>8} {'Test':>8}")
    print("-" * 30)
    total_train = total_test = 0
    for emotion in emotions:
        train_count = len(list((FER_DIR / "train" / emotion).glob("*")))
        test_count = len(list((FER_DIR / "test" / emotion).glob("*")))
        total_train += train_count
        total_test += test_count
        print(f"{emotion:<12} {train_count:>8} {test_count:>8}")
    print("-" * 30)
    print(f"{'TOTAL':<12} {total_train:>8} {total_test:>8}")
    print()
    print("Note: FER2013 is imbalanced -- 'happy' has 3x more samples than 'disgust'.")
    print("The training script handles this with weighted sampling.")


if __name__ == "__main__":
    print("=== EmotionMirror data setup ===\n")

    if FER_DIR.exists() and (FER_DIR / "train").exists():
        print("FER2013 already exists -- skipping download")
    else:
        if not check_kaggle_credentials():
            exit(1)
        if not download_fer2013():
            exit(1)

    print("\nVerifying dataset structure...")
    if verify_structure():
        print("\nAll files present")
        print_dataset_stats()
    else:
        print("\nSome files are missing -- re-download the dataset")
        exit(1)
