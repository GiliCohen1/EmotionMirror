"""
Download FER2013 from HuggingFace (clip-benchmark/wds_fer2013).
No credentials required. Saves images to model/data/fer2013/{train,test}/<emotion>/*.jpg
matching the folder structure expected by FER2013Dataset.

Original FER2013 label mapping:
  0=angry  1=disgust  2=fear  3=happy  4=sad  5=surprise  6=neutral
"""

import sys
from pathlib import Path

# Run from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LABEL_MAP = {0: "angry", 1: "disgust", 2: "fear",
             3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}

FER_DIR = ROOT / "model" / "data" / "fer2013"


def download_split(split: str):
    from datasets import load_dataset
    import warnings
    warnings.filterwarnings("ignore")

    # Create output directories
    for emotion in LABEL_MAP.values():
        (FER_DIR / split / emotion).mkdir(parents=True, exist_ok=True)

    print(f"\n[{split}] Streaming from HuggingFace...")
    ds = load_dataset("clip-benchmark/wds_fer2013", split=split, streaming=True)

    counts = {e: 0 for e in LABEL_MAP.values()}
    for i, sample in enumerate(ds):
        emotion = LABEL_MAP[sample["cls"]]
        img = sample["jpg"]  # PIL Image, 48x48
        out_path = FER_DIR / split / emotion / f"{i:05d}.jpg"
        img.save(out_path, "JPEG")
        counts[emotion] += 1

        if (i + 1) % 1000 == 0:
            total = sum(counts.values())
            print(f"  {total} images saved...", end="\r")

    total = sum(counts.values())
    print(f"  {total} images saved.     ")
    print(f"  Distribution:")
    for emotion, count in sorted(counts.items()):
        bar = "#" * (count // 200)
        print(f"    {emotion:<10} {count:>5}  {bar}")
    return total


if __name__ == "__main__":
    print("=== FER2013 download via HuggingFace ===")

    if FER_DIR.exists() and (FER_DIR / "train" / "happy").exists():
        happy_count = len(list((FER_DIR / "train" / "happy").glob("*.jpg")))
        if happy_count > 100:
            print(f"Dataset already present ({happy_count} happy train images). Skipping.")
            sys.exit(0)

    train_total = download_split("train")
    test_total  = download_split("test")

    print(f"\nDone. Train: {train_total}  Test: {test_total}")
