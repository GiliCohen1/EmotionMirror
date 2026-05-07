"""
model/data/dataset.py

FER2013 PyTorch Dataset with:
  - Albumentations augmentation pipeline
  - Class-weighted sampler for imbalance
  - Support for grayscale and RGB (for EfficientNet)
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}


def get_transforms(split: str, image_size: int, normalize_mean: List, normalize_std: List) -> A.Compose:
    """
    Returns augmentation pipeline for train/val/test.

    Train gets heavy augmentation. Val/test get only normalize + resize.
    Using albumentations instead of torchvision transforms because:
      - Faster (OpenCV under the hood)
      - More augmentation options
      - Better for CV tasks
    """
    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 8), hole_height_range=(4, 8),
                hole_width_range=(4, 8), fill=0, p=0.3
            ),
            A.Normalize(mean=normalize_mean, std=normalize_std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=normalize_mean, std=normalize_std),
            ToTensorV2(),
        ])


class FER2013Dataset(Dataset):
    """
    Loads FER2013 from folder structure:
      data_dir/
        train/angry/*.jpg
        train/happy/*.jpg
        ...
        test/angry/*.jpg
        ...
    """

    def __init__(
        self,
        data_dir: str,
        split: str,              # "train" or "test"
        image_size: int = 48,
        normalize_mean: List = [0.507, 0.507, 0.507],
        normalize_std: List = [0.255, 0.255, 0.255],
        rgb: bool = False,       # True for EfficientNet (needs 3 channels)
    ):
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.rgb = rgb
        self.transform = get_transforms(split, image_size, normalize_mean, normalize_std)

        # Build file list
        self.samples: List[Tuple[Path, int]] = []
        for emotion in EMOTIONS:
            emotion_dir = self.data_dir / emotion
            if not emotion_dir.exists():
                raise FileNotFoundError(
                    f"Missing directory: {emotion_dir}\n"
                    f"Run: python scripts/download_data.py"
                )
            for img_path in emotion_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((img_path, EMOTION_TO_IDX[emotion]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.data_dir}")

        print(f"[Dataset] {split}: {len(self.samples)} images across {len(EMOTIONS)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB" if self.rgb else "L")
        img_array = np.array(img)

        # Albumentations expects (H, W, C) or (H, W) for grayscale
        if not self.rgb:
            # Repeat grayscale to 3 channels so normalize works
            img_array = np.stack([img_array] * 3, axis=-1)

        augmented = self.transform(image=img_array)
        image = augmented["image"]

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Computes per-class weights for WeightedRandomSampler.
        Inverse frequency weighting handles FER2013's class imbalance.
        """
        class_counts = torch.zeros(len(EMOTIONS))
        for _, label in self.samples:
            class_counts[label] += 1

        weights = 1.0 / class_counts
        sample_weights = torch.tensor([weights[label] for _, label in self.samples])
        return sample_weights


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds train/val/test DataLoaders from config dict.

    Train uses WeightedRandomSampler to handle class imbalance.
    Val/test use sequential sampler (no shuffling for reproducibility).
    """
    data_cfg = cfg["data"]
    aug_cfg = cfg["augmentation"]

    # FER2013 only has train/test splits — we carve val from train
    full_train = FER2013Dataset(
        data_dir=data_cfg["data_dir"],
        split="train",
        image_size=data_cfg["image_size"],
        normalize_mean=aug_cfg["normalize_mean"],
        normalize_std=aug_cfg["normalize_std"],
        rgb=(data_cfg["image_size"] == 224),
    )

    test_dataset = FER2013Dataset(
        data_dir=data_cfg["data_dir"],
        split="test",
        image_size=data_cfg["image_size"],
        normalize_mean=aug_cfg["normalize_mean"],
        normalize_std=aug_cfg["normalize_std"],
        rgb=(data_cfg["image_size"] == 224),
    )

    # Carve val from train (last 10%)
    n_total = len(full_train)
    n_val = int(n_total * data_cfg["val_split"])
    n_train = n_total - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # WeightedRandomSampler for imbalanced training set
    train_indices = train_dataset.indices
    all_weights = full_train.get_class_weights()
    train_weights = all_weights[train_indices]

    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True,
    )

    batch_size = cfg["training"]["batch_size"]
    num_workers = 0  # Windows multiprocessing DataLoader causes memory issues

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[DataLoader] train: {len(train_loader)} batches | "
          f"val: {len(val_loader)} batches | test: {len(test_loader)} batches")

    return train_loader, val_loader, test_loader
