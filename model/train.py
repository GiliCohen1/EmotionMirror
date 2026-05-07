#!/usr/bin/env python3
"""
model/train.py

Main training script for EmotionMirror.

Usage:
    python model/train.py --config model/configs/baseline.yaml
    python model/train.py --config model/configs/efficientnet.yaml

Features:
  - Train/val/test split with proper evaluation
  - Weighted sampler for class imbalance
  - Cosine or OneCycle LR scheduling
  - Early stopping
  - Per-epoch confusion matrix
  - GradCAM samples saved every 5 epochs
  - Best model checkpoint + ONNX export
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is on sys.path so `model.*` imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from model.data.dataset import build_dataloaders, EMOTIONS
from model.model import build_model, EfficientNetEmotion


# ──────────────────────────────────────────────
# Device setup
# ──────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Apple MPS (M-series GPU)")
    else:
        device = torch.device("cpu")
        print("[Device] CPU — training will be slow, consider Google Colab for GPU")
    return device


# ──────────────────────────────────────────────
# Optimizer & scheduler
# ──────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: dict) -> optim.Optimizer:
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    # Differential learning rates: lower LR for pretrained backbone
    if isinstance(model, EfficientNetEmotion) and "backbone_lr_multiplier" in train_cfg:
        backbone_params = list(model.features.parameters())
        head_params = list(model.classifier.parameters())
        param_groups = [
            {"params": backbone_params, "lr": train_cfg["learning_rate"] * train_cfg["backbone_lr_multiplier"]},
            {"params": head_params, "lr": train_cfg["learning_rate"]},
        ]
    else:
        param_groups = model.parameters()

    optimizer_name = train_cfg.get("optimizer", "adam").lower()
    if optimizer_name == "adam":
        return optim.Adam(param_groups, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    elif optimizer_name == "adamw":
        return optim.AdamW(param_groups, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    elif optimizer_name == "sgd":
        return optim.SGD(param_groups, lr=train_cfg["learning_rate"], momentum=0.9, weight_decay=train_cfg["weight_decay"])
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(optimizer: optim.Optimizer, cfg: dict, train_loader_len: int):
    train_cfg = cfg["training"]
    scheduler_name = train_cfg.get("scheduler", "cosine").lower()

    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"], eta_min=1e-6
        )
    elif scheduler_name == "one_cycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=train_cfg["learning_rate"],
            steps_per_epoch=train_loader_len,
            epochs=train_cfg["epochs"],
        )
    elif scheduler_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        return None


# ──────────────────────────────────────────────
# Training step
# ──────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler=None,
    gradient_clip: float = 1.0,
    scheduler_step_per_batch: bool = False,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping prevents exploding gradients
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        if scheduler_step_per_batch and scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / len(loader),
        "accuracy": correct / total,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()

    return (
        {"loss": total_loss / len(loader), "accuracy": accuracy},
        all_preds,
        all_labels,
    )


# ──────────────────────────────────────────────
# Visualizations
# ──────────────────────────────────────────────

def save_confusion_matrix(preds: np.ndarray, labels: np.ndarray, save_path: str):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm, annot=True, fmt="d", xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=axes[0])
    axes[0].set_title("Confusion matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=axes[1])
    axes[1].set_title("Confusion matrix (normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def save_training_curves(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.savefig(save_path, dpi=100)
    plt.close()


# ──────────────────────────────────────────────
# ONNX export
# ──────────────────────────────────────────────

def export_onnx(model: nn.Module, cfg: dict, device: torch.device):
    model.eval()
    image_size = cfg["data"]["image_size"]
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    onnx_path = cfg["output"]["onnx_path"]
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model.cpu(),
        dummy_input.cpu(),
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"[ONNX] Exported to {onnx_path}")


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────

def train(cfg: dict, resume: bool = False):
    device = get_device()
    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    exp_name = cfg["experiment_name"]

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # Model
    model = build_model(cfg).to(device)

    # Loss — label smoothing reduces overconfidence and helps with FER2013 label noise
    label_smoothing = cfg["training"].get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer & scheduler
    optimizer = build_optimizer(model, cfg)
    is_one_cycle = cfg["training"].get("scheduler", "") == "one_cycle"
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    # Training state
    best_val_acc = 0.0
    patience_counter = 0
    start_epoch = 1
    patience = cfg["training"]["early_stopping_patience"]
    freeze_epochs = cfg["model"].get("freeze_backbone_epochs", 0)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Resume from best checkpoint if requested
    checkpoint_path = checkpoint_dir / f"{exp_name}_best.pt"
    if resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_val_acc = ckpt["val_accuracy"]
        start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] Loaded checkpoint from epoch {ckpt['epoch']} (val acc {best_val_acc:.3f})")
        print(f"[Resume] Continuing from epoch {start_epoch}")

    print(f"\n{'='*60}")
    print(f"  Training: {exp_name}")
    print(f"  Epochs: {cfg['training']['epochs']} | Batch: {cfg['training']['batch_size']}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):

        # Unfreeze EfficientNet backbone after warm-up
        if epoch == freeze_epochs + 1 and isinstance(model, EfficientNetEmotion):
            model.unfreeze_backbone()
            # Rebuild optimizer with backbone params
            optimizer = build_optimizer(model, cfg)
            scheduler = build_scheduler(optimizer, cfg, len(train_loader))

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler if is_one_cycle else None,
            gradient_clip=cfg["training"].get("gradient_clip", 1.0),
            scheduler_step_per_batch=is_one_cycle,
        )

        # Validate
        val_metrics, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

        # Step epoch-level scheduler
        if scheduler is not None and not is_one_cycle:
            scheduler.step()

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        elapsed = time.time() - start_time
        lr = optimizer.param_groups[-1]["lr"]
        print(
            f"Epoch {epoch:3d}/{cfg['training']['epochs']} | "
            f"train loss: {train_metrics['loss']:.4f} acc: {train_metrics['accuracy']:.3f} | "
            f"val loss: {val_metrics['loss']:.4f} acc: {val_metrics['accuracy']:.3f} | "
            f"lr: {lr:.2e} | {elapsed/60:.1f}m"
        )

        # Save GradCAM samples
        if epoch % 5 == 0 and cfg["logging"]["save_gradcam_samples"]:
            try:
                from model.gradcam import run_gradcam_on_batch
                sample_images, sample_labels = next(iter(val_loader))
                gradcam_path = checkpoint_dir / f"{exp_name}_gradcam_epoch{epoch:03d}.png"
                run_gradcam_on_batch(model, sample_images, sample_labels, save_path=str(gradcam_path))
            except Exception as e:
                print(f"[GradCAM] Skipped: {e}")

        # Checkpoint
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            checkpoint_path = checkpoint_dir / f"{exp_name}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "config": cfg,
            }, checkpoint_path)
            print(f"  [best]  New best! Saved to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EarlyStopping] No improvement for {patience} epochs. Stopping.")
                break

    # ── Final evaluation on test set ──
    print(f"\n{'='*60}")
    print("  Final evaluation on test set")
    print(f"{'='*60}")

    checkpoint = torch.load(checkpoint_dir / f"{exp_name}_best.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"\nTest accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print()
    print(classification_report(test_labels, test_preds, target_names=EMOTIONS))

    # Save artifacts
    if cfg["logging"]["save_confusion_matrix"]:
        cm_path = checkpoint_dir / f"{exp_name}_confusion_matrix.png"
        save_confusion_matrix(test_preds, test_labels, str(cm_path))

    curves_path = checkpoint_dir / f"{exp_name}_training_curves.png"
    save_training_curves(history, str(curves_path))

    if cfg["output"]["export_onnx"]:
        export_onnx(model, cfg, device)

    print(f"\nTotal training time: {(time.time()-start_time)/60:.1f} minutes")
    print(f"Artifacts saved to: {checkpoint_dir}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

class _TeeWriter:
    """Writes to both a file and the original stream (line-buffered)."""
    def __init__(self, stream, path):
        self._stream = stream
        self._file = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
    def write(self, msg):
        try:
            if self._stream is not None:
                self._stream.write(msg)
        except Exception:
            pass
        self._file.write(msg)
    def flush(self):
        try:
            if self._stream is not None:
                self._stream.flush()
        except Exception:
            pass
        self._file.flush()
    def fileno(self):
        try:
            return self._stream.fileno()
        except Exception:
            return -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EmotionMirror emotion classifier")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Mirror all stdout/stderr to a log file that can be tailed without locking
    log_path = Path(cfg["output"]["checkpoint_dir"]) / "train_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = _TeeWriter(sys.stdout, str(log_path))
    sys.stderr = _TeeWriter(sys.stderr, str(log_path))

    try:
        train(cfg, resume=args.resume)
    except Exception as e:
        import traceback
        print(f"\n[FATAL] Training crashed: {e}")
        traceback.print_exc()
        raise
