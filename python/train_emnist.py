"""
train_emnist.py - EMNIST ByClass Training Pipeline

Downloads the EMNIST ByClass split (62 classes: digits 0-9, uppercase A-Z,
lowercase a-z), applies the necessary transpose fix for torchvision, trains
the HandwrittenCNN model, and exports evaluation metrics and checkpoints.

IMPORTANT: EMNIST images loaded via torchvision are transposed (rotated 90°
and flipped). This pipeline applies a correction transform to fix orientation.

Outputs:
    - ../web/data/model/emnist_model.pth   (model weights)
    - ../web/data/training_metrics_emnist.json   (training + evaluation metrics)

Usage:
    python train_emnist.py
"""

from __future__ import annotations

import json
import os
import string
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from evaluate import evaluate_model, export_metrics, generate_sample_predictions
from model import HandwrittenCNN

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.001
EPOCHS: int = 20
SCHEDULER_STEP: int = 7
SCHEDULER_GAMMA: float = 0.5
VAL_SPLIT: float = 0.1
NUM_CLASSES: int = 62
RANDOM_SEED: int = 42

# EMNIST ByClass label mapping: 0-9 → digits, 10-35 → A-Z, 36-61 → a-z
CLASS_NAMES: List[str] = (
    [str(i) for i in range(10)]
    + list(string.ascii_uppercase)
    + list(string.ascii_lowercase)
)

# Output paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "web", "data", "model")
METRICS_PATH = os.path.join(
    SCRIPT_DIR, "..", "web", "data", "training_metrics_emnist.json"
)
MODEL_PATH = os.path.join(MODEL_DIR, "emnist_model.pth")


class EMNISTTransposeFix:
    """Fix the EMNIST image orientation issue in torchvision.

    EMNIST images as loaded by torchvision are transposed (90° rotation
    + horizontal flip) compared to their intended orientation. This
    transform corrects the orientation by transposing the image.

    This should be applied as the FIRST transform in the pipeline,
    before any augmentation or normalization.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        """Transpose (rotate 90° + flip) the image to correct orientation.

        Args:
            img: PIL Image loaded by torchvision.

        Returns:
            Corrected PIL Image with proper orientation.
        """
        return img.transpose(Image.TRANSPOSE)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Build training and evaluation transform pipelines.

    Both pipelines start with the EMNIST transpose fix. Training
    transforms include random rotation and affine translation for
    data augmentation.

    Returns:
        Tuple of (train_transform, eval_transform).
    """
    train_transform = transforms.Compose(
        [
            EMNISTTransposeFix(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1751,), std=(0.3332,)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            EMNISTTransposeFix(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1751,), std=(0.3332,)),
        ]
    )

    return train_transform, eval_transform


def prepare_data(
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Download EMNIST ByClass and create train/val/test dataloaders.

    The EMNIST ByClass split contains 62 classes:
        - 0-9: Digits
        - 10-35: Uppercase letters A-Z
        - 36-61: Lowercase letters a-z

    Total: ~814k samples (697k train + 116k test).

    Args:
        train_transform: Augmentation pipeline for training data.
        eval_transform: Normalization-only pipeline for val/test data.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_dir = os.path.join(SCRIPT_DIR, "data")

    # Download EMNIST ByClass split
    print("[DATA] Downloading EMNIST ByClass... (this may take a few minutes)")
    full_train = datasets.EMNIST(
        root=data_dir,
        split="byclass",
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.EMNIST(
        root=data_dir,
        split="byclass",
        train=False,
        download=True,
        transform=eval_transform,
    )

    # Split training into train + validation
    val_size = int(len(full_train) * VAL_SPLIT)
    train_size = len(full_train) - val_size

    torch.manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(
        f"[DATA] Train: {train_size:,}  |  Val: {val_size:,}  |  "
        f"Test: {len(test_dataset):,}"
    )
    print(f"[DATA] Classes: {NUM_CLASSES} ({CLASS_NAMES[:5]}...{CLASS_NAMES[-5:]})")
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: HandwrittenCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train the model for one epoch.

    Args:
        model: The CNN model.
        loader: Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Computation device.

    Returns:
        Tuple of (average_loss, accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # Progress indicator for large dataset
        if (batch_idx + 1) % 500 == 0:
            print(
                f"    Batch {batch_idx + 1}/{len(loader)} | "
                f"Running Acc: {correct/total:.4f}"
            )

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: HandwrittenCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on a validation/test set.

    Args:
        model: The CNN model.
        loader: Validation or test dataloader.
        criterion: Loss function.
        device: Computation device.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main() -> None:
    """Main EMNIST training and evaluation pipeline."""
    print("=" * 65)
    print("  EMNIST ByClass Character Recognition - Training Pipeline")
    print("  62 classes: 0-9, A-Z, a-z")
    print("=" * 65)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using: {device}")

    # Reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Data
    train_transform, eval_transform = get_transforms()
    train_loader, val_loader, test_loader = prepare_data(
        train_transform, eval_transform
    )

    # Model (62 classes for EMNIST ByClass)
    model = HandwrittenCNN(num_classes=NUM_CLASSES).to(device)
    summary = model.get_model_summary()
    print(f"[MODEL] Total parameters: {summary['total_params']:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA
    )

    # ── Training loop ─────────────────────────────────────────────
    epoch_metrics: List[Dict[str, Any]] = []
    best_val_acc = 0.0
    start_time = time.time()

    print(
        f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
        f"{'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8}"
    )
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        print(
            f"{epoch:>6d} | {train_loss:>10.4f} | {train_acc:>8.4f} | "
            f"{val_loss:>8.4f} | {val_acc:>7.4f} | {current_lr:>8.6f}  "
            f"({epoch_time:.0f}s)"
        )

        epoch_metrics.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "learning_rate": current_lr,
            }
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"         ↳ Saved best model (val_acc={val_acc:.4f})")

        scheduler.step()

    training_time = time.time() - start_time
    print(f"\n[DONE] Training complete in {training_time:.1f}s")
    print(f"[DONE] Best validation accuracy: {best_val_acc:.4f}")

    # ── Final Evaluation on Test Set ──────────────────────────────
    print("\n" + "=" * 65)
    print("  Evaluating on EMNIST Test Set")
    print("=" * 65)

    # Load best model for evaluation
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    eval_metrics = evaluate_model(
        model, test_loader, NUM_CLASSES, CLASS_NAMES, device
    )

    print(f"\nTest Accuracy: {eval_metrics['overall_accuracy']:.4f}")

    # Show per-class metrics (grouped by category)
    print(f"\nPer-class metrics (showing worst 10 classes):")
    print(f"  {'Class':>5} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'Support':>7}")
    print(f"  {'-'*40}")

    # Sort by F1 score to show hardest classes
    sorted_metrics = sorted(
        eval_metrics["per_class_metrics"], key=lambda x: x["f1"]
    )
    for m in sorted_metrics[:10]:
        print(
            f"  {m['class']:>5} | {m['precision']:>6.3f} | "
            f"{m['recall']:>6.3f} | {m['f1']:>6.3f} | {m['support']:>7d}"
        )

    # ── Export all metrics ────────────────────────────────────────
    all_metrics: Dict[str, Any] = {
        "summary": {
            "test_accuracy": eval_metrics["overall_accuracy"],
            "total_params": summary["total_params"],
            "training_epochs": EPOCHS,
            "training_time_seconds": round(training_time),
        },
        "epoch_metrics": epoch_metrics,
        "per_class_metrics": eval_metrics["per_class_metrics"],
        "confusion_matrix": eval_metrics["confusion_matrix"],
        "roc_data": eval_metrics["roc_data"],
    }

    export_metrics(all_metrics, METRICS_PATH)
    print(f"\n[EXPORT] Metrics saved to: {METRICS_PATH}")
    print(f"[EXPORT] Model saved to:   {MODEL_PATH}")


if __name__ == "__main__":
    main()
