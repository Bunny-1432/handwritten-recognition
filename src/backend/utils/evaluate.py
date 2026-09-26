"""
evaluate.py - Evaluation Utilities for Handwritten Character Recognition

Provides functions for comprehensive model evaluation including per-class
metrics, confusion matrices, ROC curves, and sample prediction generation.

Usage:
    from evaluate import evaluate_model, export_metrics

    metrics = evaluate_model(model, test_loader, num_classes=10, class_names=DIGITS)
    export_metrics(metrics, "metrics.json")
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from torch.utils.data import DataLoader


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    class_names: List[str],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run comprehensive evaluation on a trained model.

    Collects predictions over the entire dataloader and computes accuracy,
    per-class precision/recall/F1, confusion matrix, and one-vs-rest ROC
    curves with AUC for every class.

    Args:
        model: Trained PyTorch model in eval mode.
        dataloader: DataLoader for the evaluation dataset.
        num_classes: Total number of classes (10 for MNIST, 62 for EMNIST).
        class_names: Human-readable labels for each class index.
        device: Device to run inference on. Defaults to CUDA if available.

    Returns:
        Dictionary containing:
            - ``overall_accuracy`` (float): Top-1 accuracy on the full set.
            - ``per_class_metrics`` (list[dict]): For each class, a dict with
              ``class``, ``precision``, ``recall``, ``f1``, ``support``.
            - ``confusion_matrix`` (list[list[int]]): ``num_classes × num_classes``
              confusion matrix as nested Python lists.
            - ``roc_data`` (list[dict]): For each class, a dict with
              ``class``, ``auc``, ``fpr`` (list[float]), ``tpr`` (list[float]).

    Example:
        >>> metrics = evaluate_model(model, test_loader, 10, [str(i) for i in range(10)])
        >>> print(f"Accuracy: {metrics['overall_accuracy']:.4f}")
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.vstack(all_probs)

    # --- Overall accuracy ---
    overall_accuracy = float(accuracy_score(y_true, y_pred))

    # --- Per-class precision, recall, F1, support ---
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )

    per_class_metrics: List[Dict[str, Any]] = []
    for i in range(num_classes):
        per_class_metrics.append(
            {
                "class": class_names[i],
                "precision": float(round(precision[i], 4)),
                "recall": float(round(recall[i], 4)),
                "f1": float(round(f1[i], 4)),
                "support": int(support[i]),
            }
        )

    # --- Confusion matrix ---
    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(num_classes))
    )
    confusion_matrix_list: List[List[int]] = cm.tolist()

    # --- ROC curves (one-vs-rest for each class) ---
    roc_data: List[Dict[str, Any]] = []
    for i in range(num_classes):
        # Binary labels for one-vs-rest
        y_binary = (y_true == i).astype(int)

        # Skip classes with no positive samples
        if y_binary.sum() == 0:
            roc_data.append(
                {
                    "class": class_names[i],
                    "auc": 0.0,
                    "fpr": [0.0, 1.0],
                    "tpr": [0.0, 1.0],
                }
            )
            continue

        fpr, tpr, _ = roc_curve(y_binary, y_probs[:, i])
        roc_auc = float(auc(fpr, tpr))

        # Subsample to ~20 points for JSON compactness
        if len(fpr) > 20:
            indices = np.linspace(0, len(fpr) - 1, 20, dtype=int)
            fpr = fpr[indices]
            tpr = tpr[indices]

        roc_data.append(
            {
                "class": class_names[i],
                "auc": round(roc_auc, 4),
                "fpr": [round(float(v), 6) for v in fpr],
                "tpr": [round(float(v), 6) for v in tpr],
            }
        )

    return {
        "overall_accuracy": round(overall_accuracy, 4),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix_list,
        "roc_data": roc_data,
    }


def export_metrics(
    metrics: Dict[str, Any],
    filepath: str,
    indent: int = 2,
) -> None:
    """Export evaluation metrics to a JSON file.

    Handles conversion of NumPy types (int64, float64, ndarray) to
    JSON-serializable Python types automatically.

    Args:
        metrics: Dictionary of evaluation results (from ``evaluate_model``).
        filepath: Output JSON file path.
        indent: JSON indentation level. Defaults to 2.
    """

    class NumpyEncoder(json.JSONEncoder):
        """Custom encoder that converts numpy types to native Python."""

        def default(self, obj: Any) -> Any:
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=indent, cls=NumpyEncoder)

    print(f"[INFO] Metrics exported to: {filepath}")


def generate_sample_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_samples: int = 20,
    class_names: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """Generate sample predictions with images, labels, and confidences.

    Runs inference on the first ``num_samples`` items from the dataloader
    and returns a list of prediction records.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader to draw samples from.
        num_samples: Number of sample predictions to generate.
        class_names: Optional human-readable names for class indices.
        device: Inference device. Defaults to CUDA if available.

    Returns:
        List of dicts, each containing:
            - ``image`` (list[list[float]]): 28×28 pixel values [0, 1].
            - ``true_label`` (int): Ground-truth class index.
            - ``true_name`` (str): Ground-truth class name.
            - ``predicted_label`` (int): Predicted class index.
            - ``predicted_name`` (str): Predicted class name.
            - ``confidence`` (float): Softmax probability for the prediction.
            - ``correct`` (bool): Whether prediction matches ground truth.
            - ``top3`` (list[dict]): Top-3 predictions with class and prob.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    samples: List[Dict[str, Any]] = []
    collected = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            for i in range(images.size(0)):
                if collected >= num_samples:
                    return samples

                prob = probs[i].cpu().numpy()
                pred = int(np.argmax(prob))
                true = int(labels[i].item())

                # Top-3 predictions
                top3_indices = np.argsort(prob)[::-1][:3]
                top3 = [
                    {
                        "class": int(idx),
                        "name": (
                            class_names[idx] if class_names else str(idx)
                        ),
                        "probability": round(float(prob[idx]), 4),
                    }
                    for idx in top3_indices
                ]

                true_name = class_names[true] if class_names else str(true)
                pred_name = class_names[pred] if class_names else str(pred)

                samples.append(
                    {
                        "image": images[i]
                        .squeeze()
                        .cpu()
                        .numpy()
                        .tolist(),
                        "true_label": true,
                        "true_name": true_name,
                        "predicted_label": pred,
                        "predicted_name": pred_name,
                        "confidence": round(float(prob[pred]), 4),
                        "correct": pred == true,
                        "top3": top3,
                    }
                )
                collected += 1

    return samples
