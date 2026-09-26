"""
fast_train.py
===============
A high-speed training pipeline for rapid deployment.
Uses fewer epochs and optimized settings to get a working model quickly.
"""
from __future__ import annotations
import os
import argparse
import string
import time
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image

from src.backend.core.model import HandwrittenCNN
from src.backend.utils.evaluate import evaluate_model, export_metrics

# --- EMNIST Fix ---
class EMNISTTransposeFix:
    def __call__(self, img: Image.Image) -> Image.Image:
        return img.transpose(Image.TRANSPOSE)
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

def get_config(dataset: str) -> Dict[str, Any]:
    if dataset == "mnist":
        return {
            "num_classes": 10,
            "class_names": [str(i) for i in range(10)],
            "mean": (0.1307,),
            "std": (0.3081,),
            "epochs": 5, # FAST VERSION: Reduced from 25
            "split": None
        }
    elif dataset == "emnist":
        return {
            "num_classes": 62,
            "class_names": [str(i) for i in range(10)] + list(string.ascii_uppercase) + list(string.ascii_lowercase),
            "mean": (0.1751,),
            "std": (0.3332,),
            "epochs": 10, # FAST VERSION: Reduced from 30
            "split": "byclass"
        }
    raise ValueError(f"Unsupported dataset: {dataset}")

def get_transforms(dataset: str, is_train: bool) -> transforms.Compose:
    cfg = get_config(dataset)
    t_list = []
    if dataset == "emnist":
        t_list.append(EMNISTTransposeFix())
    if is_train:
        t_list.append(transforms.RandomRotation(degrees=10))
        t_list.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5)))
    t_list.append(transforms.ToTensor())
    t_list.append(transforms.Normalize(mean=cfg["mean"], std=cfg["std"]))
    return transforms.Compose(t_list)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
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
    return running_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["mnist", "emnist"], required=True)
    args = parser.parse_args()
    cfg = get_config(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf, eval_tf = get_transforms(args.dataset, True), get_transforms(args.dataset, False)
    if args.dataset == "mnist":
        full_train = datasets.MNIST(root="data/raw", train=True, download=True, transform=train_tf)
        test_dataset = datasets.MNIST(root="data/raw", train=False, download=True, transform=eval_tf)
    else:
        full_train = datasets.EMNIST(root="data/raw", split=cfg["split"], train=True, download=True, transform=train_tf)
        test_dataset = datasets.EMNIST(root="data/raw", split=cfg["split"], train=False, download=True, transform=eval_tf)

    val_size = int(len(full_train) * 0.1)
    train_ds, val_ds = random_split(full_train, [len(full_train)-val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    model = HandwrittenCNN(num_classes=cfg["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.002, steps_per_epoch=len(train_loader), epochs=cfg["epochs"])

    best_acc = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{cfg['epochs']} | Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f}")
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), f"data/model/{args.dataset}_model.pth")
        scheduler.step()

    print(f"[DONE] {args.dataset} trained. Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()
