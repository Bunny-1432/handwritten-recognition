"""
export_onnx.py
==============
Exports PyTorch .pth weights to ONNX format for browser deployment via onnxruntime-web.
"""
from __future__ import annotations
import os
import torch
import argparse
from src.backend.core.model import build_model

def export_to_onnx(dataset: str, pth_path: str, onnx_path: str, num_classes: int):
    print(f"Exporting {dataset} model to ONNX...")

    # 1. Build model and load weights
    device = torch.device("cpu")
    model = build_model(num_classes=num_classes, pretrained_path=pth_path, device=device)
    model.eval()

    # 2. Create dummy input matching the model's expected input shape (1, 1, 28, 28)
    dummy_input = torch.randn(1, 1, 28, 28).to(device)

    # 3. Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11, # Widely supported by ONNX Runtime Web
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Successfully exported {dataset} model to {onnx_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["mnist", "emnist"], required=True)
    args = parser.parse_args()

    configs = {
        "mnist": {"num_classes": 10, "pth": "data/model/mnist_model.pth", "onnx": "data/model/mnist.onnx"},
        "emnist": {"num_classes": 62, "pth": "data/model/emnist_model.pth", "onnx": "data/model/emnist.onnx"}
    }

    cfg = configs[args.dataset]

    if not os.path.exists(cfg["pth"]):
        print(f"Error: Weights file not found at {cfg['pth']}. Train the model first.")
        return

    export_to_onnx(args.dataset, cfg["pth"], cfg["onnx"], cfg["num_classes"])

if __name__ == "__main__":
    main()
