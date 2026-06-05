"""
model.py - CNN Architecture for Handwritten Character Recognition

Defines a configurable Convolutional Neural Network (CNN) for recognizing
handwritten characters from 28×28 grayscale images. Supports both MNIST
(10 digit classes) and EMNIST (62 alphanumeric classes).

Architecture Overview:
    Input (1×28×28) → Conv Blocks (32→64→128) → Adaptive Pool → FC → Output

Author: Handwritten Character Recognition Project
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HandwrittenCNN(nn.Module):
    """Convolutional Neural Network for handwritten character classification.

    A 3-block CNN with batch normalization, adaptive pooling, and dropout
    regularization. Designed for 28×28 single-channel grayscale images.

    Args:
        num_classes: Number of output classes.
            - 10 for MNIST (digits 0-9)
            - 62 for EMNIST ByClass (digits + uppercase + lowercase)

    Attributes:
        features: Sequential feature extraction layers (conv blocks).
        adaptive_pool: Adaptive average pooling to fixed spatial size.
        classifier: Fully connected classification head.
        _feature_maps: Storage for intermediate feature maps when hooks
            are registered via ``register_feature_hooks()``.

    Example:
        >>> model = HandwrittenCNN(num_classes=10)
        >>> x = torch.randn(1, 1, 28, 28)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([1, 10])
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes

        # --- Feature Extraction Blocks ---

        # Conv Block 1: 1 → 64 channels, spatial: 28→14
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv Block 2: 64 → 128 channels, spatial: 14→7
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv Block 3: 128 → 256 channels, spatial: 7→7 (no pooling)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Adaptive average pooling to fixed 3×3 spatial resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))

        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

        # Storage for feature map extraction via hooks
        self._feature_maps: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape ``(batch, 1, 28, 28)``.

        Returns:
            Raw logits of shape ``(batch, num_classes)``.
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference and return predicted classes with probabilities.

        Args:
            x: Input tensor of shape ``(batch, 1, 28, 28)``.

        Returns:
            Tuple of (predicted_classes, probabilities) where:
                - predicted_classes: shape ``(batch,)``
                - probabilities: shape ``(batch, num_classes)``
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs

    def get_model_summary(self) -> Dict[str, Any]:
        """Return a structured summary of the model architecture.

        Returns:
            Dictionary containing:
                - ``num_classes``: Number of output classes.
                - ``total_params``: Total number of learnable parameters.
                - ``trainable_params``: Number of trainable parameters.
                - ``layers``: List of dicts describing each layer with name,
                  type, output shape, parameter count, and extra info.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        layers: List[Dict[str, Any]] = [
            {"name": "Input", "type": "input", "output_shape": [1, 28, 28], "params": 0},
            {"name": "Conv1_1", "type": "conv", "output_shape": [32, 28, 28], "params": self._count_params(self.conv_block1[0])},
            {"name": "Conv1_2", "type": "conv", "output_shape": [64, 28, 28], "params": self._count_params(self.conv_block1[3])},
            {"name": "MaxPool1", "type": "pool", "output_shape": [64, 14, 14], "params": 0},
            {"name": "Conv2_1", "type": "conv", "output_shape": [128, 14, 14], "params": self._count_params(self.conv_block2[0])},
            {"name": "Conv2_2", "type": "conv", "output_shape": [128, 14, 14], "params": self._count_params(self.conv_block2[3])},
            {"name": "MaxPool2", "type": "pool", "output_shape": [128, 7, 7], "params": 0},
            {"name": "Conv3_1", "type": "conv", "output_shape": [256, 7, 7], "params": self._count_params(self.conv_block3[0])},
            {"name": "Conv3_2", "type": "conv", "output_shape": [256, 7, 7], "params": self._count_params(self.conv_block3[3])},
            {"name": "AdaptivePool", "type": "pool", "output_shape": [256, 3, 3], "params": 0},
            {"name": "Flatten", "type": "flatten", "output_shape": [2304], "params": 0},
            {"name": "FC1", "type": "dense", "output_shape": [512], "params": self._count_params(self.classifier[1])},
            {"name": "Dropout", "type": "dropout", "output_shape": [512], "params": 0, "rate": 0.5},
            {"name": "FC2", "type": "dense", "output_shape": [self.num_classes], "params": self._count_params(self.classifier[4])},
        ]

        return {
            "num_classes": self.num_classes,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "layers": layers,
        }

    def register_feature_hooks(self) -> None:
        """Register forward hooks on each conv block for feature map extraction.

        After calling this method, running a forward pass will populate
        ``self._feature_maps`` with intermediate activations keyed by
        block name (``'conv_block1'``, ``'conv_block2'``, ``'conv_block3'``).

        Example:
            >>> model.register_feature_hooks()
            >>> _ = model(x)
            >>> feature_maps = model.get_feature_maps()
        """
        self.remove_feature_hooks()  # Clean up any existing hooks

        def _make_hook(name: str):
            def hook(
                module: nn.Module,
                input: Tuple[torch.Tensor, ...],
                output: torch.Tensor,
            ) -> None:
                self._feature_maps[name] = output.detach()
            return hook

        for name, block in [
            ("conv_block1", self.conv_block1),
            ("conv_block2", self.conv_block2),
            ("conv_block3", self.conv_block3),
        ]:
            handle = block.register_forward_hook(_make_hook(name))
            self._hooks.append(handle)

    def remove_feature_hooks(self) -> None:
        """Remove all registered feature extraction hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._feature_maps.clear()

    def get_feature_maps(self) -> Dict[str, torch.Tensor]:
        """Return captured feature maps from the last forward pass.

        Returns:
            Dictionary mapping block names to their output tensors.
            Empty if ``register_feature_hooks()`` has not been called or
            no forward pass has been executed since registration.
        """
        return self._feature_maps.copy()

    @staticmethod
    def _count_params(module: nn.Module) -> int:
        """Count the total number of parameters in a single module."""
        return sum(p.numel() for p in module.parameters())


def build_model(
    num_classes: int = 10,
    pretrained_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> HandwrittenCNN:
    """Factory function to build and optionally load a pretrained model.

    Args:
        num_classes: Number of output classes (10 for MNIST, 62 for EMNIST).
        pretrained_path: Path to a saved state dict. If provided, the weights
            are loaded into the model.
        device: Device to place the model on. Defaults to CUDA if available.

    Returns:
        An initialized ``HandwrittenCNN`` instance.

    Raises:
        FileNotFoundError: If ``pretrained_path`` is given but does not exist.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HandwrittenCNN(num_classes=num_classes).to(device)

    if pretrained_path is not None:
        import os

        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(
                f"Pretrained model not found: {pretrained_path}"
            )
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[INFO] Loaded pretrained weights from: {pretrained_path}")

    return model


if __name__ == "__main__":
    # Quick sanity check
    for n_cls, label in [(10, "MNIST"), (62, "EMNIST")]:
        model = HandwrittenCNN(num_classes=n_cls)
        summary = model.get_model_summary()
        print(f"\n{'='*50}")
        print(f"Model: {label} ({n_cls} classes)")
        print(f"Total parameters: {summary['total_params']:,}")
        print(f"{'='*50}")
        for layer in summary["layers"]:
            print(
                f"  {layer['name']:15s} | {str(layer['type']):12s} | "
                f"Shape: {str(layer['output_shape']):20s} | "
                f"Params: {layer['params']:,}"
            )

    # Test forward pass
    model = HandwrittenCNN(num_classes=10)
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    print(f"\nForward pass: input {x.shape} -> output {out.shape}")

    # Test feature hook extraction
    model.register_feature_hooks()
    _ = model(x)
    fmaps = model.get_feature_maps()
    for name, fmap in fmaps.items():
        print(f"Feature map '{name}': {fmap.shape}")
    model.remove_feature_hooks()
