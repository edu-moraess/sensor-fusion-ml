"""
Minimal, deterministic-friendly encoders for software testing and early experiments.

These are intentionally small so that unit tests and Colab CPU runs remain fast.
They are NOT intended as high-accuracy perception backbones.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Tiny CNN for RGB feature extraction (test / prototype only)."""

    def __init__(self, in_channels: int = 3, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        return self.net(x)


class SmallPointNet(nn.Module):
    """Minimal PointNet-style encoder (global feature only)."""

    def __init__(self, in_dim: int = 3, out_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(64, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, in_dim) point cloud.
        Returns global feature (B, out_dim).
        """
        x = self.mlp(x)               # (B, N, 64)
        x = torch.max(x, dim=1)[0]    # (B, 64)
        return self.fc(x)
