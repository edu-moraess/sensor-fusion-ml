"""
Minimal trainable fusion model (Phase D).

- Explicit handling of missing modalities via learned or zero embeddings.
- Separate from classical baseline.
- Designed for unit tests with synthetic data and small-batch gradient checks.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from sensor_fusion_ml.encoders.minimal import SmallCNN, SmallPointNet


class MinimalFusionMLP(nn.Module):
    """
    Prototype multi-modal fusion network.

    Inputs (all optional except that at least one modality should be present):
        rgb:     (B, 3, H, W)
        points:  (B, N, 3)
        gnss:    (B, 3) or (B, 2)   – treated as a direct feature
        imu:     (B, 6)             – acc + gyro (optional)

    Outputs:
        position:   (B, dim)
        velocity:   (B, dim)
        trajectory: (B, horizon, dim)  if enabled
    """

    def __init__(
        self,
        position_dim: int = 2,
        hidden_dim: int = 128,
        trajectory_horizon: int = 10,
        use_gru: bool = False,
        dropout: float = 0.1,
        rgb_out_dim: int = 64,
        point_out_dim: int = 64,
    ):
        super().__init__()
        self.position_dim = position_dim
        self.trajectory_horizon = trajectory_horizon
        self.use_gru = use_gru

        self.rgb_encoder = SmallCNN(out_dim=rgb_out_dim)
        self.point_encoder = SmallPointNet(out_dim=point_out_dim)

        # Learned embeddings for missing modalities
        self.rgb_missing = nn.Parameter(torch.zeros(1, rgb_out_dim))
        self.point_missing = nn.Parameter(torch.zeros(1, point_out_dim))
        self.gnss_missing = nn.Parameter(torch.zeros(1, position_dim))
        self.imu_missing = nn.Parameter(torch.zeros(1, 6))

        fusion_in = rgb_out_dim + point_out_dim + position_dim + 6
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.pos_head = nn.Linear(hidden_dim, position_dim)
        self.vel_head = nn.Linear(hidden_dim, position_dim)

        if trajectory_horizon > 0:
            self.traj_head = nn.Linear(hidden_dim, trajectory_horizon * position_dim)
        else:
            self.traj_head = None

        if use_gru:
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        else:
            self.gru = None

    def forward(
        self,
        rgb: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        gnss: Optional[torch.Tensor] = None,
        imu: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = self._infer_batch_size(rgb, points, gnss, imu)

        # Encode or use missing embedding
        if rgb is not None:
            rgb_feat = self.rgb_encoder(rgb)
        else:
            rgb_feat = self.rgb_missing.expand(B, -1)

        if points is not None:
            point_feat = self.point_encoder(points)
        else:
            point_feat = self.point_missing.expand(B, -1)

        if gnss is not None:
            gnss_feat = gnss
            if gnss_feat.shape[-1] != self.position_dim:
                # pad or truncate to expected dim
                if gnss_feat.shape[-1] > self.position_dim:
                    gnss_feat = gnss_feat[..., :self.position_dim]
                else:
                    pad = torch.zeros(B, self.position_dim - gnss_feat.shape[-1], device=gnss_feat.device)
                    gnss_feat = torch.cat([gnss_feat, pad], dim=-1)
        else:
            gnss_feat = self.gnss_missing.expand(B, -1)

        if imu is not None:
            imu_feat = imu
        else:
            imu_feat = self.imu_missing.expand(B, -1)

        fused = torch.cat([rgb_feat, point_feat, gnss_feat, imu_feat], dim=-1)
        h = self.fusion(fused)

        if self.use_gru and history is not None:
            # history: (B, T, hidden_dim) – advanced usage; not required for basic tests
            h_seq, _ = self.gru(history)
            h = h_seq[:, -1]

        out = {
            "position": self.pos_head(h),
            "velocity": self.vel_head(h),
        }
        if self.traj_head is not None:
            traj = self.traj_head(h).view(B, self.trajectory_horizon, self.position_dim)
            out["trajectory"] = traj
        return out

    @staticmethod
    def _infer_batch_size(*tensors: Optional[torch.Tensor]) -> int:
        for t in tensors:
            if t is not None:
                return t.shape[0]
        raise ValueError("At least one modality tensor must be provided")
