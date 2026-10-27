"""
Standard metrics for position, velocity and trajectory prediction.

All functions expect numpy arrays or torch tensors that can be converted.
They raise if shapes are incompatible. They do NOT invent ground truth.
"""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[np.ndarray, "torch.Tensor"]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mae(pred: ArrayLike, target: ArrayLike) -> float:
    """Mean Absolute Error (any shape, element-wise)."""
    p, t = _to_numpy(pred), _to_numpy(target)
    if p.shape != t.shape:
        raise ValueError(f"Shape mismatch: pred {p.shape} vs target {t.shape}")
    return float(np.mean(np.abs(p - t)))


def rmse(pred: ArrayLike, target: ArrayLike) -> float:
    """Root Mean Square Error."""
    p, t = _to_numpy(pred), _to_numpy(target)
    if p.shape != t.shape:
        raise ValueError(f"Shape mismatch: pred {p.shape} vs target {t.shape}")
    return float(np.sqrt(np.mean((p - t) ** 2)))


def average_displacement_error(
    pred_traj: ArrayLike,
    gt_traj: ArrayLike,
) -> float:
    """
    ADE: mean L2 distance over all timesteps and samples.
    pred_traj, gt_traj: (..., T, D)
    """
    p, t = _to_numpy(pred_traj), _to_numpy(gt_traj)
    if p.shape != t.shape:
        raise ValueError(f"Shape mismatch: pred {p.shape} vs target {t.shape}")
    return float(np.mean(np.linalg.norm(p - t, axis=-1)))


def final_displacement_error(
    pred_traj: ArrayLike,
    gt_traj: ArrayLike,
) -> float:
    """
    FDE: L2 distance at the final timestep, averaged over samples.
    pred_traj, gt_traj: (..., T, D)
    """
    p, t = _to_numpy(pred_traj), _to_numpy(gt_traj)
    if p.shape != t.shape:
        raise ValueError(f"Shape mismatch: pred {p.shape} vs target {t.shape}")
    return float(np.mean(np.linalg.norm(p[..., -1, :] - t[..., -1, :], axis=-1)))
