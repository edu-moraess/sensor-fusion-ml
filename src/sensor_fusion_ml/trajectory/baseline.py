"""
Simple kinematic trajectory extrapolation (Phase E baseline).

Used as a non-learned reference to compare against learned predictors.
Does not claim future motion guarantees.
"""

from __future__ import annotations

import numpy as np


def kinematic_extrapolation(
    position: np.ndarray,
    velocity: np.ndarray,
    horizon: int,
    dt: float = 0.1,
) -> np.ndarray:
    """
    Constant-velocity extrapolation.

    Parameters
    ----------
    position : (D,) or (B, D)
    velocity : (D,) or (B, D)
    horizon : number of future steps
    dt : time step in seconds

    Returns
    -------
    traj : (horizon, D) or (B, horizon, D)
    """
    pos = np.asarray(position, dtype=np.float64)
    vel = np.asarray(velocity, dtype=np.float64)

    single = pos.ndim == 1
    if single:
        pos = pos[None, :]
        vel = vel[None, :]

    B, D = pos.shape
    steps = np.arange(1, horizon + 1, dtype=np.float64)[None, :, None]  # (1, H, 1)
    traj = pos[:, None, :] + vel[:, None, :] * steps * dt               # (B, H, D)

    if single:
        return traj[0]
    return traj
