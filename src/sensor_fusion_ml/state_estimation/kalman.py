"""
Constant-velocity Kalman Filter baseline.

State vector (2D): [x, y, vx, vy]
State vector (3D): [x, y, z, vx, vy, vz]

This is a classical reference, not a learned model.
All estimates are explicitly marked as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class KalmanState:
    mean: np.ndarray          # (state_dim,)
    covariance: np.ndarray    # (state_dim, state_dim)
    timestamp: float          # seconds
    track_id: Optional[int] = None
    is_estimate: bool = True  # always True for filtered state
    hits: int = 0
    age: int = 0
    time_since_update: int = 0

    def position(self) -> np.ndarray:
        dim = 2 if self.mean.shape[0] == 4 else 3
        return self.mean[:dim].copy()

    def velocity(self) -> np.ndarray:
        dim = 2 if self.mean.shape[0] == 4 else 3
        return self.mean[dim:2 * dim].copy()


class ConstantVelocityKF:
    """
    Linear constant-velocity Kalman Filter.

    Parameters
    ----------
    dim : int
        Spatial dimension (2 or 3).
    process_noise : float
        Scalar process noise intensity.
    measurement_noise : float
        Scalar measurement noise (position only).
    """

    def __init__(
        self,
        dim: int = 2,
        process_noise: float = 0.1,
        measurement_noise: float = 1.0,
    ):
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        self.dim = dim
        self.state_dim = 2 * dim
        self.q = float(process_noise)
        self.r = float(measurement_noise)

        # Measurement matrix: we observe position only
        self.H = np.zeros((dim, self.state_dim), dtype=np.float64)
        self.H[:, :dim] = np.eye(dim)

        self.R = np.eye(dim, dtype=np.float64) * self.r

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(self.state_dim, dtype=np.float64)
        F[:self.dim, self.dim:] = np.eye(self.dim) * dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        # Simple continuous white-noise acceleration model discretized
        q = self.q
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q11 = dt4 / 4.0 * q
        q12 = dt3 / 2.0 * q
        q22 = dt2 * q
        Q = np.zeros((self.state_dim, self.state_dim), dtype=np.float64)
        for i in range(self.dim):
            Q[i, i] = q11
            Q[i, i + self.dim] = q12
            Q[i + self.dim, i] = q12
            Q[i + self.dim, i + self.dim] = q22
        return Q

    def initiate(self, measurement: np.ndarray, timestamp: float, track_id: Optional[int] = None) -> KalmanState:
        """Create a new track from a position measurement."""
        measurement = np.asarray(measurement, dtype=np.float64).ravel()
        if measurement.shape[0] != self.dim:
            raise ValueError(f"measurement must have shape ({self.dim},)")

        mean = np.zeros(self.state_dim, dtype=np.float64)
        mean[:self.dim] = measurement
        # High initial velocity uncertainty
        cov = np.eye(self.state_dim, dtype=np.float64) * 10.0
        cov[:self.dim, :self.dim] = np.eye(self.dim) * self.r

        return KalmanState(
            mean=mean,
            covariance=cov,
            timestamp=timestamp,
            track_id=track_id,
            hits=1,
            age=1,
            time_since_update=0,
        )

    def predict(self, state: KalmanState, timestamp: float) -> KalmanState:
        dt = timestamp - state.timestamp
        if dt < 0:
            raise ValueError("Cannot predict backwards in time")
        F = self._F(dt)
        Q = self._Q(dt)
        mean = F @ state.mean
        cov = F @ state.covariance @ F.T + Q
        return KalmanState(
            mean=mean,
            covariance=cov,
            timestamp=timestamp,
            track_id=state.track_id,
            hits=state.hits,
            age=state.age + 1,
            time_since_update=state.time_since_update + 1,
        )

    def update(self, state: KalmanState, measurement: np.ndarray) -> KalmanState:
        measurement = np.asarray(measurement, dtype=np.float64).ravel()
        if measurement.shape[0] != self.dim:
            raise ValueError(f"measurement must have shape ({self.dim},)")

        # Innovation
        z_pred = self.H @ state.mean
        y = measurement - z_pred
        S = self.H @ state.covariance @ self.H.T + self.R
        K = state.covariance @ self.H.T @ np.linalg.inv(S)

        mean = state.mean + K @ y
        # Joseph form for covariance (more stable)
        I = np.eye(self.state_dim)
        cov = (I - K @ self.H) @ state.covariance @ (I - K @ self.H).T + K @ self.R @ K.T

        return KalmanState(
            mean=mean,
            covariance=cov,
            timestamp=state.timestamp,
            track_id=state.track_id,
            hits=state.hits + 1,
            age=state.age,
            time_since_update=0,
        )

    def predict_and_update(
        self,
        state: KalmanState,
        measurement: np.ndarray,
        timestamp: float,
    ) -> KalmanState:
        predicted = self.predict(state, timestamp)
        return self.update(predicted, measurement)
