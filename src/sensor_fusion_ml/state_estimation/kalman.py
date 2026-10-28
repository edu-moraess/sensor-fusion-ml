"""
Constant-velocity Kalman Filter baseline.

State vector (2D): [x, y, vx, vy]
State vector (3D): [x, y, z, vx, vy, vz]

This is a classical reference, not a learned model.
All estimates are explicitly marked as such.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class KalmanState:
    mean: np.ndarray
    covariance: np.ndarray
    timestamp: float
    track_id: Optional[int] = None
    is_estimate: bool = True
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
    """Linear constant-velocity Kalman Filter."""

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
        if not np.isfinite(self.q) or self.q < 0:
            raise ValueError("process_noise must be finite and >= 0")
        if not np.isfinite(self.r) or self.r <= 0:
            raise ValueError("measurement_noise must be finite and > 0")

        self.H = np.zeros((dim, self.state_dim), dtype=np.float64)
        self.H[:, :dim] = np.eye(dim)
        self.R = np.eye(dim, dtype=np.float64) * self.r

    @staticmethod
    def _validate_timestamp(timestamp: float) -> float:
        value = float(timestamp)
        if not np.isfinite(value):
            raise ValueError("timestamp must be finite")
        return value

    def _validate_measurement(self, measurement: np.ndarray) -> np.ndarray:
        value = np.asarray(measurement, dtype=np.float64)
        if value.shape != (self.dim,):
            raise ValueError(f"measurement must have shape ({self.dim},)")
        if not np.all(np.isfinite(value)):
            raise ValueError("measurement must contain only finite values")
        return value

    def _validate_state(self, state: KalmanState) -> None:
        mean = np.asarray(state.mean, dtype=np.float64)
        cov = np.asarray(state.covariance, dtype=np.float64)
        if mean.shape != (self.state_dim,):
            raise ValueError(f"state.mean must have shape ({self.state_dim},)")
        if cov.shape != (self.state_dim, self.state_dim):
            raise ValueError(
                f"state.covariance must have shape ({self.state_dim}, {self.state_dim})"
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(cov)):
            raise ValueError("state mean and covariance must contain only finite values")
        if not np.isfinite(float(state.timestamp)):
            raise ValueError("state timestamp must be finite")
        if not np.allclose(cov, cov.T, rtol=1e-7, atol=1e-9):
            raise ValueError("state covariance must be symmetric")
        # Permit tiny floating-point negative eigenvalues, reject materially invalid covariance.
        if np.linalg.eigvalsh((cov + cov.T) / 2.0).min() < -1e-8:
            raise ValueError("state covariance must be positive semidefinite")

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(self.state_dim, dtype=np.float64)
        F[:self.dim, self.dim:] = np.eye(self.dim) * dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
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

    def initiate(
        self,
        measurement: np.ndarray,
        timestamp: float,
        track_id: Optional[int] = None,
    ) -> KalmanState:
        """Create a new track from a validated position measurement."""
        measurement = self._validate_measurement(measurement)
        timestamp = self._validate_timestamp(timestamp)
        mean = np.zeros(self.state_dim, dtype=np.float64)
        mean[:self.dim] = measurement
        cov = np.eye(self.state_dim, dtype=np.float64) * 10.0
        cov[:self.dim, :self.dim] = np.eye(self.dim) * self.r
        return KalmanState(
            mean=mean, covariance=cov, timestamp=timestamp, track_id=track_id,
            hits=1, age=1, time_since_update=0,
        )

    def predict(self, state: KalmanState, timestamp: float) -> KalmanState:
        self._validate_state(state)
        timestamp = self._validate_timestamp(timestamp)
        dt = timestamp - float(state.timestamp)
        if dt < 0:
            raise ValueError("Cannot predict backwards in time")
        F = self._F(dt)
        mean = F @ np.asarray(state.mean, dtype=np.float64)
        cov = F @ np.asarray(state.covariance, dtype=np.float64) @ F.T + self._Q(dt)
        cov = (cov + cov.T) / 2.0
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(cov)):
            raise ValueError("prediction produced non-finite state values")
        return KalmanState(
            mean=mean, covariance=cov, timestamp=timestamp, track_id=state.track_id,
            hits=state.hits, age=state.age + 1,
            time_since_update=state.time_since_update + 1,
        )

    def update(self, state: KalmanState, measurement: np.ndarray) -> KalmanState:
        self._validate_state(state)
        measurement = self._validate_measurement(measurement)
        mean_prior = np.asarray(state.mean, dtype=np.float64)
        cov_prior = np.asarray(state.covariance, dtype=np.float64)
        innovation = measurement - self.H @ mean_prior
        S = self.H @ cov_prior @ self.H.T + self.R
        if not np.all(np.isfinite(S)):
            raise ValueError("innovation covariance contains non-finite values")
        try:
            # Solve S X = (P H^T)^T to avoid explicitly forming S inverse.
            K = np.linalg.solve(S, (cov_prior @ self.H.T).T).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("innovation covariance is singular") from exc

        mean = mean_prior + K @ innovation
        I_KH = np.eye(self.state_dim) - K @ self.H
        # Joseph form is more numerically stable than the simplified update.
        cov = I_KH @ cov_prior @ I_KH.T + K @ self.R @ K.T
        cov = (cov + cov.T) / 2.0
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(cov)):
            raise ValueError("update produced non-finite state values")
        if np.linalg.eigvalsh(cov).min() < -1e-8:
            raise ValueError("update produced an invalid covariance")
        return KalmanState(
            mean=mean, covariance=cov, timestamp=state.timestamp,
            track_id=state.track_id, hits=state.hits + 1, age=state.age,
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
