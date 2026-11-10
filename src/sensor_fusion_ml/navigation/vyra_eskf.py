"""Vyra Navigation Core v0.1.1: local ENU INS and 15-state ESKF primitives.

Quaternion convention: Hamilton, scalar-first [w, x, y, z].
Attitude maps body coordinates into the local navigation frame.
Error convention: right-multiplicative body-frame attitude error.
Gravity: [0, 0, -g] in ENU.

Noise defaults are software-test placeholders, not calibrated sensor values.
This module does not perform RINEX processing, coordinate conversion, or
stationary detection. ZUPT must only be called after an independent detector.
"""
from __future__ import annotations

import numpy as np

G = 9.80665
EPS = 1e-12
STATE_DIM = 15


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < EPS:
        raise ValueError("Invalid quaternion norm.")
    return q / norm


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product; scalar-first quaternion representation."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_from_rotation_vector(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    angle = np.linalg.norm(theta)
    if angle < 1e-10:
        return normalize_quaternion(np.concatenate(([1.0], 0.5 * theta)))
    return np.concatenate(([np.cos(angle / 2.0)],
                           theta / angle * np.sin(angle / 2.0)))


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def quat_rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return quat_to_rotation_matrix(q) @ np.asarray(vector, dtype=float)


class NavigationState:
    """Nominal p [m], v [m/s], q body-to-ENU, ba [m/s²], bg [rad/s]."""

    def __init__(self, p=None, v=None, q=None, ba=None, bg=None):
        self.p = np.zeros(3) if p is None else np.array(p, dtype=float)
        self.v = np.zeros(3) if v is None else np.array(v, dtype=float)
        self.q = (np.array([1., 0., 0., 0.]) if q is None
                  else normalize_quaternion(q))
        self.ba = np.zeros(3) if ba is None else np.array(ba, dtype=float)
        self.bg = np.zeros(3) if bg is None else np.array(bg, dtype=float)

    def copy(self):
        return NavigationState(self.p.copy(), self.v.copy(), self.q.copy(),
                               self.ba.copy(), self.bg.copy())


class IMUNoise:
    """Continuous noise densities; defaults are placeholders, not calibration."""

    def __init__(self, accel_noise_density=0.01, gyro_noise_density=0.001,
                 accel_bias_rw=0.0001, gyro_bias_rw=0.00001):
        self.sigmas = np.array([
            accel_noise_density, accel_noise_density, accel_noise_density,
            gyro_noise_density, gyro_noise_density, gyro_noise_density,
            accel_bias_rw, accel_bias_rw, accel_bias_rw,
            gyro_bias_rw, gyro_bias_rw, gyro_bias_rw,
        ], dtype=float)
        if not np.all(np.isfinite(self.sigmas)) or np.any(self.sigmas < 0):
            raise ValueError("Invalid IMU noise parameters.")

    def continuous_spectral_density(self):
        return np.diag(self.sigmas ** 2)


def continuous_error_matrices(state: NavigationState,
                              specific_force: np.ndarray,
                              angular_rate: np.ndarray):
    """Continuous F and G for [dp,dv,dtheta_body,dba,dbg], right error."""
    R = quat_to_rotation_matrix(state.q)
    f = np.asarray(specific_force, dtype=float) - state.ba
    w = np.asarray(angular_rate, dtype=float) - state.bg
    F = np.zeros((15, 15))
    Gm = np.zeros((15, 12))
    F[0:3, 3:6] = np.eye(3)
    F[3:6, 6:9] = -R @ skew(f)
    F[3:6, 9:12] = -R
    F[6:9, 6:9] = -skew(w)
    F[6:9, 12:15] = -np.eye(3)
    Gm[3:6, 0:3] = -R
    Gm[6:9, 3:6] = -np.eye(3)
    Gm[9:12, 6:9] = np.eye(3)
    Gm[12:15, 9:12] = np.eye(3)
    return F, Gm


def propagate_nominal(state: NavigationState, accel_measured: np.ndarray,
                      gyro_measured: np.ndarray, dt: float):
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt must be positive and finite.")
    state = state.copy()
    f_b = np.asarray(accel_measured, dtype=float) - state.ba
    omega_b = np.asarray(gyro_measured, dtype=float) - state.bg
    dq_half = quat_from_rotation_vector(omega_b * dt * 0.5)
    q_mid = normalize_quaternion(quat_multiply(state.q, dq_half))
    f_n = quat_rotate(q_mid, f_b)
    acceleration_n = f_n + np.array([0.0, 0.0, -G])
    state.p = state.p + state.v * dt + 0.5 * acceleration_n * dt**2
    state.v = state.v + acceleration_n * dt
    state.q = normalize_quaternion(quat_multiply(
        state.q, quat_from_rotation_vector(omega_b * dt)))
    return state, acceleration_n


def propagate_covariance(P: np.ndarray, F: np.ndarray, Gm: np.ndarray,
                         noise_model: IMUNoise, dt: float):
    """First-order Phi and Qd discretization."""
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt must be positive and finite.")
    Phi = np.eye(15) + F * dt
    Qd = Gm @ noise_model.continuous_spectral_density() @ Gm.T * dt
    P_next = Phi @ P @ Phi.T + Qd
    return 0.5 * (P_next + P_next.T), Phi, Qd


def inject_error(state: NavigationState, dx: np.ndarray, P: np.ndarray):
    dx = np.asarray(dx, dtype=float)
    if dx.shape != (15,):
        raise ValueError("Error vector must have dimension 15.")
    state = state.copy()
    state.p += dx[0:3]
    state.v += dx[3:6]
    state.q = normalize_quaternion(quat_multiply(
        state.q, quat_from_rotation_vector(dx[6:9])))
    state.ba += dx[9:12]
    state.bg += dx[12:15]
    J = np.eye(15)
    J[6:9, 6:9] -= 0.5 * skew(dx[6:9])
    P = J @ P @ J.T
    return state, 0.5 * (P + P.T)


def measurement_update(state: NavigationState, P: np.ndarray,
                       residual: np.ndarray, H: np.ndarray, R: np.ndarray):
    residual = np.asarray(residual, dtype=float)
    H = np.asarray(H, dtype=float)
    R = np.asarray(R, dtype=float)
    if not (np.all(np.isfinite(residual)) and np.all(np.isfinite(H))
            and np.all(np.isfinite(R))):
        raise ValueError("Non-finite measurement input.")
    if not np.allclose(R, R.T, atol=1e-9):
        raise ValueError("Measurement covariance must be symmetric.")
    if np.linalg.eigvalsh(R).min() <= 0:
        raise ValueError("Measurement covariance must be positive definite.")
    S = H @ P @ H.T + R
    S = 0.5 * (S + S.T)
    K = np.linalg.solve(S, H @ P).T
    dx = K @ residual
    A = np.eye(15) - K @ H
    P_updated = A @ P @ A.T + K @ R @ K.T
    P_updated = 0.5 * (P_updated + P_updated.T)
    state, P_updated = inject_error(state, dx, P_updated)
    nis = float(residual.T @ np.linalg.solve(S, residual))
    return state, P_updated, {
        "innovation": residual.tolist(),
        "nis": nis,
        "innovation_covariance": S.tolist(),
    }


def gnss_position_update(state: NavigationState, P: np.ndarray,
                         position_enu: np.ndarray, R_gnss: np.ndarray):
    z = np.asarray(position_enu, dtype=float)
    if z.shape != (3,):
        raise ValueError("GNSS position must have dimension 3.")
    H = np.zeros((3, 15))
    H[:, 0:3] = np.eye(3)
    return measurement_update(state, P, z - state.p, H, R_gnss)


def zupt_update(state: NavigationState, P: np.ndarray, R_zupt: np.ndarray):
    """Zero-velocity update; caller must supply independent rest detection."""
    H = np.zeros((3, 15))
    H[:, 3:6] = np.eye(3)
    return measurement_update(state, P, -state.v, H, R_zupt)


class VyraESKF:
    def __init__(self, state=None, P=None, noise_model=None):
        self.state = NavigationState() if state is None else state.copy()
        self.P = np.eye(15) if P is None else np.asarray(P, dtype=float).copy()
        if self.P.shape != (15, 15) or not np.all(np.isfinite(self.P)):
            raise ValueError("P must be a finite 15x15 matrix.")
        if not np.allclose(self.P, self.P.T, atol=1e-9):
            raise ValueError("P must be symmetric.")
        if np.linalg.eigvalsh(self.P).min() < -1e-9:
            raise ValueError("P must be positive semidefinite.")
        self.noise = IMUNoise() if noise_model is None else noise_model
        self.history = []

    def predict(self, accel, gyro, dt):
        accel = np.asarray(accel, dtype=float)
        gyro = np.asarray(gyro, dtype=float)
        if accel.shape != (3,) or gyro.shape != (3,):
            raise ValueError("IMU measurements must be 3D vectors.")
        F, Gm = continuous_error_matrices(self.state, accel, gyro)
        self.P, _, _ = propagate_covariance(self.P, F, Gm, self.noise, dt)
        self.state, acceleration = propagate_nominal(self.state, accel, gyro, dt)
        self.history.append({"event": "predict", "dt": float(dt)})
        return acceleration

    def update_gnss(self, position_enu, covariance):
        self.state, self.P, report = gnss_position_update(
            self.state, self.P, position_enu, covariance)
        self.history.append({"event": "gnss", "nis": report["nis"]})
        return report

    def update_zupt(self, covariance):
        self.state, self.P, report = zupt_update(
            self.state, self.P, covariance)
        self.history.append({"event": "zupt", "nis": report["nis"]})
        return report


def covariance_diagnostics(P: np.ndarray):
    P = np.asarray(P, dtype=float)
    eig = np.linalg.eigvalsh(0.5 * (P + P.T))
    return {
        "shape": list(P.shape),
        "finite": bool(np.all(np.isfinite(P))),
        "symmetric": bool(np.allclose(P, P.T, atol=1e-9)),
        "minimum_eigenvalue": float(eig.min()),
        "positive_semidefinite": bool(eig.min() >= -1e-9),
        "trace": float(np.trace(P)),
    }
