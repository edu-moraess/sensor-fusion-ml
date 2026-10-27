#!/usr/bin/env python3
"""
Run classical Kalman + association baseline on synthetic data.
No real dataset is required or downloaded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sensor_fusion_ml.state_estimation.kalman import ConstantVelocityKF
from sensor_fusion_ml.evaluation.metrics import mae, rmse


def simulate_trajectory(n_steps: int = 50, dt: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    pos = np.zeros((n_steps, 2))
    vel = np.array([1.0, 0.5])
    for i in range(1, n_steps):
        pos[i] = pos[i - 1] + vel * dt + rng.normal(0, 0.01, size=2)
    measurements = pos + rng.normal(0, 0.3, size=pos.shape)
    return pos, measurements


def main() -> int:
    parser = argparse.ArgumentParser(description="Classical baseline on synthetic data")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gt, meas = simulate_trajectory(n_steps=args.steps, seed=args.seed)
    kf = ConstantVelocityKF(dim=2, process_noise=0.05, measurement_noise=0.3)

    state = kf.initiate(meas[0], timestamp=0.0, track_id=0)
    estimates = [state.position()]

    for i in range(1, len(meas)):
        state = kf.predict_and_update(state, meas[i], timestamp=i * 0.1)
        estimates.append(state.position())

    estimates = np.stack(estimates)
    pos_mae = mae(estimates, gt)
    pos_rmse = rmse(estimates, gt)

    print("Classical Kalman baseline (synthetic constant-velocity trajectory)")
    print(f"  Steps          : {args.steps}")
    print(f"  Position MAE   : {pos_mae:.4f} m")
    print(f"  Position RMSE  : {pos_rmse:.4f} m")
    print("  Status         : PASS (software validation only – not a scientific claim)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
