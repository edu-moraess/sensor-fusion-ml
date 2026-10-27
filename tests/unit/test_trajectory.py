"""Unit tests for kinematic trajectory baseline."""

import numpy as np

from sensor_fusion_ml.trajectory.baseline import kinematic_extrapolation


def test_single():
    pos = np.array([0.0, 0.0])
    vel = np.array([1.0, 0.0])
    traj = kinematic_extrapolation(pos, vel, horizon=5, dt=0.1)
    assert traj.shape == (5, 2)
    assert abs(traj[-1, 0] - 0.5) < 1e-6


def test_batch():
    pos = np.zeros((3, 2))
    vel = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    traj = kinematic_extrapolation(pos, vel, horizon=4, dt=1.0)
    assert traj.shape == (3, 4, 2)
    assert abs(traj[0, -1, 0] - 4.0) < 1e-6
