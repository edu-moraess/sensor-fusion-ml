"""Unit tests for ConstantVelocityKF."""

import numpy as np
import pytest

from sensor_fusion_ml.state_estimation.kalman import ConstantVelocityKF


def test_initiate_and_predict():
    kf = ConstantVelocityKF(dim=2, process_noise=0.1, measurement_noise=1.0)
    state = kf.initiate(np.array([0.0, 0.0]), timestamp=0.0, track_id=1)
    assert state.position().shape == (2,)
    assert state.track_id == 1
    pred = kf.predict(state, timestamp=0.1)
    assert pred.timestamp == 0.1
    assert pred.time_since_update == 1


def test_update_reduces_uncertainty_on_position():
    kf = ConstantVelocityKF(dim=2, process_noise=0.01, measurement_noise=0.1)
    state = kf.initiate(np.array([1.0, 2.0]), timestamp=0.0)
    # Predict then update with same measurement
    state = kf.predict(state, 0.1)
    updated = kf.update(state, np.array([1.0, 2.0]))
    # Position variance should decrease
    assert updated.covariance[0, 0] < state.covariance[0, 0]


def test_full_filter_on_synthetic():
    kf = ConstantVelocityKF(dim=2, process_noise=0.05, measurement_noise=0.2)
    true_pos = np.array([0.0, 0.0])
    true_vel = np.array([1.0, 0.0])
    dt = 0.1
    state = kf.initiate(true_pos + np.array([0.1, -0.1]), 0.0)

    errors = []
    for i in range(1, 30):
        true_pos = true_pos + true_vel * dt
        meas = true_pos + np.random.randn(2) * 0.15
        state = kf.predict_and_update(state, meas, i * dt)
        errors.append(np.linalg.norm(state.position() - true_pos))

    # After a few steps the filter should be reasonably close
    assert np.mean(errors[-10:]) < 0.5


def test_dim3():
    kf = ConstantVelocityKF(dim=3)
    state = kf.initiate(np.zeros(3), 0.0)
    assert state.mean.shape == (6,)
    pred = kf.predict(state, 0.2)
    assert pred.mean.shape == (6,)
