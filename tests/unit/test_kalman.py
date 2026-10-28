"""Unit and regression tests for ConstantVelocityKF."""

import numpy as np
import pytest

from sensor_fusion_ml.state_estimation.kalman import ConstantVelocityKF, KalmanState


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
    state = kf.predict(state, 0.1)
    updated = kf.update(state, np.array([1.0, 2.0]))
    assert updated.covariance[0, 0] < state.covariance[0, 0]


def test_full_filter_on_synthetic():
    rng = np.random.default_rng(42)
    kf = ConstantVelocityKF(dim=2, process_noise=0.05, measurement_noise=0.2)
    true_pos = np.array([0.0, 0.0])
    true_vel = np.array([1.0, 0.0])
    dt = 0.1
    state = kf.initiate(true_pos + np.array([0.1, -0.1]), 0.0)
    errors = []
    for i in range(1, 30):
        true_pos = true_pos + true_vel * dt
        meas = true_pos + rng.standard_normal(2) * 0.15
        state = kf.predict_and_update(state, meas, i * dt)
        errors.append(np.linalg.norm(state.position() - true_pos))
    assert np.mean(errors[-10:]) < 0.5


def test_dim3():
    kf = ConstantVelocityKF(dim=3)
    state = kf.initiate(np.zeros(3), 0.0)
    assert state.mean.shape == (6,)
    assert kf.predict(state, 0.2).mean.shape == (6,)


@pytest.mark.parametrize("kwargs", [
    {"process_noise": -0.1},
    {"process_noise": float("nan")},
    {"process_noise": float("inf")},
    {"measurement_noise": 0.0},
    {"measurement_noise": -1.0},
    {"measurement_noise": float("nan")},
    {"measurement_noise": float("inf")},
])
def test_rejects_invalid_noise_parameters(kwargs):
    with pytest.raises(ValueError):
        ConstantVelocityKF(**kwargs)


@pytest.mark.parametrize("measurement", [
    [float("nan"), 0.0],
    [float("inf"), 0.0],
    [0.0],
    [0.0, 1.0, 2.0],
])
def test_initiate_rejects_invalid_measurements(measurement):
    with pytest.raises(ValueError):
        ConstantVelocityKF().initiate(measurement, 0.0)


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -float("inf")])
def test_initiate_rejects_nonfinite_timestamp(timestamp):
    with pytest.raises(ValueError):
        ConstantVelocityKF().initiate([0.0, 0.0], timestamp)


def test_predict_rejects_nonfinite_timestamp():
    kf = ConstantVelocityKF()
    state = kf.initiate([0.0, 0.0], 0.0)
    with pytest.raises(ValueError):
        kf.predict(state, float("nan"))


def test_predict_rejects_nonfinite_state():
    kf = ConstantVelocityKF()
    state = kf.initiate([0.0, 0.0], 0.0)
    state.mean[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        kf.predict(state, 1.0)


def test_update_rejects_nonfinite_measurement():
    kf = ConstantVelocityKF()
    state = kf.initiate([0.0, 0.0], 0.0)
    with pytest.raises(ValueError, match="finite"):
        kf.update(state, [float("nan"), 1.0])


def test_update_rejects_invalid_covariance():
    kf = ConstantVelocityKF()
    state = kf.initiate([0.0, 0.0], 0.0)
    state.covariance[0, 1] = 2.0
    with pytest.raises(ValueError, match="symmetric"):
        kf.update(state, [0.1, 0.2])


def test_predict_rejects_backwards_time():
    kf = ConstantVelocityKF()
    state = kf.initiate([0.0, 0.0], 1.0)
    with pytest.raises(ValueError, match="backwards"):
        kf.predict(state, 0.5)


def test_covariance_remains_symmetric_psd_after_update():
    kf = ConstantVelocityKF()
    state = kf.initiate([0.0, 0.0], 0.0)
    state = kf.predict_and_update(state, [0.2, -0.1], 0.1)
    assert np.allclose(state.covariance, state.covariance.T)
    assert np.linalg.eigvalsh(state.covariance).min() >= -1e-8
