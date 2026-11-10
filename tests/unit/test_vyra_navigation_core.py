import numpy as np

from sensor_fusion_ml.navigation.vyra_eskf import (
    G, NavigationState, IMUNoise, VyraESKF, quat_rotate,
    quat_from_rotation_vector, quat_to_rotation_matrix,
    propagate_nominal, continuous_error_matrices,
    propagate_covariance, covariance_diagnostics,
    gnss_position_update, zupt_update,
)


def test_identity_quaternion():
    q = np.array([1., 0., 0., 0.])
    v = np.array([1., 2., 3.])
    np.testing.assert_allclose(quat_rotate(q, v), v, atol=1e-10)


def test_90_degree_rotation_about_up():
    q = quat_from_rotation_vector([0., 0., np.pi / 2])
    np.testing.assert_allclose(quat_rotate(q, [1., 0., 0.]),
                               [0., 1., 0.], atol=1e-9)


def test_rotation_matrix_is_orthogonal():
    q = quat_from_rotation_vector([0.2, -0.1, 0.4])
    R = quat_to_rotation_matrix(q)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9


def test_stationary_specific_force():
    state = NavigationState()
    for _ in range(100):
        state, _ = propagate_nominal(state, [0., 0., G], [0., 0., 0.], 0.01)
    np.testing.assert_allclose(state.v, np.zeros(3), atol=1e-9)
    np.testing.assert_allclose(state.p, np.zeros(3), atol=1e-9)


def test_constant_east_acceleration():
    state = NavigationState()
    for _ in range(100):
        state, _ = propagate_nominal(state, [1., 0., G], [0., 0., 0.], 0.01)
    assert abs(state.v[0] - 1.0) < 1e-6
    assert abs(state.p[0] - 0.5) < 1e-4
    assert abs(state.v[2]) < 1e-8


def test_error_matrix_dimensions():
    F, Gm = continuous_error_matrices(
        NavigationState(), [0., 0., G], [0., 0., 0.])
    assert F.shape == (15, 15)
    assert Gm.shape == (15, 12)


def test_covariance_propagation():
    state = NavigationState()
    P = np.eye(15) * 0.1
    F, Gm = continuous_error_matrices(state, [0., 0., G], [0., 0., 0.])
    P2, _, _ = propagate_covariance(P, F, Gm, IMUNoise(), 0.01)
    d = covariance_diagnostics(P2)
    assert d["finite"] and d["symmetric"] and d["positive_semidefinite"]


def test_gnss_reduces_position_uncertainty():
    state = NavigationState()
    P = np.eye(15) * 10.0
    before = np.trace(P[0:3, 0:3])
    state, P, report = gnss_position_update(
        state, P, [1., 2., 3.], np.eye(3) * 0.1)
    assert np.trace(P[0:3, 0:3]) < before
    np.testing.assert_allclose(state.p, [1/1.01, 2/1.01, 3/1.01], atol=1e-8)
    assert np.isfinite(report["nis"])


def test_zupt_reduces_velocity_uncertainty():
    initial_velocity = np.array([2., -1., 0.5])
    initial_covariance = 2.0
    measurement_variance = 0.01
    state = NavigationState(v=initial_velocity)
    P = np.eye(15) * initial_covariance
    before = np.trace(P[3:6, 3:6])
    state, P, report = zupt_update(state, P, np.eye(3) * measurement_variance)
    after = np.trace(P[3:6, 3:6])
    expected_factor = measurement_variance / (initial_covariance + measurement_variance)
    np.testing.assert_allclose(
        state.v, initial_velocity * expected_factor, rtol=1e-9, atol=1e-12)
    assert after < before
    assert np.linalg.norm(state.v) < 0.006 * np.linalg.norm(initial_velocity)
    assert np.isfinite(report["nis"])


def test_filter_pipeline():
    filt = VyraESKF()
    for _ in range(100):
        filt.predict([0., 0., G], [0., 0., 0.], 0.01)
    filt.update_gnss([0.1, 0., 0.], np.eye(3) * 0.05)
    d = covariance_diagnostics(filt.P)
    assert d["finite"] and d["symmetric"] and d["positive_semidefinite"]
    assert np.all(np.isfinite(filt.state.p))
