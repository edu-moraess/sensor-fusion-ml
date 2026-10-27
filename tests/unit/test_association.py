"""Unit tests for data association."""

import numpy as np

from sensor_fusion_ml.tracking.associator import hungarian_association, greedy_association


def test_hungarian_perfect():
    tracks = np.array([[0.0, 0.0], [10.0, 0.0]])
    dets = np.array([[0.1, 0.0], [10.1, 0.0]])
    matches, u_t, u_d = hungarian_association(tracks, dets, max_cost=1.0)
    assert len(matches) == 2
    assert len(u_t) == 0
    assert len(u_d) == 0


def test_hungarian_max_cost():
    tracks = np.array([[0.0, 0.0]])
    dets = np.array([[100.0, 0.0]])
    matches, u_t, u_d = hungarian_association(tracks, dets, max_cost=5.0)
    assert len(matches) == 0
    assert u_t == [0]
    assert u_d == [0]


def test_greedy_fallback():
    tracks = np.array([[0.0, 0.0], [5.0, 0.0]])
    dets = np.array([[0.2, 0.0], [5.1, 0.0]])
    matches, _, _ = greedy_association(tracks, dets, max_cost=1.0)
    assert len(matches) == 2
