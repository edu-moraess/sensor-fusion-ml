"""Unit tests for data association."""

import builtins

import numpy as np
import pytest

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


def test_hungarian_gate_maximizes_valid_cardinality_before_cost():
    tracks = np.array([
        [4.9540994, 6.77729536],
        [4.77183156, 4.84933585],
    ])
    dets = np.array([
        [4.81267514, 3.96718471],
        [0.36535598, 3.05544734],
    ])
    matches, unmatched_tracks, unmatched_detections = hungarian_association(
        tracks, dets, max_cost=5.0
    )
    assert matches == [(0, 0), (1, 1)]
    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_greedy_never_returns_an_out_of_gate_pair():
    tracks = np.array([[0.0, 0.0], [10.0, 0.0]])
    dets = np.array([[0.0, 0.0], [100.0, 0.0]])
    matches, unmatched_tracks, unmatched_detections = greedy_association(
        tracks, dets, max_cost=0.0
    )
    assert matches == [(0, 0)]
    assert unmatched_tracks == [1]
    assert unmatched_detections == [1]


def test_hungarian_fallback_is_exact_for_small_matrices(monkeypatch):
    original_import = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name == "scipy.optimize" or name.startswith("scipy."):
            raise ImportError("scipy unavailable for fallback test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    tracks = np.array([
        [4.9540994, 6.77729536],
        [4.77183156, 4.84933585],
    ])
    dets = np.array([
        [4.81267514, 3.96718471],
        [0.36535598, 3.05544734],
    ])
    assert hungarian_association(tracks, dets, max_cost=5.0) == (
        [(0, 0), (1, 1)], [], []
    )


@pytest.mark.parametrize("bad_tracks, bad_dets", [
    (np.array([0.0, 1.0]), np.array([[0.0, 1.0]])),
    (np.array([[0.0, np.nan]]), np.array([[0.0, 1.0]])),
    (np.array([[0.0, 1.0]]), np.array([[0.0, np.inf]])),
    (np.empty((0, 2)), np.empty((0, 3))),
])
def test_association_rejects_invalid_inputs(bad_tracks, bad_dets):
    with pytest.raises(ValueError):
        hungarian_association(bad_tracks, bad_dets)


def test_association_accepts_empty_rectangular_inputs():
    assert hungarian_association(np.empty((0, 2)), np.zeros((3, 2))) == (
        [], [], [0, 1, 2]
    )
    assert hungarian_association(np.empty((2, 2)), np.empty((0, 2))) == (
        [], [0, 1], []
    )


def test_equal_cost_assignment_is_deterministic():
    tracks = np.array([[0.0, 0.0], [0.0, 0.0]])
    dets = np.array([[1.0, 0.0], [1.0, 0.0]])
    results = [hungarian_association(tracks, dets, max_cost=1.0) for _ in range(5)]
    assert all(result == results[0] for result in results)
    assert results[0][0] == [(0, 0), (1, 1)]
