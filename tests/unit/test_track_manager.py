import numpy as np
import pytest

from sensor_fusion_ml.tracking import TrackManager


def detections(*rows):
    return np.asarray(rows, dtype=float)


def test_first_track_and_multiple_tracks_are_created_deterministically():
    manager = TrackManager(dim=2)
    tracks = manager.update(detections((0, 0), (10, 10)), 0.0)
    assert [track.track_id for track in tracks] == [0, 1]
    assert [track.position().tolist() for track in tracks] == [[0.0, 0.0], [10.0, 10.0]]
    assert all(track.hits == 1 for track in tracks)


def test_association_preserves_ids_and_updates_states():
    manager = TrackManager(dim=2, max_cost=2.0)
    manager.update(detections((0, 0), (10, 10)), 0.0)
    tracks = manager.update(detections((10.2, 10), (0.1, 0)), 1.0)
    assert [track.track_id for track in tracks] == [0, 1]
    assert [track.hits for track in tracks] == [2, 2]
    assert tracks[0].position()[0] < 2.0
    assert tracks[1].position()[0] > 8.0


def test_distant_detection_creates_new_id_without_reusing_expired_id():
    manager = TrackManager(dim=2, max_age=0, max_cost=1.0)
    manager.update(detections((0, 0)), 0.0)
    tracks = manager.update(detections((10, 10)), 1.0)
    assert [track.track_id for track in tracks] == [1]
    manager.update(np.empty((0, 2)), 2.0)
    manager.update(detections((20, 20)), 3.0)
    assert [track.track_id for track in manager.tracks] == [2]


def test_unmatched_track_is_retained_then_expires_after_max_age():
    manager = TrackManager(dim=2, max_age=2)
    manager.update(detections((0, 0)), 0.0)
    assert len(manager.update(np.empty((0, 2)), 1.0)) == 1
    assert len(manager.update(np.empty((0, 2)), 2.0)) == 1
    assert manager.update(np.empty((0, 2)), 3.0) == []


def test_empty_input_without_previous_tracks_is_valid():
    manager = TrackManager(dim=2)
    assert manager.update(np.empty((0, 2)), 0.0) == []


def test_supports_3d():
    manager = TrackManager(dim=3)
    track = manager.update(detections((1, 2, 3)), 2.0)[0]
    assert track.position().shape == (3,)
    assert np.allclose(track.position(), [1, 2, 3])


def test_equal_timestamps_are_allowed():
    manager = TrackManager()
    manager.update(detections((0, 0)), 1.0)
    assert len(manager.update(detections((0.1, 0)), 1.0)) == 1


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_timestamp_is_rejected_without_mutation(timestamp):
    manager = TrackManager()
    manager.update(detections((1, 2)), 1.0)
    before = manager.tracks
    with pytest.raises(ValueError):
        manager.update(detections((2, 3)), timestamp)
    assert manager.tracks[0].track_id == before[0].track_id
    assert manager.tracks[0].timestamp == before[0].timestamp


def test_regressive_timestamp_is_rejected_without_mutation():
    manager = TrackManager()
    manager.update(detections((1, 2)), 2.0)
    before = manager.tracks[0]
    with pytest.raises(ValueError, match="regress"):
        manager.update(detections((2, 3)), 1.0)
    after = manager.tracks[0]
    assert after.timestamp == before.timestamp
    assert np.array_equal(after.mean, before.mean)


@pytest.mark.parametrize("bad", [np.array([[np.nan, 0.0]]), np.array([[0.0, np.inf]])])
def test_nonfinite_detections_are_rejected(bad):
    with pytest.raises(ValueError):
        TrackManager().update(bad, 0.0)


@pytest.mark.parametrize("bad", [np.array([1.0, 2.0]), np.empty((1, 3)), np.empty((1, 1))])
def test_invalid_detection_dimensions_are_rejected(bad):
    with pytest.raises(ValueError):
        TrackManager(dim=2).update(bad, 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim": 4}, {"dim": True}, {"max_age": -1}, {"max_age": True},
        {"max_cost": -1.0}, {"max_cost": np.inf}, {"max_cost": True},
        {"process_noise": -1.0}, {"process_noise": np.nan},
        {"measurement_noise": 0.0}, {"measurement_noise": np.inf},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TrackManager(**kwargs)


def test_tracks_property_is_isolated():
    manager = TrackManager()
    manager.update(detections((1, 2)), 0.0)
    snapshot = manager.tracks
    snapshot.clear()
    snapshot = manager.tracks
    snapshot[0].mean[0] = 999.0
    assert len(manager.tracks) == 1
    assert manager.tracks[0].position()[0] == 1.0


def test_reset_clears_tracks_and_restarts_ids():
    manager = TrackManager()
    manager.update(detections((1, 2)), 0.0)
    manager.reset()
    assert manager.tracks == []
    assert manager.update(detections((3, 4)), 0.0)[0].track_id == 0


def test_failure_during_filter_operation_is_atomic(monkeypatch):
    manager = TrackManager()
    manager.update(detections((1, 2)), 0.0)
    before = manager.tracks[0]
    original = manager._kf.predict

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(manager._kf, "predict", fail)
    with pytest.raises(RuntimeError):
        manager.update(detections((3, 4)), 1.0)
    assert manager.tracks[0].track_id == before.track_id
    assert manager.tracks[0].timestamp == before.timestamp
    assert manager._next_track_id == 1
    monkeypatch.setattr(manager._kf, "predict", original)
    assert manager.update(detections((3, 4)), 1.0)[0].track_id == 0


def test_public_import():
    from sensor_fusion_ml.tracking import TrackManager as PublicTrackManager
    assert PublicTrackManager is TrackManager


def test_counterexample_preserves_both_existing_ids_without_new_tracks():
    manager = TrackManager(max_cost=5.0)
    manager.update(
        detections((4.9540994, 6.77729536), (4.77183156, 4.84933585)),
        0.0,
    )
    tracks = manager.update(
        detections((4.81267514, 3.96718471), (0.36535598, 3.05544734)),
        1.0,
    )
    assert [track.track_id for track in tracks] == [0, 1]
    assert [track.hits for track in tracks] == [2, 2]
    assert manager._next_track_id == 2


def test_max_cost_zero_only_associates_exact_positions():
    manager = TrackManager(max_cost=0.0, max_age=0)
    manager.update(detections((0.0, 0.0)), 0.0)
    tracks = manager.update(detections((1.0, 0.0)), 1.0)
    assert [track.track_id for track in tracks] == [1]


def test_consecutive_empty_detections_preserve_prediction_then_expire():
    manager = TrackManager(max_age=1)
    manager.update(detections((0.0, 0.0)), 0.0)
    assert len(manager.update(np.empty((0, 2)), 1.0)) == 1
    assert manager.update(np.empty((0, 2)), 2.0) == []


def test_invalid_association_contract_is_rejected_atomically(monkeypatch):
    manager = TrackManager()
    manager.update(detections((0.0, 0.0)), 0.0)
    before = manager.tracks[0]

    def invalid_association(*args, **kwargs):
        return [(0, 0), (0, 0)], [], []

    monkeypatch.setattr("sensor_fusion_ml.tracking.manager.hungarian_association", invalid_association)
    with pytest.raises(ValueError, match="one-to-one"):
        manager.update(detections((1.0, 0.0)), 1.0)
    after = manager.tracks[0]
    assert after.timestamp == before.timestamp
    assert np.array_equal(after.mean, before.mean)
    assert manager._next_track_id == 1


def test_failure_during_kalman_update_is_atomic(monkeypatch):
    manager = TrackManager()
    manager.update(detections((0.0, 0.0)), 0.0)
    before = manager.tracks[0]

    def fail_update(*args, **kwargs):
        raise RuntimeError("synthetic update failure")

    monkeypatch.setattr(manager._kf, "update", fail_update)
    with pytest.raises(RuntimeError):
        manager.update(detections((1.0, 0.0)), 1.0)
    after = manager.tracks[0]
    assert after.timestamp == before.timestamp
    assert np.array_equal(after.mean, before.mean)
    assert manager._next_track_id == 1


def test_failure_during_new_track_creation_is_atomic(monkeypatch):
    manager = TrackManager()
    manager.update(detections((0.0, 0.0)), 0.0)
    before = manager.tracks[0]

    def fail_initiate(*args, **kwargs):
        raise RuntimeError("synthetic creation failure")

    monkeypatch.setattr(manager._kf, "initiate", fail_initiate)
    with pytest.raises(RuntimeError):
        manager.update(detections((100.0, 100.0)), 1.0)
    after = manager.tracks[0]
    assert after.timestamp == before.timestamp
    assert np.array_equal(after.mean, before.mean)
    assert manager._next_track_id == 1
