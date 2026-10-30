"""Lifecycle management for constant-velocity multi-object tracks."""
from __future__ import annotations
from copy import deepcopy
from numbers import Real
import numpy as np

from ..state_estimation.kalman import ConstantVelocityKF, KalmanState
from .associator import hungarian_association


class TrackManager:
    """Manage track prediction, association, filtering, and expiration.

    Each call to :meth:`update` is committed atomically: validation and all
    filter operations happen on staged state, and the manager is changed only
    after the complete update succeeds.  Track IDs are monotonically
    increasing until :meth:`reset` is called.
    """

    def __init__(
        self,
        dim: int = 2,
        max_age: int = 2,
        max_cost: float = 5.0,
        process_noise: float = 0.1,
        measurement_noise: float = 1.0,
    ) -> None:
        self._validate_dim(dim)
        self._validate_nonnegative_int("max_age", max_age)
        self._validate_nonnegative_real("max_cost", max_cost)
        self._validate_nonnegative_real("process_noise", process_noise)
        self._validate_positive_real("measurement_noise", measurement_noise)

        self.dim = dim
        self.max_age = max_age
        self.max_cost = float(max_cost)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self._kf = ConstantVelocityKF(
            dim=dim,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
        )
        self._tracks: list[KalmanState] = []
        self._next_track_id = 0
        self._last_timestamp: float | None = None

    @staticmethod
    def _validate_dim(value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value not in (2, 3):
            raise ValueError("dim must be 2 or 3")

    @staticmethod
    def _validate_nonnegative_int(name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be an integer >= 0")

    @staticmethod
    def _validate_real(name: str, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be numeric")
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    @classmethod
    def _validate_nonnegative_real(cls, name: str, value: object) -> None:
        if cls._validate_real(name, value) < 0:
            raise ValueError(f"{name} must be finite and >= 0")

    @classmethod
    def _validate_positive_real(cls, name: str, value: object) -> None:
        if cls._validate_real(name, value) <= 0:
            raise ValueError(f"{name} must be finite and > 0")

    def _validate_timestamp(self, timestamp: object) -> float:
        value = self._validate_real("timestamp", timestamp)
        if self._last_timestamp is not None and value < self._last_timestamp:
            raise ValueError("timestamp cannot regress")
        return value

    def _validate_detections(self, detections: object) -> np.ndarray:
        try:
            raw = np.asarray(detections)
        except (TypeError, ValueError) as exc:
            raise ValueError("detections must be a numeric array of shape (N, dim)") from exc
        if raw.dtype.kind in {"b", "c", "O", "U", "S"}:
            raise ValueError("detections must contain real numeric values")
        if raw.ndim != 2 or raw.shape[1] != self.dim:
            raise ValueError(f"detections must have shape (N, {self.dim})")
        try:
            value = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("detections must contain real numeric values") from exc
        if not np.all(np.isfinite(value)):
            raise ValueError("detections must contain only finite values")
        return value.copy()

    @staticmethod
    def _copy_state(state: KalmanState) -> KalmanState:
        return deepcopy(state)

    @staticmethod
    def _index(value: object, name: str, upper_bound: int) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"association {name} indices must be integers")
        index = int(value)
        if index < 0 or index >= upper_bound:
            raise ValueError(f"association {name} index out of bounds")
        return index

    def _validate_association_result(
        self,
        result: tuple[object, object, object],
        n_tracks: int,
        n_detections: int,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        try:
            raw_matches, raw_unmatched_tracks, raw_unmatched_detections = result
            matches = list(raw_matches)
            unmatched_tracks = list(raw_unmatched_tracks)
            unmatched_detections = list(raw_unmatched_detections)
        except (TypeError, ValueError) as exc:
            raise ValueError("association result must contain three index collections") from exc

        normalized_matches: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for pair in matches:
            try:
                track_idx, detection_idx = pair
            except (TypeError, ValueError) as exc:
                raise ValueError("each association match must contain two indices") from exc
            track_idx = self._index(track_idx, "track", n_tracks)
            detection_idx = self._index(detection_idx, "detection", n_detections)
            if track_idx in matched_tracks or detection_idx in matched_detections:
                raise ValueError("association matches must be one-to-one")
            matched_tracks.add(track_idx)
            matched_detections.add(detection_idx)
            normalized_matches.append((track_idx, detection_idx))

        unmatched_tracks = [
            self._index(index, "unmatched track", n_tracks) for index in unmatched_tracks
        ]
        unmatched_detections = [
            self._index(index, "unmatched detection", n_detections)
            for index in unmatched_detections
        ]
        if len(set(unmatched_tracks)) != len(unmatched_tracks):
            raise ValueError("unmatched track indices must be unique")
        if len(set(unmatched_detections)) != len(unmatched_detections):
            raise ValueError("unmatched detection indices must be unique")
        if matched_tracks.intersection(unmatched_tracks):
            raise ValueError("track cannot be both matched and unmatched")
        if matched_detections.intersection(unmatched_detections):
            raise ValueError("detection cannot be both matched and unmatched")
        if matched_tracks.union(unmatched_tracks) != set(range(n_tracks)):
            raise ValueError("association result does not cover all tracks")
        if matched_detections.union(unmatched_detections) != set(range(n_detections)):
            raise ValueError("association result does not cover all detections")
        return normalized_matches, unmatched_tracks, unmatched_detections

    @property
    def tracks(self) -> list[KalmanState]:
        """Return isolated, deterministic snapshots ordered by track ID."""
        return [self._copy_state(track) for track in self._tracks]

    def update(self, detections: object, timestamp: object) -> list[KalmanState]:
        """Process one timestamp and return snapshots of active tracks."""
        value = self._validate_detections(detections)
        current_timestamp = self._validate_timestamp(timestamp)

        staged_tracks: list[KalmanState] = []
        for track in self._tracks:
            staged_tracks.append(self._kf.predict(self._copy_state(track), current_timestamp))

        if staged_tracks and len(value):
            positions = np.asarray([track.position() for track in staged_tracks])
            matches, unmatched_tracks, unmatched_detections = hungarian_association(
                positions, value, max_cost=self.max_cost
            )
        else:
            matches = []
            unmatched_tracks = list(range(len(staged_tracks)))
            unmatched_detections = list(range(len(value)))

        matches, unmatched_tracks, unmatched_detections = self._validate_association_result(
            (matches, unmatched_tracks, unmatched_detections),
            len(staged_tracks),
            len(value),
        )

        matched_by_track = {track_idx: detection_idx for track_idx, detection_idx in matches}
        for track_idx, detection_idx in matched_by_track.items():
            staged_tracks[track_idx] = self._kf.update(
                staged_tracks[track_idx], value[detection_idx]
            )

        retained_tracks = [
            track for index, track in enumerate(staged_tracks)
            if index not in unmatched_tracks or track.time_since_update <= self.max_age
        ]

        next_id = self._next_track_id
        for detection_idx in unmatched_detections:
            retained_tracks.append(
                self._kf.initiate(value[detection_idx], current_timestamp, track_id=next_id)
            )
            next_id += 1

        retained_tracks.sort(key=lambda track: int(track.track_id))
        self._tracks = retained_tracks
        self._next_track_id = next_id
        self._last_timestamp = current_timestamp
        return self.tracks

    def reset(self) -> None:
        """Remove all tracks and restart track IDs from zero."""
        self._tracks = []
        self._next_track_id = 0
        self._last_timestamp = None
