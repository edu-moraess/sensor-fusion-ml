"""Data association for multi-object tracking.

The public association functions use Euclidean distance and a hard gate.  A
valid assignment is one whose distance is at most ``max_cost``.  The primary
solver maximizes the number of valid matches first and minimizes total
matching cost second.  The returned pairs are sorted by track and detection
index for deterministic output.
"""
from __future__ import annotations

from functools import lru_cache
from numbers import Real
from typing import List, Tuple

import numpy as np


def _validate_positions(
    track_positions: np.ndarray,
    detection_positions: np.ndarray,
    max_cost: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if isinstance(max_cost, bool) or not isinstance(max_cost, Real):
        raise ValueError("max_cost must be numeric")
    gate = float(max_cost)
    if not np.isfinite(gate) or gate < 0:
        raise ValueError("max_cost must be finite and >= 0")

    def convert(value: np.ndarray, name: str) -> np.ndarray:
        raw = np.asarray(value)
        if raw.dtype.kind in {"b", "c", "O", "U", "S"}:
            raise ValueError(f"{name} must contain real numeric values")
        if raw.ndim != 2:
            raise ValueError(f"{name} must have shape (N, D)")
        try:
            converted = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain real numeric values") from exc
        if not np.all(np.isfinite(converted)):
            raise ValueError(f"{name} must contain only finite values")
        return converted.copy()

    tracks = convert(track_positions, "track_positions")
    detections = convert(detection_positions, "detection_positions")
    if tracks.shape[1] != detections.shape[1]:
        raise ValueError("track_positions and detection_positions must have the same dimension")
    return tracks, detections, gate


def _cost_matrix(
    track_positions: np.ndarray,
    detection_positions: np.ndarray,
) -> np.ndarray:
    """Euclidean distance cost matrix (N_tracks x N_detections)."""
    diff = track_positions[:, None, :] - detection_positions[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _empty_result(
    n_tracks: int, n_detections: int
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    return [], list(range(n_tracks)), list(range(n_detections))


def _result_from_matches(
    matches: list[tuple[int, int]], n_tracks: int, n_detections: int
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    ordered = sorted(matches)
    used_tracks = {track_idx for track_idx, _ in ordered}
    used_detections = {detection_idx for _, detection_idx in ordered}
    return (
        ordered,
        [idx for idx in range(n_tracks) if idx not in used_tracks],
        [idx for idx in range(n_detections) if idx not in used_detections],
    )


def _scipy_assignment(
    cost: np.ndarray, max_cost: float
) -> list[tuple[int, int]]:
    """Solve gated assignment with SciPy using explicit non-association nodes.

    A real track-to-detection edge has its Euclidean cost when it is inside the
    gate and is forbidden otherwise.  Every unmatched track and detection is
    represented by a dummy edge with penalty P=max_cost+1.  Replacing one
    unmatched track/detection pair costs 2P, which is strictly greater than
    every valid edge cost, so cardinality is optimized before distance.
    """
    from scipy.optimize import linear_sum_assignment

    n_tracks, n_detections = cost.shape
    size = n_tracks + n_detections
    penalty = max_cost + 1.0
    augmented = np.full((size, size), np.inf, dtype=np.float64)
    valid = cost <= max_cost
    augmented[:n_tracks, :n_detections] = np.where(valid, cost, np.inf)
    augmented[:n_tracks, n_detections:] = penalty
    augmented[n_tracks:, :n_detections] = penalty
    augmented[n_tracks:, n_detections:] = 0.0

    rows, columns = linear_sum_assignment(augmented)
    return [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if row < n_tracks and column < n_detections and valid[row, column]
    ]


def _exact_assignment(cost: np.ndarray, max_cost: float) -> list[tuple[int, int]]:
    """Exact deterministic gated assignment for small matrices.

    Dynamic programming maximizes cardinality, then minimizes cost, then uses
    lexicographically smallest pairs as a deterministic tie-breaker.
    """
    n_tracks, n_detections = cost.shape
    valid = cost <= max_cost

    @lru_cache(maxsize=None)
    def solve(track_idx: int, used_mask: int) -> tuple[int, float, tuple[tuple[int, int], ...]]:
        if track_idx == n_tracks:
            return 0, 0.0, ()

        best = solve(track_idx + 1, used_mask)
        best_count, best_cost, best_pairs = best
        for detection_idx in range(n_detections):
            if used_mask & (1 << detection_idx) or not valid[track_idx, detection_idx]:
                continue
            count, total, pairs = solve(track_idx + 1, used_mask | (1 << detection_idx))
            candidate = (
                count + 1,
                total + float(cost[track_idx, detection_idx]),
                ((track_idx, detection_idx),) + pairs,
            )
            if candidate[0] > best_count:
                best = candidate
                best_count, best_cost, best_pairs = best
            elif candidate[0] == best_count:
                tolerance = 1e-12 * max(1.0, abs(best_cost), abs(candidate[1]))
                if candidate[1] < best_cost - tolerance or (
                    abs(candidate[1] - best_cost) <= tolerance and candidate[2] < best_pairs
                ):
                    best = candidate
                    best_count, best_cost, best_pairs = best
        return best

    return list(solve(0, 0)[2])


def _greedy_assignment(cost: np.ndarray, max_cost: float) -> list[tuple[int, int]]:
    """Deterministic heuristic fallback for large matrices without SciPy.

    This fallback is not globally optimal, but it only considers gated edges;
    therefore it never returns an association above ``max_cost``.
    """
    valid_edges = [
        (float(cost[track_idx, detection_idx]), track_idx, detection_idx)
        for track_idx in range(cost.shape[0])
        for detection_idx in range(cost.shape[1])
        if cost[track_idx, detection_idx] <= max_cost
    ]
    valid_edges.sort()
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, track_idx, detection_idx in valid_edges:
        if track_idx not in used_tracks and detection_idx not in used_detections:
            matches.append((track_idx, detection_idx))
            used_tracks.add(track_idx)
            used_detections.add(detection_idx)
    return matches


def hungarian_association(
    track_positions: np.ndarray,
    detection_positions: np.ndarray,
    max_cost: float = 5.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Return a maximum-cardinality, minimum-cost gated assignment.

    SciPy is used when available.  Without SciPy, the fallback is exact for
    matrices with at most 20 tracks and 20 detections, and deterministic
    gated-greedy for larger matrices.  The latter is explicitly not claimed
    to be globally optimal.
    """
    tracks, detections, gate = _validate_positions(
        track_positions, detection_positions, max_cost
    )
    n_tracks, n_detections = len(tracks), len(detections)
    if n_tracks == 0 or n_detections == 0:
        return _empty_result(n_tracks, n_detections)
    cost = _cost_matrix(tracks, detections)
    try:
        matches = _scipy_assignment(cost, gate)
    except ImportError:
        if max(n_tracks, n_detections) <= 20:
            matches = _exact_assignment(cost, gate)
        else:
            matches = _greedy_assignment(cost, gate)
    return _result_from_matches(matches, n_tracks, n_detections)


def greedy_association(
    track_positions: np.ndarray,
    detection_positions: np.ndarray,
    max_cost: float = 5.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Return a deterministic gated nearest-neighbor heuristic assignment."""
    tracks, detections, gate = _validate_positions(
        track_positions, detection_positions, max_cost
    )
    n_tracks, n_detections = len(tracks), len(detections)
    if n_tracks == 0 or n_detections == 0:
        return _empty_result(n_tracks, n_detections)
    matches = _greedy_assignment(_cost_matrix(tracks, detections), gate)
    return _result_from_matches(matches, n_tracks, n_detections)
