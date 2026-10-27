"""
Data association for multi-object tracking.

Hungarian algorithm (optimal) and greedy fallback.
Uses scipy.optimize.linear_sum_assignment when available.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def _cost_matrix(
    track_positions: np.ndarray,
    detection_positions: np.ndarray,
) -> np.ndarray:
    """Euclidean distance cost matrix (N_tracks x N_detections)."""
    # track_positions: (T, D), detection_positions: (M, D)
    diff = track_positions[:, None, :] - detection_positions[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def hungarian_association(
    track_positions: np.ndarray,
    detection_positions: np.ndarray,
    max_cost: float = 5.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Optimal assignment via Hungarian algorithm.

    Returns
    -------
    matches : list of (track_idx, det_idx)
    unmatched_tracks : list of track indices
    unmatched_detections : list of detection indices
    """
    if len(track_positions) == 0 or len(detection_positions) == 0:
        return [], list(range(len(track_positions))), list(range(len(detection_positions)))

    cost = _cost_matrix(track_positions, detection_positions)

    try:
        from scipy.optimize import linear_sum_assignment
        row_idx, col_idx = linear_sum_assignment(cost)
    except ImportError:
        # Fallback to greedy if scipy missing (should not happen with requirements)
        return greedy_association(track_positions, detection_positions, max_cost)

    matches = []
    unmatched_tracks = set(range(len(track_positions)))
    unmatched_dets = set(range(len(detection_positions)))

    for r, c in zip(row_idx, col_idx):
        if cost[r, c] <= max_cost:
            matches.append((int(r), int(c)))
            unmatched_tracks.discard(int(r))
            unmatched_dets.discard(int(c))

    return matches, sorted(unmatched_tracks), sorted(unmatched_dets)


def greedy_association(
    track_positions: np.ndarray,
    detection_positions: np.ndarray,
    max_cost: float = 5.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy nearest-neighbor association (fallback)."""
    if len(track_positions) == 0 or len(detection_positions) == 0:
        return [], list(range(len(track_positions))), list(range(len(detection_positions)))

    cost = _cost_matrix(track_positions, detection_positions)
    matches = []
    used_tracks = set()
    used_dets = set()

    # Flatten and sort by cost
    flat = [(cost[i, j], i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
    flat.sort()

    for c, i, j in flat:
        if c > max_cost:
            break
        if i not in used_tracks and j not in used_dets:
            matches.append((i, j))
            used_tracks.add(i)
            used_dets.add(j)

    unmatched_tracks = [i for i in range(len(track_positions)) if i not in used_tracks]
    unmatched_dets = [j for j in range(len(detection_positions)) if j not in used_dets]
    return matches, unmatched_tracks, unmatched_dets
