"""Multi-object tracking utilities (Phase C)."""

from .associator import hungarian_association, greedy_association
from .manager import TrackManager

__all__ = ["hungarian_association", "greedy_association", "TrackManager"]
