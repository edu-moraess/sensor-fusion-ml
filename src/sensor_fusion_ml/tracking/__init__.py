"""Multi-object tracking utilities (Phase C)."""

from .associator import hungarian_association, greedy_association

__all__ = ["hungarian_association", "greedy_association"]
