"""Unit tests for evaluation metrics."""

import numpy as np
import pytest

from sensor_fusion_ml.evaluation.metrics import (
    mae,
    rmse,
    average_displacement_error,
    final_displacement_error,
)


def test_mae_rmse():
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.0, 2.0, 4.0])
    assert abs(mae(pred, target) - 1.0 / 3.0) < 1e-6
    assert abs(rmse(pred, target) - np.sqrt(1.0 / 3.0)) < 1e-6


def test_ade_fde():
    # (B=1, T=3, D=2)
    pred = np.array([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    gt = np.array([[[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]])
    assert abs(average_displacement_error(pred, gt) - 1.0 / 3.0) < 1e-6
    assert abs(final_displacement_error(pred, gt) - 1.0) < 1e-6


def test_shape_mismatch():
    with pytest.raises(ValueError):
        mae(np.zeros(3), np.zeros(4))
