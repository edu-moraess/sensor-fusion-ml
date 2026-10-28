"""Unit tests for LiDAR-to-camera projection."""

import numpy as np
import pytest

from sensor_fusion_ml.geometry import project_lidar_to_image


IDENTITY = np.eye(3)
PROJECTION = np.array([
    [100.0, 0.0, 50.0, 0.0],
    [0.0, 100.0, 40.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
])


def project(points, **kwargs):
    defaults = dict(
        projection_matrix=PROJECTION,
        rectification_rotation=IDENTITY,
        lidar_to_camera_rotation=IDENTITY,
        lidar_to_camera_translation=np.zeros(3),
        image_shape=(80, 100),
    )
    defaults.update(kwargs)
    return project_lidar_to_image(points, **defaults)


def test_projects_known_point_and_preserves_row_alignment():
    result = project([[1.0, 2.0, 10.0, 0.75]])
    np.testing.assert_allclose(result.pixels, [[60.0, 60.0]])
    np.testing.assert_allclose(result.depth, [10.0])
    assert result.finite_input.tolist() == [True]
    assert result.positive_depth.tolist() == [True]
    assert result.inside_image.tolist() == [True]


def test_filters_points_behind_camera_and_outside_image():
    result = project([
        [0.0, 0.0, 10.0],
        [0.0, 0.0, -1.0],
        [10.0, 0.0, 10.0],
    ])
    assert result.positive_depth.tolist() == [True, False, True]
    assert result.inside_image.tolist() == [True, False, False]
    assert np.isnan(result.pixels[1]).all()
    assert np.isfinite(result.pixels[2]).all()


def test_nonfinite_input_is_masked_without_shifting_rows():
    result = project([
        [0.0, 0.0, 10.0],
        [np.nan, 0.0, 1.0],
        [1.0, 0.0, 10.0],
    ])
    assert result.finite_input.tolist() == [True, False, True]
    assert result.positive_depth.tolist() == [True, False, True]
    assert np.isnan(result.pixels[1]).all()
    assert np.isnan(result.depth[1])


def test_optional_image_shape_leaves_inside_mask_none():
    result = project([[0.0, 0.0, 10.0]], image_shape=None)
    assert result.inside_image is None
    np.testing.assert_allclose(result.pixels, [[50.0, 40.0]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"projection_matrix": np.eye(3)},
        {"rectification_rotation": np.zeros((3, 3))},
        {"lidar_to_camera_rotation": np.diag([1.0, 1.0, -1.0])},
        {"lidar_to_camera_translation": [0.0, np.inf, 0.0]},
        {"min_depth": -1.0},
        {"image_shape": (0, 100)},
    ],
)
def test_rejects_invalid_calibration_or_arguments(kwargs):
    with pytest.raises(ValueError):
        project([[0.0, 0.0, 10.0]], **kwargs)


def test_rejects_invalid_point_array_shape():
    with pytest.raises(ValueError):
        project([1.0, 2.0, 3.0])
