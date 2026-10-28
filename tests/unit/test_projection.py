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


def test_rejects_nonpositive_projective_denominator():
    bad_projection = PROJECTION.copy()
    bad_projection[2] = 0.0
    result = project(
        [[0.0, 0.0, 10.0]],
        projection_matrix=bad_projection,
    )
    assert result.positive_depth.tolist() == [False]
    assert np.isnan(result.pixels[0]).all()


def test_positive_depth_uses_rectified_z_and_projective_denominator():
    # A projeção tem denominador w = Z - 5, diferente de Z retificada.
    projection = PROJECTION.copy()
    projection[2] = [0.0, 0.0, 1.0, -5.0]

    result = project(
        [
            [0.0, 0.0, 10.0],  # Z=10, w=5: válido
            [0.0, 0.0, 4.0],   # Z=4: profundidade inválida
            [0.0, 0.0, 5.0],   # w=0: denominador inválido
        ],
        projection_matrix=projection,
        image_shape=None,
    )

    assert result.depth.tolist() == [10.0, 4.0, 5.0]
    assert result.positive_depth.tolist() == [True, False, False]
    assert np.isfinite(result.pixels[0]).all()
    assert np.isnan(result.pixels[1:]).all()


def test_positive_depth_rejects_negative_projective_denominator():
    # Z é positivo, mas o denominador projetivo é negativo.
    projection = PROJECTION.copy()
    projection[2] = [0.0, 0.0, 1.0, -11.0]

    result = project(
        [[0.0, 0.0, 10.0]],
        projection_matrix=projection,
        image_shape=None,
    )

    assert result.depth.tolist() == [10.0]
    assert result.positive_depth.tolist() == [False]
    assert np.isnan(result.pixels[0]).all()


def test_min_depth_boundary_is_strict():
    result = project(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0001],
        ],
        min_depth=1.0,
        image_shape=None,
    )

    assert result.positive_depth.tolist() == [False, True]


def test_rejects_invalid_point_array_shape():
    with pytest.raises(ValueError):
        project([1.0, 2.0, 3.0])


def test_projection_matches_independent_homogeneous_reference():
    # Rotação de 90 graus em torno de Z, com translação não nula.
    angle = np.pi / 2
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle),  np.cos(angle), 0.0],
        [0.0,            0.0,           1.0],
    ])
    translation = np.array([0.5, -0.25, 1.0])
    rectification = np.eye(3)

    projection = np.array([
        [120.0, 0.0, 60.0, 0.0],
        [0.0, 110.0, 40.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])

    points = np.array([
        [1.0, 2.0, 8.0],
        [-1.0, 0.5, 5.0],
        [0.5, -1.0, 3.0],
        [1.0, 1.0, -4.0],
    ])

    result = project_lidar_to_image(
        points=points,
        projection_matrix=projection,
        rectification_rotation=rectification,
        lidar_to_camera_rotation=rotation,
        lidar_to_camera_translation=translation,
        image_shape=(80, 120),
    )

    # Referência independente por transformação homogênea.
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    rect = np.eye(4)
    rect[:3, :3] = rectification

    homogeneous_points = np.column_stack(
        [points, np.ones(len(points))]
    )
    camera_rect = (
        rect @ transform @ homogeneous_points.T
    ).T[:, :3]

    projected_h = (
        projection
        @ np.column_stack(
            [camera_rect, np.ones(len(camera_rect))]
        ).T
    ).T

    denominator = projected_h[:, 2]
    reference_valid = (
        np.isfinite(camera_rect).all(axis=1)
        & np.isfinite(projected_h).all(axis=1)
        & (camera_rect[:, 2] > 1e-6)
        & (denominator > 1e-6)
    )

    expected_pixels = np.full((len(points), 2), np.nan)
    expected_pixels[reference_valid] = (
        projected_h[reference_valid, :2]
        / denominator[reference_valid, None]
    )

    np.testing.assert_array_equal(
        result.positive_depth, reference_valid
    )
    np.testing.assert_allclose(
        result.pixels[reference_valid],
        expected_pixels[reference_valid],
        rtol=0.0,
        atol=1e-6,
    )
    assert np.isnan(result.pixels[~reference_valid]).all()

    # Valores esperados calculados independentemente:
    # câmera: Xc = -Y + 0.5, Yc = X - 0.25, Zc = Z + 1.
    expected_manual = np.array([
        [40.0, 49.166666666666664],
        [60.0, 17.083333333333332],
        [105.0, 46.875],
    ])

    np.testing.assert_allclose(
        result.pixels[:3],
        expected_manual,
        rtol=0.0,
        atol=1e-6,
    )
    assert np.isnan(result.pixels[3]).all()
