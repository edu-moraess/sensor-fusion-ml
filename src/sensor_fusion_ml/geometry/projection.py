"""Validated LiDAR-to-camera projection primitives.

Inputs use metric XYZ coordinates in the LiDAR sensor frame. Calibration
matrices must describe LiDAR-to-camera and camera-rectification transforms;
the 3x4 projection matrix maps rectified-camera coordinates to pixels.
This module does not synchronize sensors or estimate calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ProjectionResult:
    """Projection outputs aligned row-for-row with the input point array.

    pixels contains NaNs for non-finite input points or points with camera
    depth at or behind min_depth. Points in front of the camera but outside
    the image retain their pixel coordinates.
    """

    pixels: FloatArray
    depth: FloatArray
    finite_input: BoolArray
    positive_depth: BoolArray
    inside_image: Optional[BoolArray]


def _matrix(value: object, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _rotation(value: object, name: str, tolerance: float = 1e-4) -> FloatArray:
    rotation = _matrix(value, (3, 3), name)
    orthogonality_error = float(np.linalg.norm(rotation @ rotation.T - np.eye(3)))
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > tolerance or abs(determinant - 1.0) > tolerance:
        raise ValueError(
            f"{name} must be a proper rotation matrix "
            f"(orthogonality_error={orthogonality_error:.3g}, "
            f"determinant={determinant:.6g})"
        )
    return rotation


def project_lidar_to_image(
    points: object,
    projection_matrix: object,
    rectification_rotation: object,
    lidar_to_camera_rotation: object,
    lidar_to_camera_translation: object,
    image_shape: Optional[Sequence[int]] = None,
    *,
    min_depth: float = 1e-6,
) -> ProjectionResult:
    """Transform LiDAR XYZ points and project them into a rectified image.

    Args:
        points: Array shaped (N, 3) (XYZ) or (N, >=4) (XYZ plus optional
            attributes such as intensity). Only the first three columns are
            transformed.
        projection_matrix: Finite 3x4 camera projection matrix P.
        rectification_rotation: Proper 3x3 rotation into the rectified camera.
        lidar_to_camera_rotation: Proper 3x3 rotation from LiDAR to camera.
        lidar_to_camera_translation: Three-element translation in the same
            metric unit as the input XYZ coordinates.
        image_shape: Optional (height, width) for the in-image mask. If omitted,
            inside_image is None.
        min_depth: Minimum accepted rectified-camera Z; finite and non-negative.

    Returns:
        ProjectionResult with arrays aligned to the original input rows.
        Pixel coordinates are (u, v); depth is rectified-camera Z.

    Raises:
        ValueError: If array shapes, calibration matrices, or arguments are
            invalid. Non-finite point rows are masked, not treated as errors.
    """
    raw_points = np.asarray(points, dtype=np.float64)
    if raw_points.ndim != 2 or raw_points.shape[1] < 3:
        raise ValueError("points must have shape (N, 3) or (N, >=4)")
    if not np.isfinite(min_depth) or min_depth < 0:
        raise ValueError("min_depth must be finite and non-negative")

    projection = _matrix(projection_matrix, (3, 4), "projection_matrix")
    rectification = _rotation(rectification_rotation, "rectification_rotation")
    rotation = _rotation(lidar_to_camera_rotation, "lidar_to_camera_rotation")
    translation = _matrix(
        lidar_to_camera_translation, (3,), "lidar_to_camera_translation"
    )

    height = width = None
    if image_shape is not None:
        if len(image_shape) != 2:
            raise ValueError("image_shape must be (height, width)")
        height, width = image_shape
        if (
            isinstance(height, bool) or isinstance(width, bool)
            or int(height) != height or int(width) != width
            or height <= 0 or width <= 0
        ):
            raise ValueError("image_shape dimensions must be positive integers")
        height, width = int(height), int(width)

    count = raw_points.shape[0]
    xyz = raw_points[:, :3]
    finite_input = np.isfinite(xyz).all(axis=1)

    # Replace invalid rows before matrix multiplication, then mask their outputs.
    safe_xyz = np.where(finite_input[:, None], xyz, 0.0)
    camera_points = (rotation @ safe_xyz.T) + translation[:, None]
    rectified_points = rectification @ camera_points

    homogeneous = np.vstack(
        [rectified_points, np.ones((1, count), dtype=np.float64)]
    )
    projected = projection @ homogeneous
    depth = rectified_points[2].copy()

    positive_depth = (
        finite_input
        & np.isfinite(rectified_points).all(axis=0)
        & np.isfinite(projected).all(axis=0)
        & (depth > min_depth)
        & (projected[2] > min_depth)
    )

    pixels = np.full((count, 2), np.nan, dtype=np.float64)
    pixels[positive_depth] = (
        projected[:2, positive_depth] / projected[2, positive_depth]
    ).T

    depth[~finite_input] = np.nan

    inside_image: Optional[BoolArray] = None
    if image_shape is not None:
        inside_image = (
            positive_depth
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )

    return ProjectionResult(
        pixels=pixels,
        depth=depth,
        finite_input=finite_input,
        positive_depth=positive_depth,
        inside_image=inside_image,
    )
