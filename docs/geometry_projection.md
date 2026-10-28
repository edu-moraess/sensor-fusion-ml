# LiDAR-to-camera projection

The reusable primitive is sensor_fusion_ml.geometry.project_lidar_to_image.
It accepts LiDAR points in metric XYZ coordinates, a LiDAR-to-camera rigid
transform, a camera rectification rotation, and a 3x4 projection matrix.

## Coordinate convention

The transform is applied in this order:

1. p_camera = R_lidar_to_camera @ p_lidar + t_lidar_to_camera
2. p_rectified = R_rectification @ p_camera
3. q = P @ [p_rectified_x, p_rectified_y, p_rectified_z, 1]
4. u = q_x / q_z, v = q_y / q_z

The translation and XYZ coordinates must use the same length unit (KITTI uses
meters). The supplied projection matrix must correspond to the rectified camera
whose image is being used. The module validates matrix shapes, finiteness, and
rotation-matrix properties, but does not prove that calibration belongs to the
selected sensor or dataset.

## Output and masking

The returned ProjectionResult preserves input row order and exposes:

- pixels: Nx2 pixel coordinates. Rows with invalid input or depth at/below
  min_depth contain NaNs. Valid points outside the image retain coordinates.
- depth: rectified-camera Z for finite input rows.
- finite_input: mask for finite XYZ rows.
- positive_depth: mask for finite points in front of the camera.
- inside_image: mask for points within (height, width), or None when no image
  shape was supplied.

This function is a geometric transform only. It does not synchronize sensor
timestamps, compensate for motion between captures, estimate calibration, or
guarantee visual alignment. Validate calibration provenance and inspect
projections across multiple frames before using them for fusion or training.

## Example

    import numpy as np
    from sensor_fusion_ml.geometry import project_lidar_to_image

    result = project_lidar_to_image(
        points=lidar_xyz,
        projection_matrix=P,
        rectification_rotation=R_rect,
        lidar_to_camera_rotation=R_velo_to_cam,
        lidar_to_camera_translation=t_velo_to_cam,
        image_shape=(375, 1242),
    )

    visible_pixels = result.pixels[result.inside_image]
