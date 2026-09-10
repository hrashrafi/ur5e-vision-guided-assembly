"""Pixel -> world-position back-projection for a known Z-height plane.

The shape detectors in shape_detector.py only give a single 2D pixel
centroid per detected feature - there's no full pose to solve for, and
none is needed: the resting/insertion surface heights in this scene are
fixed and already known (QDs rest on a known floor height; the manifold's
top face height is fixed), so vision only has to refine X/Y. This
ray-casts a detected pixel back through the camera to the one known
world-Z plane it must lie on.
"""

import numpy as np


def pixel_to_world_on_plane(px, py, K, cam_world_pos, cam_world_mat_mj, target_z):
    """(px, py): detected pixel (OpenCV convention - origin top-left, +y
    down). K: 3x3 intrinsics (qd_sim/vision/camera_intrinsics.py).
    cam_world_pos/cam_world_mat_mj: the camera's world pose (data.cam_xpos,
    data.cam_xmat.reshape(3,3) - MuJoCo's own OpenGL-style camera
    convention: local -Z forward, +X right, +Y up). target_z: the known
    world Z-height the real feature lies on.

    Returns the 3D world point where the ray through that pixel crosses
    target_z.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Pixel -> normalized camera-local direction. Forward is -Z (MuJoCo's
    # own camera convention), and pixel-Y increasing downward is
    # camera-local -Y (up), hence the sign flip.
    x_ndc = (px - cx) / fx
    y_ndc = -(py - cy) / fy
    dir_cam = np.array([x_ndc, y_ndc, -1.0])

    R_cam_world = np.asarray(cam_world_mat_mj).reshape(3, 3)
    dir_world = R_cam_world @ dir_cam
    cam_pos = np.asarray(cam_world_pos, dtype=float)

    if abs(dir_world[2]) < 1e-9:
        raise ValueError("camera ray is parallel to the target Z-plane - can't intersect")
    t = (target_z - cam_pos[2]) / dir_world[2]
    if t <= 0:
        raise ValueError(f"target Z-plane ({target_z}) is behind the camera along this ray (t={t:.4f})")
    return cam_pos + t * dir_world
