"""
Coordinate transformation for robot navigation.

Converts pixel coordinates from camera frame to world frame for navigation.
No ROS2 dependencies - pure math implementation.

Usage:
    world_x, world_y = pixel_to_world(u, v, depth, K, T_cam_to_base, robot_pose)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CameraParams:
    """Camera intrinsic and extrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float

    K: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.K is None:
            self.K = np.array([
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]
            ], dtype=np.float64)


@dataclass
class RobotPose:
    """Robot pose in world frame."""

    x: float  # meters
    y: float  # meters
    theta: float  # radians


def pixel_to_camera(
    u: float,
    v: float,
    depth: float,
    K: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Convert pixel coordinate to camera frame 3D point.
    Optimized: Uses algebraic expansion instead of matrix inversion.

    Args:
        u, v: Pixel coordinates (image space)
        depth: Depth in meters
        K: Camera intrinsic matrix 3x3

    Returns:
        (x_c, y_c, z_c) in camera frame (meters)
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    x_c = (u - cx) * depth / fx
    y_c = (v - cy) * depth / fy
    z_c = depth

    return x_c, y_c, z_c


def camera_to_base(
    x_c: float,
    y_c: float,
    z_c: float,
    T_cam_to_base: np.ndarray,
) -> Tuple[float, float]:
    """
    Transform point from camera frame to robot base frame.

    Args:
        x_c, y_c, z_c: Point in camera frame
        T_cam_to_base: 4x4 transformation matrix from camera to base

    Returns:
        (x_b, y_b) in robot base frame (meters)
    """
    point_cam = np.array([x_c, y_c, z_c, 1.0])
    point_base = T_cam_to_base @ point_cam
    return point_base[0], point_base[1]


def base_to_world(
    x_b: float,
    y_b: float,
    pose: RobotPose,
) -> Tuple[float, float]:
    """
    Transform point from robot base frame to world frame.

    Args:
        x_b, y_b: Point in robot base frame
        pose: Robot pose in world frame (x, y, theta)

    Returns:
        (x_w, y_w) in world frame (meters)
    """
    cos_t = np.cos(pose.theta)
    sin_t = np.sin(pose.theta)

    # Rotation matrix from base to world (inverse of world to base)
    R = np.array([
        [cos_t, -sin_t],
        [sin_t, cos_t]
    ])

    point_b = np.array([x_b, y_b])
    point_w = R @ point_b + np.array([pose.x, pose.y])

    return point_w[0], point_w[1]


def pixel_to_world(
    u: float,
    v: float,
    depth: float,
    K: np.ndarray,
    T_cam_to_base: np.ndarray,
    robot_pose: RobotPose,
) -> Tuple[float, float]:
    """
    Complete pipeline: pixel -> camera -> base -> world.

    Args:
        u, v: Pixel coordinates
        depth: Depth in meters
        K: Camera intrinsic matrix 3x3
        T_cam_to_base: 4x4 transformation matrix (camera to robot base)
        robot_pose: RobotPose (x, y, theta) in world frame

    Returns:
        (world_x, world_y) in meters
    """
    x_c, y_c, z_c = pixel_to_camera(u, v, depth, K)
    x_b, y_b = camera_to_base(x_c, y_c, z_c, T_cam_to_base)
    x_w, y_w = base_to_world(x_b, y_b, robot_pose)
    return x_w, y_w


def region_centroid_to_world(
    mask: np.ndarray,
    depth: float,
    K: np.ndarray,
    T_cam_to_base: np.ndarray,
    robot_pose: RobotPose,
) -> Optional[Tuple[float, float]]:
    """
    Convert region centroid to world coordinates.
    Handles non-convex shapes by checking if centroid falls within mask.

    Args:
        mask: Binary mask of region (H, W)
        depth: Depth value to use for all pixels in region
        K: Camera intrinsic matrix
        T_cam_to_base: Transformation matrix
        robot_pose: Robot pose in world frame

    Returns:
        (world_x, world_y) or None if mask is empty
    """
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    u_mean = np.mean(xs)
    v_mean = np.mean(ys)

    u_int, v_int = int(u_mean), int(v_mean)
    if 0 <= v_int < mask.shape[0] and 0 <= u_int < mask.shape[1] and mask[v_int, u_int]:
        u, v = float(u_mean), float(v_mean)
    else:
        distances = (xs - u_mean)**2 + (ys - v_mean)**2
        closest_idx = np.argmin(distances)
        u, v = float(xs[closest_idx]), float(ys[closest_idx])

    return pixel_to_world(u, v, depth, K, T_cam_to_base, robot_pose)


def create_default_transform(
    camera_height: float = 0.3,
    camera_pitch: float = 0.0,
) -> np.ndarray:
    """
    Create default camera-to-base transform (for testing/placeholder).

    Assumes camera is mounted at height 'camera_height' above ground,
    with optional pitch angle.

    Args:
        camera_height: Height of camera above robot base (meters)
        camera_pitch: Pitch angle in radians (positive = camera tilted up)

    Returns:
        4x4 transformation matrix T_cam_to_base
    """
    cy = np.cos(camera_pitch)
    sy = np.sin(camera_pitch)

    # Rotation: pitch around X axis
    R = np.array([
        [1, 0, 0],
        [0, cy, -sy],
        [0, sy, cy]
    ])

    # Translation: camera is above base
    t = np.array([0, 0, camera_height])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def create_default_camera_params(
    width: int = 640,
    height: int = 480,
    hfov: float = 90.0,
) -> CameraParams:
    """
    Create default camera params (for testing/placeholder).

    Args:
        width, height: Image resolution
        hfov: Horizontal field of view in degrees

    Returns:
        CameraParams with approximate values
    """
    fx = width / (2 * np.tan(np.radians(hfov) / 2))
    fy = fx  # Assume square pixels

    cx = width / 2
    cy = height / 2

    return CameraParams(fx=fx, fy=fy, cx=cx, cy=cy)