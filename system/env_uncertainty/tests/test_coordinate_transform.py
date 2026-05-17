"""
Tests for coordinate transformation (Step 10).
"""

import numpy as np
import pytest

from system.env_uncertainty.coordinate_transform import (
    CameraParams,
    RobotPose,
    pixel_to_camera,
    camera_to_base,
    base_to_world,
    pixel_to_world,
    region_centroid_to_world,
    create_default_transform,
    create_default_camera_params,
)


class TestCameraParams:
    def test_create_with_focal_params(self):
        params = CameraParams(fx=320, fy=320, cx=320, cy=240)
        assert params.fx == 320
        assert params.fy == 320
        assert params.K.shape == (3, 3)

    def test_K_matrix_creation(self):
        params = CameraParams(fx=320, fy=320, cx=320, cy=240)
        K = params.K
        assert K[0, 0] == 320  # fx
        assert K[1, 1] == 320  # fy
        assert K[0, 2] == 320  # cx
        assert K[1, 2] == 240  # cy
        assert K[2, 2] == 1    # homogeneous


class TestPixelToCamera:
    def test_center_pixel_at_depth_1(self):
        K = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]], dtype=np.float64)
        x, y, z = pixel_to_camera(320, 240, 1.0, K)
        assert z == pytest.approx(1.0)
        assert x == pytest.approx(0.0, abs=0.01)
        assert y == pytest.approx(0.0, abs=0.01)

    def test_right_pixel_at_depth_2(self):
        K = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]], dtype=np.float64)
        x, y, z = pixel_to_camera(640, 240, 2.0, K)
        assert z == pytest.approx(2.0)
        # (640-320)/320 * 2 = 2.0
        assert x == pytest.approx(2.0, abs=0.01)

    def test_top_pixel_at_depth_1(self):
        K = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]], dtype=np.float64)
        x, y, z = pixel_to_camera(320, 0, 1.0, K)
        # (0-240)/320 * 1 = -0.75 (image y=0 is top, so negative in camera frame)
        assert y == pytest.approx(-0.75, abs=0.01)


class TestCameraToBase:
    def test_identity_transform(self):
        T = np.eye(4)
        x_b, y_b = camera_to_base(1.0, 2.0, 3.0, T)
        assert x_b == pytest.approx(1.0)
        assert y_b == pytest.approx(2.0)

    def test_translation_transform(self):
        T = np.array([
            [1, 0, 0, 0.5],
            [0, 1, 0, 0.2],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        x_b, y_b = camera_to_base(1.0, 2.0, 3.0, T)
        assert x_b == pytest.approx(1.5)
        assert y_b == pytest.approx(2.2)


class TestBaseToWorld:
    def test_identity_pose(self):
        pose = RobotPose(x=0, y=0, theta=0)
        x_w, y_w = base_to_world(1.0, 2.0, pose)
        assert x_w == pytest.approx(1.0)
        assert y_w == pytest.approx(2.0)

    def test_translation_only(self):
        pose = RobotPose(x=5, y=10, theta=0)
        x_w, y_w = base_to_world(1.0, 2.0, pose)
        assert x_w == pytest.approx(6.0)
        assert y_w == pytest.approx(12.0)

    def test_rotation_90_degrees(self):
        pose = RobotPose(x=0, y=0, theta=np.pi/2)
        x_w, y_w = base_to_world(1.0, 0.0, pose)
        assert x_w == pytest.approx(0.0, abs=0.01)
        assert y_w == pytest.approx(1.0, abs=0.01)

    def test_combined_translation_and_rotation(self):
        pose = RobotPose(x=5, y=10, theta=np.pi/2)
        x_w, y_w = base_to_world(1.0, 2.0, pose)
        assert x_w == pytest.approx(5.0 - 2.0, abs=0.01)
        assert y_w == pytest.approx(10.0 + 1.0, abs=0.01)


class TestPixelToWorld:
    def test_full_pipeline_default_params(self):
        K = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]], dtype=np.float64)
        T = create_default_transform(camera_height=0.3)
        pose = RobotPose(x=0, y=0, theta=0)

        x_w, y_w = pixel_to_world(320, 240, 2.0, K, T, pose)
        assert x_w == pytest.approx(0.0, abs=0.1)
        assert y_w == pytest.approx(0.0, abs=0.1)

    def test_full_pipeline_off_center(self):
        K = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]], dtype=np.float64)
        T = create_default_transform(camera_height=0.3)
        pose = RobotPose(x=0, y=0, theta=0)

        x_w, y_w = pixel_to_world(640, 240, 2.0, K, T, pose)
        assert x_w > 0


class TestRegionCentroidToWorld:
    def test_empty_mask_returns_none(self):
        K = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]], dtype=np.float64)
        T = create_default_transform()
        pose = RobotPose(x=0, y=0, theta=0)
        mask = np.zeros((100, 100), dtype=bool)

        result = region_centroid_to_world(mask, 2.0, K, T, pose)
        assert result is None

    def test_single_pixel_mask(self):
        K = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]], dtype=np.float64)
        T = create_default_transform()
        pose = RobotPose(x=0, y=0, theta=0)
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 50] = True

        result = region_centroid_to_world(mask, 2.0, K, T, pose)
        assert result is not None

    def test_multi_pixel_mask_returns_centroid(self):
        K = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]], dtype=np.float64)
        T = create_default_transform()
        pose = RobotPose(x=0, y=0, theta=0)
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True  # Center region

        result = region_centroid_to_world(mask, 2.0, K, T, pose)
        assert result is not None
        x_w, y_w = result
        assert isinstance(x_w, float)
        assert isinstance(y_w, float)


class TestCreateDefaultTransform:
    def test_identity_height(self):
        T = create_default_transform(camera_height=0.0)
        assert T[2, 3] == pytest.approx(0.0)

    def test_nonzero_height(self):
        T = create_default_transform(camera_height=0.5)
        assert T[2, 3] == pytest.approx(0.5)

    def test_rotation_matrix_valid(self):
        T = create_default_transform(camera_pitch=np.pi/4)
        R = T[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=0.01)


class TestCreateDefaultCameraParams:
    def test_90_deg_hfov(self):
        params = create_default_camera_params(640, 480, 90.0)
        expected_fx = 640 / (2 * np.tan(np.pi/4))
        assert params.fx == pytest.approx(expected_fx)
        assert params.fy == params.fx

    def test_center_principal_point(self):
        params = create_default_camera_params(640, 480, 90.0)
        assert params.cx == 320
        assert params.cy == 240