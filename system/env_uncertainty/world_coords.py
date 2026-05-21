"""
World coordinate transforms for multi-frame GP and scene graph accumulation.

Converts image pixel coordinates + robot pose → metric world coordinates,
enabling the GP traversability map to accumulate observations across frames
as the robot moves, and the scene graph to span multiple camera views.

Without world coordinates
--------------------------
The GP is reset each frame because its observations are image-relative:
pixel (row=100, col=200) in frame 2 refers to a completely different physical
location than the same pixel in frame 1 after the robot has moved 1 metre.

With world coordinates
-----------------------
GP observations are stored as (x_world, y_world) in metres. Observations from
all frames accumulate in the same metric map. The scene graph tiles are also
world-absolute (0.5 m × 0.5 m), so a "grass" patch seen in frame 1 is the same
tile as the same patch revisited in frame 10.

Coordinate systems
------------------
Image frame:   (row, col),  (0, 0) = top-left.
Camera frame:  ROS REP 103 — x=right, y=down, z=forward.
Robot frame:   x=forward, y=left (ground robot, ROS REP 103).
World frame:   x=East/forward, y=North/left (2D local map frame).
               Origin = robot position when the first frame was processed.

Transform chain (one pixel to world)
--------------------------------------
  pixel (row, col)
    └─ + depth_m + CameraIntrinsics → camera frame (X_c, Y_c, Z_c)
  camera frame
    └─ + CameraMount (pitch, yaw, height) → robot frame (dx_fwd, dy_left)
  robot frame
    └─ + RobotPose (x, y, theta) → world frame (x_w, y_w)

Depth estimation (monocular — no RGB-D required)
-------------------------------------------------
For a camera mounted at known height h above level ground and pitched down by
angle p from horizontal:

  elevation_angle(row) = arctan((cy - row) / fy)        # positive = below horizon
  total_depression = p + elevation_angle(row)            # angle below horizontal
  depth_m = h / tan(total_depression)                    # distance to ground point

This is the standard monocular BEV-projection assumption (flat-ground).
Valid for level outdoor terrain; degrades on steep slopes.
Pixels above the horizon (elevation_angle < -p) produce negative depth → skipped.

Visual odometry (optional)
---------------------------
OpticalFlowOdometry estimates inter-frame robot displacement from OpenCV sparse
Lucas-Kanade optical flow on the bottom half of the image (ground region).
Horizontal flow → lateral displacement. Vertical flow → forward displacement via
the ground-plane homography assumption. Output is a delta-pose that is integrated
into the running world pose.

MockForwardOdometry provides a deterministic pose sequence for testing on
datasets where no real odometry is available. It assumes the robot drives forward
at a fixed speed with zero lateral drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RobotPose:
    """
    2D robot pose in the world frame.

    x:         Metres east/forward from the map origin.
    y:         Metres north/left from the map origin.
    theta:     Heading in radians (0 = east/forward, counter-clockwise positive).
    timestamp: Unix time in seconds; 0.0 = unknown / not set.
    source:    Odometry source: "gps", "visual_odometry", "wheel_odometry", "mock".
    """

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    timestamp: float = 0.0
    source: str = "mock"

    def advance(self, dx_fwd: float, dy_left: float, dtheta: float = 0.0) -> "RobotPose":
        """
        Return a new pose after a relative motion in robot frame.

        Args:
            dx_fwd:   Forward displacement in metres.
            dy_left:  Left displacement in metres.
            dtheta:   Heading change in radians (counter-clockwise positive).

        Returns:
            New RobotPose with updated (x, y, theta).
        """
        cos_t, sin_t = math.cos(self.theta), math.sin(self.theta)
        new_x = self.x + cos_t * dx_fwd - sin_t * dy_left
        new_y = self.y + sin_t * dx_fwd + cos_t * dy_left
        return RobotPose(
            x=new_x,
            y=new_y,
            theta=self.theta + dtheta,
            timestamp=self.timestamp,
            source=self.source,
        )


@dataclass
class CameraMount:
    """
    Camera mounting parameters for pixel→ground-plane depth estimation.

    height_m:     Camera optical centre height above ground in metres.
    pitch_rad:    Camera pitch below horizontal in radians (positive = looking down).
                  A typical forward-facing camera on a ground robot pitches ~10-20°.
    yaw_rad:      Camera yaw relative to robot forward (0 = forward-facing).
    """

    height_m: float = 0.50      # typical robot camera height
    pitch_rad: float = 0.26     # ~15° forward/downward pitch
    yaw_rad: float = 0.0        # forward-facing


@dataclass
class WorldObservation:
    """
    A single terrain observation in the world coordinate frame.

    x_w, y_w:      World position in metres.
    traversability: Score ∈ [0, 1].
    label:          Terrain vocabulary label (e.g. "grass", "unknown").
    frame_id:       Source image identifier (for debugging).
    """

    x_w: float
    y_w: float
    traversability: float
    label: str = "unknown"
    frame_id: str = ""


# ── Depth estimation ──────────────────────────────────────────────────────────

def monocular_ground_depth(
    pixel_row: int,
    fy: float,
    cy_principal: float,
    mount: CameraMount,
) -> Optional[float]:
    """
    Estimate metric ground-plane depth for a pixel using the flat-ground assumption.

    Uses the camera's known mounting height and pitch angle to project the pixel
    onto a horizontal ground plane via perspective geometry:

        elevation_angle = arctan((cy - row) / fy)   (positive = below horizon)
        total_depression = pitch + elevation_angle
        depth_m = height / tan(total_depression)

    Args:
        pixel_row:      Image row index (0 = top).
        fy:             Focal length in pixels (vertical axis).
        cy_principal:   Principal point y in pixels.
        mount:          CameraMount with height_m and pitch_rad.

    Returns:
        Metric depth in metres, or None if the pixel is at or above the horizon
        (no intersection with ground plane).
    """
    elevation = math.atan2(cy_principal - pixel_row, fy)  # below horizon = positive
    total_depression = mount.pitch_rad + elevation
    if total_depression <= 0.01:   # pixel is at or above the horizon
        return None
    return mount.height_m / math.tan(total_depression)


# ── Pixel → world transform ───────────────────────────────────────────────────

def pixel_to_world(
    pixel_row: int,
    pixel_col: int,
    pose: RobotPose,
    fx: float,
    fy: float,
    cx_principal: float,
    cy_principal: float,
    mount: CameraMount,
    depth_m: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """
    Convert an image pixel to (x_world, y_world) in metres.

    Transform chain:
      pixel → camera frame  (X_c, Z_c)
      camera frame → robot frame  (dx_fwd, dy_left)  [via mount yaw/pitch]
      robot frame → world frame  (x_w, y_w)  [via robot pose]

    When depth_m is None, uses monocular_ground_depth() (flat-ground assumption).
    When depth_m is provided (from RGB-D or lidar), uses it directly.

    Args:
        pixel_row:    Image row index.
        pixel_col:    Image column index.
        pose:         Current robot pose in world frame.
        fx, fy:       Camera focal lengths in pixels.
        cx_principal, cy_principal: Camera principal point.
        mount:        CameraMount specifying height, pitch, yaw.
        depth_m:      Optional override depth (metres); None = monocular estimate.

    Returns:
        (x_world, y_world) in metres, or None if depth cannot be estimated
        (pixel above horizon in monocular mode).
    """
    if depth_m is None:
        depth_m = monocular_ground_depth(pixel_row, fy, cy_principal, mount)
    if depth_m is None or depth_m <= 0.0 or depth_m > 50.0:
        return None

    # Pixel → camera frame (pinhole, z=forward)
    X_c = (pixel_col - cx_principal) * depth_m / fx   # camera-frame right
    Z_c = depth_m                                       # camera-frame forward

    # Camera frame → robot frame
    # Camera yaw rotates the horizontal plane; camera pitch already used for depth.
    cos_y, sin_y = math.cos(mount.yaw_rad), math.sin(mount.yaw_rad)
    dx_fwd = cos_y * Z_c - sin_y * X_c    # robot forward
    dy_left = -(sin_y * Z_c + cos_y * X_c)  # robot left (camera +x = robot -y)

    # Robot frame → world frame using robot pose
    cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
    x_w = pose.x + cos_t * dx_fwd - sin_t * dy_left
    y_w = pose.y + sin_t * dx_fwd + cos_t * dy_left

    return x_w, y_w


def mask_centroids_to_world(
    mask: np.ndarray,
    pose: RobotPose,
    fx: float,
    fy: float,
    cx_principal: float,
    cy_principal: float,
    mount: CameraMount,
    n_samples: int = 9,
    depth_map: Optional[np.ndarray] = None,
) -> List[Tuple[float, float]]:
    """
    Convert a boolean mask to a list of world-coordinate sample points.

    Samples n_samples evenly-spaced pixels from the mask and projects each
    to world coordinates. Returns only the successful projections (above-horizon
    pixels are silently dropped).

    Args:
        mask:       (H, W) bool numpy array.
        pose:       Current robot pose.
        fx, fy:     Focal lengths.
        cx_principal, cy_principal: Principal point.
        mount:      CameraMount.
        n_samples:  Number of mask pixels to sample (default: 9, a 3×3 grid).
        depth_map:  Optional (H, W) float depth image in metres. None = monocular.

    Returns:
        List of (x_w, y_w) world-coordinate pairs.
    """
    arr = np.asarray(mask)
    ys, xs = np.where(arr)
    if len(ys) == 0:
        return []

    indices = np.linspace(0, len(ys) - 1, min(n_samples, len(ys)), dtype=int)
    world_pts = []
    for idx in indices:
        row, col = int(ys[idx]), int(xs[idx])
        d = float(depth_map[row, col]) if depth_map is not None else None
        pt = pixel_to_world(row, col, pose, fx, fy, cx_principal, cy_principal, mount, depth_m=d)
        if pt is not None:
            world_pts.append(pt)
    return world_pts


# ── Pose integrators ──────────────────────────────────────────────────────────

class MockForwardOdometry:
    """
    Deterministic pose sequence for dataset evaluation without real odometry.

    Assumes the robot drives forward at a constant speed with zero lateral drift.
    Useful for testing multi-frame GP accumulation on GOOSE/RELLIS-3D sequences
    where real odometry data is not available.

    Args:
        speed_mps:      Forward speed in metres per second (default 0.5 m/s).
        fps:            Camera frame rate (default 5 fps = one frame per 0.2 s).
        initial_pose:   Starting pose (default: origin, heading east).
    """

    def __init__(
        self,
        speed_mps: float = 0.5,
        fps: float = 5.0,
        initial_pose: Optional[RobotPose] = None,
    ) -> None:
        self._speed = speed_mps
        self._dt = 1.0 / fps
        self._pose = initial_pose or RobotPose(source="mock")
        self._frame_idx = 0

    def next_pose(self) -> RobotPose:
        """Advance one frame and return the new pose."""
        dx_fwd = self._speed * self._dt
        self._pose = self._pose.advance(dx_fwd, 0.0)
        self._pose = RobotPose(
            x=self._pose.x,
            y=self._pose.y,
            theta=self._pose.theta,
            timestamp=self._frame_idx * self._dt,
            source="mock",
        )
        self._frame_idx += 1
        return self._pose

    @property
    def current_pose(self) -> RobotPose:
        return self._pose

    def reset(self, pose: Optional[RobotPose] = None) -> None:
        self._pose = pose or RobotPose(source="mock")
        self._frame_idx = 0


class OpticalFlowOdometry:
    """
    Visual odometry from sparse Lucas-Kanade optical flow (OpenCV).

    Estimates inter-frame robot displacement by tracking feature points in the
    bottom half of the image (ground region, excluding sky and distant objects).

    Horizontal flow → lateral (y) displacement in robot frame.
    Vertical flow   → forward (x) displacement via ground-plane scale.

    The scale from vertical pixel flow to forward metres uses the same flat-ground
    assumption as monocular_ground_depth():
      forward_m = (drow_px / fy) * depth_at_row_m

    Integrates delta-poses to maintain a running world pose.

    Args:
        fx, fy:       Camera focal lengths in pixels.
        cx, cy:       Principal point in pixels.
        mount:        CameraMount parameters.
        initial_pose: Starting pose (default: origin).
        lk_params:    Lucas-Kanade optical flow parameters (dict).
        feature_params: Shi-Tomasi corner detector parameters (dict).
    """

    def __init__(
        self,
        fx: float = 615.0,
        fy: float = 615.0,
        cx: float = 320.0,
        cy: float = 240.0,
        mount: Optional[CameraMount] = None,
        initial_pose: Optional[RobotPose] = None,
    ) -> None:
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._mount = mount or CameraMount()
        self._pose = initial_pose or RobotPose(source="visual_odometry")
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self._frame_idx = 0

        # Lucas-Kanade parameters
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(3, 30, 0.01),  # cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT
        )
        # Shi-Tomasi feature parameters
        self._feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7,
        )

    def update(
        self,
        image_rgb: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        dt: float = 0.2,
    ) -> RobotPose:
        """
        Process a new frame and return the updated world pose.

        On the first call (no prior frame), returns the initial pose unchanged.

        Args:
            image_rgb:  (H, W, 3) uint8 RGB image.
            depth_map:  Optional (H, W) float depth in metres. When None, monocular
                        depth estimation is used for scale recovery.
            dt:         Time delta since last frame in seconds (for timestamp).

        Returns:
            Updated RobotPose in the world frame.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python required for OpticalFlowOdometry: pip install opencv-python")

        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Restrict feature tracking to the bottom half (ground region)
        ground_half = gray[h // 2:, :]

        if self._prev_gray is None or self._prev_pts is None or len(self._prev_pts) < 5:
            # First frame or feature starvation — re-detect features
            pts = cv2.goodFeaturesToTrack(ground_half, mask=None, **self._feature_params)
            if pts is not None:
                # Shift row indices back to full image coordinates
                pts[:, 0, 1] += h // 2
                self._prev_pts = pts
            self._prev_gray = gray
            self._frame_idx += 1
            self._pose = RobotPose(
                x=self._pose.x, y=self._pose.y, theta=self._pose.theta,
                timestamp=self._frame_idx * dt, source="visual_odometry",
            )
            return self._pose

        # Track features from previous to current frame
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **self._lk_params
        )

        # Keep only successfully tracked points
        good_prev = self._prev_pts[status.ravel() == 1]
        good_curr = curr_pts[status.ravel() == 1]

        if len(good_prev) < 4:
            # Not enough matches — no pose update, re-seed features
            pts = cv2.goodFeaturesToTrack(ground_half, mask=None, **self._feature_params)
            if pts is not None:
                pts[:, 0, 1] += h // 2
                self._prev_pts = pts
            self._prev_gray = gray
            return self._pose

        # Compute median pixel displacement (robust to outliers)
        flow = good_curr - good_prev               # (N, 1, 2)
        flow_np = flow.reshape(-1, 2)
        median_dcol = float(np.median(flow_np[:, 0]))  # horizontal (x in image)
        median_drow = float(np.median(flow_np[:, 1]))  # vertical   (y in image)

        # Horizontal pixel flow → lateral robot displacement
        # Small angle: dcol_px / fx ≈ sin(lateral_angle) ≈ lateral_angle
        # dy_left = -dcol_px * depth / fx  (camera +x = robot -y)
        mean_row = float(np.mean(good_prev[:, 0, 1]))
        ref_depth = None
        if depth_map is not None:
            r = min(int(mean_row), h - 1)
            ref_depth = float(np.median(depth_map[r, :]))
        else:
            ref_depth = monocular_ground_depth(int(mean_row), self._fy, self._cy, self._mount)

        if ref_depth is None or ref_depth <= 0:
            ref_depth = 2.0   # fallback: 2m if depth unavailable

        dy_left = -median_dcol * ref_depth / self._fx

        # Vertical pixel flow → forward displacement via ground-plane scale
        # drow_px / fy ≈ angular change in depression → scale change → forward motion
        # dx_fwd = -drow_px * ref_depth / fy  (moving forward → features move up = drow < 0)
        dx_fwd = -median_drow * ref_depth / self._fy

        # Update pose
        self._pose = self._pose.advance(dx_fwd, dy_left)
        self._frame_idx += 1
        self._pose = RobotPose(
            x=self._pose.x, y=self._pose.y, theta=self._pose.theta,
            timestamp=self._frame_idx * dt, source="visual_odometry",
        )

        # Re-seed features every 10 frames to prevent drift from track aging
        if self._frame_idx % 10 == 0:
            pts = cv2.goodFeaturesToTrack(ground_half, mask=None, **self._feature_params)
            if pts is not None:
                pts[:, 0, 1] += h // 2
                self._prev_pts = pts
            else:
                self._prev_pts = good_curr.reshape(-1, 1, 2)
        else:
            self._prev_pts = good_curr.reshape(-1, 1, 2)

        self._prev_gray = gray
        return self._pose

    @property
    def current_pose(self) -> RobotPose:
        return self._pose

    def reset(self, pose: Optional[RobotPose] = None) -> None:
        self._pose = pose or RobotPose(source="visual_odometry")
        self._prev_gray = None
        self._prev_pts = None
        self._frame_idx = 0


class GPSOdometry:
    """
    Robot pose from GPS (latitude/longitude) + compass heading.

    Converts GPS coordinates to a local metric frame using an equirectangular
    projection around the first fix. Accurate to ~1 m over distances < 1 km.

    Args:
        initial_pose: Override the map origin pose (default: origin).
    """

    METRES_PER_DEG_LAT: float = 111_320.0

    def __init__(self, initial_pose: Optional[RobotPose] = None) -> None:
        self._origin_lat: Optional[float] = None
        self._origin_lon: Optional[float] = None
        self._pose = initial_pose or RobotPose(source="gps")

    def update(
        self,
        lat: float,
        lon: float,
        heading_rad: float = 0.0,
        timestamp: float = 0.0,
    ) -> RobotPose:
        """
        Update pose from a GPS fix.

        Args:
            lat:          Latitude in decimal degrees.
            lon:          Longitude in decimal degrees.
            heading_rad:  Compass heading in radians (0 = north, counter-clockwise).
            timestamp:    Unix timestamp in seconds.

        Returns:
            Updated RobotPose in local metric frame.
        """
        if self._origin_lat is None:
            self._origin_lat = lat
            self._origin_lon = lon

        lat_m_per_deg = self.METRES_PER_DEG_LAT
        lon_m_per_deg = self.METRES_PER_DEG_LAT * math.cos(math.radians(self._origin_lat))

        x_w = (lon - self._origin_lon) * lon_m_per_deg
        y_w = (lat - self._origin_lat) * lat_m_per_deg

        self._pose = RobotPose(
            x=x_w, y=y_w, theta=heading_rad,
            timestamp=timestamp, source="gps",
        )
        return self._pose

    @property
    def current_pose(self) -> RobotPose:
        return self._pose

    def reset(self) -> None:
        self._origin_lat = None
        self._origin_lon = None
        self._pose = RobotPose(source="gps")
