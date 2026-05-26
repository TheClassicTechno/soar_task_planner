"""
Navigation stack interface — publishes subgoals to /task_subgoal.

Converts the best trajectory waypoint from image pixel coordinates to
robot-frame metric (dx, dy) offsets and publishes a PoseStamped to the
DWA motion planner.

Coordinate transform chain (Step 10 of the pipeline):
  image pixel (y_px, x_px)  +  depth_m (from RGB-D or lidar)
  → pinhole projection into camera frame:
        X_c = (x_px − cx) * depth / fx      (right in camera frame)
        Z_c = depth_m                         (forward in camera frame)
  → robot frame (ROS REP 103, right-hand, x=forward, y=left):
        dx =  Z_c     (camera forward  → robot forward)
        dy = −X_c     (camera right    → robot right = −robot left)
  → PoseStamped(frame_id="robot_frame", x=dx, y=dy, z=0)
  → publish to /task_subgoal

The motion planner (jingGM/navigation_stack) subscribes to /task_subgoal
and passes the goal to the DWA planner, which outputs /gita_2/twist_cmd
velocity commands.

ROS2 is an optional dependency — the module can be imported and unit-tested
without a ROS2 installation.  All rclpy/geometry_msgs imports are guarded
by a try/except so pure-Python test environments work correctly.

Default camera intrinsics are from a typical Intel RealSense D435i at
640×480 resolution.  Override via CameraIntrinsics for a different sensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ROS2 imports are optional — guarded so tests work without a ROS2 install.
try:
    from geometry_msgs.msg import PoseStamped  # type: ignore
    import rclpy  # type: ignore  # noqa: F401
    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False


# ── Camera intrinsics ─────────────────────────────────────────────────────────

@dataclass
class CameraIntrinsics:
    """
    Pinhole camera intrinsics for pixel → camera-frame projection.

    Default values match a RealSense D435i at 640×480.

    Attributes:
        fx: Focal length in pixels (x-axis).
        fy: Focal length in pixels (y-axis).
        cx: Principal point x (pixels from left).
        cy: Principal point y (pixels from top).
    """
    fx: float = 615.0
    fy: float = 615.0
    cx: float = 320.0
    cy: float = 240.0


# ── Coordinate transforms (pure Python, no ROS2) ─────────────────────────────

def pixel_to_robot_frame(
    pixel_y: int,
    pixel_x: int,
    depth_m: float,
    intrinsics: Optional[CameraIntrinsics] = None,
) -> Tuple[float, float]:
    """
    Convert a depth-registered image pixel to (dx, dy) in the robot's frame.

    Uses pinhole projection followed by the standard camera→robot TF:
        dx  =  Z_c  (forward)
        dy  = −X_c  (left; camera +x is right, robot +y is left)

    Args:
        pixel_y:    Row index in the image (0-indexed from top).
        pixel_x:    Column index in the image (0-indexed from left).
        depth_m:    Metric depth at this pixel in metres (from RGB-D or lidar).
        intrinsics: CameraIntrinsics; uses D435i defaults when None.

    Returns:
        (dx, dy) — metres in the robot body frame (dx=forward, dy=left).
    """
    intr = intrinsics or CameraIntrinsics()
    X_c = (pixel_x - intr.cx) * depth_m / intr.fx   # camera-frame right
    Z_c = depth_m                                      # camera-frame forward
    dx = Z_c
    dy = -X_c
    return dx, dy


def make_posestamped(
    dx: float,
    dy: float,
    frame_id: str = "robot_frame",
) -> "PoseStamped":
    """
    Build a geometry_msgs/PoseStamped from (dx, dy) robot-frame offsets.

    Matches the format expected by jingGM/navigation_stack GoalInterface,
    which subscribes to /task_subgoal with frame_id="robot_frame".

    ROS2 must be installed to call this function.  Use pixel_to_robot_frame()
    alone if you only need the coordinate math without a ROS2 message.

    Args:
        dx:       Metres forward in robot frame (+x = forward).
        dy:       Metres left in robot frame (+y = left, -y = right).
        frame_id: TF frame name; must match motion_planner.yaml (default "robot_frame").

    Returns:
        PoseStamped with identity quaternion (heading unknown, DWA infers it).

    Raises:
        ImportError: If geometry_msgs is not installed (no ROS2 environment).
    """
    if not _ROS2_AVAILABLE:
        raise ImportError(
            "geometry_msgs is not installed. "
            "Source your ROS2 workspace before calling make_posestamped()."
        )

    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = float(dx)
    pose.pose.position.y = float(dy)
    pose.pose.position.z = 0.0

    # Identity quaternion — DWA planner infers heading from direction of motion
    pose.pose.orientation.x = 0.0
    pose.pose.orientation.y = 0.0
    pose.pose.orientation.z = 0.0
    pose.pose.orientation.w = 1.0

    return pose


# ── NavPublisher: ROS2-aware goal publisher ───────────────────────────────────

class NavPublisher:
    """
    Publishes navigation subgoals to /task_subgoal.

    Wraps the coordinate transform (pixel → robot frame) and the ROS2 publisher.
    Instantiate once per pipeline run; call publish_waypoint() each time the
    robot selects a new best trajectory waypoint after replanning (Step 10).

    Usage::

        import rclpy
        rclpy.init()
        node = rclpy.create_node("soar_task_planner")
        pub = NavPublisher(node)
        pub.publish_waypoint(best_y, best_x, depth_at_waypoint)

    Args:
        node:        An rclpy Node to create the publisher on.
        topic:       Topic name (default "/task_subgoal").
        frame_id:    TF frame (default "robot_frame").
        intrinsics:  Camera intrinsics; uses D435i defaults when None.
        queue_size:  ROS2 publisher queue size.
    """

    def __init__(
        self,
        node,  # rclpy.node.Node — not typed to avoid import at module level
        topic: str = "/task_subgoal",
        frame_id: str = "robot_frame",
        intrinsics: Optional[CameraIntrinsics] = None,
        queue_size: int = 10,
    ) -> None:
        if not _ROS2_AVAILABLE:
            raise ImportError("rclpy and geometry_msgs are required for NavPublisher.")

        self._node = node
        self._frame_id = frame_id
        self._intrinsics = intrinsics or CameraIntrinsics()
        self._pub = node.create_publisher(PoseStamped, topic, queue_size)

    def publish_waypoint(
        self,
        pixel_y: int,
        pixel_x: int,
        depth_m: float,
    ) -> Tuple[float, float]:
        """
        Convert a trajectory waypoint to robot-frame offsets and publish.

        Args:
            pixel_y: Row index of the best waypoint in the current image.
            pixel_x: Column index of the best waypoint.
            depth_m: Metric depth at that pixel (from RGB-D sensor).

        Returns:
            (dx, dy) in metres — the offsets that were published.
        """
        dx, dy = pixel_to_robot_frame(
            pixel_y, pixel_x, depth_m, self._intrinsics
        )
        msg = make_posestamped(dx, dy, self._frame_id)
        msg.header.stamp = self._node.get_clock().now().to_msg()
        self._pub.publish(msg)
        return dx, dy

    @property
    def ros2_available(self) -> bool:
        """True when ROS2 is installed and this publisher can send messages."""
        return _ROS2_AVAILABLE
