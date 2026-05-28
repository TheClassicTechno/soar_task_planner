"""
Candidate trajectory generation and traversability scoring.

Two generators are provided:

  TrajectoryGenerator         — original fixed-geometry generator.  Produces
                                 three paths (forward, left_arc, right_arc) from
                                 the bottom-center of the image toward the top.
                                 Used when no explicit navigation goal is available.

  GoalDirectedTrajectoryGenerator — goal-directed generator (May 19 mentor
                                 feedback).  Given an explicit goal pixel, produces
                                 three quadratic Bézier paths that all lead toward
                                 that goal: one straight and two with lateral detours.
                                 Use this when the robot has a navigation goal.

Trajectories are represented as sequences of (y, x) pixel coordinates —
a simplified 2D model of the robot's footprint over the image plane.

After scoring against a TraversabilityMap, the runner selects the trajectory
with the highest minimum traversability (safety-first selection).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from system.env_uncertainty.traversability import TraversabilityMap


@dataclass
class Trajectory:
    """
    One candidate robot path through a scene.

    waypoints: ordered list of (y, x) pixel coordinates the robot would visit
    name:      human-readable identifier ("forward", "left_arc", "right_arc")
    mean_traversability: average score over all waypoints (computed after scoring)
    min_traversability:  worst-case score along the path (used for safety selection)
    passes_through_unknown: True if any waypoint has traversability == 0.0
    """

    name: str
    waypoints: List[Tuple[int, int]]
    mean_traversability: float = 0.0
    min_traversability: float = 0.0
    passes_through_unknown: bool = True


class TrajectoryGenerator:
    """
    Generates and scores three candidate trajectories for a scene.

    All trajectories start at the bottom-center of the image (simulating the
    robot's current position as seen in its front-facing camera) and end at
    different horizontal positions near the top of the image.

    The generator does NOT account for real 3D geometry — it works entirely
    in image pixel space as a simplified proxy for spatial trajectory planning.
    """

    def __init__(self, image_height: int, image_width: int, n_waypoints: int = 20):
        """
        Args:
            image_height: Image height in pixels.
            image_width:  Image width in pixels.
            n_waypoints:  Number of waypoints to sample along each trajectory.
        """
        if n_waypoints < 2:
            raise ValueError("n_waypoints must be at least 2")
        self._h = image_height
        self._w = image_width
        self._n = n_waypoints

    def generate_trajectories(self) -> List[Trajectory]:
        """
        Return the three candidate trajectories (unscored).

        Each trajectory starts at (h-1, w//2) — bottom center — and ends at
        a different target determined by the trajectory name.
        """
        start_y = self._h - 1
        start_x = self._w // 2
        horizon_y = self._h // 5   # aim for near the top of the scene

        return [
            Trajectory(
                name="forward",
                waypoints=self._straight_line(start_y, start_x, horizon_y, self._w // 2),
            ),
            Trajectory(
                name="left_arc",
                waypoints=self._arc(start_y, start_x, horizon_y, self._w // 4),
            ),
            Trajectory(
                name="right_arc",
                waypoints=self._arc(start_y, start_x, horizon_y, 3 * self._w // 4),
            ),
        ]

    def score_trajectory(
        self, traj: Trajectory, tmap: TraversabilityMap
    ) -> Trajectory:
        """
        Score a trajectory against a traversability map.

        Computes mean and minimum traversability across all waypoints and
        checks whether any waypoint falls in an unknown region (score == 0.0).

        Args:
            traj: Trajectory to score (waypoints must already be set).
            tmap: TraversabilityMap for the current scene.

        Returns:
            A new Trajectory with mean_traversability, min_traversability,
            and passes_through_unknown filled in.
        """
        scores = [tmap.score_at(y, x) for y, x in traj.waypoints]
        mean_t = float(np.mean(scores)) if scores else 0.0
        min_t = float(np.min(scores)) if scores else 0.0
        has_unknown = any(s == 0.0 for s in scores)

        return Trajectory(
            name=traj.name,
            waypoints=traj.waypoints,
            mean_traversability=mean_t,
            min_traversability=min_t,
            passes_through_unknown=has_unknown,
        )

    def select_best_trajectory(
        self, trajectories: List[Trajectory]
    ) -> Optional[Trajectory]:
        """
        Choose the safest trajectory from a scored list.

        Selection rule (safety-first):
          1. Prefer trajectories that avoid unknown regions entirely.
          2. Among those, pick the one with the highest minimum traversability.
          3. If all trajectories pass through unknown regions, return None
             (robot cannot safely proceed without user input).

        Args:
            trajectories: List of scored Trajectory objects.

        Returns:
            The best Trajectory, or None if all paths are unknown.
        """
        safe_paths = [t for t in trajectories if not t.passes_through_unknown]
        if not safe_paths:
            return None
        return max(safe_paths, key=lambda t: t.min_traversability)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _straight_line(
        self, y0: int, x0: int, y1: int, x1: int
    ) -> List[Tuple[int, int]]:
        """
        Sample n_waypoints along a straight line from (y0,x0) to (y1,x1).
        Uses linspace so waypoints are evenly spaced.
        """
        ys = np.linspace(y0, y1, self._n).astype(int)
        xs = np.linspace(x0, x1, self._n).astype(int)
        return [(int(y), int(x)) for y, x in zip(ys, xs)]

    def _arc(
        self, y0: int, x0: int, y1: int, x1: int
    ) -> List[Tuple[int, int]]:
        """
        Sample n_waypoints along a parabolic arc from (y0,x0) to (y1,x1).

        The arc bends toward the target horizontally — a simple quadratic
        interpolation that curves more than a straight line, approximating
        a turning trajectory.
        """
        t_vals = np.linspace(0.0, 1.0, self._n)
        # Quadratic interpolation: x curves faster than y
        ys = (y0 * (1 - t_vals) + y1 * t_vals).astype(int)
        # x follows a quadratic ease-in: starts slow, ends near target
        xs = (x0 + (x1 - x0) * t_vals ** 2).astype(int)
        # Clamp to image bounds
        ys = np.clip(ys, 0, self._h - 1)
        xs = np.clip(xs, 0, self._w - 1)
        return [(int(y), int(x)) for y, x in zip(ys, xs)]


# ═══════════════════════════════════════════════════════════════════════════════
# Goal-directed trajectory generator (May 19 mentor requirement)
# ═══════════════════════════════════════════════════════════════════════════════

class GoalDirectedTrajectoryGenerator:
    """
    Generates candidate trajectories that all lead toward a specified goal pixel.

    Per mentor feedback (May 19 meeting): the robot should have a navigation goal
    and evaluate uncertainty along the path TO that goal — not along arbitrary
    directions. This class replaces the fixed forward/left_arc/right_arc geometry
    when the robot knows where it wants to go.

    Three path variants via quadratic Bézier curves:
      "direct"        — straight line from start to goal (zero curvature)
      "left_detour"   — curves left of the direct line, then to goal
      "right_detour"  — curves right of the direct line, then to goal

    The lateral offset for detour paths equals detour_fraction * path_length,
    so curvature naturally scales with how far away the goal is.

    All three paths start and end at the same pixels; they only differ in the
    intermediate waypoints. The runner scores each for traversability uncertainty
    and picks the safest one.

    Args:
        image_height:    Image height in pixels.
        image_width:     Image width in pixels.
        n_waypoints:     Waypoints per trajectory (must be >= 2).
        detour_fraction: Lateral offset as a fraction of path length (default 0.25).
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        n_waypoints: int = 20,
        detour_fraction: float = 0.25,
    ) -> None:
        if n_waypoints < 2:
            raise ValueError("n_waypoints must be at least 2")
        self._h = image_height
        self._w = image_width
        self._n = n_waypoints
        self._detour = detour_fraction

    def generate_toward_goal(
        self,
        start_pixel: Tuple[int, int],
        goal_pixel: Tuple[int, int],
    ) -> List[Trajectory]:
        """
        Return three candidate trajectories from start_pixel to goal_pixel.

        All three paths use quadratic Bézier curves ending at goal_pixel.
        The quadratic Bézier formula is:

            B(t) = (1−t)²·P₀  +  2(1−t)t·P₁  +  t²·P₂,   t ∈ [0, 1]

        where P₀=start, P₂=goal, and P₁ is the control point.

        Control point calculation for detour paths:
          1. Compute midpoint M = (P₀ + P₂) / 2
          2. Compute unit direction d̂ = (P₂ − P₀) / ‖P₂ − P₀‖
          3. Rotate 90°: unit perpendicular n̂ = (−d̂_x, d̂_y)
          4. Offset: P₁ = M ± (detour_fraction × ‖P₂ − P₀‖) × n̂

        Geometry (image coordinates, y increases downward):

            P₀ (start, bottom-center)
             |
             |── direct path (P₁ = M, no offset)
            / \\
           /   \\
          L     R  ← left/right control points at midpoint ± perpendicular offset
           \\   /
            \\ /
             P₂ (goal)

        Note: control points are chosen based on path geometry only, not terrain.
        If a control point falls over a non-traversable region, the Bézier curve
        may still pass through it. The GP LCB scoring step (S3) handles this by
        giving that trajectory a low score — the robot then picks a different path
        or falls back to ASK if all paths are blocked.

        Args:
            start_pixel: (y, x) current robot position in image coordinates.
            goal_pixel:  (y, x) navigation goal in image coordinates.

        Returns:
            List of three unscored Trajectory objects: direct, left_detour, right_detour.
        """
        y0, x0 = float(start_pixel[0]), float(start_pixel[1])
        y1, x1 = float(goal_pixel[0]), float(goal_pixel[1])

        mid_y = (y0 + y1) / 2.0
        mid_x = (x0 + x1) / 2.0

        # Unit direction along start→goal, then rotate 90° CCW for perpendicular.
        # In image coords (y down, x right): rotate (dy,dx) 90° CCW → (−dx, dy).
        dy, dx = y1 - y0, x1 - x0
        path_length = float(np.sqrt(dy**2 + dx**2)) + 1e-6
        perp_y = -dx / path_length   # perpendicular component in y
        perp_x = dy / path_length    # perpendicular component in x

        # Lateral offset = detour_fraction × path_length so curvature scales with distance
        offset = self._detour * path_length
        left_ctrl = (mid_y + offset * perp_y, mid_x + offset * perp_x)
        right_ctrl = (mid_y - offset * perp_y, mid_x - offset * perp_x)

        return [
            Trajectory(
                name="direct",
                waypoints=self._line(y0, x0, y1, x1),
            ),
            Trajectory(
                name="left_detour",
                waypoints=self._bezier((y0, x0), left_ctrl, (y1, x1)),
            ),
            Trajectory(
                name="right_detour",
                waypoints=self._bezier((y0, x0), right_ctrl, (y1, x1)),
            ),
        ]

    def score_trajectory(
        self, traj: Trajectory, tmap: TraversabilityMap
    ) -> Trajectory:
        """
        Score a trajectory against a traversability map.

        Same logic as TrajectoryGenerator.score_trajectory — computes mean/min
        traversability and detects unknown regions (score == 0.0).
        """
        scores = [tmap.score_at(y, x) for y, x in traj.waypoints]
        mean_t = float(np.mean(scores)) if scores else 0.0
        min_t = float(np.min(scores)) if scores else 0.0
        has_unknown = any(s == 0.0 for s in scores)
        return Trajectory(
            name=traj.name,
            waypoints=traj.waypoints,
            mean_traversability=mean_t,
            min_traversability=min_t,
            passes_through_unknown=has_unknown,
        )

    def select_best_trajectory(
        self, trajectories: List[Trajectory]
    ) -> Optional[Trajectory]:
        """
        Choose the safest trajectory (safety-first: highest min_traversability
        among paths that avoid all unknown regions).

        Returns None if every path passes through an unknown region.
        """
        safe_paths = [t for t in trajectories if not t.passes_through_unknown]
        if not safe_paths:
            return None
        return max(safe_paths, key=lambda t: t.min_traversability)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _line(
        self, y0: float, x0: float, y1: float, x1: float
    ) -> List[Tuple[int, int]]:
        ys = np.clip(np.linspace(y0, y1, self._n), 0, self._h - 1).astype(int)
        xs = np.clip(np.linspace(x0, x1, self._n), 0, self._w - 1).astype(int)
        return [(int(y), int(x)) for y, x in zip(ys, xs)]

    def _bezier(
        self,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> List[Tuple[int, int]]:
        """
        Quadratic Bézier from p0 to p2 via control point p1.

            B(t) = (1−t)²·p0  +  2(1−t)t·p1  +  t²·p2,   t ∈ [0, 1]

        At t=0: B=p0 (start).  At t=1: B=p2 (goal).
        The control point p1 pulls the curve toward it without the curve
        actually passing through p1 — it is a "magnet" that determines
        how much the path bends.
        """
        t = np.linspace(0.0, 1.0, self._n)
        ys = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
        xs = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
        ys = np.clip(ys, 0, self._h - 1).astype(int)
        xs = np.clip(xs, 0, self._w - 1).astype(int)
        return [(int(y), int(x)) for y, x in zip(ys, xs)]
