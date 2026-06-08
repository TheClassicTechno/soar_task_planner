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
    Generates a fan of candidate trajectories toward a specified goal pixel.

    Per CRESTE (Shah et al., ICRA 2023) and similar RGB-based navigation work,
    pre-chosen trajectories must cover the full lateral range so that if ANY
    traversable corridor exists, at least one trajectory finds it.  CRESTE uses
    31 constant-curvature arcs; we use a parametric fan of N quadratic Bézier
    curves with evenly-spaced lateral offsets from -max_offset to +max_offset.

    With default n_trajectories=7, max_offset=0.75 the fan covers:
      right_3 (-0.75), right_2 (-0.50), right_1 (-0.25),
      direct  ( 0.00),
      left_1  (+0.25), left_2  (+0.50), left_3  (+0.75)

    The lateral offset for each arc = offset_fraction × path_length × n̂, where
    n̂ is the unit perpendicular to the start→goal direction.  The direct path
    (offset=0) uses a straight line; all others use quadratic Bézier curves.

    Args:
        image_height:   Image height in pixels.
        image_width:    Image width in pixels.
        n_waypoints:    Waypoints per trajectory (must be >= 2, default 20).
        n_trajectories: Number of fan trajectories (must be odd ≥ 3, default 7).
        max_offset:     Maximum lateral offset as fraction of path length (default 0.75).
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        n_waypoints: int = 20,
        n_trajectories: int = 7,
        max_offset: float = 0.75,
    ) -> None:
        if n_waypoints < 2:
            raise ValueError("n_waypoints must be at least 2")
        if n_trajectories < 3 or n_trajectories % 2 == 0:
            raise ValueError("n_trajectories must be an odd number >= 3")
        self._h = image_height
        self._w = image_width
        self._n = n_waypoints
        self._n_traj = n_trajectories
        self._max_offset = max_offset

    def generate_toward_goal(
        self,
        start_pixel: Tuple[int, int],
        goal_pixel: Tuple[int, int],
    ) -> List[Trajectory]:
        """
        Return a fan of n_trajectories candidate paths from start_pixel to goal_pixel.

        Fan geometry (n_trajectories=7, max_offset=0.75, image y increases downward):

            P₀ (start, bottom-center)
            |  \\  |  /  |
           R3  R2 | L2  L3
             R1   |   L1
                direct
                  |
                 P₂ (goal)

        Bézier formula for all non-direct paths:
            B(t) = (1−t)²·P₀  +  2(1−t)t·P₁  +  t²·P₂,   t ∈ [0, 1]
        where P₁ = M + offset_fraction × path_length × n̂
              M  = midpoint of P₀→P₂
              n̂  = unit perpendicular (positive = left, negative = right)

        The direct path (offset=0) uses a straight line for zero curvature.

        Args:
            start_pixel: (y, x) current robot position in image coordinates.
            goal_pixel:  (y, x) navigation goal in image coordinates.

        Returns:
            List of n_trajectories unscored Trajectory objects, ordered from
            rightmost (most negative offset) to leftmost (most positive offset),
            with "direct" at center.  Names: right_N, ..., right_1, direct,
            left_1, ..., left_N  where N = (n_trajectories - 1) // 2.
        """
        y0, x0 = float(start_pixel[0]), float(start_pixel[1])
        y1, x1 = float(goal_pixel[0]), float(goal_pixel[1])

        mid_y = (y0 + y1) / 2.0
        mid_x = (x0 + x1) / 2.0

        # Unit perpendicular to start→goal direction.
        # In image coords (y down, x right): rotate (dy,dx) 90° CCW → (−dx, dy).
        dy, dx = y1 - y0, x1 - x0
        path_length = float(np.sqrt(dy**2 + dx**2)) + 1e-6
        perp_y = -dx / path_length
        perp_x = dy / path_length

        # Evenly-spaced offsets from -max_offset to +max_offset.
        # Positive = left, negative = right (in image perpendicular convention).
        offsets = np.linspace(-self._max_offset, self._max_offset, self._n_traj)
        n_side = (self._n_traj - 1) // 2  # number of arcs on each side

        trajectories = []
        for offset_frac in offsets:
            lateral = offset_frac * path_length
            if abs(offset_frac) < 1e-9:
                name = "direct"
                waypoints = self._line(y0, x0, y1, x1)
            else:
                ctrl = (
                    mid_y + lateral * perp_y,
                    mid_x + lateral * perp_x,
                )
                if offset_frac > 0:
                    # Rank by distance from center: left_1 (closest) … left_N (farthest)
                    rank = round(offset_frac / self._max_offset * n_side)
                    name = f"left_{rank}"
                else:
                    rank = round(-offset_frac / self._max_offset * n_side)
                    name = f"right_{rank}"
                waypoints = self._bezier((y0, x0), ctrl, (y1, x1))
            trajectories.append(Trajectory(name=name, waypoints=waypoints))

        return trajectories

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
