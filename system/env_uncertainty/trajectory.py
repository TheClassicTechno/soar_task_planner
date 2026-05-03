"""
Candidate trajectory generation and traversability scoring.

The robot considers three simulated trajectories through each scene:
  - forward:    straight line from robot position to scene center-top
  - left_arc:   arc curving from center to left edge of scene
  - right_arc:  arc curving from center to right edge of scene

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
