"""
Unit tests for system/env_uncertainty/trajectory.py

Tests:
  TrajectoryGenerator:
    - generate_trajectories returns exactly 3 trajectories
    - trajectory names are "forward", "left_arc", "right_arc"
    - waypoints count matches n_waypoints
    - all waypoints are within image bounds
    - first waypoint starts at bottom-center
    - invalid n_waypoints raises ValueError

  score_trajectory:
    - fully-known path gets non-zero mean and min traversability
    - fully-unknown path gets mean 0.0 and passes_through_unknown True
    - mixed path: min < mean, passes_through_unknown True

  select_best_trajectory:
    - returns None when all paths pass through unknown
    - returns best (highest min_traversability) safe path
    - ignores trajectories with passes_through_unknown=True
"""

import numpy as np
import pytest

from system.env_uncertainty.trajectory import Trajectory, TrajectoryGenerator
from system.env_uncertainty.traversability import TraversabilityMap


H, W = 100, 150


# ── Fixture helpers ───────────────────────────────────────────────────────────

def _all_known_map(h=H, w=W):
    tmap = TraversabilityMap.create(h, w)
    full = np.ones((h, w), dtype=bool)
    return tmap.update_region(full, "grass")  # score 0.9 everywhere


def _all_unknown_map(h=H, w=W):
    return TraversabilityMap.create(h, w)  # all zeros


def _gen(h=H, w=W, n=20):
    return TrajectoryGenerator(h, w, n_waypoints=n)


# ── generate_trajectories ─────────────────────────────────────────────────────

def test_returns_exactly_three_trajectories():
    trajs = _gen().generate_trajectories()
    assert len(trajs) == 3


def test_trajectory_names():
    trajs = _gen().generate_trajectories()
    names = {t.name for t in trajs}
    assert names == {"forward", "left_arc", "right_arc"}


def test_waypoint_count_matches_n_waypoints():
    n = 15
    trajs = TrajectoryGenerator(H, W, n_waypoints=n).generate_trajectories()
    for t in trajs:
        assert len(t.waypoints) == n


def test_all_waypoints_within_image_bounds():
    trajs = _gen().generate_trajectories()
    for traj in trajs:
        for y, x in traj.waypoints:
            assert 0 <= y < H, f"{traj.name}: y={y} out of bounds"
            assert 0 <= x < W, f"{traj.name}: x={x} out of bounds"


def test_first_waypoint_near_bottom_center():
    trajs = _gen().generate_trajectories()
    for traj in trajs:
        y0, x0 = traj.waypoints[0]
        # Start point should be near bottom of image
        assert y0 >= H - 5, f"{traj.name}: start y={y0} not near bottom"


def test_invalid_n_waypoints_raises():
    with pytest.raises(ValueError):
        TrajectoryGenerator(H, W, n_waypoints=1)


# ── score_trajectory ──────────────────────────────────────────────────────────

def test_score_fully_known_path():
    gen = _gen()
    tmap = _all_known_map()
    traj = gen.generate_trajectories()[0]  # forward
    scored = gen.score_trajectory(traj, tmap)
    assert scored.mean_traversability > 0.0
    assert scored.min_traversability > 0.0
    assert scored.passes_through_unknown is False


def test_score_fully_unknown_path():
    gen = _gen()
    tmap = _all_unknown_map()
    traj = gen.generate_trajectories()[0]
    scored = gen.score_trajectory(traj, tmap)
    assert scored.mean_traversability == pytest.approx(0.0)
    assert scored.min_traversability == pytest.approx(0.0)
    assert scored.passes_through_unknown is True


def test_score_mixed_path_has_passes_through_unknown_true():
    gen = _gen()
    # Top half unknown, bottom half grass
    tmap = TraversabilityMap.create(H, W)
    bottom_mask = np.zeros((H, W), dtype=bool)
    bottom_mask[H // 2:, :] = True
    tmap = tmap.update_region(bottom_mask, "grass")
    # Forward trajectory goes from bottom to top — will cross unknown zone
    traj = gen.generate_trajectories()[0]
    scored = gen.score_trajectory(traj, tmap)
    assert scored.passes_through_unknown is True


def test_score_returns_new_trajectory_instance():
    gen = _gen()
    tmap = _all_known_map()
    traj = gen.generate_trajectories()[0]
    scored = gen.score_trajectory(traj, tmap)
    assert scored is not traj


# ── select_best_trajectory ────────────────────────────────────────────────────

def test_select_returns_none_when_all_unknown():
    gen = _gen()
    tmap = _all_unknown_map()
    trajs = [gen.score_trajectory(t, tmap) for t in gen.generate_trajectories()]
    assert gen.select_best_trajectory(trajs) is None


def test_select_returns_best_safe_trajectory():
    gen = _gen()
    # Manually construct two scored trajectories: one safe (min=0.9), one unsafe
    safe = Trajectory("safe", [(0, 0)], mean_traversability=0.9, min_traversability=0.9, passes_through_unknown=False)
    unsafe = Trajectory("unsafe", [(0, 0)], mean_traversability=0.0, min_traversability=0.0, passes_through_unknown=True)
    result = gen.select_best_trajectory([safe, unsafe])
    assert result is safe


def test_select_picks_highest_min_among_safe():
    gen = _gen()
    t1 = Trajectory("a", [], mean_traversability=0.7, min_traversability=0.6, passes_through_unknown=False)
    t2 = Trajectory("b", [], mean_traversability=0.9, min_traversability=0.8, passes_through_unknown=False)
    t3 = Trajectory("c", [], mean_traversability=0.5, min_traversability=0.3, passes_through_unknown=False)
    result = gen.select_best_trajectory([t1, t2, t3])
    assert result is t2


def test_select_ignores_unknown_trajectories():
    gen = _gen()
    bad = Trajectory("bad", [], mean_traversability=1.0, min_traversability=1.0, passes_through_unknown=True)
    good = Trajectory("good", [], mean_traversability=0.5, min_traversability=0.5, passes_through_unknown=False)
    result = gen.select_best_trajectory([bad, good])
    assert result is good
