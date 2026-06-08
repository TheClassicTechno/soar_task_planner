"""
Unit tests for GoalDirectedTrajectoryGenerator and nav_interface helpers.

Covers:
  GoalDirectedTrajectoryGenerator:
    - generate_toward_goal returns exactly n_trajectories (default 7)
    - trajectory names follow fan convention: right_N…right_1, direct, left_1…left_N
    - waypoint count matches n_waypoints
    - all waypoints are within image bounds
    - first waypoint is near start_pixel, last is near goal_pixel
    - left_detour and right_detour curve differently (distinct midpoints)
    - direct path is straight (midpoint is near line midpoint)
    - invalid n_waypoints raises ValueError
    - zero-length path (start == goal) produces valid waypoints
    - score_trajectory works on known and unknown terrain
    - select_best_trajectory returns safest path, None when all unknown

  nav_interface helpers (pure Python, no ROS2):
    - pixel_to_robot_frame correct geometry
    - dx > 0 for any pixel (depth always positive)
    - dy = 0 when pixel_x == cx (centre of image)
    - dy < 0 for pixel_x > cx (right of centre → negative robot y)
    - dy > 0 for pixel_x < cx (left of centre → positive robot y)
"""

import numpy as np
import pytest

from system.env_uncertainty.trajectory import GoalDirectedTrajectoryGenerator, Trajectory
from system.env_uncertainty.traversability import TraversabilityMap
from system.env_uncertainty.nav_interface import (
    CameraIntrinsics,
    pixel_to_robot_frame,
)

H, W = 120, 160


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _gen(h=H, w=W, n=20, n_traj=7, max_offset=0.75):
    return GoalDirectedTrajectoryGenerator(h, w, n_waypoints=n, n_trajectories=n_traj, max_offset=max_offset)


def _all_known_map(h=H, w=W):
    tmap = TraversabilityMap.create(h, w)
    full_mask = np.ones((h, w), dtype=bool)
    return tmap.update_region(full_mask, "grass")


def _all_unknown_map(h=H, w=W):
    return TraversabilityMap.create(h, w)  # all zeros


# ── generate_toward_goal ──────────────────────────────────────────────────────

def test_returns_n_trajectories_default_seven():
    gen = _gen()
    trajs = gen.generate_toward_goal((H - 1, W // 2), (H // 5, W // 2))
    assert len(trajs) == 7


def test_returns_custom_n_trajectories():
    gen = _gen(n_traj=9)
    trajs = gen.generate_toward_goal((H - 1, W // 2), (H // 5, W // 2))
    assert len(trajs) == 9


def test_trajectory_names_fan_convention():
    gen = _gen()  # 7 trajectories
    trajs = gen.generate_toward_goal((H - 1, W // 2), (H // 5, W // 2))
    names = {t.name for t in trajs}
    assert "direct" in names
    assert "left_1" in names and "left_2" in names and "left_3" in names
    assert "right_1" in names and "right_2" in names and "right_3" in names


def test_direct_is_center_of_fan():
    gen = _gen()
    trajs = gen.generate_toward_goal((H - 1, W // 2), (H // 5, W // 2))
    names = [t.name for t in trajs]
    center_idx = len(names) // 2
    assert names[center_idx] == "direct", f"Center of fan should be 'direct', got {names[center_idx]}"


def test_invalid_even_n_trajectories_raises():
    with pytest.raises(ValueError):
        GoalDirectedTrajectoryGenerator(H, W, n_trajectories=6)


def test_invalid_n_trajectories_too_small_raises():
    with pytest.raises(ValueError):
        GoalDirectedTrajectoryGenerator(H, W, n_trajectories=1)


def test_waypoint_count_matches_n_waypoints():
    n = 15
    gen = _gen(n=n)
    trajs = gen.generate_toward_goal((H - 1, W // 2), (H // 5, W // 2))
    for traj in trajs:
        assert len(traj.waypoints) == n


def test_all_waypoints_within_image_bounds():
    gen = _gen()
    trajs = gen.generate_toward_goal((H - 1, 10), (5, W - 10))
    for traj in trajs:
        for y, x in traj.waypoints:
            assert 0 <= y < H, f"{traj.name}: y={y} out of bounds"
            assert 0 <= x < W, f"{traj.name}: x={x} out of bounds"


def test_first_waypoint_near_start():
    start = (H - 1, W // 2)
    gen = _gen()
    trajs = gen.generate_toward_goal(start, (H // 5, W // 2))
    for traj in trajs:
        y0, x0 = traj.waypoints[0]
        assert abs(y0 - start[0]) <= 2, f"{traj.name}: start y mismatch"
        assert abs(x0 - start[1]) <= 2, f"{traj.name}: start x mismatch"


def test_last_waypoint_near_goal():
    goal = (10, 30)
    gen = _gen()
    trajs = gen.generate_toward_goal((H - 1, W // 2), goal)
    for traj in trajs:
        y_last, x_last = traj.waypoints[-1]
        assert abs(y_last - goal[0]) <= 3, f"{traj.name}: end y mismatch"
        assert abs(x_last - goal[1]) <= 3, f"{traj.name}: end x mismatch"


def test_direct_path_is_roughly_straight():
    start = (H - 1, W // 2)
    goal = (0, W // 2)
    gen = _gen(n=21)
    trajs = gen.generate_toward_goal(start, goal)
    direct = next(t for t in trajs if t.name == "direct")
    mid = direct.waypoints[len(direct.waypoints) // 2]
    expected_mid_y = (start[0] + goal[0]) // 2
    expected_mid_x = (start[1] + goal[1]) // 2
    assert abs(mid[0] - expected_mid_y) <= 3
    assert abs(mid[1] - expected_mid_x) <= 3


def test_left_and_right_arcs_have_different_midpoints():
    start = (H - 1, W // 2)
    goal = (0, W // 2)
    gen = _gen(n=21)
    trajs = gen.generate_toward_goal(start, goal)
    left = next(t for t in trajs if t.name == "left_1")
    right = next(t for t in trajs if t.name == "right_1")
    mid_idx = len(left.waypoints) // 2
    assert left.waypoints[mid_idx] != right.waypoints[mid_idx], (
        "left_1 and right_1 should curve to opposite sides"
    )


def test_non_direct_paths_deviate_from_center():
    start = (H - 1, W // 2)
    goal = (0, W // 2)
    gen = _gen(n=21, max_offset=0.50)
    trajs = gen.generate_toward_goal(start, goal)
    for traj in (t for t in trajs if t.name != "direct"):
        mid_idx = len(traj.waypoints) // 2
        mid_x = traj.waypoints[mid_idx][1]
        assert abs(mid_x - W // 2) > 2, f"{traj.name} did not deviate from center"


def test_fan_finds_safe_arc_when_center_path_blocked():
    """
    Item 5.3 validation: if the direct/center path is blocked by an obstacle
    but the sides are clear, at least one fan arc must find a safe path.

    This is the key correctness guarantee Jing asked for:
    'if there is any traversable area, you should be able to find it.'

    Design: initialize full grass map (all clear), then place a rectangular
    obstacle at mid-height center only — so start/end pixels (x=50) are clear
    but the mid-path center is blocked.  Wide arcs (left_2/right_2+) curve
    around the obstacle at mid-height.
    """
    gen = _gen()  # 7-arc fan, max_offset=0.75

    # Step 1: set entire map to grass (traversable)
    tmap = TraversabilityMap.create(H, W)
    full_mask = np.ones((H, W), dtype=bool)
    tmap = tmap.update_region(full_mask, "grass")

    # Step 2: block a rectangular obstacle at mid-height center.
    # y=[H//3, 2*H//3] ensures start (y=H-1) and goal (y=H//5) are outside obstacle.
    # x=[2*W//5, 3*W//5] blocks the center column that direct path travels through.
    obstacle = np.zeros((H, W), dtype=bool)
    obstacle[H // 3 : 2 * H // 3, 2 * W // 5 : 3 * W // 5] = True
    tmap = tmap.update_region(obstacle, "unknown")

    start = (H - 1, W // 2)
    goal = (H // 5, W // 2)
    raw = gen.generate_toward_goal(start, goal)
    scored = [gen.score_trajectory(t, tmap) for t in raw]
    best = gen.select_best_trajectory(scored)

    assert best is not None, (
        "Fan should find a safe arc when center is blocked but sides are clear"
    )
    assert best.name != "direct", (
        f"Direct path should be blocked; fan should pick a side arc, got '{best.name}'"
    )


def test_invalid_n_waypoints_raises():
    with pytest.raises(ValueError):
        GoalDirectedTrajectoryGenerator(H, W, n_waypoints=1)


def test_zero_length_path_produces_valid_waypoints():
    # start == goal: all waypoints should be at the same pixel
    pos = (H // 2, W // 2)
    gen = _gen(n=10)
    trajs = gen.generate_toward_goal(pos, pos)
    for traj in trajs:
        assert len(traj.waypoints) == 10
        for y, x in traj.waypoints:
            assert 0 <= y < H
            assert 0 <= x < W


# ── score_trajectory ──────────────────────────────────────────────────────────

def test_score_known_terrain_has_positive_traversability():
    gen = _gen()
    tmap = _all_known_map()
    trajs = gen.generate_toward_goal((H - 1, W // 2), (H // 5, W // 2))
    scored = gen.score_trajectory(trajs[0], tmap)
    assert scored.mean_traversability > 0.0
    assert scored.min_traversability > 0.0
    assert scored.passes_through_unknown is False


def test_score_unknown_terrain_is_zero():
    gen = _gen()
    tmap = _all_unknown_map()
    trajs = gen.generate_toward_goal((H - 1, W // 2), (H // 5, W // 2))
    scored = gen.score_trajectory(trajs[0], tmap)
    assert scored.mean_traversability == pytest.approx(0.0)
    assert scored.passes_through_unknown is True


def test_score_returns_new_trajectory_instance():
    gen = _gen()
    tmap = _all_known_map()
    traj = gen.generate_toward_goal((H - 1, W // 2), (H // 5, W // 2))[0]
    scored = gen.score_trajectory(traj, tmap)
    assert scored is not traj


# ── select_best_trajectory ────────────────────────────────────────────────────

def test_select_returns_none_when_all_unknown():
    gen = _gen()
    tmap = _all_unknown_map()
    trajs = [gen.score_trajectory(t, tmap) for t in gen.generate_toward_goal((H-1, W//2), (0, W//2))]
    assert gen.select_best_trajectory(trajs) is None


def test_select_returns_safest_path():
    gen = _gen()
    safe = Trajectory("a", [(0, 0)], mean_traversability=0.9, min_traversability=0.9, passes_through_unknown=False)
    unsafe = Trajectory("b", [(0, 0)], mean_traversability=0.0, min_traversability=0.0, passes_through_unknown=True)
    best = gen.select_best_trajectory([safe, unsafe])
    assert best is safe


def test_select_picks_highest_min_among_safe():
    gen = _gen()
    t1 = Trajectory("a", [], mean_traversability=0.7, min_traversability=0.5, passes_through_unknown=False)
    t2 = Trajectory("b", [], mean_traversability=0.9, min_traversability=0.85, passes_through_unknown=False)
    t3 = Trajectory("c", [], mean_traversability=0.6, min_traversability=0.2, passes_through_unknown=False)
    best = gen.select_best_trajectory([t1, t2, t3])
    assert best is t2


# ── nav_interface: pixel_to_robot_frame ──────────────────────────────────────

def test_pixel_at_center_gives_zero_dy():
    intr = CameraIntrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0)
    dx, dy = pixel_to_robot_frame(240, 320, depth_m=2.0, intrinsics=intr)
    assert dx == pytest.approx(2.0)    # depth goes forward
    assert dy == pytest.approx(0.0)    # centre pixel → no lateral offset


def test_pixel_right_of_center_gives_negative_dy():
    intr = CameraIntrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0)
    _, dy = pixel_to_robot_frame(240, 400, depth_m=2.0, intrinsics=intr)
    assert dy < 0.0, "pixel right of centre should give negative dy (rightward)"


def test_pixel_left_of_center_gives_positive_dy():
    intr = CameraIntrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0)
    _, dy = pixel_to_robot_frame(240, 200, depth_m=2.0, intrinsics=intr)
    assert dy > 0.0, "pixel left of centre should give positive dy (leftward)"


def test_dx_equals_depth():
    intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    depth = 3.5
    dx, _ = pixel_to_robot_frame(240, 320, depth_m=depth, intrinsics=intr)
    assert dx == pytest.approx(depth)


def test_dy_magnitude_scales_with_depth():
    intr = CameraIntrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0)
    _, dy1 = pixel_to_robot_frame(240, 400, depth_m=1.0, intrinsics=intr)
    _, dy2 = pixel_to_robot_frame(240, 400, depth_m=2.0, intrinsics=intr)
    assert abs(dy2) == pytest.approx(abs(dy1) * 2.0, rel=1e-5)
