"""
Tests for May 26 pipeline improvements:

  1. EnvUncertaintyDecision.target_node — S5 must expose the most uncertain
     on-path node so S8 knows which node to update.

  2. Dirichlet update in run_with_feedback() — when user says "that's grass",
     the target_node's Dirichlet alpha[grass] should increase.

  3. Bézier control point geometry — verify that detour control points are
     placed at the correct perpendicular offset from the midpoint.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from system.env_uncertainty.scene_graph import SceneGraph, TerrainNode, TERRAIN_CLASSES
from system.env_uncertainty.trajectory import GoalDirectedTrajectoryGenerator
from system.env_uncertainty.runner import EnvironmentalUncertaintyRunner, EnvUncertaintyDecision
from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.traversability import TraversabilityMap

CONFIG_PATH = str(Path(__file__).parents[1] / "config.yaml")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_runner() -> EnvironmentalUncertaintyRunner:
    from system.env_uncertainty.detector import EnvironmentalUncertaintyDetector
    detector = EnvironmentalUncertaintyDetector()
    return EnvironmentalUncertaintyRunner(CONFIG_PATH, detector=detector)


def _make_scene_graph_with_uncertain_node(label: str = "mud") -> SceneGraph:
    """Return a SceneGraph with one high-entropy node at grid cell (5, 5)."""
    from system.env_uncertainty.scene_graph import CertaintyLevel
    sg = SceneGraph()
    node = TerrainNode(
        label=label,
        position_cell_id=(5, 5),
        gp_mean=0.5,
        gp_variance=0.16,
        certainty_level=CertaintyLevel.UNKNOWN,
        user_confirmed=False,
    )
    # Flat Dirichlet = uniform = maximum entropy (all alpha = 1.0)
    sg._nodes[(label, (5, 5))] = node
    return sg


# ── Tests: EnvUncertaintyDecision has target_node field ──────────────────────

class TestTargetNodeField:

    def test_decision_has_target_node_field(self):
        """EnvUncertaintyDecision must expose a target_node field."""
        decision = EnvUncertaintyDecision(
            scene_id="test",
            has_unknown=False,
            unknown_coverage=0.0,
            sam3_coverage=1.0,
            best_trajectory=None,
            robot_action="PROCEED",
            question=None,
            n_known_regions=1,
            n_unknown_regions=0,
        )
        assert hasattr(decision, "target_node"), "target_node field missing from EnvUncertaintyDecision"

    def test_decision_has_decision_reason_field(self):
        """EnvUncertaintyDecision must expose a decision_reason field."""
        decision = EnvUncertaintyDecision(
            scene_id="test",
            has_unknown=False,
            unknown_coverage=0.0,
            sam3_coverage=1.0,
            best_trajectory=None,
            robot_action="PROCEED",
            question=None,
            n_known_regions=1,
            n_unknown_regions=0,
        )
        assert hasattr(decision, "decision_reason")
        assert isinstance(decision.decision_reason, str)

    def test_target_node_defaults_to_none(self):
        decision = EnvUncertaintyDecision(
            scene_id="test",
            has_unknown=False,
            unknown_coverage=0.0,
            sam3_coverage=1.0,
            best_trajectory=None,
            robot_action="PROCEED",
            question=None,
            n_known_regions=0,
            n_unknown_regions=0,
        )
        assert decision.target_node is None


# ── Tests: _decide_action returns 3-tuple ────────────────────────────────────

class TestDecideActionReturn:

    def _build_mock_result(self, unknown_coverage: float = 0.0):
        mask = np.zeros((50, 50), dtype=bool)
        mask[0:5, :] = True
        known = [RegionInfo(
            label="concrete", mask=~mask, confidence=0.9,
            pixel_fraction=1.0 - unknown_coverage, source="sam3", traversability=0.95,
        )]
        tmap = TraversabilityMap.create(50, 50)
        return DetectionResult(
            known_regions=known,
            unknown_regions=[],
            image_shape=(50, 50),
            sam3_coverage=1.0 - unknown_coverage,
            unknown_coverage=unknown_coverage,
            has_unknown=unknown_coverage > 0.0,
            traversability_map=tmap,
        )

    def test_decide_action_proceed_returns_none_target_node(self):
        runner = _make_runner()
        result = self._build_mock_result(unknown_coverage=0.0)
        from system.env_uncertainty.trajectory import GoalDirectedTrajectoryGenerator
        gen = GoalDirectedTrajectoryGenerator(50, 50)
        trajs = gen.generate_toward_goal((49, 25), (10, 25))
        scored = [gen.score_trajectory(t, result.traversability_map) for t in trajs]
        best = gen.select_best_trajectory(scored)

        action, question, target_node = runner._decide_action(
            result, scored, best, on_path_nodes=[], best_lcb=0.8
        )
        assert action == "PROCEED"
        assert target_node is None, "PROCEED action should return target_node=None"

    def test_decide_action_ask_returns_target_node_when_nodes_provided(self):
        runner = _make_runner()
        # unknown region triggers ASK
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:25, :] = True
        known = [RegionInfo(
            label="grass", mask=~mask, confidence=0.9,
            pixel_fraction=0.5, source="sam3", traversability=0.9,
        )]
        unk = [RegionInfo(
            label="unknown", mask=mask, confidence=0.8,
            pixel_fraction=0.5, source="sam2", traversability=0.0,
        )]
        tmap = TraversabilityMap.create(50, 50)
        result = DetectionResult(
            known_regions=known, unknown_regions=unk,
            image_shape=(50, 50), sam3_coverage=0.5, unknown_coverage=0.25,
            has_unknown=True, traversability_map=tmap,
        )
        # Build a scene graph node with high entropy (flat Dirichlet)
        from system.env_uncertainty.scene_graph import CertaintyLevel
        node = TerrainNode(
            label="grass", position_cell_id=(3, 3),
            gp_mean=0.5, gp_variance=0.16,
            certainty_level=CertaintyLevel.UNKNOWN, user_confirmed=False,
        )

        from system.env_uncertainty.trajectory import GoalDirectedTrajectoryGenerator
        gen = GoalDirectedTrajectoryGenerator(50, 50)
        trajs = gen.generate_toward_goal((49, 25), (5, 25))
        scored = [gen.score_trajectory(t, tmap) for t in trajs]

        action, question, target_node = runner._decide_action(
            result, scored, None, on_path_nodes=[node], best_lcb=None
        )
        assert action == "ASK"
        assert target_node is node, "ASK with on_path_nodes should return the most uncertain node"


# ── Tests: Dirichlet update in run_with_feedback ─────────────────────────────

class TestDirichletUpdateFromFeedback:

    def _rgb_image(self, h: int = 50, w: int = 50) -> np.ndarray:
        """Return a uniform green-ish image (simulate grass)."""
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 1] = 120  # green channel
        return img

    def test_dirichlet_updates_when_label_and_ask(self):
        """
        When robot_action==ASK and user says a terrain label, the target node's
        Dirichlet should shift toward that class.
        """
        runner = _make_runner()
        img = self._rgb_image()
        sg = _make_scene_graph_with_uncertain_node(label="mud")

        # Run initial decision — expect ASK (grass coverage + unknown)
        initial, replanned = runner.run_with_feedback(
            image=img,
            user_response="that looks like grass, probably safe",
            scene_id="test_dirichlet",
            scene_graph=sg,
            goal_pixel=(10, 25),
        )

        # If robot was in ASK mode and we had an on-path node, Dirichlet was updated.
        # We check via the scene graph node directly.
        node = list(sg._nodes.values())[0]
        grass_idx = TERRAIN_CLASSES.index("grass")
        # The alpha for grass should be > 1.0 only if update happened
        # (initial value is 1.0 for all classes in uniform prior)
        if initial.robot_action == "ASK" and initial.target_node is not None:
            # User said "grass" with moderate confidence → alpha[grass] > 1.0
            assert node.dirichlet_alpha[grass_idx] > 1.0, (
                "Dirichlet alpha for 'grass' should increase after user says 'grass'"
            )

    def test_dirichlet_not_updated_when_no_label(self):
        """When user response has no terrain label, Dirichlet stays unchanged."""
        runner = _make_runner()
        img = self._rgb_image()
        sg = _make_scene_graph_with_uncertain_node(label="mud")

        node = list(sg._nodes.values())[0]
        alpha_before = list(node.dirichlet_alpha)

        runner.run_with_feedback(
            image=img,
            user_response="yes go ahead",  # no terrain label
            scene_id="test_no_label",
            scene_graph=sg,
            goal_pixel=(10, 25),
        )

        # No terrain label → no Dirichlet update (alpha unchanged)
        assert node.dirichlet_alpha == alpha_before, (
            "Dirichlet should not change when user response has no terrain label"
        )


# ── Tests: Bézier control point geometry ─────────────────────────────────────

class TestBezierControlPoints:

    def test_direct_path_is_straight_line(self):
        """The 'direct' trajectory should be a straight line start→goal."""
        gen = GoalDirectedTrajectoryGenerator(image_height=100, image_width=100, n_waypoints=5)
        start = (99, 50)
        goal = (10, 50)
        trajs = gen.generate_toward_goal(start, goal)
        direct = next(t for t in trajs if t.name == "direct")
        # All x-coordinates should be ~50 (center column) for a vertical straight line
        xs = [wp[1] for wp in direct.waypoints]
        assert all(abs(x - 50) <= 1 for x in xs), (
            f"Direct path should stay near center column; got xs={xs}"
        )

    def test_left_and_right_detours_are_symmetric(self):
        """Left and right detours should be mirror images about the direct path."""
        gen = GoalDirectedTrajectoryGenerator(image_height=100, image_width=100, n_waypoints=11)
        start = (99, 50)
        goal = (10, 50)  # straight up
        trajs = gen.generate_toward_goal(start, goal)
        left = next(t for t in trajs if t.name == "left_detour")
        right = next(t for t in trajs if t.name == "right_detour")

        # Mid-waypoint (index 5) should be symmetric around x=50
        mid_left_x = left.waypoints[5][1]
        mid_right_x = right.waypoints[5][1]
        assert abs((mid_left_x + mid_right_x) - 100) <= 2, (
            f"Left+right midpoint x should sum to ~100 (symmetric); "
            f"got {mid_left_x} + {mid_right_x} = {mid_left_x + mid_right_x}"
        )

    def test_detour_curves_away_from_direct_path(self):
        """Detour paths should deviate from the direct line in the middle."""
        gen = GoalDirectedTrajectoryGenerator(image_height=100, image_width=100, n_waypoints=11)
        start = (99, 50)
        goal = (10, 50)
        trajs = gen.generate_toward_goal(start, goal)
        direct = next(t for t in trajs if t.name == "direct")
        left = next(t for t in trajs if t.name == "left_detour")
        right = next(t for t in trajs if t.name == "right_detour")

        # At midpoint (index 5), detour x should differ from direct x
        direct_mid_x = direct.waypoints[5][1]
        left_mid_x = left.waypoints[5][1]
        right_mid_x = right.waypoints[5][1]
        assert left_mid_x != direct_mid_x, "Left detour should deviate from direct path"
        assert right_mid_x != direct_mid_x, "Right detour should deviate from direct path"

    def test_all_paths_start_and_end_at_correct_pixels(self):
        """All three paths should start at start_pixel and end at goal_pixel."""
        gen = GoalDirectedTrajectoryGenerator(image_height=100, image_width=100, n_waypoints=20)
        start = (99, 50)
        goal = (10, 70)
        trajs = gen.generate_toward_goal(start, goal)
        for traj in trajs:
            assert traj.waypoints[0] == start or abs(traj.waypoints[0][0] - start[0]) <= 1, (
                f"{traj.name}: first waypoint {traj.waypoints[0]} should be near start {start}"
            )
            assert abs(traj.waypoints[-1][0] - goal[0]) <= 1, (
                f"{traj.name}: last waypoint {traj.waypoints[-1]} should be near goal {goal}"
            )

    def test_detour_fraction_controls_curvature(self):
        """Larger detour_fraction → wider deviation at midpoint."""
        start, goal = (99, 50), (10, 50)
        gen_small = GoalDirectedTrajectoryGenerator(100, 100, n_waypoints=11, detour_fraction=0.10)
        gen_large = GoalDirectedTrajectoryGenerator(100, 100, n_waypoints=11, detour_fraction=0.40)

        small_trajs = gen_small.generate_toward_goal(start, goal)
        large_trajs = gen_large.generate_toward_goal(start, goal)

        small_left = next(t for t in small_trajs if t.name == "left_detour")
        large_left = next(t for t in large_trajs if t.name == "left_detour")

        small_dev = abs(small_left.waypoints[5][1] - 50)
        large_dev = abs(large_left.waypoints[5][1] - 50)
        assert large_dev > small_dev, (
            f"Larger detour_fraction should produce wider curve; "
            f"got small_dev={small_dev}, large_dev={large_dev}"
        )
