"""
Integration tests for multi-pose world terrain accumulation.

Covers run_scene_with_pose(), WorldSceneGraph tile semantics, and
WorldGPTraversabilityMap accumulation across frames:

  Class 1 — MockForwardOdometry: deterministic pose sequence
  Class 2 — WorldSceneGraph:     tile arithmetic, upsert, blending
  Class 3 — run_scene_with_pose() basic:  valid decision, PROCEED path
  Class 4 — World accumulation:   world_gp + world_scene_graph grow across frames
  Class 5 — Different poses:      far-apart poses → different tiles
  Class 6 — reset_world_knowledge():  clears both stores, leaves terrain_knowledge
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.runner import EnvUncertaintyDecision, EnvironmentalUncertaintyRunner
from system.env_uncertainty.scene_graph import WorldSceneGraph
from system.env_uncertainty.traversability import TraversabilityMap
from system.env_uncertainty.world_coords import CameraMount, MockForwardOdometry, RobotPose

CONFIG_PATH = str(Path(__file__).parents[1] / "config.yaml")
H, W = 100, 100
IMAGE = np.zeros((H, W, 3), dtype=np.uint8)


# ── Shared helpers ──────────────────────────────────────────────────────────────

def _region(label: str, mask: np.ndarray, traversability: float) -> RegionInfo:
    return RegionInfo(
        label=label,
        mask=mask,
        confidence=0.85,
        pixel_fraction=float(mask.sum()) / (H * W),
        source="sam3",
        traversability=traversability,
    )


def _full_grass_region() -> RegionInfo:
    """Grass covering the entire image — max GP coverage at trajectory start."""
    return _region("grass", np.ones((H, W), dtype=bool), traversability=0.90)


def _make_known_detector(known_regions) -> MagicMock:
    """Mock detector with only known regions — no unknowns → clean PROCEED path."""
    mock = MagicMock()
    tmap = TraversabilityMap.create(H, W)
    for r in known_regions:
        tmap = tmap.update_region(r.mask, r.label)
    mock.detect.return_value = DetectionResult(
        known_regions=known_regions,
        unknown_regions=[],
        image_shape=(H, W),
        sam3_coverage=sum(r.pixel_fraction for r in known_regions),
        unknown_coverage=0.0,
        has_unknown=False,
        traversability_map=tmap,
    )
    return mock


def _grass_runner() -> EnvironmentalUncertaintyRunner:
    """Runner backed by a full-image grass detector."""
    return EnvironmentalUncertaintyRunner(
        CONFIG_PATH, detector=_make_known_detector([_full_grass_region()])
    )


def _pose(x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> RobotPose:
    return RobotPose(x=x, y=y, theta=theta)


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 1 — MockForwardOdometry
# ══════════════════════════════════════════════════════════════════════════════

class TestMockForwardOdometry:
    """MockForwardOdometry produces a correct deterministic straight-line sequence."""

    def test_initial_pose_is_origin(self):
        odo = MockForwardOdometry(speed_mps=0.5, fps=5.0)
        assert odo.current_pose.x == pytest.approx(0.0)
        assert odo.current_pose.y == pytest.approx(0.0)
        assert odo.current_pose.theta == pytest.approx(0.0)

    def test_first_next_pose_advances_x(self):
        # dt = 1/5 = 0.2 s, speed = 0.5 m/s → dx = 0.1 m
        odo = MockForwardOdometry(speed_mps=0.5, fps=5.0)
        pose = odo.next_pose()
        assert pose.x == pytest.approx(0.1, abs=1e-6)

    def test_y_stays_zero_for_straight_line(self):
        odo = MockForwardOdometry(speed_mps=0.5, fps=5.0)
        for _ in range(5):
            pose = odo.next_pose()
        assert pose.y == pytest.approx(0.0, abs=1e-6)

    def test_five_poses_accumulate_to_half_metre(self):
        odo = MockForwardOdometry(speed_mps=0.5, fps=5.0)
        for _ in range(5):
            pose = odo.next_pose()
        assert pose.x == pytest.approx(0.5, abs=1e-6)

    def test_poses_are_monotonically_increasing(self):
        odo = MockForwardOdometry(speed_mps=0.5, fps=5.0)
        xs = [odo.next_pose().x for _ in range(5)]
        assert all(xs[i] < xs[i + 1] for i in range(len(xs) - 1))

    def test_reset_returns_to_origin(self):
        odo = MockForwardOdometry(speed_mps=0.5, fps=5.0)
        odo.next_pose()
        odo.next_pose()
        odo.reset()
        assert odo.current_pose.x == pytest.approx(0.0)

    def test_source_is_mock(self):
        odo = MockForwardOdometry()
        pose = odo.next_pose()
        assert pose.source == "mock"


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 2 — WorldSceneGraph tile arithmetic and node management
# ══════════════════════════════════════════════════════════════════════════════

class TestWorldSceneGraphTiles:
    """WorldSceneGraph: tile index computation, upsert, recall, and blending."""

    def test_world_to_tile_origin(self):
        sg = WorldSceneGraph(tile_size_m=0.5)
        assert sg.world_to_tile(0.0, 0.0) == (0, 0)

    def test_world_to_tile_positive_coords(self):
        sg = WorldSceneGraph(tile_size_m=0.5)
        assert sg.world_to_tile(0.6, 0.0) == (1, 0)
        assert sg.world_to_tile(1.1, 0.75) == (2, 1)

    def test_world_to_tile_negative_coords(self):
        sg = WorldSceneGraph(tile_size_m=0.5)
        # -0.1 / 0.5 = -0.2 → floor(-0.2) = -1
        assert sg.world_to_tile(-0.1, 0.0) == (-1, 0)

    def test_empty_graph_node_count_is_zero(self):
        sg = WorldSceneGraph()
        assert sg.node_count == 0

    def test_upsert_creates_node_with_correct_label(self):
        sg = WorldSceneGraph()
        node = sg.upsert_world_region("grass", x_w=1.0, y_w=0.0, gp_mean=0.8)
        assert node is not None
        assert node.label == "grass"

    def test_upsert_stores_gp_mean(self):
        sg = WorldSceneGraph()
        node = sg.upsert_world_region("grass", x_w=1.0, y_w=0.0, gp_mean=0.8)
        assert node.gp_mean == pytest.approx(0.8)

    def test_node_count_grows_with_each_new_tile(self):
        sg = WorldSceneGraph()
        assert sg.node_count == 0
        sg.upsert_world_region("grass", x_w=1.0, y_w=0.0, gp_mean=0.8)
        assert sg.node_count == 1
        sg.upsert_world_region("road", x_w=5.0, y_w=0.0, gp_mean=0.9)
        assert sg.node_count == 2

    def test_recall_world_finds_inserted_node(self):
        sg = WorldSceneGraph()
        sg.upsert_world_region("grass", x_w=1.0, y_w=0.0, gp_mean=0.8)
        node = sg.recall_world("grass", x_w=1.0, y_w=0.0)
        assert node is not None
        assert node.label == "grass"

    def test_recall_world_returns_none_for_unknown_position(self):
        sg = WorldSceneGraph()
        sg.upsert_world_region("grass", x_w=1.0, y_w=0.0)
        assert sg.recall_world("grass", x_w=100.0, y_w=0.0) is None

    def test_same_tile_twice_blends_gp_mean(self):
        """Second observation: gp_mean = 0.9 * old + 0.1 * new."""
        sg = WorldSceneGraph(confidence_decay=0.9)
        sg.upsert_world_region("grass", x_w=1.0, y_w=0.0, gp_mean=0.8)
        sg.upsert_world_region("grass", x_w=1.0, y_w=0.0, gp_mean=0.2)
        node = sg.recall_world("grass", x_w=1.0, y_w=0.0)
        expected = 0.9 * 0.8 + 0.1 * 0.2  # = 0.74
        assert node.gp_mean == pytest.approx(expected, abs=1e-6)

    def test_same_tile_does_not_create_duplicate_node(self):
        """Re-observing the same (label, tile) updates in place — node_count stays 1."""
        sg = WorldSceneGraph()
        sg.upsert_world_region("grass", x_w=1.0, y_w=0.0, gp_mean=0.8)
        sg.upsert_world_region("grass", x_w=1.0, y_w=0.0, gp_mean=0.5)
        assert sg.node_count == 1

    def test_different_labels_same_tile_are_separate_nodes(self):
        """Same world tile, different label → two independent nodes."""
        sg = WorldSceneGraph()
        sg.upsert_world_region("grass", x_w=1.0, y_w=0.0, gp_mean=0.8)
        sg.upsert_world_region("dirt", x_w=1.0, y_w=0.0, gp_mean=0.4)
        assert sg.node_count == 2

    def test_nodes_near_world_returns_node_within_radius(self):
        sg = WorldSceneGraph(tile_size_m=0.5)
        sg.upsert_world_region("grass", x_w=1.0, y_w=0.0, gp_mean=0.8)
        nodes = sg.nodes_near_world(1.1, 0.0, radius_m=0.6)
        assert any(n.label == "grass" for n in nodes)

    def test_nodes_near_world_excludes_far_nodes(self):
        sg = WorldSceneGraph(tile_size_m=0.5)
        sg.upsert_world_region("grass", x_w=1.0, y_w=0.0)
        nodes = sg.nodes_near_world(20.0, 0.0, radius_m=0.5)
        assert len(nodes) == 0


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 3 — run_scene_with_pose() basic contract
# ══════════════════════════════════════════════════════════════════════════════

class TestRunSceneWithPoseBasic:
    """run_scene_with_pose() returns a valid EnvUncertaintyDecision."""

    def test_returns_env_uncertainty_decision(self):
        runner = _grass_runner()
        d = runner.run_scene_with_pose(IMAGE, _pose())
        assert isinstance(d, EnvUncertaintyDecision)

    def test_robot_action_is_valid_string(self):
        runner = _grass_runner()
        d = runner.run_scene_with_pose(IMAGE, _pose())
        assert d.robot_action in ("PROCEED", "ASK", "STOP")

    def test_no_unknown_regions_reported(self):
        runner = _grass_runner()
        d = runner.run_scene_with_pose(IMAGE, _pose())
        assert d.n_unknown_regions == 0
        assert not d.has_unknown

    def test_scene_id_forwarded(self):
        runner = _grass_runner()
        d = runner.run_scene_with_pose(IMAGE, _pose(), scene_id="frame_42")
        assert d.scene_id == "frame_42"

    def test_first_frame_asks_about_newly_seen_terrain(self):
        """First observation of any terrain type triggers ASK.

        Fresh WorldSceneGraph nodes have Dirichlet alpha=[1,...,2,...,1] over 21
        classes. Shannon entropy ≈ 3.0 >> entropy_ask_threshold (1.5). The robot
        correctly asks for confirmation before committing to a path through terrain
        it has never visited before. PROCEED becomes possible only after the user
        confirms the terrain (mark_confirmed_world) or many observations lower entropy.
        """
        runner = _grass_runner()
        d = runner.run_scene_with_pose(IMAGE, _pose())
        assert d.robot_action == "ASK"

    def test_detector_called_once_per_frame(self):
        detector = _make_known_detector([_full_grass_region()])
        runner = EnvironmentalUncertaintyRunner(CONFIG_PATH, detector=detector)
        runner.run_scene_with_pose(IMAGE, _pose())
        assert detector.detect.call_count == 1


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 4 — World GP and scene graph accumulate across frames
# ══════════════════════════════════════════════════════════════════════════════

class TestWorldAccumulationAcrossFrames:
    """world_gp.n_observations and world_scene_graph.node_count grow persistently."""

    def test_world_gp_nonempty_after_first_frame(self):
        runner = _grass_runner()
        runner.run_scene_with_pose(IMAGE, _pose(x=0.0))
        assert runner.world_gp.n_observations > 0

    def test_world_gp_grows_after_second_far_frame(self):
        runner = _grass_runner()
        runner.run_scene_with_pose(IMAGE, _pose(x=0.0))
        n1 = runner.world_gp.n_observations
        runner.run_scene_with_pose(IMAGE, _pose(x=20.0))
        assert runner.world_gp.n_observations > n1

    def test_world_scene_graph_nonempty_after_first_frame(self):
        runner = _grass_runner()
        runner.run_scene_with_pose(IMAGE, _pose(x=0.0))
        assert runner.world_scene_graph.node_count > 0

    def test_world_scene_graph_grows_with_far_second_frame(self):
        runner = _grass_runner()
        runner.run_scene_with_pose(IMAGE, _pose(x=0.0))
        n1 = runner.world_scene_graph.node_count
        runner.run_scene_with_pose(IMAGE, _pose(x=20.0))
        assert runner.world_scene_graph.node_count > n1

    def test_world_scene_graph_stable_for_same_pose(self):
        """Exact same pose → same tiles → blending, not new nodes."""
        runner = _grass_runner()
        runner.run_scene_with_pose(IMAGE, _pose(x=0.0))
        n1 = runner.world_scene_graph.node_count
        runner.run_scene_with_pose(IMAGE, _pose(x=0.0))
        assert runner.world_scene_graph.node_count == n1

    def test_five_frame_odometry_accumulates_observations(self):
        runner = _grass_runner()
        odo = MockForwardOdometry(speed_mps=0.5, fps=5.0)
        for _ in range(5):
            runner.run_scene_with_pose(IMAGE, odo.next_pose())
        # 5 frames each contributing samples — well above zero
        assert runner.world_gp.n_observations >= 5
        assert runner.world_scene_graph.node_count >= 1

    def test_five_frame_sequence_returns_valid_decisions(self):
        runner = _grass_runner()
        odo = MockForwardOdometry(speed_mps=0.5, fps=5.0)
        decisions = [
            runner.run_scene_with_pose(IMAGE, odo.next_pose()) for _ in range(5)
        ]
        assert all(d.robot_action in ("PROCEED", "ASK", "STOP") for d in decisions)


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 5 — Different poses produce different world tiles
# ══════════════════════════════════════════════════════════════════════════════

class TestDifferentPosesDifferentTiles:
    """Robot at two far-apart positions projects terrain to non-overlapping tiles."""

    def test_far_apart_poses_expand_node_count(self):
        """20 m displacement → new tiles, not blending into existing ones."""
        runner = _grass_runner()
        runner.run_scene_with_pose(IMAGE, _pose(x=0.0))
        n1 = runner.world_scene_graph.node_count
        runner.run_scene_with_pose(IMAGE, _pose(x=20.0))
        assert runner.world_scene_graph.node_count > n1

    def test_tile_indices_differ_for_far_poses(self):
        """Directly verify that two very different poses produce different tile indices."""
        sg = WorldSceneGraph(tile_size_m=0.5)
        tile_near = sg.world_to_tile(1.0, 0.0)   # x=1.0 → tile_ix=2
        tile_far  = sg.world_to_tile(20.0, 0.0)  # x=20.0 → tile_ix=40
        assert tile_near != tile_far

    def test_close_poses_reuse_tiles(self):
        """Poses 0.05 m apart project to the same tile (tile_size=0.5 m)."""
        sg = WorldSceneGraph(tile_size_m=0.5)
        # Two poses separated by 0.05 m, both projecting a point ~1 m ahead
        tile_a = sg.world_to_tile(0.0 + 1.0, 0.0)
        tile_b = sg.world_to_tile(0.05 + 1.0, 0.0)
        # 1.0 and 1.05 both fall in tile_ix = floor(x/0.5) = 2
        assert tile_a == tile_b


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 6 — reset_world_knowledge()
# ══════════════════════════════════════════════════════════════════════════════

class TestResetWorldKnowledge:
    """reset_world_knowledge() clears world_gp and world_scene_graph only."""

    def test_reset_empties_world_gp(self):
        runner = _grass_runner()
        runner.run_scene_with_pose(IMAGE, _pose(x=0.0))
        assert runner.world_gp.n_observations > 0
        runner.reset_world_knowledge()
        assert runner.world_gp.n_observations == 0

    def test_reset_empties_world_scene_graph(self):
        runner = _grass_runner()
        runner.run_scene_with_pose(IMAGE, _pose(x=0.0))
        assert runner.world_scene_graph.node_count > 0
        runner.reset_world_knowledge()
        assert runner.world_scene_graph.node_count == 0

    def test_observations_rebuild_after_reset(self):
        runner = _grass_runner()
        runner.run_scene_with_pose(IMAGE, _pose(x=0.0))
        runner.reset_world_knowledge()
        runner.run_scene_with_pose(IMAGE, _pose(x=0.0))
        assert runner.world_gp.n_observations > 0
        assert runner.world_scene_graph.node_count > 0

    def test_reset_does_not_clear_terrain_knowledge(self):
        """PersistentTerrainKnowledge (label beliefs) must survive reset_world_knowledge."""
        runner = _grass_runner()
        runner.terrain_knowledge.update_from_feedback(
            "grass", is_traversable=False, confidence=0.95
        )
        prior_score = runner.terrain_knowledge.adjusted_traversability("grass")
        runner.reset_world_knowledge()
        # Score must be unchanged — terrain_knowledge is unaffected
        assert runner.terrain_knowledge.adjusted_traversability("grass") == pytest.approx(
            prior_score, abs=1e-6
        )

    def test_decision_valid_after_reset(self):
        runner = _grass_runner()
        runner.run_scene_with_pose(IMAGE, _pose(x=0.0))
        runner.reset_world_knowledge()
        d = runner.run_scene_with_pose(IMAGE, _pose(x=0.0))
        assert d.robot_action in ("PROCEED", "ASK", "STOP")
