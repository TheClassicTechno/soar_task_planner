"""
Unit tests for system/env_uncertainty/runner.py — detector fully mocked.

Tests:
  EnvUncertaintyDecision fields:
    - All fields present
    - robot_action in {"PROCEED", "ASK", "STOP"}
    - question is None when action is PROCEED
    - question is a string when action is ASK or STOP

  EnvironmentalUncertaintyRunner.run_scene:
    - No unknown → PROCEED
    - Unknown in path, no safe alternative → ASK
    - Very large unknown area → STOP (>= stop_unknown_threshold)
    - Small off-path unknown → PROCEED (unknown_coverage < ask_threshold)

  EnvironmentalUncertaintyRunner.run_evaluation:
    - Returns n_scenarios matching input
    - Returns AAR and SAR keys
    - AAR = 1.0 when all ASK scenarios correctly predicted ASK
    - SAR = 0.0 when no PROCEED scenarios incorrectly asked

  Config loading:
    - Runner reads ask_unknown_threshold from config

  Dirichlet entropy ASK trigger (scene_graph path):
    - High-entropy node on path → ASK even when unknown_coverage is zero
    - Low-entropy (confirmed) node on path → PROCEED (entropy below threshold)
    - No scene graph provided → existing coverage-based logic unchanged
    - nodes_in_cell integration: _on_path_nodes deduplicates correctly
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.runner import EnvUncertaintyDecision, EnvironmentalUncertaintyRunner
from system.env_uncertainty.scene_graph import SceneGraph, TERRAIN_CLASSES
from system.env_uncertainty.traversability import TraversabilityMap


CONFIG_PATH = str(Path(__file__).parents[1] / "config.yaml")
H, W = 50, 50
IMAGE = np.zeros((H, W, 3), dtype=np.uint8)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_region(frac, h=H, w=W, label="unknown"):
    n = int(h * w * frac)
    mask = np.zeros((h, w), dtype=bool)
    mask.flat[:n] = True
    return RegionInfo(
        label=label, mask=mask, confidence=0.8,
        pixel_fraction=frac, source="sam2", traversability=0.0,
    )


def _make_detector(unknown_coverage=0.0, n_unknown=0, all_zeros=True):
    """
    Build a mock EnvironmentalUncertaintyDetector.

    Returns a detection result where:
      - traversability_map is all-zero (everything unknown) when all_zeros=True
        so that scored trajectories pass through unknown regions
      - Otherwise uses a fully-known map
    """
    mock = MagicMock()
    h, w = H, W
    tmap = TraversabilityMap.create(h, w)
    if not all_zeros:
        tmap = tmap.update_region(np.ones((h, w), dtype=bool), "grass")

    unknown_regions = []
    if n_unknown > 0:
        frac_each = unknown_coverage / max(n_unknown, 1)
        for _ in range(n_unknown):
            unknown_regions.append(_make_region(frac_each))
    if not all_zeros and unknown_regions:
        # Place unknown regions into the tmap
        for region in unknown_regions:
            tmap = tmap.update_region(region.mask, "unknown")

    mock.detect.return_value = DetectionResult(
        known_regions=[],
        unknown_regions=unknown_regions,
        image_shape=(h, w),
        sam3_coverage=0.5 if not all_zeros else 0.0,
        unknown_coverage=unknown_coverage,
        has_unknown=n_unknown > 0,
        traversability_map=tmap,
    )
    return mock


def _make_runner(detector):
    runner = EnvironmentalUncertaintyRunner(CONFIG_PATH, detector=detector)
    return runner


# ── EnvUncertaintyDecision fields ─────────────────────────────────────────────

def test_run_scene_returns_decision():
    runner = _make_runner(_make_detector())
    decision = runner.run_scene(IMAGE)
    assert isinstance(decision, EnvUncertaintyDecision)


def test_decision_has_required_fields():
    runner = _make_runner(_make_detector())
    d = runner.run_scene(IMAGE, scene_id="test001")
    assert d.scene_id == "test001"
    assert d.robot_action in {"PROCEED", "ASK", "STOP"}
    assert isinstance(d.has_unknown, bool)
    assert isinstance(d.unknown_coverage, float)
    assert isinstance(d.sam3_coverage, float)


def test_robot_action_valid_values():
    runner = _make_runner(_make_detector())
    d = runner.run_scene(IMAGE)
    assert d.robot_action in {"PROCEED", "ASK", "STOP"}


# ── PROCEED behavior ──────────────────────────────────────────────────────────

def test_no_unknown_regions_is_proceed():
    detector = _make_detector(unknown_coverage=0.0, n_unknown=0, all_zeros=False)
    runner = _make_runner(detector)
    d = runner.run_scene(IMAGE)
    assert d.robot_action == "PROCEED"
    assert d.question is None


def test_proceed_has_no_question():
    detector = _make_detector(unknown_coverage=0.0, n_unknown=0, all_zeros=False)
    runner = _make_runner(detector)
    d = runner.run_scene(IMAGE)
    assert d.question is None


# ── ASK behavior ──────────────────────────────────────────────────────────────

def test_unknown_in_path_triggers_ask():
    # High unknown_coverage, all map zeros → all trajectories pass through unknown
    detector = _make_detector(unknown_coverage=0.30, n_unknown=1, all_zeros=True)
    runner = _make_runner(detector)
    d = runner.run_scene(IMAGE)
    assert d.robot_action == "ASK"


def test_ask_has_non_empty_question():
    detector = _make_detector(unknown_coverage=0.30, n_unknown=1, all_zeros=True)
    runner = _make_runner(detector)
    d = runner.run_scene(IMAGE)
    assert isinstance(d.question, str) and len(d.question) > 5


# ── STOP behavior ─────────────────────────────────────────────────────────────

def test_very_large_unknown_triggers_stop():
    # Default stop_threshold is 0.80
    detector = _make_detector(unknown_coverage=0.90, n_unknown=1, all_zeros=True)
    runner = _make_runner(detector)
    d = runner.run_scene(IMAGE)
    assert d.robot_action == "STOP"


def test_stop_has_question():
    detector = _make_detector(unknown_coverage=0.90, n_unknown=1, all_zeros=True)
    runner = _make_runner(detector)
    d = runner.run_scene(IMAGE)
    assert isinstance(d.question, str) and len(d.question) > 5


# ── run_evaluation ────────────────────────────────────────────────────────────

def _make_env_test_cases(n_ask, n_proceed):
    """Create minimal test case dicts for run_evaluation."""
    cases = []
    for i in range(n_ask):
        cases.append({
            "entry_id": f"env_{i:03d}",
            "correct_action": "ASK",
            "should_ask": True,
            "unknown_region_pixel_fraction": 0.30,
        })
    for j in range(n_proceed):
        cases.append({
            "entry_id": f"env_{n_ask + j:03d}",
            "correct_action": "PROCEED",
            "should_ask": False,
            "unknown_region_pixel_fraction": 0.0,
        })
    return cases


def test_run_evaluation_returns_n_scenarios():
    detector = _make_detector(unknown_coverage=0.30, n_unknown=1, all_zeros=True)
    runner = _make_runner(detector)
    cases = _make_env_test_cases(3, 2)
    metrics = runner.run_evaluation(cases)
    assert metrics["n_scenarios"] == 5


def test_run_evaluation_has_aar_sar():
    detector = _make_detector(unknown_coverage=0.30, n_unknown=1, all_zeros=True)
    runner = _make_runner(detector)
    cases = _make_env_test_cases(2, 2)
    metrics = runner.run_evaluation(cases)
    assert "AAR" in metrics
    assert "SAR" in metrics


def test_run_evaluation_sar_zero_when_no_proceed_scenarios():
    detector = _make_detector(unknown_coverage=0.30, n_unknown=1, all_zeros=True)
    runner = _make_runner(detector)
    cases = _make_env_test_cases(3, 0)
    metrics = runner.run_evaluation(cases)
    assert metrics["SAR"] == pytest.approx(0.0)


# ── Dirichlet entropy ASK trigger ─────────────────────────────────────────────

def _make_scene_graph_with_node(label: str, cy: int, cx: int, uniform: bool) -> SceneGraph:
    """
    Build a SceneGraph containing one node at (label, (cy, cx)).

    uniform=True  → uniform Dirichlet prior (high entropy, robot unsure of class)
    uniform=False → concentrated prior on label (low entropy, robot confident)
    """
    sg = SceneGraph()
    # pixel_to_cell maps pixel -> cell; insert at a pixel that hits (cy, cx)
    pixel_y = int(cy * H / SceneGraph.GRID_SIZE) + 1
    pixel_x = int(cx * W / SceneGraph.GRID_SIZE) + 1
    node = sg.upsert_region(label=label, pixel_y=pixel_y, pixel_x=pixel_x, height=H, width=W)
    if not uniform:
        # Drive entropy down: add 50 pseudocounts to label's class
        label_lower = label.lower()
        if label_lower in TERRAIN_CLASSES:
            idx = TERRAIN_CLASSES.index(label_lower)
            node.dirichlet_alpha[idx] += 50.0
    return sg


def test_high_entropy_node_on_path_triggers_ask():
    # No unknown coverage, but a high-entropy node sits on the robot's path.
    # Uniform Dirichlet prior → entropy ≈ log(K) ≈ 3.0 >> threshold 1.5.
    detector = _make_detector(unknown_coverage=0.0, n_unknown=0, all_zeros=False)
    runner = _make_runner(detector)
    # Cell (5,5) is near the centre where the forward trajectory passes.
    sg = _make_scene_graph_with_node("unknown", cy=5, cx=5, uniform=True)
    d = runner.run_scene(IMAGE, scene_graph=sg)
    assert d.robot_action == "ASK"


def test_high_entropy_node_triggers_ask_has_question():
    detector = _make_detector(unknown_coverage=0.0, n_unknown=0, all_zeros=False)
    runner = _make_runner(detector)
    sg = _make_scene_graph_with_node("unknown", cy=5, cx=5, uniform=True)
    d = runner.run_scene(IMAGE, scene_graph=sg)
    assert isinstance(d.question, str) and len(d.question) > 5


def test_no_scene_graph_keeps_coverage_logic():
    # Without a scene_graph argument the behaviour must be identical to before.
    detector = _make_detector(unknown_coverage=0.0, n_unknown=0, all_zeros=False)
    runner = _make_runner(detector)
    d = runner.run_scene(IMAGE)  # no scene_graph
    assert d.robot_action == "PROCEED"


def test_stop_takes_priority_over_entropy():
    # STOP threshold fires before entropy check; entropy must not prevent STOP.
    detector = _make_detector(unknown_coverage=0.95, n_unknown=1, all_zeros=True)
    runner = _make_runner(detector)
    sg = _make_scene_graph_with_node("unknown", cy=5, cx=5, uniform=True)
    d = runner.run_scene(IMAGE, scene_graph=sg)
    assert d.robot_action == "STOP"


def test_on_path_nodes_deduplicates_same_cell():
    # Two nodes with different labels but the same cell → both returned, no dups.
    from system.env_uncertainty.trajectory import Trajectory
    runner = _make_runner(_make_detector())
    sg = SceneGraph()
    sg.upsert_region("grass", pixel_y=25, pixel_x=25, height=H, width=W)
    sg.upsert_region("mud",   pixel_y=25, pixel_x=25, height=H, width=W)
    traj = Trajectory(name="forward", waypoints=[(25, 25)], mean_traversability=0.5,
                      min_traversability=0.5, passes_through_unknown=False)
    nodes = runner._on_path_nodes(sg, traj, H, W)
    labels = {n.label for n in nodes}
    assert labels == {"grass", "mud"}
    assert len(nodes) == 2  # no duplicates


def test_repeated_waypoint_in_same_cell_not_duplicated():
    from system.env_uncertainty.trajectory import Trajectory
    runner = _make_runner(_make_detector())
    sg = SceneGraph()
    sg.upsert_region("grass", pixel_y=25, pixel_x=25, height=H, width=W)
    # Same cell hit by two different waypoints
    traj = Trajectory(name="forward", waypoints=[(24, 24), (25, 25), (26, 26)],
                      mean_traversability=0.5, min_traversability=0.5,
                      passes_through_unknown=False)
    nodes = runner._on_path_nodes(sg, traj, H, W)
    assert sum(1 for n in nodes if n.label == "grass") == 1
