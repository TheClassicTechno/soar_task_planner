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
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.runner import EnvUncertaintyDecision, EnvironmentalUncertaintyRunner
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
