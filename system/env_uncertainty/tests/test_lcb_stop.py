"""
LCB-threshold STOP tests.

Verifies the new GP LCB-based STOP branch added to EnvironmentalUncertaintyRunner
(runner.py _decide_action).  This branch fires when the best safe trajectory's
GP lower confidence bound (μ − β·σ) falls below lcb_stop_threshold (default 0.20),
signalling that known terrain on the path is dangerously untraversable.

Six tests:

  1. Config loads lcb_stop_threshold as 0.20 by default.
  2. GP seeded with low traversability (mud, 0.05) → STOP.
  3. GP seeded with high traversability (sidewalk, 0.95) → PROCEED.
  4. GP has zero observations → LCB STOP guard suppressed → PROCEED (not STOP).
  5. STOP action always includes a non-None question.
  6. Low unknown_coverage (would normally PROCEED) but low LCB → still STOP.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.runner import EnvironmentalUncertaintyRunner
from system.env_uncertainty.traversability import TraversabilityMap

CONFIG_PATH = str(Path(__file__).parents[1] / "config.yaml")
H, W = 100, 100
IMAGE = np.zeros((H, W, 3), dtype=np.uint8)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _full_mask() -> np.ndarray:
    """Boolean mask covering the entire image."""
    return np.ones((H, W), dtype=bool)


def _top_mask(frac: float = 0.05) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[: int(H * frac), :] = True
    return mask


def _region(label: str, mask: np.ndarray, traversability: float) -> RegionInfo:
    return RegionInfo(
        label=label,
        mask=mask,
        confidence=0.85,
        pixel_fraction=float(mask.sum()) / (H * W),
        source="sam3" if label != "unknown" else "sam2",
        traversability=traversability,
    )


def _make_detector(
    known_regions,
    unknown_regions,
    unknown_coverage: float,
    all_zeros: bool = False,
) -> MagicMock:
    mock = MagicMock()
    tmap = TraversabilityMap.create(H, W)
    if not all_zeros:
        for r in known_regions:
            tmap = tmap.update_region(r.mask, r.label)
    for r in unknown_regions:
        tmap = tmap.update_region(r.mask, "unknown")

    mock.detect.return_value = DetectionResult(
        known_regions=known_regions,
        unknown_regions=unknown_regions,
        image_shape=(H, W),
        sam3_coverage=sum(r.pixel_fraction for r in known_regions),
        unknown_coverage=unknown_coverage,
        has_unknown=len(unknown_regions) > 0,
        traversability_map=tmap,
    )
    return mock


def _runner(detector) -> EnvironmentalUncertaintyRunner:
    return EnvironmentalUncertaintyRunner(CONFIG_PATH, detector=detector)


# ── Test 1: Config threshold ───────────────────────────────────────────────────

def test_lcb_stop_threshold_loaded_from_config():
    # Runner must read lcb_stop_threshold from config and default to 0.20.
    detector = _make_detector([], [], 0.0)
    runner = _runner(detector)
    assert runner._lcb_stop_threshold == pytest.approx(0.20)


# ── Test 2: Low LCB → STOP ─────────────────────────────────────────────────────

def test_lcb_stop_triggers_when_seeded_with_low_traversability():
    # Mud covering the full image.  The GP is seeded with traversability=0.05
    # (dangerously low). All trajectories are through known (non-zero) terrain,
    # so passes_through_unknown=False.  GP LCB will be << 0.20 → STOP.
    mud = _region("mud", _full_mask(), traversability=0.05)
    detector = _make_detector([mud], [], unknown_coverage=0.0)
    runner = _runner(detector)
    decision = runner.run_scene(IMAGE)
    assert decision.robot_action == "STOP", (
        f"Expected STOP on dangerously low LCB, got {decision.robot_action}"
    )


# ── Test 3: High LCB → PROCEED ─────────────────────────────────────────────────

def _strip_regions(label: str, traversability: float, n_strips: int = 8):
    """
    Create n_strips small horizontal-band regions spread from row 0 to H-1.
    Each strip seeds one GP observation at a different y position, giving the
    GP enough coverage to be confident everywhere along the trajectory.
    A single centroid observation at (50,50) leaves the GP uncertain near
    row 99 where trajectories start (σ≈1.0 there), so LCB goes negative
    even for high-traversability terrain.
    """
    regions = []
    step = H // n_strips
    for i in range(n_strips):
        mask = np.zeros((H, W), dtype=bool)
        row_start = i * step
        row_end = min(row_start + step, H)
        mask[row_start:row_end, :] = True
        regions.append(_region(label, mask, traversability))
    return regions


def test_lcb_above_threshold_gives_proceed():
    # Sidewalk strip-seeded across the full image height so GP is confident
    # along the entire trajectory (rows 20-99). LCB >> 0.20 → PROCEED.
    sidewalk_regions = _strip_regions("sidewalk", traversability=0.95)
    detector = _make_detector(sidewalk_regions, [], unknown_coverage=0.0)
    runner = _runner(detector)
    decision = runner.run_scene(IMAGE)
    assert decision.robot_action == "PROCEED", (
        f"Expected PROCEED on high-traversability path, got {decision.robot_action}"
    )


# ── Test 4: Zero GP observations → guard suppresses LCB STOP ──────────────────

def test_lcb_stop_requires_gp_observation():
    # No known regions → GP is never seeded (n_observations == 0).
    # Prior LCB = 0.5 - 1.5*0.4 = -0.1 < 0.20, but the guard must block the STOP.
    # With no unknown regions, the runner should PROCEED.
    detector = _make_detector([], [], unknown_coverage=0.0)
    runner = _runner(detector)
    decision = runner.run_scene(IMAGE)
    assert decision.robot_action == "PROCEED", (
        f"LCB STOP must not fire without GP observations, got {decision.robot_action}"
    )


# ── Test 5: STOP always has a question ─────────────────────────────────────────

def test_lcb_stop_has_question():
    # When LCB STOP fires, _decide_action must generate a clarifying question.
    mud = _region("mud", _full_mask(), traversability=0.05)
    detector = _make_detector([mud], [], unknown_coverage=0.0)
    runner = _runner(detector)
    decision = runner.run_scene(IMAGE)
    assert decision.robot_action == "STOP"
    assert decision.question is not None and len(decision.question) > 5, (
        "STOP decision must include a non-empty clarification question"
    )


# ── Test 6: LCB STOP fires even with low unknown_coverage ──────────────────────

def test_lcb_stop_independent_of_unknown_coverage():
    # Small unknown region (5% → unknown_coverage=0.05, below ask_threshold=0.10).
    # Dangerous mud covers the rest (traversability=0.05).
    # Coverage check alone would give PROCEED, but LCB STOP should fire first.
    top_unk = _top_mask(0.05)
    bot_mud = ~top_unk
    unknown = _region("unknown", top_unk, traversability=0.0)
    mud = _region("mud", bot_mud, traversability=0.05)
    detector = _make_detector([mud], [unknown], unknown_coverage=0.05)
    runner = _runner(detector)
    decision = runner.run_scene(IMAGE)
    # Coverage (0.05) < ask_threshold (0.10), so without LCB STOP it would PROCEED.
    # With LCB STOP, dangerous mud forces STOP.
    assert decision.robot_action == "STOP", (
        f"Expected STOP from low LCB despite low coverage, got {decision.robot_action}"
    )
