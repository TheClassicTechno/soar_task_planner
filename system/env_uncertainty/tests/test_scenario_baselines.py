"""
Unit tests for eval_scenario_baselines.py.

Tests:
  - _make_synthetic_detection(): correct unknown_coverage, correct n_known_regions
  - always_ask baseline gets 100% on ASK scenarios (sanity check)
  - always_proceed baseline fails on all STOP/ASK scenarios
  - OurSystemEquivalent matches expected_action for every scenario
"""

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from system.env_uncertainty.scenarios import SCENARIOS
from scripts.eval_scenario_baselines import (
    _make_synthetic_detection,
    OurSystemEquivalent,
)


# ── Tests: _make_synthetic_detection ─────────────────────────────────────────

class TestMakeSyntheticDetection:

    def test_unknown_coverage_matches_scenario(self):
        for scenario in SCENARIOS:
            det = _make_synthetic_detection(scenario)
            assert abs(det.unknown_coverage - scenario.unknown_coverage) < 0.01, (
                f"{scenario.name}: expected unknown_coverage={scenario.unknown_coverage}, "
                f"got {det.unknown_coverage:.3f}"
            )

    def test_known_regions_present_when_coverage_less_than_one(self):
        for scenario in SCENARIOS:
            if scenario.unknown_coverage < 1.0:
                det = _make_synthetic_detection(scenario)
                assert len(det.known_regions) > 0, (
                    f"{scenario.name}: expected known regions but got none"
                )

    def test_unknown_region_present_when_coverage_positive(self):
        for scenario in SCENARIOS:
            if scenario.unknown_coverage > 0.0:
                det = _make_synthetic_detection(scenario)
                assert det.has_unknown, (
                    f"{scenario.name}: expected has_unknown=True for coverage={scenario.unknown_coverage}"
                )
                assert len(det.unknown_regions) > 0

    def test_no_unknown_region_when_coverage_zero(self):
        # All real-image scenarios now have small but nonzero GT coverage.
        # Verify the builder correctly omits unknown regions when coverage=0.
        from system.env_uncertainty.scenarios import Scenario
        synthetic = Scenario(
            name="test_zero_coverage",
            description="synthetic zero-coverage fixture",
            uncertainty_trigger="none",
            expected_action="PROCEED",
            unknown_coverage=0.0,
            terrain_on_path=["concrete"],
            user_response_example=None,
            expected_post_action=None,
            goal_description="test",
        )
        det = _make_synthetic_detection(synthetic)
        assert not det.has_unknown, "Coverage=0.0 should produce no unknown regions"

    def test_terrain_labels_in_known_regions(self):
        for scenario in SCENARIOS:
            det = _make_synthetic_detection(scenario)
            known_labels = {r.label for r in det.known_regions}
            for lbl in scenario.terrain_on_path:
                if lbl != "unknown":
                    assert lbl in known_labels, (
                        f"{scenario.name}: expected label '{lbl}' in known regions, "
                        f"got {known_labels}"
                    )

    def test_image_shape_is_set(self):
        for scenario in SCENARIOS:
            det = _make_synthetic_detection(scenario)
            assert len(det.image_shape) == 2
            assert det.image_shape[0] > 0
            assert det.image_shape[1] > 0


# ── Tests: always_ask baseline ───────────────────────────────────────────────

class TestAlwaysAskBaseline:

    def _get_always_ask(self):
        from scripts.eval_env_baselines import AlwaysAsk
        return AlwaysAsk()

    def test_always_ask_correct_on_all_ask_scenarios(self):
        ask_scenarios = [s for s in SCENARIOS if s.expected_action == "ASK"]
        assert len(ask_scenarios) >= 2, "Expected at least 2 ASK scenarios"
        baseline = self._get_always_ask()
        H, W = 50, 50
        goal = (int(H * 0.20), W // 2)
        for scenario in ask_scenarios:
            det = _make_synthetic_detection(scenario)
            result = baseline.decide(det, H, W, goal)
            action = result[0] if isinstance(result, tuple) else result
            assert action == "ASK", (
                f"AlwaysAsk should return ASK on '{scenario.name}', got '{action}'"
            )

    def test_always_ask_never_returns_proceed(self):
        baseline = self._get_always_ask()
        H, W = 50, 50
        goal = (int(H * 0.20), W // 2)
        for scenario in SCENARIOS:
            det = _make_synthetic_detection(scenario)
            result = baseline.decide(det, H, W, goal)
            action = result[0] if isinstance(result, tuple) else result
            assert action != "PROCEED", (
                f"AlwaysAsk returned PROCEED on '{scenario.name}' — impossible"
            )


# ── Tests: always_proceed baseline ───────────────────────────────────────────

class TestAlwaysProceedBaseline:

    def _get_always_proceed(self):
        from scripts.eval_env_baselines import AlwaysProceed
        return AlwaysProceed()

    def test_always_proceed_wrong_on_stop_scenarios(self):
        stop_scenarios = [s for s in SCENARIOS if s.expected_action == "STOP"]
        assert len(stop_scenarios) >= 1, "Expected at least 1 STOP scenario"
        baseline = self._get_always_proceed()
        H, W = 50, 50
        goal = (int(H * 0.20), W // 2)
        for scenario in stop_scenarios:
            det = _make_synthetic_detection(scenario)
            result = baseline.decide(det, H, W, goal)
            action = result[0] if isinstance(result, tuple) else result
            assert action != "STOP", f"AlwaysProceed should never return STOP (scenario: {scenario.name})"
            assert action == "PROCEED"

    def test_always_proceed_wrong_on_ask_scenarios(self):
        ask_scenarios = [s for s in SCENARIOS if s.expected_action == "ASK"]
        baseline = self._get_always_proceed()
        H, W = 50, 50
        goal = (int(H * 0.20), W // 2)
        for scenario in ask_scenarios:
            det = _make_synthetic_detection(scenario)
            result = baseline.decide(det, H, W, goal)
            action = result[0] if isinstance(result, tuple) else result
            assert action != scenario.expected_action, (
                f"AlwaysProceed should be wrong on ASK scenario '{scenario.name}'"
            )

    def test_always_proceed_correct_on_proceed_scenarios(self):
        proceed_scenarios = [s for s in SCENARIOS if s.expected_action == "PROCEED"]
        assert len(proceed_scenarios) >= 3, "Expected at least 3 PROCEED scenarios"
        baseline = self._get_always_proceed()
        H, W = 50, 50
        goal = (int(H * 0.20), W // 2)
        for scenario in proceed_scenarios:
            det = _make_synthetic_detection(scenario)
            result = baseline.decide(det, H, W, goal)
            action = result[0] if isinstance(result, tuple) else result
            assert action == "PROCEED", (
                f"AlwaysProceed should be correct on PROCEED scenario '{scenario.name}', got '{action}'"
            )


# ── Tests: OurSystemEquivalent ────────────────────────────────────────────────

class TestOurSystemEquivalent:

    def test_our_system_correct_on_all_scenarios(self):
        our_system = OurSystemEquivalent()
        for scenario in SCENARIOS:
            det = _make_synthetic_detection(scenario)
            action, _ = our_system.decide(scenario, det)
            assert action == scenario.expected_action, (
                f"OurSystemEquivalent wrong on '{scenario.name}': "
                f"expected '{scenario.expected_action}', got '{action}'"
            )

    def test_our_system_achieves_100_percent_accuracy(self):
        our_system = OurSystemEquivalent()
        n_correct = 0
        for scenario in SCENARIOS:
            det = _make_synthetic_detection(scenario)
            action, _ = our_system.decide(scenario, det)
            if action == scenario.expected_action:
                n_correct += 1
        accuracy = n_correct / len(SCENARIOS)
        assert accuracy == pytest.approx(1.0, abs=0.001), (
            f"OurSystemEquivalent accuracy should be 100%, got {accuracy*100:.1f}%"
        )

    def test_wet_grass_returns_proceed(self):
        # wet_grass_low_lcb is now PROCEED: GT image has 3.5% unknown (below
        # path_unknown_tolerance), all known terrain is safe (grass ≥ 0.30),
        # so LCB STOP is bypassed and the robot proceeds.
        our_system = OurSystemEquivalent()
        wet_grass = next(s for s in SCENARIOS if s.name == "wet_grass_low_lcb")
        det = _make_synthetic_detection(wet_grass)
        action, _ = our_system.decide(wet_grass, det)
        assert action == "PROCEED"
