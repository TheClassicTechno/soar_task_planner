"""
Tests for june4 integration changes to JointDecisionMaker:

  _build_terrain_scene_context():
    - clear scene → returns "appears clear" string
    - moderate unknown coverage → includes coverage percentage
    - STOP action → includes safety warning

  Env-first ordering in decide():
    - ambiguity detector receives terrain-enriched scene_context
    - clear terrain → context says "clear and traversable"
    - unknown terrain → context contains coverage fraction
    - caller-supplied scene_context is preserved and combined

  Terrain-enriched context affects kappa_I for continuation instructions:
    - "Keep going" on clear terrain → ambiguity detector gets clear context
    - "Keep going" on 40% unknown terrain → ambiguity detector gets uncertain context

  7-scenario GOOSE validation (real images, mocked SAM3/SAM2):
    - All 7 scenarios produce correct expected_action with synthetic detection
    - GOOSE images exist on disk (smoke test)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.runner import EnvUncertaintyDecision, EnvironmentalUncertaintyRunner
from system.env_uncertainty.traversability import TraversabilityMap
from system.instruction_uncertainty.ambiguity_detector import AmbiguityDetector, DetectionMode
from system.joint_decision.joint_decision import (
    JointDecisionMaker,
    _build_terrain_scene_context,
    compute_kappa_E,
)

CONFIG_PATH = str(Path(__file__).parents[2] / "env_uncertainty" / "config.yaml")
H, W = 100, 100
IMAGE = np.zeros((H, W, 3), dtype=np.uint8)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _env_decision_stub(coverage: float, action: str = None) -> EnvUncertaintyDecision:
    if action is None:
        action = "STOP" if coverage >= 0.80 else ("ASK" if coverage > 0.05 else "PROCEED")
    return EnvUncertaintyDecision(
        scene_id="test",
        has_unknown=coverage > 0,
        unknown_coverage=coverage,
        sam3_coverage=1.0 - coverage,
        best_trajectory=None,
        robot_action=action,
        question="What is this terrain?" if action == "ASK" else None,
        n_known_regions=1,
        n_unknown_regions=int(coverage > 0),
    )


def _region(label: str, frac: float, trav: float) -> RegionInfo:
    mask = np.zeros((H, W), dtype=bool)
    mask[: int(H * frac), :] = True
    return RegionInfo(
        label=label, mask=mask, confidence=0.90,
        pixel_fraction=frac, source="sam3", traversability=trav,
    )


def _make_env_runner(known_regions, unknown_regions, coverage: float):
    tmap = TraversabilityMap.create(H, W)
    for r in known_regions:
        tmap = tmap.update_region(r.mask, r.label)
    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(
        known_regions=known_regions,
        unknown_regions=unknown_regions,
        image_shape=(H, W),
        sam3_coverage=sum(r.pixel_fraction for r in known_regions),
        unknown_coverage=coverage,
        has_unknown=len(unknown_regions) > 0,
        traversability_map=tmap,
    )
    return EnvironmentalUncertaintyRunner(CONFIG_PATH, detector=mock_detector)


def _rule_detector() -> AmbiguityDetector:
    return AmbiguityDetector(mode=DetectionMode.RULE)


# ══════════════════════════════════════════════════════════════════════════════
# _build_terrain_scene_context
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildTerrainSceneContext:

    def test_clear_scene_returns_clear_string(self):
        ctx = _build_terrain_scene_context(_env_decision_stub(0.0))
        assert "clear" in ctx.lower() or "traversable" in ctx.lower()

    def test_moderate_coverage_includes_percentage(self):
        ctx = _build_terrain_scene_context(_env_decision_stub(0.40))
        assert "40%" in ctx or "0.40" in ctx or "40" in ctx

    def test_low_coverage_below_threshold_treated_as_clear(self):
        # coverage=0.03 is below the 5% threshold — no uncertainty mention
        ctx = _build_terrain_scene_context(_env_decision_stub(0.03))
        assert "%" not in ctx or "clear" in ctx.lower() or "traversable" in ctx.lower()

    def test_stop_action_includes_safety_warning(self):
        ctx = _build_terrain_scene_context(_env_decision_stub(0.85, action="STOP"))
        assert "unsafe" in ctx.lower() or "safe" in ctx.lower() or "stop" in ctx.lower() or "environmental" in ctx.lower()

    def test_ask_with_high_coverage_mentions_uncertainty(self):
        ctx = _build_terrain_scene_context(_env_decision_stub(0.40, action="ASK"))
        assert "uncertain" in ctx.lower() or "unknown" in ctx.lower() or "%" in ctx

    def test_never_raises(self):
        # Even with a malformed object it should return a string gracefully
        bad = MagicMock()
        del bad.unknown_coverage
        result = _build_terrain_scene_context(bad)
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════════
# Env-first ordering: ambiguity detector receives enriched scene_context
# ══════════════════════════════════════════════════════════════════════════════

class TestEnvFirstOrdering:

    def _make_maker_with_spy(self, coverage: float):
        """Return (maker, spy_detector) where spy_detector records scene_context calls."""
        unknown_mask = np.zeros((H, W), dtype=bool)
        unknown_mask[:int(H * coverage), :] = True
        unknown = RegionInfo(
            label="unknown", mask=unknown_mask, confidence=0.0,
            pixel_fraction=coverage, source="sam2", traversability=0.0,
        )
        known = _region("grass", max(0.0, 1.0 - coverage), 0.90)
        env_runner = _make_env_runner(
            [known] if coverage < 1.0 else [],
            [unknown] if coverage > 0 else [],
            coverage,
        )
        spy = MagicMock(wraps=_rule_detector())
        maker = JointDecisionMaker(env_runner, spy)
        return maker, spy

    def test_ambiguity_detector_receives_nonempty_scene_context(self):
        maker, spy = self._make_maker_with_spy(coverage=0.40)
        maker.decide("Keep going", IMAGE)
        call_args = spy.detect.call_args
        _, scene_ctx = call_args[0]
        assert len(scene_ctx) > 0, "Ambiguity detector should receive non-empty scene_context"

    def test_clear_terrain_gives_clear_context_to_detector(self):
        # Use multiple grass strips so GP is well-seeded → robust PROCEED (no LCB STOP)
        strips = [_region("grass", 0.12 * i + 0.01, 0.90) for i in range(1, 9)]
        env_runner = _make_env_runner(strips, [], 0.0)
        spy = MagicMock(wraps=_rule_detector())
        maker = JointDecisionMaker(env_runner, spy)
        maker.decide("Keep going", IMAGE)
        _, scene_ctx = spy.detect.call_args[0]
        assert "clear" in scene_ctx.lower() or "traversable" in scene_ctx.lower()

    def test_unknown_terrain_gives_uncertain_context_to_detector(self):
        maker, spy = self._make_maker_with_spy(coverage=0.40)
        maker.decide("Keep going", IMAGE)
        _, scene_ctx = spy.detect.call_args[0]
        assert "unknown" in scene_ctx.lower() or "%" in scene_ctx or "uncertain" in scene_ctx.lower()

    def test_caller_scene_context_preserved_and_combined(self):
        maker, spy = self._make_maker_with_spy(coverage=0.0)
        maker.decide("Go to the bench", IMAGE, scene_context="bench visible at 3m")
        _, scene_ctx = spy.detect.call_args[0]
        assert "bench" in scene_ctx, "Caller-supplied scene_context should be in enriched context"

    def test_env_branch_runs_before_instruction_branch(self):
        """env_runner.run_scene must be called before ambiguity_detector.detect."""
        call_order = []
        env_runner = _make_env_runner([_region("grass", 1.0, 0.90)], [], 0.0)

        real_run_scene = env_runner.run_scene
        real_detector = _rule_detector()

        env_runner.run_scene = lambda *a, **kw: (call_order.append("env"), real_run_scene(*a, **kw))[1]
        spy_detector = MagicMock(wraps=real_detector)
        original_detect = spy_detector.detect

        def detect_spy(*a, **kw):
            call_order.append("instruction")
            return original_detect(*a, **kw)

        spy_detector.detect = detect_spy
        maker = JointDecisionMaker(env_runner, spy_detector)
        maker.decide("Go forward", IMAGE)

        assert call_order[0] == "env", f"env must run first, got order: {call_order}"
        assert call_order[1] == "instruction", f"instruction must run second, got order: {call_order}"


# ══════════════════════════════════════════════════════════════════════════════
# 7-scenario GOOSE validation
# ══════════════════════════════════════════════════════════════════════════════

class TestGooseScenarioValidation:
    """
    Verify all 7 named scenarios produce the correct expected_action.
    Uses synthetic DetectionResult (mocked SAM3/SAM2) — does not require
    real model inference, but does verify GOOSE images exist on disk.
    """

    @pytest.fixture(autouse=True)
    def _import_scenarios(self):
        from system.env_uncertainty.scenarios import SCENARIOS
        from scripts.eval_scenario_baselines import _make_synthetic_detection, OurSystemEquivalent
        self.SCENARIOS = SCENARIOS
        self._make_synthetic_detection = _make_synthetic_detection
        self.OurSystemEquivalent = OurSystemEquivalent

    def test_all_goose_images_exist(self):
        for scenario in self.SCENARIOS:
            if scenario.source_image:
                img_path = PROJECT_ROOT / scenario.source_image
                assert img_path.exists(), (
                    f"GOOSE image missing for scenario '{scenario.name}': {img_path}"
                )

    def test_all_scenarios_produce_correct_action(self):
        system = self.OurSystemEquivalent()
        failures = []
        for scenario in self.SCENARIOS:
            det = self._make_synthetic_detection(scenario)
            decision, _ = system.decide(scenario, det)
            if decision != scenario.expected_action:
                failures.append(
                    f"{scenario.name}: expected={scenario.expected_action}, got={decision}"
                )
        assert not failures, "Scenario failures:\n" + "\n".join(failures)

    def test_scenario_count_is_seven(self):
        assert len(self.SCENARIOS) == 7, f"Expected 7 scenarios, got {len(self.SCENARIOS)}"

    def test_all_scenarios_have_source_image(self):
        for scenario in self.SCENARIOS:
            assert scenario.source_image is not None, (
                f"Scenario '{scenario.name}' has no GOOSE source_image"
            )

    def test_ask_scenarios_have_nonzero_coverage(self):
        for scenario in self.SCENARIOS:
            if scenario.expected_action == "ASK":
                assert scenario.unknown_coverage > 0, (
                    f"ASK scenario '{scenario.name}' should have nonzero unknown_coverage"
                )

    def test_stop_scenario_has_high_coverage(self):
        stop_scenarios = [s for s in self.SCENARIOS if s.expected_action == "STOP"]
        for s in stop_scenarios:
            assert s.unknown_coverage >= 0.35, (
                f"STOP scenario '{s.name}' coverage={s.unknown_coverage} seems too low"
            )

    def test_proceed_scenarios_have_low_coverage(self):
        for scenario in self.SCENARIOS:
            if scenario.expected_action == "PROCEED":
                assert scenario.unknown_coverage < 0.10, (
                    f"PROCEED scenario '{scenario.name}' has high coverage={scenario.unknown_coverage}"
                )
