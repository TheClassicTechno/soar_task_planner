"""
Tests for combined env+instruction uncertainty scenarios (june1actionitems item 19+20).

Verifies that JointDecisionMaker produces the correct final_action and routes
the question to the correct branch for all three scenario types:
  Type A — ambiguous instruction + uncertain terrain → instruction dominates
  Type B — clear instruction + uncertain terrain    → environment dominates
  Type C — ambiguous instruction + clear terrain    → instruction dominates
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.runner import EnvUncertaintyDecision, EnvironmentalUncertaintyRunner
from system.env_uncertainty.traversability import TraversabilityMap
from system.instruction_uncertainty.ambiguity_detector import AmbiguityDetector, DetectionMode
from system.joint_decision.joint_decision import JointDecisionMaker
from system.joint_decision.combined_scenarios import COMBINED_SCENARIOS

CONFIG_PATH = str(Path(__file__).parents[2] / "env_uncertainty" / "config.yaml")
H, W = 100, 100


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_env_runner_with_coverage(coverage: float) -> EnvironmentalUncertaintyRunner:
    """
    Build a mock env runner with the given unknown_coverage fraction.

    Uses 8 horizontal concrete strips to seed the GP across the full image
    height so LCB is reliable and doesn't spuriously trigger STOP on clear terrain.
    This matches the pattern used in TestEnvFirstOrdering.
    """
    tmap = TraversabilityMap.create(H, W)
    n_strips = 8
    step = H // n_strips
    known_regions = []
    for i in range(n_strips):
        y_start = i * step
        y_end = min((i + 1) * step, H)
        mask = np.zeros((H, W), dtype=bool)
        mask[y_start:y_end, :] = True
        tmap = tmap.update_region(mask, "concrete")
        known_regions.append(RegionInfo(
            label="concrete", mask=mask, confidence=0.90,
            pixel_fraction=float(mask.sum()) / (H * W),
            source="sam3", traversability=0.95,
        ))

    unknown_mask = np.zeros((H, W), dtype=bool)
    unknown_regions = []
    if coverage > 0.0:
        unknown_rows = max(1, int(H * coverage))
        unknown_mask[:unknown_rows, :] = True
        unknown_regions = [RegionInfo(
            label="unknown", mask=unknown_mask, confidence=0.0,
            pixel_fraction=coverage, source="sam2", traversability=0.0,
        )]

    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(
        known_regions=known_regions,
        unknown_regions=unknown_regions,
        image_shape=(H, W),
        sam3_coverage=1.0 - coverage,
        unknown_coverage=coverage,
        has_unknown=coverage > 0,
        traversability_map=tmap,
    )
    return EnvironmentalUncertaintyRunner(CONFIG_PATH, detector=mock_detector)


def _maker(coverage: float) -> JointDecisionMaker:
    env_runner = _make_env_runner_with_coverage(coverage)
    detector = AmbiguityDetector(mode=DetectionMode.RULE)
    return JointDecisionMaker(env_runner, detector)


IMAGE = np.zeros((H, W, 3), dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# Combined scenario correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestCombinedScenarios:

    def test_three_scenarios_defined(self):
        assert len(COMBINED_SCENARIOS) == 3

    def test_all_scenarios_produce_ask(self):
        """All three combined scenarios should require ASK — neither pure PROCEED nor STOP."""
        for scenario in COMBINED_SCENARIOS:
            maker = _maker(scenario.unknown_coverage)
            decision = maker.decide(scenario.instruction, IMAGE,
                                    scene_context=scenario.terrain_description)
            assert decision.final_action == "ASK", (
                f"{scenario.name}: expected ASK, got {decision.final_action}"
            )

    def test_type_A_instruction_dominates(self):
        """Ambiguous instruction + uncertain terrain → instruction branch wins."""
        s = next(sc for sc in COMBINED_SCENARIOS if sc.name == "ambiguous_target_unknown_terrain")
        maker = _maker(s.unknown_coverage)
        decision = maker.decide(s.instruction, IMAGE, scene_context=s.terrain_description)
        assert decision.dominant_branch == "instruction", (
            f"Expected instruction to dominate, got '{decision.dominant_branch}'. "
            f"κ_I={decision.kappa_I:.3f}, κ_E={decision.kappa_E:.3f}"
        )
        assert decision.kappa_I > decision.kappa_E, (
            f"κ_I ({decision.kappa_I:.3f}) should exceed κ_E ({decision.kappa_E:.3f})"
        )

    def test_type_B_environment_dominates(self):
        """Clear instruction + uncertain terrain → environment branch wins."""
        s = next(sc for sc in COMBINED_SCENARIOS if sc.name == "clear_instruction_uncertain_path")
        maker = _maker(s.unknown_coverage)
        decision = maker.decide(s.instruction, IMAGE, scene_context=s.terrain_description)
        assert decision.dominant_branch == "environment", (
            f"Expected environment to dominate, got '{decision.dominant_branch}'. "
            f"κ_I={decision.kappa_I:.3f}, κ_E={decision.kappa_E:.3f}"
        )
        assert decision.kappa_E >= decision.kappa_I, (
            f"κ_E ({decision.kappa_E:.3f}) should be >= κ_I ({decision.kappa_I:.3f})"
        )

    def test_type_C_instruction_dominates_on_clear_terrain(self):
        """Ambiguous instruction + clear terrain → instruction branch wins."""
        s = next(sc for sc in COMBINED_SCENARIOS if sc.name == "ambiguous_target_clear_path")
        maker = _maker(s.unknown_coverage)
        decision = maker.decide(s.instruction, IMAGE, scene_context=s.terrain_description)
        assert decision.dominant_branch == "instruction", (
            f"Expected instruction to dominate on clear terrain, got '{decision.dominant_branch}'. "
            f"κ_I={decision.kappa_I:.3f}, κ_E={decision.kappa_E:.3f}"
        )

    def test_type_A_question_is_about_instruction(self):
        """Type A: robot should ask about the instruction (ambiguous target), not terrain."""
        s = next(sc for sc in COMBINED_SCENARIOS if sc.name == "ambiguous_target_unknown_terrain")
        maker = _maker(s.unknown_coverage)
        decision = maker.decide(s.instruction, IMAGE, scene_context=s.terrain_description)
        assert decision.question is not None
        question_lower = decision.question.lower()
        # Instruction question should reference direction/location/object, not terrain
        assert any(w in question_lower for w in ["location", "object", "landmark", "direction", "mean", "clarify"]), (
            f"Type A question should be about instruction ambiguity, got: '{decision.question}'"
        )

    def test_type_B_question_is_about_terrain(self):
        """Type B: robot should ask about the terrain, not the instruction."""
        s = next(sc for sc in COMBINED_SCENARIOS if sc.name == "clear_instruction_uncertain_path")
        maker = _maker(s.unknown_coverage)
        decision = maker.decide(s.instruction, IMAGE, scene_context=s.terrain_description)
        assert decision.question is not None
        # Env question should mention terrain/surface/area — not instruction grammar
        question_lower = decision.question.lower()
        assert any(w in question_lower for w in ["terrain", "surface", "area", "safe", "region", "path", "ground"]), (
            f"Type B question should be about terrain, got: '{decision.question}'"
        )

    def test_kappa_joint_is_max_of_both(self):
        """κ_joint = max(κ_I, κ_E) for all combined scenarios."""
        for scenario in COMBINED_SCENARIOS:
            maker = _maker(scenario.unknown_coverage)
            decision = maker.decide(scenario.instruction, IMAGE,
                                    scene_context=scenario.terrain_description)
            expected = max(decision.kappa_I, decision.kappa_E)
            assert abs(decision.kappa_joint - expected) < 1e-9, (
                f"{scenario.name}: κ_joint={decision.kappa_joint:.4f} != "
                f"max(κ_I={decision.kappa_I:.4f}, κ_E={decision.kappa_E:.4f})"
            )

    def test_terrain_context_enriches_instruction_detection(self):
        """
        Items 12/14/18: _build_terrain_scene_context now includes target_node.label.
        Verify that scene_context passed to ambiguity detector contains terrain info.
        """
        from unittest.mock import patch, MagicMock as MM
        s = COMBINED_SCENARIOS[1]  # clear instruction + uncertain terrain
        env_runner = _make_env_runner_with_coverage(s.unknown_coverage)
        real_detector = AmbiguityDetector(mode=DetectionMode.RULE)
        spy = MM(wraps=real_detector)
        maker = JointDecisionMaker(env_runner, spy)
        maker.decide(s.instruction, IMAGE)
        _, received_ctx = spy.detect.call_args[0]
        # Context should contain terrain info (coverage or clear)
        assert len(received_ctx) > 0, "Instruction detector should receive non-empty terrain context"
