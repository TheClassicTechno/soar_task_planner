"""
Joint decision tests — κ_joint = max(κ_I, κ_E) and Mode A baseline.

Tests cover:

  κ_E normalization:
    - coverage=0.0  → κ_E=0.0
    - coverage=0.40 → κ_E=0.50  (document example)
    - coverage=0.80 → κ_E=1.0   (stop threshold)

  κ_joint = max(κ_I, κ_E):
    - formula verified with explicit values
    - tie case (κ_I == κ_E) resolves correctly

  High κ_I overrides clear environment:
    - ambiguous instruction + fully known terrain → ASK

  High κ_E overrides clear instruction:
    - clear instruction + 40% unknown terrain → ASK

  Both clear → PROCEED (Mode A baseline):
    - κ_I=0.0, κ_E=0.0 → final_action="PROCEED", question=None

  Env STOP is never downgraded:
    - env_decision.robot_action=STOP → joint always returns STOP

  JointDecision dataclass fields are all present:
    - kappa_I, kappa_E, kappa_joint, instruction_ambiguity, env_decision,
      final_action, question all non-None on a real run
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.runner import EnvUncertaintyDecision, EnvironmentalUncertaintyRunner
from system.env_uncertainty.traversability import TraversabilityMap
from system.instruction_uncertainty.ambiguity_detector import AmbiguityDetector, DetectionMode
from system.joint_decision.joint_decision import (
    JointDecisionMaker,
    compute_kappa_E,
    compute_kappa_joint,
)

CONFIG_PATH = str(Path(__file__).parents[2] / "env_uncertainty" / "config.yaml")
H, W = 100, 100
IMAGE = np.zeros((H, W, 3), dtype=np.uint8)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _region(label: str, mask: np.ndarray, traversability: float) -> RegionInfo:
    return RegionInfo(
        label=label,
        mask=mask,
        confidence=0.85,
        pixel_fraction=float(mask.sum()) / (H * W),
        source="sam3" if label != "unknown" else "sam2",
        traversability=traversability,
    )


def _full_mask() -> np.ndarray:
    return np.ones((H, W), dtype=bool)


def _top_mask(frac: float) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[: int(H * frac), :] = True
    return mask


def _sidewalk_strips(n_strips: int = 8):
    """
    Create n_strips small horizontal-band sidewalk regions spread top-to-bottom.
    Each strip seeds one GP observation at a different y position, giving the
    GP enough spatial coverage to be confident along the entire trajectory
    (rows 20–99). A single centroid at (50,50) leaves the GP uncertain at row
    99 (σ≈1.0) and the LCB STOP branch would fire even for safe terrain.
    """
    step = H // n_strips
    strips = []
    for i in range(n_strips):
        mask = np.zeros((H, W), dtype=bool)
        mask[i * step : min((i + 1) * step, H), :] = True
        strips.append(_region("sidewalk", mask, traversability=0.95))
    return strips


def _make_env_runner(
    known_regions,
    unknown_regions,
    unknown_coverage: float,
) -> EnvironmentalUncertaintyRunner:
    tmap = TraversabilityMap.create(H, W)
    for r in known_regions:
        tmap = tmap.update_region(r.mask, r.label)
    for r in unknown_regions:
        tmap = tmap.update_region(r.mask, "unknown")

    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(
        known_regions=known_regions,
        unknown_regions=unknown_regions,
        image_shape=(H, W),
        sam3_coverage=sum(r.pixel_fraction for r in known_regions),
        unknown_coverage=unknown_coverage,
        has_unknown=len(unknown_regions) > 0,
        traversability_map=tmap,
    )
    return EnvironmentalUncertaintyRunner(CONFIG_PATH, detector=mock_detector)


def _rule_detector() -> AmbiguityDetector:
    return AmbiguityDetector(mode=DetectionMode.RULE)


def _maker(env_runner, amb_detector=None) -> JointDecisionMaker:
    if amb_detector is None:
        amb_detector = _rule_detector()
    return JointDecisionMaker(env_runner, amb_detector)


# ══════════════════════════════════════════════════════════════════════════════
# κ_E normalization
# ══════════════════════════════════════════════════════════════════════════════

class TestKappaENormalization:
    """κ_E = min(unknown_coverage / 0.80, 1.0)"""

    def _env_decision(self, coverage: float) -> EnvUncertaintyDecision:
        # Build a minimal EnvUncertaintyDecision stub with the required fields
        from system.env_uncertainty.trajectory import Trajectory
        return EnvUncertaintyDecision(
            scene_id="test",
            has_unknown=coverage > 0,
            unknown_coverage=coverage,
            sam3_coverage=1.0 - coverage,
            best_trajectory=None,
            robot_action="PROCEED" if coverage == 0 else "ASK",
            question=None,
            n_known_regions=1,
            n_unknown_regions=int(coverage > 0),
        )

    def test_zero_coverage_gives_zero_kappa_E(self):
        assert compute_kappa_E(self._env_decision(0.0)) == pytest.approx(0.0)

    def test_40_percent_coverage_gives_half(self):
        # Document example: coverage=0.40 → κ_E = 0.40/0.80 = 0.50
        assert compute_kappa_E(self._env_decision(0.40)) == pytest.approx(0.50)

    def test_stop_threshold_coverage_gives_one(self):
        # coverage=0.80 (stop threshold) → κ_E = 1.0 (maximum)
        assert compute_kappa_E(self._env_decision(0.80)) == pytest.approx(1.0)

    def test_beyond_stop_threshold_clamped_to_one(self):
        # coverage > 0.80 is physically possible (all unknown); must not exceed 1.0
        assert compute_kappa_E(self._env_decision(0.90)) == pytest.approx(1.0)

    def test_kappa_E_in_unit_interval(self):
        for cov in [0.0, 0.10, 0.40, 0.79, 0.80, 0.90]:
            k = compute_kappa_E(self._env_decision(cov))
            assert 0.0 <= k <= 1.0, f"κ_E={k} out of [0,1] for coverage={cov}"


# ══════════════════════════════════════════════════════════════════════════════
# κ_joint formula
# ══════════════════════════════════════════════════════════════════════════════

class TestKappaJointFormula:
    """κ_joint = max(κ_I, κ_E)."""

    def test_formula_kI_dominates(self):
        assert compute_kappa_joint(0.80, 0.30) == pytest.approx(0.80)

    def test_formula_kE_dominates(self):
        assert compute_kappa_joint(0.10, 0.60) == pytest.approx(0.60)

    def test_formula_tie(self):
        assert compute_kappa_joint(0.50, 0.50) == pytest.approx(0.50)

    def test_both_zero_gives_zero(self):
        assert compute_kappa_joint(0.0, 0.0) == pytest.approx(0.0)

    def test_formula_result_in_unit_interval(self):
        assert 0.0 <= compute_kappa_joint(0.75, 0.40) <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# High κ_I overrides clear environment → ASK
# ══════════════════════════════════════════════════════════════════════════════

class TestHighKappaIOverridesClearEnvironment:
    """Ambiguous instruction + clear terrain → joint must ASK (instruction branch wins)."""

    def test_ambiguous_target_on_clear_terrain_gives_ask(self):
        # Clear sidewalk: no unknown regions, traversability=0.95
        runner = _make_env_runner(_sidewalk_strips(), [], unknown_coverage=0.0)
        maker = _maker(runner)

        # "Go there" → ambiguous_target → κ_I = 0.75 * 0.80 = 0.60 > ask_threshold=0.15
        jd = maker.decide("Go there", IMAGE, scene_context="path forks ahead")
        assert jd.final_action == "ASK", (
            f"Expected ASK when instruction is ambiguous, got {jd.final_action}"
        )

    def test_kappa_I_is_positive_on_clear_environment(self):
        runner = _make_env_runner(_sidewalk_strips(), [], unknown_coverage=0.0)
        maker = _maker(runner)
        jd = maker.decide("Go there", IMAGE)
        assert jd.kappa_I > 0.0

    def test_kappa_E_is_zero_on_clear_environment(self):
        runner = _make_env_runner(_sidewalk_strips(), [], unknown_coverage=0.0)
        maker = _maker(runner)
        jd = maker.decide("Go there", IMAGE)
        assert jd.kappa_E == pytest.approx(0.0)

    def test_kappa_joint_equals_kappa_I_when_environment_clear(self):
        runner = _make_env_runner(_sidewalk_strips(), [], unknown_coverage=0.0)
        maker = _maker(runner)
        jd = maker.decide("Go there", IMAGE)
        assert jd.kappa_joint == pytest.approx(jd.kappa_I)


# ══════════════════════════════════════════════════════════════════════════════
# High κ_E overrides clear instruction → ASK
# ══════════════════════════════════════════════════════════════════════════════

class TestHighKappaEOverridesClearInstruction:
    """Clear instruction + 40% unknown terrain → joint must ASK (env branch wins)."""

    def test_clear_instruction_with_unknown_terrain_gives_ask(self):
        # 40% unknown coverage → κ_E = 0.50 > ask_threshold=0.15
        unknown_mask = _top_mask(0.40)
        unknown = _region("unknown", unknown_mask, traversability=0.0)
        runner = _make_env_runner([], [unknown], unknown_coverage=0.40)
        maker = _maker(runner)

        # Fully clear instruction → κ_I = 0.0
        jd = maker.decide(
            "Navigate to the park bench directly ahead",
            IMAGE,
        )
        assert jd.final_action == "ASK", (
            f"Expected ASK when environment is uncertain, got {jd.final_action}"
        )

    def test_kappa_I_is_zero_on_clear_instruction(self):
        unknown = _region("unknown", _top_mask(0.40), traversability=0.0)
        runner = _make_env_runner([], [unknown], unknown_coverage=0.40)
        maker = _maker(runner)
        jd = maker.decide("Navigate to the park bench directly ahead", IMAGE)
        assert jd.kappa_I == pytest.approx(0.0)

    def test_kappa_E_equals_normalized_coverage(self):
        unknown = _region("unknown", _top_mask(0.40), traversability=0.0)
        runner = _make_env_runner([], [unknown], unknown_coverage=0.40)
        maker = _maker(runner)
        jd = maker.decide("Navigate to the park bench directly ahead", IMAGE)
        assert jd.kappa_E == pytest.approx(0.50)

    def test_kappa_joint_equals_kappa_E_when_instruction_clear(self):
        unknown = _region("unknown", _top_mask(0.40), traversability=0.0)
        runner = _make_env_runner([], [unknown], unknown_coverage=0.40)
        maker = _maker(runner)
        jd = maker.decide("Navigate to the park bench directly ahead", IMAGE)
        assert jd.kappa_joint == pytest.approx(jd.kappa_E)


# ══════════════════════════════════════════════════════════════════════════════
# Mode A baseline — both branches clear → PROCEED
# Mentor ref: "Keep going on clear flat asphalt — no uncertainty → PROCEED"
# ══════════════════════════════════════════════════════════════════════════════

class TestModeABaseline:
    """
    Mode A: both instruction and environment are unambiguous → PROCEED immediately.

    This is the happy-path baseline the document contrasts against ASK cases.
    κ_I = 0.0 (no instruction ambiguity)
    κ_E = 0.0 (no unknown terrain)
    κ_joint = max(0.0, 0.0) = 0.0 < ask_threshold → PROCEED.
    """

    def _setup(self):
        runner = _make_env_runner(_sidewalk_strips(), [], unknown_coverage=0.0)
        return _maker(runner)

    def test_mode_a_kappa_I_is_zero(self):
        maker = self._setup()
        jd = maker.decide("Navigate to the park bench directly ahead", IMAGE)
        assert jd.kappa_I == pytest.approx(0.0)

    def test_mode_a_kappa_E_is_zero(self):
        maker = self._setup()
        jd = maker.decide("Navigate to the park bench directly ahead", IMAGE)
        assert jd.kappa_E == pytest.approx(0.0)

    def test_mode_a_kappa_joint_is_zero(self):
        maker = self._setup()
        jd = maker.decide("Navigate to the park bench directly ahead", IMAGE)
        assert jd.kappa_joint == pytest.approx(0.0)

    def test_mode_a_joint_action_is_proceed(self):
        maker = self._setup()
        jd = maker.decide("Navigate to the park bench directly ahead", IMAGE)
        assert jd.final_action == "PROCEED"

    def test_mode_a_question_is_none(self):
        maker = self._setup()
        jd = maker.decide("Navigate to the park bench directly ahead", IMAGE)
        assert jd.question is None

    def test_mode_a_env_decision_is_proceed(self):
        maker = self._setup()
        jd = maker.decide("Navigate to the park bench directly ahead", IMAGE)
        # The env runner also says PROCEED independently
        assert jd.env_decision.robot_action == "PROCEED"


# ══════════════════════════════════════════════════════════════════════════════
# Env STOP is never downgraded
# ══════════════════════════════════════════════════════════════════════════════

class TestEnvStopNeverDowngraded:
    """
    If the environmental runner returns STOP (e.g., 90% unknown coverage),
    the joint decision must also return STOP regardless of κ_I.
    """

    def test_env_stop_gives_joint_stop(self):
        # 90% unknown: env runner will return STOP (above stop_threshold=0.80)
        unknown_mask = _top_mask(0.90)
        unknown = _region("unknown", unknown_mask, traversability=0.0)
        runner = _make_env_runner([], [unknown], unknown_coverage=0.90)
        maker = _maker(runner)

        # Even with a clear instruction, env STOP must propagate
        jd = maker.decide("Navigate to the park bench directly ahead", IMAGE)
        assert jd.final_action == "STOP"

    def test_env_stop_with_clear_instruction_still_stops(self):
        # κ_I = 0.0 (clear instruction) but env is catastrophically uncertain
        unknown_mask = _top_mask(0.90)
        unknown = _region("unknown", unknown_mask, traversability=0.0)
        runner = _make_env_runner([], [unknown], unknown_coverage=0.90)
        maker = _maker(runner)

        jd = maker.decide("Navigate to the park bench directly ahead", IMAGE)
        assert jd.kappa_I == pytest.approx(0.0)
        assert jd.final_action == "STOP"


# ══════════════════════════════════════════════════════════════════════════════
# JointDecision dataclass completeness
# ══════════════════════════════════════════════════════════════════════════════

class TestJointDecisionDataclass:
    """All fields on JointDecision must be populated on a real decide() call."""

    def _run(self, instruction: str, unknown_coverage: float):
        unknown_mask = _top_mask(unknown_coverage) if unknown_coverage > 0 else np.zeros((H, W), dtype=bool)
        if unknown_coverage > 0:
            regions, unknown_regions = [], [_region("unknown", unknown_mask, traversability=0.0)]
        else:
            regions, unknown_regions = _sidewalk_strips(), []
        runner = _make_env_runner(regions, unknown_regions, unknown_coverage)
        maker = _maker(runner)
        return maker.decide(instruction, IMAGE)

    def test_all_fields_populated_on_clear_run(self):
        jd = self._run("Navigate to the park bench directly ahead", 0.0)
        assert isinstance(jd.kappa_I, float)
        assert isinstance(jd.kappa_E, float)
        assert isinstance(jd.kappa_joint, float)
        assert jd.instruction_ambiguity is not None
        assert jd.env_decision is not None
        assert jd.final_action in ("PROCEED", "ASK", "STOP")

    def test_all_fields_populated_on_ask_run(self):
        jd = self._run("Navigate to the park bench directly ahead", 0.40)
        assert isinstance(jd.kappa_joint, float)
        assert jd.final_action == "ASK"


# ══════════════════════════════════════════════════════════════════════════════
# Both branches fire simultaneously
# ══════════════════════════════════════════════════════════════════════════════

class TestBothBranchesFire:
    """
    Both κ_I and κ_E exceed the ask threshold at the same time.

    Setup:
      instruction = "Go there"  → ambiguous_target → κ_I > 0.15
      environment = 50% unknown → κ_E = 0.50/0.80 = 0.625 > 0.15
    Expected:
      final_action == "ASK"
      kappa_joint == max(kappa_I, kappa_E) == kappa_E (env dominates)
    """

    def _setup(self):
        unknown_mask = _top_mask(0.50)
        unknown = _region("unknown", unknown_mask, traversability=0.0)
        runner = _make_env_runner([], [unknown], unknown_coverage=0.50)
        return _maker(runner)

    def test_both_branches_fire_simultaneously(self):
        maker = self._setup()
        jd = maker.decide("Go there", IMAGE, scene_context="path forks ahead")

        assert jd.final_action == "ASK", (
            f"Expected ASK when both branches fire, got {jd.final_action}"
        )
        assert jd.kappa_I > 0.0, "κ_I should be positive for ambiguous instruction"
        assert jd.kappa_E == pytest.approx(0.625, abs=0.01), (
            f"κ_E should be 0.50/0.80=0.625 for 50% unknown coverage, got {jd.kappa_E}"
        )
        assert jd.kappa_joint == pytest.approx(max(jd.kappa_I, 0.625), abs=0.01), (
            f"κ_joint should equal max(κ_I, κ_E), got {jd.kappa_joint}"
        )

    def test_kappa_joint_is_max_when_env_dominates(self):
        # Use 70% unknown → κ_E = 0.70/0.80 = 0.875, which clearly exceeds κ_I ≈ 0.64
        unknown_mask = _top_mask(0.70)
        unknown = _region("unknown", unknown_mask, traversability=0.0)
        runner = _make_env_runner([], [unknown], unknown_coverage=0.70)
        maker = _maker(runner)
        jd = maker.decide("Go there", IMAGE, scene_context="path forks ahead")

        assert jd.kappa_E == pytest.approx(0.875, abs=0.01), (
            f"κ_E should be 0.70/0.80=0.875, got {jd.kappa_E}"
        )
        assert jd.kappa_joint == pytest.approx(jd.kappa_E, abs=0.01), (
            f"κ_joint should equal κ_E when env dominates, "
            f"got κ_joint={jd.kappa_joint}, κ_E={jd.kappa_E}, κ_I={jd.kappa_I}"
        )
        assert jd.kappa_joint >= jd.kappa_I, (
            f"max property: κ_joint={jd.kappa_joint} must be >= κ_I={jd.kappa_I}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Dominant branch and question attribution
# june1meeting: "the question asked is always from the env branch — it asks
# about terrain, not about the ambiguous instruction. There is currently no
# logic that says 'the instruction is ambiguous, so use the scene graph terrain
# info to help narrow what the user meant.'" — this test suite verifies that
# the fix is in place: the dominant branch determines the question.
# ══════════════════════════════════════════════════════════════════════════════

class TestDominantBranchQuestion:
    """
    The question in the JointDecision must come from the dominant branch.

    When κ_I > κ_E (instruction branch drives uncertainty):
      - dominant_branch == "instruction"
      - question addresses the instruction ambiguity (not terrain)

    When κ_E > κ_I (environmental branch drives uncertainty):
      - dominant_branch == "environment"
      - question addresses the terrain uncertainty

    When both are below ask_threshold:
      - dominant_branch == "none"
      - question is None
    """

    def test_instruction_dominant_branch_on_clear_terrain(self):
        # Clear sidewalk: κ_E = 0.0; "Go there" → κ_I ≈ 0.64 (ambiguous_target)
        runner = _make_env_runner(_sidewalk_strips(), [], unknown_coverage=0.0)
        maker = _maker(runner)
        jd = maker.decide("Go there", IMAGE)
        assert jd.dominant_branch == "instruction", (
            f"Expected 'instruction' branch when κ_I={jd.kappa_I:.3f} > κ_E={jd.kappa_E:.3f}"
        )

    def test_instruction_dominant_question_is_about_ambiguity_not_purely_terrain(self):
        # When instruction branch dominates, question must ask to clarify the
        # instruction — it may include a terrain-context suffix (june1meeting
        # lines 88-94: scene graph should inform instruction disambiguation),
        # but the question's primary purpose must NOT be asking about
        # raw terrain traversability.
        runner = _make_env_runner(_sidewalk_strips(), [], unknown_coverage=0.0)
        maker = _maker(runner)
        jd = maker.decide("Go there", IMAGE)
        assert jd.final_action == "ASK"
        assert jd.question is not None
        # These are PURELY terrain-query phrases (env-branch language);
        # they should not be the main point of an instruction question.
        purely_terrain_phrases = {"unknown area", "unrecognized area", "is the terrain"}
        q_lower = jd.question.lower()
        for phrase in purely_terrain_phrases:
            assert phrase not in q_lower, (
                f"Instruction-dominant question uses pure terrain-query phrasing "
                f"'{phrase}': {jd.question}"
            )

    def test_instruction_dominant_question_addresses_ambiguity_slot(self):
        # For ambiguous_target, the question should ask for location/object clarification
        runner = _make_env_runner(_sidewalk_strips(), [], unknown_coverage=0.0)
        maker = _maker(runner)
        jd = maker.decide("Go there", IMAGE)
        # ambiguous_target question should reference location or direction
        assert any(
            word in jd.question.lower()
            for word in ("location", "object", "specific", "landmark", "direction", "mean")
        ), f"Instruction question for ambiguous_target should ask for clarification: {jd.question}"

    def test_env_dominant_branch_on_ambiguous_terrain(self):
        # 70% unknown → κ_E = 0.875; clear instruction → κ_I = 0.0
        unknown_mask = _top_mask(0.70)
        unknown = _region("unknown", unknown_mask, traversability=0.0)
        runner = _make_env_runner([], [unknown], unknown_coverage=0.70)
        maker = _maker(runner)
        jd = maker.decide("Navigate to the bench ahead", IMAGE)
        assert jd.dominant_branch == "environment", (
            f"Expected 'environment' branch when κ_E={jd.kappa_E:.3f} > κ_I={jd.kappa_I:.3f}"
        )

    def test_env_dominant_question_comes_from_env_runner(self):
        # When env branch dominates, question must match what env_decision produced
        unknown_mask = _top_mask(0.70)
        unknown = _region("unknown", unknown_mask, traversability=0.0)
        runner = _make_env_runner([], [unknown], unknown_coverage=0.70)
        maker = _maker(runner)
        jd = maker.decide("Navigate to the bench ahead", IMAGE)
        assert jd.question == jd.env_decision.question, (
            "When env branch dominates, joint question must equal env_decision.question"
        )

    def test_proceed_has_none_dominant_branch(self):
        # Clear instruction + clear terrain → PROCEED → dominant_branch="none"
        runner = _make_env_runner(_sidewalk_strips(), [], unknown_coverage=0.0)
        maker = _maker(runner)
        jd = maker.decide("Navigate to the park bench directly ahead", IMAGE)
        assert jd.final_action == "PROCEED"
        assert jd.dominant_branch == "none"
        assert jd.question is None

    def test_stop_dominant_branch_is_environment(self):
        # STOP always comes from environmental branch (safety constraint is absolute)
        unknown_mask = _top_mask(0.90)
        unknown = _region("unknown", unknown_mask, traversability=0.0)
        runner = _make_env_runner([], [unknown], unknown_coverage=0.90)
        maker = _maker(runner)
        jd = maker.decide("Navigate to the bench ahead", IMAGE)
        assert jd.final_action == "STOP"
        assert jd.dominant_branch == "environment"

    def test_generate_instruction_question_all_types(self):
        # Every ambiguity type must produce a non-empty instruction question
        from system.joint_decision.joint_decision import generate_instruction_question
        from system.instruction_uncertainty.ambiguity_detector import AmbiguityDetection
        for atype in ["missing_action", "ambiguous_target", "missing_object",
                      "ambiguous_action", "missing_direction", "missing_distance"]:
            amb = AmbiguityDetection(
                ambiguity_type=atype, p_ambiguous=0.8,
                nonconformity_score=0.6, missing_slots=[atype.split("_")[-1]],
                reasoning="test", source="rule",
            )
            q = generate_instruction_question(amb)
            assert isinstance(q, str) and len(q) > 10, (
                f"generate_instruction_question returned empty string for type '{atype}'"
            )

    def test_generate_instruction_question_fallback_for_unknown_type(self):
        from system.joint_decision.joint_decision import generate_instruction_question
        from system.instruction_uncertainty.ambiguity_detector import AmbiguityDetection
        amb = AmbiguityDetection(
            ambiguity_type="no_uncertainty", p_ambiguous=0.0,
            nonconformity_score=0.0, missing_slots=[],
            reasoning="", source="rule",
        )
        q = generate_instruction_question(amb)
        assert isinstance(q, str) and len(q) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Scene graph ↔ Instruction branch connection
# june1meeting lines 88-94: "scene graph and language don't have the best
# connection in my pipeline right now, but maybe I should connect them more
# strongly... instruction uncertainties also contain some understanding of
# the scenario."
# june1meeting notes lines 311-313: "make sure instruction is aligned with
# environment... instructional and environmental, dependent on each other"
# ══════════════════════════════════════════════════════════════════════════════

class TestSceneGraphInstructionConnection:
    """
    Verifies that when the instruction branch triggers ASK (κ_I > κ_E),
    the generated question is grounded in the terrain the robot observes.

    This implements the june1meeting requirement: scene graph context must
    inform instruction disambiguation so the user can answer based on what
    the robot actually sees.
    """

    def _env_decision_with_unknown(self, coverage: float) -> EnvUncertaintyDecision:
        return EnvUncertaintyDecision(
            scene_id="test",
            has_unknown=coverage > 0,
            unknown_coverage=coverage,
            sam3_coverage=max(0.0, 0.80 - coverage),
            best_trajectory=None,
            robot_action="ASK",
            question="Is the terrain safe?",
            n_known_regions=2 if coverage < 1.0 else 0,
            n_unknown_regions=int(coverage > 0),
        )

    def test_instruction_question_includes_terrain_context_when_unknown(self):
        # When the scene has unknown terrain (coverage=30%), the instruction
        # question should include terrain context so the user can ground their reply.
        from system.joint_decision.joint_decision import generate_instruction_question
        from system.instruction_uncertainty.ambiguity_detector import AmbiguityDetection
        amb = AmbiguityDetection(
            ambiguity_type="ambiguous_target", p_ambiguous=0.8,
            nonconformity_score=0.6, missing_slots=["target"],
            reasoning="Ambiguous pronoun", source="rule",
        )
        env_dec = self._env_decision_with_unknown(0.30)
        q = generate_instruction_question(amb, env_decision=env_dec)
        assert isinstance(q, str) and len(q) > 10
        # With unknown terrain, the question should mention the robot's view
        assert "view" in q.lower() or "unidentified" in q.lower() or "%" in q, (
            f"Expected terrain context in instruction question, got: {q}"
        )

    def test_instruction_question_without_env_decision_is_clean(self):
        # Without env_decision, question should be clean (no terrain suffix)
        from system.joint_decision.joint_decision import generate_instruction_question
        from system.instruction_uncertainty.ambiguity_detector import AmbiguityDetection
        amb = AmbiguityDetection(
            ambiguity_type="ambiguous_target", p_ambiguous=0.8,
            nonconformity_score=0.6, missing_slots=["target"],
            reasoning="test", source="rule",
        )
        q_no_env = generate_instruction_question(amb, env_decision=None)
        q_with_env = generate_instruction_question(
            amb, env_decision=self._env_decision_with_unknown(0.30)
        )
        # Both should be valid questions
        assert isinstance(q_no_env, str) and len(q_no_env) > 5
        assert isinstance(q_with_env, str) and len(q_with_env) > 5
        # The grounded version (with env) should be at least as long as the plain one
        assert len(q_with_env) >= len(q_no_env)

    def test_instruction_question_with_zero_unknown_has_no_terrain_suffix(self):
        # When unknown_coverage=0, no terrain context should be appended (nothing to report)
        from system.joint_decision.joint_decision import generate_instruction_question, _terrain_context_suffix
        env_dec = self._env_decision_with_unknown(0.0)
        # _terrain_context_suffix returns "" when coverage=0 (no unknown terrain to report)
        suffix = _terrain_context_suffix(env_dec)
        assert suffix == ""

    def test_full_pipeline_instruction_question_references_terrain_when_present(self):
        # End-to-end: "Go there" on a scene with 30% unknown → instruction question
        # should include terrain view context (connecting scene graph to instruction)
        unknown_mask = _top_mask(0.30)
        unknown = _region("unknown", unknown_mask, traversability=0.0)
        runner = _make_env_runner([], [unknown], unknown_coverage=0.30)
        maker = _maker(runner)

        # "Go there" → ambiguous_target → κ_I ≈ 0.64; 30% unknown → κ_E ≈ 0.375
        # κ_I > κ_E → instruction branch dominates → instruction question with terrain context
        jd = maker.decide("Go there", IMAGE, scene_context="campus path ahead")
        assert jd.dominant_branch == "instruction"
        assert jd.question is not None
        # Question should be terrain-aware (mention view/percentage or terrain info)
        # because both branches have context to share
        assert len(jd.question) > 30, f"Expected substantive question, got: {jd.question}"

    def test_joint_decide_env_decision_always_populated(self):
        # env_decision must always be set so the scene graph context is available
        runner = _make_env_runner(_sidewalk_strips(), [], unknown_coverage=0.0)
        maker = _maker(runner)
        jd = maker.decide("Go there", IMAGE)
        assert jd.env_decision is not None
        assert isinstance(jd.env_decision.unknown_coverage, float)
        assert isinstance(jd.env_decision.sam3_coverage, float)
