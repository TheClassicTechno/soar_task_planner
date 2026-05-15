"""
Instruction ambiguity scenario tests — end-to-end narrative coverage.

Based on the concrete scenarios defined across mentor meetings (april27, april29,
aprl25) and the pipeline document.  These tests wire AmbiguityDetector all the
way through to κ_I, the way Scenarios 0–5 do for the environmental branch.

All tests use DetectionMode.RULE — deterministic, no LLM mock needed.

Scenario A — "Go there" at a 3-way fork → ambiguous_target
    The robot hears a pronoun destination with no named location.
    κ_I = severity(ambiguous_target) * p_ambiguous = 0.75 * 0.80 = 0.60.
    Expected: ambiguity_type="ambiguous_target", nonconformity_score > 0.

Scenario B — "Move forward a bit" → missing_distance
    The robot hears a vague distance with no quantified value.
    κ_I = 0.25 * 0.70 = 0.175.
    Expected: ambiguity_type="missing_distance", nonconformity_score > 0.

Scenario C — "Handle the area ahead" → ambiguous_action
    The robot hears a vague verb that doesn't specify a concrete robot action.
    κ_I = 0.50 * 0.75 = 0.375.
    Expected: ambiguity_type="ambiguous_action", nonconformity_score > 0.

Scenario D — Clear instruction → no_uncertainty → PROCEED
    "Navigate to the park bench directly ahead" — every slot is filled.
    κ_I = 0.0. Expected: ambiguity_type="no_uncertainty", p_ambiguous=0.0.

Each scenario class has 4 tests:
  - ambiguity_type is correct
  - nonconformity_score sign (> 0 or == 0)
  - missing_slots contains the right slot name (or is empty for no_uncertainty)
  - κ_I magnitude matches severity_weight * p_ambiguous
"""

import pytest

from system.instruction_uncertainty.ambiguity_detector import (
    AmbiguityDetector,
    AmbiguityDetection,
    DetectionMode,
)
from system.instruction_uncertainty.intent_memory import SEVERITY_WEIGHTS


# ── Shared helper ──────────────────────────────────────────────────────────────

def _detector() -> AmbiguityDetector:
    """Rule-mode detector: deterministic, no LLM required."""
    return AmbiguityDetector(mode=DetectionMode.RULE)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO A — "Go there" at a 3-way fork → ambiguous_target
# Mentor ref: "Go there at a 3-way fork" — ambiguous_target → ASK (april29meeting)
# ══════════════════════════════════════════════════════════════════════════════

class TestScenarioAGoThere:
    """
    Instruction: "Go there"
    Scene: path forks in two directions, no named location in instruction.
    Expected: robot classifies as ambiguous_target and must ASK.
    """

    INSTRUCTION = "Go there"
    SCENE = "path forks in two directions"

    def _detect(self) -> AmbiguityDetection:
        return _detector().detect(self.INSTRUCTION, self.SCENE)

    def test_ambiguity_type_is_ambiguous_target(self):
        r = self._detect()
        assert r.ambiguity_type == "ambiguous_target", (
            f"Expected ambiguous_target, got {r.ambiguity_type}"
        )

    def test_nonconformity_score_is_positive(self):
        # κ_I > 0 means the CP predictor will consider asking
        r = self._detect()
        assert r.nonconformity_score > 0.0

    def test_target_in_missing_slots(self):
        # Missing slot must name "target" so the question generator knows what to ask
        r = self._detect()
        assert "target" in r.missing_slots

    def test_kappa_I_equals_severity_times_p_ambiguous(self):
        # κ_I = w(ambiguous_target) * p_ambiguous = 0.75 * 0.80 = 0.60
        r = self._detect()
        expected = SEVERITY_WEIGHTS["ambiguous_target"] * r.p_ambiguous
        assert r.nonconformity_score == pytest.approx(expected, abs=1e-9)

    def test_source_is_rule(self):
        # Must be deterministic rule-based, not LLM
        r = self._detect()
        assert r.source == "rule"


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO B — "Move forward a bit" → missing_distance
# Mentor ref: Vague distance reference → ASK (pipeline doc, Step 6)
# ══════════════════════════════════════════════════════════════════════════════

class TestScenarioBMoveForwardABit:
    """
    Instruction: "Move forward a bit"
    "A bit" is a vague distance with no quantified value.
    Expected: missing_distance → nonconformity_score > 0.
    """

    INSTRUCTION = "Move forward a bit"

    def _detect(self) -> AmbiguityDetection:
        return _detector().detect(self.INSTRUCTION)

    def test_ambiguity_type_is_missing_distance(self):
        r = self._detect()
        assert r.ambiguity_type == "missing_distance", (
            f"Expected missing_distance, got {r.ambiguity_type}"
        )

    def test_nonconformity_score_is_positive(self):
        r = self._detect()
        assert r.nonconformity_score > 0.0

    def test_distance_in_missing_slots(self):
        r = self._detect()
        assert "distance" in r.missing_slots

    def test_kappa_I_uses_missing_distance_weight(self):
        # severity weight for missing_distance = 0.25 (lowest — robot can guess)
        r = self._detect()
        expected = SEVERITY_WEIGHTS["missing_distance"] * r.p_ambiguous
        assert r.nonconformity_score == pytest.approx(expected, abs=1e-9)

    def test_missing_distance_weight_is_lowest(self):
        # missing_distance (0.25) is deliberately the lowest severity weight.
        # A vague distance is less dangerous than a missing action or target.
        assert SEVERITY_WEIGHTS["missing_distance"] == pytest.approx(0.25)

    def test_kappa_I_lower_than_ambiguous_target_scenario(self):
        # missing_distance has a lower κ_I than ambiguous_target for the same p_ambiguous.
        # This reflects the design: a vague distance is less safety-critical than
        # not knowing where to go at all.
        detect_a = _detector().detect("Go there", "path forks")
        detect_b = self._detect()
        assert detect_b.nonconformity_score < detect_a.nonconformity_score


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO C — "Handle the area ahead" → ambiguous_action
# Mentor ref: "Handle the area" — ambiguous_action → ASK (pipeline doc)
# ══════════════════════════════════════════════════════════════════════════════

class TestScenarioCHandleTheArea:
    """
    Instruction: "Handle the area ahead"
    "Handle" is a vague verb — the robot cannot determine what concrete action to take.
    Expected: ambiguous_action → ASK.
    """

    INSTRUCTION = "Handle the area ahead"

    def _detect(self) -> AmbiguityDetection:
        return _detector().detect(self.INSTRUCTION)

    def test_ambiguity_type_is_ambiguous_action(self):
        r = self._detect()
        assert r.ambiguity_type == "ambiguous_action", (
            f"Expected ambiguous_action, got {r.ambiguity_type}"
        )

    def test_nonconformity_score_is_positive(self):
        r = self._detect()
        assert r.nonconformity_score > 0.0

    def test_action_in_missing_slots(self):
        r = self._detect()
        assert "action" in r.missing_slots

    def test_kappa_I_uses_ambiguous_action_weight(self):
        # severity weight for ambiguous_action = 0.50
        r = self._detect()
        expected = SEVERITY_WEIGHTS["ambiguous_action"] * r.p_ambiguous
        assert r.nonconformity_score == pytest.approx(expected, abs=1e-9)

    def test_ambiguous_action_weight_is_midrange(self):
        # 0.50: higher than missing_distance, lower than ambiguous_target
        w = SEVERITY_WEIGHTS["ambiguous_action"]
        assert w > SEVERITY_WEIGHTS["missing_distance"]
        assert w < SEVERITY_WEIGHTS["ambiguous_target"]


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO D — Clear instruction → no_uncertainty → PROCEED (Mode A baseline)
# Mentor ref: "Keep going on clear flat asphalt" — no uncertainty → PROCEED
# ══════════════════════════════════════════════════════════════════════════════

class TestScenarioDClearInstruction:
    """
    Instruction: "Navigate to the park bench directly ahead"
    Every slot is filled: action=navigate, target=park bench, direction=ahead.
    Expected: no_uncertainty → κ_I = 0.0 → PROCEED.

    This is Mode A: the robot has a clear instruction AND (combined with Scenario 2,
    a clear environment) should execute immediately with no question asked.
    """

    INSTRUCTION = "Navigate to the park bench directly ahead"

    def _detect(self) -> AmbiguityDetection:
        return _detector().detect(self.INSTRUCTION)

    def test_ambiguity_type_is_no_uncertainty(self):
        r = self._detect()
        assert r.ambiguity_type == "no_uncertainty", (
            f"Expected no_uncertainty on clear instruction, got {r.ambiguity_type}"
        )

    def test_nonconformity_score_is_zero(self):
        # κ_I = 0.0 → CP predictor will not trigger ASK on this branch alone
        r = self._detect()
        assert r.nonconformity_score == pytest.approx(0.0)

    def test_p_ambiguous_is_zero(self):
        r = self._detect()
        assert r.p_ambiguous == pytest.approx(0.0)

    def test_missing_slots_is_empty(self):
        r = self._detect()
        assert r.missing_slots == []

    def test_reasoning_is_nonempty_string(self):
        r = self._detect()
        assert isinstance(r.reasoning, str) and len(r.reasoning) > 5


# ══════════════════════════════════════════════════════════════════════════════
# Cross-scenario: severity ordering
# ══════════════════════════════════════════════════════════════════════════════

class TestSeverityOrdering:
    """
    Verify that the severity weights match the pipeline document's ordering:
    missing_action (1.00) > ambiguous_target (0.75) = missing_object (0.75)
    > ambiguous_action (0.50) = missing_direction (0.50) > missing_distance (0.25)

    This ordering reflects safety priority: not knowing what to do at all is
    more dangerous than not knowing how far to go.
    """

    def test_missing_action_is_highest(self):
        # No action verb: robot literally does not know what to do
        r = _detector().detect("The park")
        assert r.ambiguity_type == "missing_action"
        assert SEVERITY_WEIGHTS["missing_action"] == pytest.approx(1.00)

    def test_kappa_I_ordering_across_scenarios(self):
        # missing_action > ambiguous_target > missing_distance for same p_ambiguous
        w_ma = SEVERITY_WEIGHTS["missing_action"]
        w_at = SEVERITY_WEIGHTS["ambiguous_target"]
        w_md = SEVERITY_WEIGHTS["missing_distance"]
        assert w_ma > w_at > w_md

    def test_no_uncertainty_has_weight_zero(self):
        # no_uncertainty is not in SEVERITY_WEIGHTS (κ_I hard-coded to 0.0)
        r = _detector().detect("Navigate to the park bench directly ahead")
        assert r.nonconformity_score == pytest.approx(0.0)
