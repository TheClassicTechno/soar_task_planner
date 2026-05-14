"""
Tests for ConsistencyChecker.

Covers:
  - Constructor: n_runs < 2 raises ValueError
  - Consistent runs: is_consistent=True, score = _compute_nonconformity(type, avg_p)
  - Inconsistent runs: is_consistent=False, nonconformity_score=1.0
  - all_types length equals n_runs
  - avg_p_ambiguous is the mean of individual p_ambiguous values
  - source field: "consistent" vs "conservative_ask"
  - final_detection.source reflects underlying detector source when consistent
  - n_runs=2 works; n_runs=5 works
  - RULE mode (no LLM needed) with consistent instructions
"""

import pytest
from unittest.mock import MagicMock, patch

from system.instruction_uncertainty.ambiguity_detector import (
    _compute_nonconformity,
    DetectionMode,
)
from system.instruction_uncertainty.consistency_check import (
    ConsistencyChecker,
    ConsistencyResult,
)
from system.instruction_uncertainty.intent_memory import SEVERITY_WEIGHTS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_llm_cycling(*responses) -> MagicMock:
    """Return an LLM mock whose predict_json cycles through the given dicts."""
    llm = MagicMock()
    llm.predict_json.side_effect = list(responses)
    return llm


def _make_llm_constant(
    ambiguity_type: str = "no_uncertainty",
    p_ambiguous: float = 0.0,
    missing_slots=None,
    reasoning: str = "Test.",
) -> MagicMock:
    llm = MagicMock()
    llm.predict_json.return_value = {
        "ambiguity_type": ambiguity_type,
        "p_ambiguous": p_ambiguous,
        "missing_slots": missing_slots or [],
        "reasoning": reasoning,
    }
    return llm


# ── Constructor ───────────────────────────────────────────────────────────────

def test_constructor_n_runs_too_small():
    llm = _make_llm_constant()
    with pytest.raises(ValueError, match="n_runs"):
        ConsistencyChecker(llm=llm, n_runs=1)


def test_constructor_n_runs_zero():
    llm = _make_llm_constant()
    with pytest.raises(ValueError):
        ConsistencyChecker(llm=llm, n_runs=0)


# ── Consistent runs ───────────────────────────────────────────────────────────

def test_consistent_all_agree_ambiguous_target():
    llm = _make_llm_constant("ambiguous_target", 0.80, ["target"])
    checker = ConsistencyChecker(llm=llm, n_runs=3)
    result = checker.check("Go there", "bench visible")

    assert result.is_consistent is True
    assert result.source == "consistent"
    assert len(result.all_types) == 3
    assert all(t == "ambiguous_target" for t in result.all_types)
    assert abs(result.avg_p_ambiguous - 0.80) < 1e-9

    expected_score = _compute_nonconformity("ambiguous_target", 0.80)
    assert abs(result.final_detection.nonconformity_score - expected_score) < 1e-9
    assert result.final_detection.ambiguity_type == "ambiguous_target"


def test_consistent_no_uncertainty():
    llm = _make_llm_constant("no_uncertainty", 0.0)
    checker = ConsistencyChecker(llm=llm, n_runs=3)
    result = checker.check("Walk to the library", "")

    assert result.is_consistent is True
    assert result.final_detection.nonconformity_score == 0.0
    assert result.source == "consistent"


def test_consistent_avg_p_used_in_score():
    """Score must use average p, not the first run's p."""
    responses = [
        {"ambiguity_type": "missing_direction", "p_ambiguous": 0.60, "missing_slots": ["direction"], "reasoning": "x"},
        {"ambiguity_type": "missing_direction", "p_ambiguous": 0.80, "missing_slots": ["direction"], "reasoning": "x"},
        {"ambiguity_type": "missing_direction", "p_ambiguous": 0.70, "missing_slots": ["direction"], "reasoning": "x"},
    ]
    llm = MagicMock()
    llm.predict_json.side_effect = responses
    checker = ConsistencyChecker(llm=llm, n_runs=3)
    result = checker.check("Move", "")

    avg_p = (0.60 + 0.80 + 0.70) / 3
    expected_score = _compute_nonconformity("missing_direction", avg_p)
    assert abs(result.avg_p_ambiguous - avg_p) < 1e-9
    assert abs(result.final_detection.nonconformity_score - expected_score) < 1e-9


# ── Inconsistent runs ─────────────────────────────────────────────────────────

def test_inconsistent_two_types_gives_conservative():
    responses = [
        {"ambiguity_type": "ambiguous_target", "p_ambiguous": 0.80, "missing_slots": ["target"], "reasoning": "a"},
        {"ambiguity_type": "missing_direction", "p_ambiguous": 0.70, "missing_slots": ["direction"], "reasoning": "b"},
        {"ambiguity_type": "ambiguous_target", "p_ambiguous": 0.75, "missing_slots": ["target"], "reasoning": "c"},
    ]
    llm = MagicMock()
    llm.predict_json.side_effect = responses
    checker = ConsistencyChecker(llm=llm, n_runs=3)
    result = checker.check("Go there", "")

    assert result.is_consistent is False
    assert result.source == "conservative_ask"
    assert result.final_detection.nonconformity_score == 1.0
    assert len(result.all_types) == 3
    assert set(result.all_types) == {"ambiguous_target", "missing_direction"}


def test_inconsistent_three_distinct_types():
    responses = [
        {"ambiguity_type": "missing_action", "p_ambiguous": 0.80, "missing_slots": [], "reasoning": "x"},
        {"ambiguity_type": "ambiguous_target", "p_ambiguous": 0.70, "missing_slots": [], "reasoning": "x"},
        {"ambiguity_type": "missing_object", "p_ambiguous": 0.60, "missing_slots": [], "reasoning": "x"},
    ]
    llm = MagicMock()
    llm.predict_json.side_effect = responses
    checker = ConsistencyChecker(llm=llm, n_runs=3)
    result = checker.check("That", "")

    assert result.is_consistent is False
    assert result.final_detection.nonconformity_score == 1.0


def test_inconsistent_source_field_on_final_detection():
    responses = [
        {"ambiguity_type": "ambiguous_target", "p_ambiguous": 0.80, "missing_slots": [], "reasoning": "x"},
        {"ambiguity_type": "missing_direction", "p_ambiguous": 0.70, "missing_slots": [], "reasoning": "x"},
        {"ambiguity_type": "ambiguous_target", "p_ambiguous": 0.75, "missing_slots": [], "reasoning": "x"},
    ]
    llm = MagicMock()
    llm.predict_json.side_effect = responses
    checker = ConsistencyChecker(llm=llm, n_runs=3)
    result = checker.check("Go there", "")

    assert result.final_detection.source == "conservative_ask"


# ── all_types length ──────────────────────────────────────────────────────────

def test_all_types_length_matches_n_runs_2():
    llm = _make_llm_constant("ambiguous_target", 0.80)
    checker = ConsistencyChecker(llm=llm, n_runs=2)
    result = checker.check("Go there", "")
    assert len(result.all_types) == 2


def test_all_types_length_matches_n_runs_5():
    llm = _make_llm_constant("ambiguous_target", 0.80)
    checker = ConsistencyChecker(llm=llm, n_runs=5)
    result = checker.check("Go there", "")
    assert len(result.all_types) == 5


# ── avg_p_ambiguous ───────────────────────────────────────────────────────────

def test_avg_p_ambiguous_computed_correctly_consistent():
    responses = [
        {"ambiguity_type": "missing_distance", "p_ambiguous": 0.50, "missing_slots": [], "reasoning": "x"},
        {"ambiguity_type": "missing_distance", "p_ambiguous": 0.90, "missing_slots": [], "reasoning": "x"},
    ]
    llm = MagicMock()
    llm.predict_json.side_effect = responses
    checker = ConsistencyChecker(llm=llm, n_runs=2)
    result = checker.check("Go a bit further", "")
    assert abs(result.avg_p_ambiguous - 0.70) < 1e-9


# ── RULE mode ─────────────────────────────────────────────────────────────────

def test_rule_mode_consistent_no_uncertainty():
    checker = ConsistencyChecker(
        llm=None,
        mode=DetectionMode.RULE,
        n_runs=3,
    )
    result = checker.check("Walk to the library on Main Street", "")
    assert result.is_consistent is True
    assert result.final_detection.ambiguity_type == "no_uncertainty"
    assert result.final_detection.nonconformity_score == 0.0


def test_rule_mode_consistent_ambiguous_target():
    checker = ConsistencyChecker(llm=None, mode=DetectionMode.RULE, n_runs=3)
    result = checker.check("Go there", "")
    assert result.is_consistent is True
    assert result.final_detection.ambiguity_type == "ambiguous_target"


# ── Score formula cross-check ─────────────────────────────────────────────────

@pytest.mark.parametrize("atype,p", [
    ("missing_action", 0.80),
    ("ambiguous_target", 0.75),
    ("missing_object", 0.60),
    ("ambiguous_action", 0.50),
    ("missing_direction", 0.70),
    ("missing_distance", 0.30),
])
def test_consistent_score_matches_formula(atype, p):
    llm = _make_llm_constant(atype, p)
    checker = ConsistencyChecker(llm=llm, n_runs=3)
    result = checker.check("test", "")
    expected = SEVERITY_WEIGHTS[atype] * p
    assert abs(result.final_detection.nonconformity_score - expected) < 1e-9
