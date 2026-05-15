"""
Unit tests for baselines/whentoask/prompt.py

Tests:
  format_intent_prompt:
    - Contains instruction, terrain, and n_intents in output
  parse_intent_response:
    - Extracts descriptions and probabilities
    - Probabilities normalize to 1.0
    - Falls back gracefully on empty/malformed response
  format_scoring_prompt:
    - Contains instruction, terrain, intent description, all option labels A-E
  parse_option_scores:
    - Returns all 5 keys A/B/C/D/E with floats in [0, 1]
    - Clamps out-of-range values
    - Missing keys default: A-D → 0.25, E → 0.0
  marginalize_scores:
    - Weighted sum is correct for simple cases
    - Single-intent case returns unweighted scores unchanged
"""

import pytest

from baselines.whentoask.prompt import (
    format_direct_scoring_prompt,
    format_intent_prompt,
    format_scoring_prompt,
    marginalize_scores,
    parse_intent_response,
    parse_option_scores,
)


BASIC_OPTIONS = {
    "A": "Continue on the current path",
    "B": "Ask the user a clarifying question",
    "C": "Reroute automatically",
    "D": "Slow down and proceed cautiously",
}


# ── format_intent_prompt ──────────────────────────────────────────────────────

def test_intent_prompt_contains_instruction():
    prompt = format_intent_prompt("Move through the park", "Wet grass")
    assert "Move through the park" in prompt


def test_intent_prompt_contains_terrain():
    prompt = format_intent_prompt("Keep going", "Bumpy gravel path")
    assert "Bumpy gravel path" in prompt


def test_intent_prompt_contains_n_intents():
    prompt = format_intent_prompt("Go ahead", "Flat concrete", n_intents=4)
    assert "4" in prompt


# ── parse_intent_response ─────────────────────────────────────────────────────

def test_parse_intent_response_extracts_intents():
    response = {
        "intents": [
            {"description": "Move toward destination quickly", "probability": 0.6},
            {"description": "Explore the area safely", "probability": 0.4},
        ]
    }
    intents = parse_intent_response(response)
    assert len(intents) == 2
    assert intents[0][0] == "Move toward destination quickly"


def test_parse_intent_response_probs_sum_to_one():
    response = {
        "intents": [
            {"description": "Intent A", "probability": 0.5},
            {"description": "Intent B", "probability": 0.3},
            {"description": "Intent C", "probability": 0.2},
        ]
    }
    intents = parse_intent_response(response)
    total = sum(p for _, p in intents)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_parse_intent_response_normalizes_uneven_probs():
    # Probabilities don't sum to 1 — should normalize
    response = {
        "intents": [
            {"description": "Intent A", "probability": 0.8},
            {"description": "Intent B", "probability": 0.8},
        ]
    }
    intents = parse_intent_response(response)
    total = sum(p for _, p in intents)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_parse_intent_response_fallback_on_empty():
    intents = parse_intent_response({})
    assert len(intents) == 1
    assert intents[0][1] == pytest.approx(1.0)


# ── format_scoring_prompt ─────────────────────────────────────────────────────

def test_scoring_prompt_contains_instruction():
    prompt = format_scoring_prompt("Turn left", "Pebble path", "Go toward library", BASIC_OPTIONS)
    assert "Turn left" in prompt


def test_scoring_prompt_contains_intent_description():
    prompt = format_scoring_prompt("Go forward", "Wet grass", "Reach the gazebo quickly", BASIC_OPTIONS)
    assert "Reach the gazebo quickly" in prompt


def test_scoring_prompt_contains_option_E_label():
    prompt = format_scoring_prompt("Move on", "Rocky trail", "Proceed to end", BASIC_OPTIONS)
    assert "E" in prompt


def test_scoring_prompt_contains_all_nav_option_labels():
    prompt = format_scoring_prompt("Go right", "Gravel", "Turn right at corner", BASIC_OPTIONS)
    for label in ["A", "B", "C", "D"]:
        assert label in prompt


# ── parse_option_scores ───────────────────────────────────────────────────────

def test_parse_option_scores_all_five_keys():
    response = {"scores": {"A": 0.8, "B": 0.2, "C": 0.1, "D": 0.05, "E": 0.0}}
    scores = parse_option_scores(response)
    assert set(scores.keys()) == {"A", "B", "C", "D", "E"}


def test_parse_option_scores_values_in_range():
    response = {"scores": {"A": 0.9, "B": 0.3, "C": 0.1, "D": 0.2, "E": 0.05}}
    scores = parse_option_scores(response)
    for label, val in scores.items():
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0, f"{label}={val} out of range"


def test_parse_option_scores_missing_abcd_defaults_to_025():
    response = {"scores": {"A": 0.9}}
    scores = parse_option_scores(response)
    for label in ["B", "C", "D"]:
        assert scores[label] == pytest.approx(0.25)


def test_parse_option_scores_missing_e_defaults_to_zero():
    # E defaults to 0.0 — incapability should be explicitly scored
    response = {"scores": {"A": 0.8, "B": 0.2, "C": 0.1, "D": 0.05}}
    scores = parse_option_scores(response)
    assert scores["E"] == pytest.approx(0.0)


def test_parse_option_scores_clamps_above_one():
    response = {"scores": {"A": 1.5, "B": 0.2, "C": 0.1, "D": 0.3, "E": 2.0}}
    scores = parse_option_scores(response)
    assert scores["A"] == pytest.approx(1.0)
    assert scores["E"] == pytest.approx(1.0)


def test_parse_option_scores_clamps_below_zero():
    response = {"scores": {"A": -0.5, "B": 0.8, "C": 0.1, "D": 0.2, "E": -0.1}}
    scores = parse_option_scores(response)
    assert scores["A"] == pytest.approx(0.0)
    assert scores["E"] == pytest.approx(0.0)


# ── marginalize_scores ────────────────────────────────────────────────────────

def test_marginalize_single_intent_returns_original_scores():
    scores = {"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.03, "E": 0.02}
    result = marginalize_scores([(scores, 1.0)])
    for label in scores:
        assert result[label] == pytest.approx(scores[label])


def test_marginalize_two_intents_weighted_sum():
    scores_1 = {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}
    scores_2 = {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0, "E": 0.0}
    result = marginalize_scores([(scores_1, 0.6), (scores_2, 0.4)])
    assert result["A"] == pytest.approx(0.6)
    assert result["B"] == pytest.approx(0.4)
    assert result["C"] == pytest.approx(0.0)


def test_marginalize_returns_all_five_keys():
    scores = {"A": 0.5, "B": 0.3, "C": 0.1, "D": 0.1, "E": 0.0}
    result = marginalize_scores([(scores, 1.0)])
    assert set(result.keys()) == {"A", "B", "C", "D", "E"}


# ── format_direct_scoring_prompt ──────────────────────────────────────────────

def test_direct_scoring_prompt_contains_option_e():
    prompt = format_direct_scoring_prompt("Go forward", "Wet trail", BASIC_OPTIONS)
    assert "E" in prompt
    assert "none" in prompt.lower()
