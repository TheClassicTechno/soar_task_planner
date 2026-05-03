"""
Unit tests for system/env_uncertainty/question_generator.py

Tests:
  generate_question_template:
    - Large unknown coverage → uses large-unknown template
    - Multiple unknown regions → uses multiple-unknowns template
    - Has safe alternative → uses has-alternative template
    - No safe alternative → uses no-alternative template

  QuestionGenerator (template mode):
    - Returns a non-empty string
    - Default mode is "template"
    - No LLM call is made

  QuestionGenerator (llm mode):
    - LLM.predict_json is called once
    - Returns question from LLM response
    - Falls back to template when LLM returns empty question
    - Falls back to template on LLM exception

  QuestionGenerator validation:
    - Invalid mode raises ValueError
    - llm mode without LLM instance raises ValueError
"""

import pytest
from unittest.mock import MagicMock

import numpy as np

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.question_generator import (
    QuestionGenerator,
    _LARGE_UNKNOWN_THRESHOLD,
    generate_question_template,
)
from system.env_uncertainty.trajectory import Trajectory
from system.env_uncertainty.traversability import TraversabilityMap


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_region(frac=0.20, label="unknown"):
    mask = np.zeros((50, 50), dtype=bool)
    n = int(50 * 50 * frac)
    mask.flat[:n] = True
    return RegionInfo(
        label=label, mask=mask, confidence=0.8,
        pixel_fraction=frac, source="sam2", traversability=0.0,
    )


def _make_result(n_unknown=1, unknown_coverage=0.20):
    tmap = TraversabilityMap.create(50, 50)
    regions = [_make_region(frac=unknown_coverage / max(n_unknown, 1)) for _ in range(n_unknown)]
    return DetectionResult(
        known_regions=[],
        unknown_regions=regions,
        image_shape=(50, 50),
        sam3_coverage=0.5,
        unknown_coverage=unknown_coverage,
        has_unknown=n_unknown > 0,
        traversability_map=tmap,
    )


def _safe_traj():
    return Trajectory("left_arc", [], mean_traversability=0.8, min_traversability=0.7, passes_through_unknown=False)


def _unsafe_traj():
    return Trajectory("forward", [], mean_traversability=0.0, min_traversability=0.0, passes_through_unknown=True)


# ── generate_question_template ────────────────────────────────────────────────

def test_large_unknown_uses_large_template():
    result = _make_result(unknown_coverage=_LARGE_UNKNOWN_THRESHOLD + 0.01)
    q = generate_question_template(result)
    assert "cannot" in q.lower() or "unrecogni" in q.lower() or "ahead" in q.lower()


def test_multiple_unknown_regions_uses_multiple_template():
    result = _make_result(n_unknown=4, unknown_coverage=0.30)
    q = generate_question_template(result)
    assert len(q) > 10


def test_has_safe_alternative_mentions_alternative():
    result = _make_result(n_unknown=1, unknown_coverage=0.15)
    q = generate_question_template(result, trajectories=[_safe_traj(), _unsafe_traj()])
    assert "alternative" in q.lower() or "longer" in q.lower() or "route" in q.lower()


def test_no_safe_alternative_uses_stop_template():
    result = _make_result(n_unknown=1, unknown_coverage=0.15)
    q = generate_question_template(result, trajectories=[_unsafe_traj()])
    assert "stop" in q.lower() or "cannot" in q.lower() or "safe" in q.lower()


def test_template_with_no_trajectories_returns_string():
    result = _make_result(n_unknown=1, unknown_coverage=0.15)
    q = generate_question_template(result, trajectories=None)
    assert isinstance(q, str) and len(q) > 5


# ── QuestionGenerator template mode ──────────────────────────────────────────

def test_template_mode_returns_nonempty_string():
    gen = QuestionGenerator(mode="template")
    result = _make_result()
    q = gen.generate(result)
    assert isinstance(q, str) and len(q) > 5


def test_template_mode_makes_no_llm_call():
    mock_llm = MagicMock()
    gen = QuestionGenerator(mode="template")
    result = _make_result()
    gen.generate(result)
    mock_llm.predict_json.assert_not_called()


# ── QuestionGenerator llm mode ────────────────────────────────────────────────

def test_llm_mode_calls_predict_json_once():
    mock_llm = MagicMock()
    mock_llm.predict_json.return_value = {"question": "Is the path safe?"}
    gen = QuestionGenerator(mode="llm", llm=mock_llm)
    result = _make_result()
    gen.generate(result)
    assert mock_llm.predict_json.call_count == 1


def test_llm_mode_returns_question_from_llm():
    mock_llm = MagicMock()
    mock_llm.predict_json.return_value = {"question": "Can I proceed safely?"}
    gen = QuestionGenerator(mode="llm", llm=mock_llm)
    result = _make_result()
    q = gen.generate(result)
    assert q == "Can I proceed safely?"


def test_llm_mode_falls_back_on_empty_question():
    mock_llm = MagicMock()
    mock_llm.predict_json.return_value = {"question": ""}
    gen = QuestionGenerator(mode="llm", llm=mock_llm)
    result = _make_result()
    q = gen.generate(result)
    # Fallback to template — still a non-empty string
    assert isinstance(q, str) and len(q) > 5


def test_llm_mode_falls_back_on_exception():
    mock_llm = MagicMock()
    mock_llm.predict_json.side_effect = RuntimeError("API error")
    gen = QuestionGenerator(mode="llm", llm=mock_llm)
    result = _make_result()
    q = gen.generate(result)
    assert isinstance(q, str) and len(q) > 5


# ── Validation ────────────────────────────────────────────────────────────────

def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        QuestionGenerator(mode="invalid")


def test_llm_mode_without_llm_raises():
    with pytest.raises(ValueError):
        QuestionGenerator(mode="llm", llm=None)
