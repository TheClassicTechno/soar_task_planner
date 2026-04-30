"""
Unit tests for baselines/knowno/runner.py — LLM fully mocked. No real API calls.

Tests:
  format_direct_scoring_prompt:
    - Contains instruction and terrain in output
    - Contains all four option labels A/B/C/D
    - Uses provided option text over defaults

  KnowNoDecision fields:
    - scenario_id, options, option_confidences, prediction_set, robot_decision all present
    - option_confidences sum to ~1.0 (normalized)
    - robot_decision ∈ {"A","B","C","D","ASK"}

  LLM call count:
    - run_scenario makes exactly 1 LLM call (no retrieval)

  Calibration:
    - calibrate() sets tau on the predictor
    - Records n_calibration samples equal to number of scenarios

  Post-calibration behavior:
    - High confidence on one option → that option in prediction set → ACT
    - Low confidence (spread evenly) → prediction set > 1 → ASK
    - Robot decision == ASK when prediction set > 1

  run_evaluation:
    - Returns standard metrics keys (SR, HR, FPR, NCR, ESR, n_scenarios)
    - n_scenarios matches input count
    - All metrics in [0, 1]
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from baselines.introplan.runner import NavigationScenario
from baselines.knowno.runner import (
    KnowNoDecision,
    KnowNoRunner,
    format_direct_scoring_prompt,
)

CONFIG_PATH = str(Path(__file__).parents[1] / "config.yaml")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_scenario(sid="k001", correct="B", utype=2):
    return NavigationScenario(
        scenario_id=sid,
        instruction="Keep going",
        terrain_description="Wet leaves covering the path",
        uncertainty_type=utype,
        correct_option=correct,
        options={
            "A": "Continue on the current path",
            "B": "Ask the user about the wet leaves",
            "C": "Reroute automatically",
            "D": "Slow down and proceed cautiously",
        },
    )


def _make_mock_llm(prediction="B", confidence=0.85):
    """Return a mock LLMInterface that always returns the given prediction."""
    mock = MagicMock()
    mock.predict_json.return_value = {
        "prediction": prediction,
        "confidence": confidence,
        "reasoning": "test reasoning",
    }
    return mock


def _runner(llm=None):
    return KnowNoRunner(config_path=CONFIG_PATH, llm=llm or _make_mock_llm())


# ── format_direct_scoring_prompt ──────────────────────────────────────────────

def test_prompt_contains_instruction():
    prompt = format_direct_scoring_prompt(
        "Move toward the park",
        "Wet grass ahead",
        {"A": "Go", "B": "Ask", "C": "Reroute", "D": "Slow"},
    )
    assert "Move toward the park" in prompt


def test_prompt_contains_terrain():
    prompt = format_direct_scoring_prompt(
        "Keep going",
        "Gravel path with loose stones",
        {"A": "Go", "B": "Ask", "C": "Reroute", "D": "Slow"},
    )
    assert "Gravel path with loose stones" in prompt


def test_prompt_contains_all_option_labels():
    prompt = format_direct_scoring_prompt(
        "Go",
        "Clear asphalt",
        {"A": "Proceed", "B": "Ask", "C": "Reroute", "D": "Slow down"},
    )
    for label in ["A", "B", "C", "D"]:
        assert label in prompt


def test_prompt_uses_provided_option_text():
    custom_b = "Request terrain guidance from user"
    prompt = format_direct_scoring_prompt(
        "Keep going", "Mud", {"A": "Go", "B": custom_b, "C": "Reroute", "D": "Slow"}
    )
    assert custom_b in prompt


# ── KnowNoDecision fields ─────────────────────────────────────────────────────

def test_run_scenario_returns_decision():
    result = _runner().run_scenario(_make_scenario())
    assert isinstance(result, KnowNoDecision)


def test_run_scenario_id_matches():
    result = _runner().run_scenario(_make_scenario(sid="k999"))
    assert result.scenario_id == "k999"


def test_run_scenario_robot_decision_valid():
    result = _runner().run_scenario(_make_scenario())
    assert result.robot_decision in {"A", "B", "C", "D", "ASK"}


def test_run_scenario_confidences_sum_to_one():
    result = _runner().run_scenario(_make_scenario())
    total = sum(result.option_confidences.values())
    assert total == pytest.approx(1.0, abs=1e-6)


def test_run_scenario_options_has_four_keys():
    result = _runner().run_scenario(_make_scenario())
    assert set(result.options.keys()) == {"A", "B", "C", "D"}


def test_run_scenario_prediction_set_is_list():
    result = _runner().run_scenario(_make_scenario())
    assert isinstance(result.prediction_set, list)


# ── LLM call count ────────────────────────────────────────────────────────────

def test_run_scenario_makes_exactly_one_llm_call():
    mock_llm = _make_mock_llm()
    runner = _runner(llm=mock_llm)
    runner.run_scenario(_make_scenario())
    assert mock_llm.predict_json.call_count == 1


def test_two_scenarios_make_two_llm_calls():
    mock_llm = _make_mock_llm()
    runner = _runner(llm=mock_llm)
    runner.run_scenario(_make_scenario(sid="k001"))
    runner.run_scenario(_make_scenario(sid="k002"))
    assert mock_llm.predict_json.call_count == 2


# ── Calibration ───────────────────────────────────────────────────────────────

def test_calibrate_sets_tau():
    runner = _runner()
    assert runner._predictor.tau is None
    runner.calibrate([_make_scenario(sid=f"c{i}") for i in range(5)])
    assert runner._predictor.tau is not None


def test_calibrate_records_n_samples():
    runner = _runner()
    n = 8
    runner.calibrate([_make_scenario(sid=f"c{i}") for i in range(n)])
    assert runner._predictor.n_calibration == n


# ── Uncalibrated fallback ─────────────────────────────────────────────────────

def test_uncalibrated_falls_back_to_argmax():
    # High-confidence B prediction without calibration → argmax = B
    runner = _runner(llm=_make_mock_llm(prediction="B", confidence=0.9))
    result = runner.run_scenario(_make_scenario())
    assert result.robot_decision == "B"


# ── Post-calibration behavior ─────────────────────────────────────────────────

def test_calibrated_high_confidence_acts_directly():
    # Calibrate on 10 high-confidence B scenarios → tau is tight
    mock_llm = _make_mock_llm(prediction="B", confidence=0.95)
    runner = _runner(llm=mock_llm)
    runner.calibrate([_make_scenario(sid=f"c{i}", correct="B") for i in range(10)])

    # High-confidence B at test time → singleton set → ACT
    result = runner.run_scenario(_make_scenario())
    assert result.robot_decision == "B"
    assert result.prediction_set == ["B"]


def test_calibrated_low_confidence_triggers_ask():
    # Calibrate on 10 high-confidence scenarios to get a tight tau
    calib_llm = _make_mock_llm(prediction="B", confidence=0.95)
    runner = _runner(llm=calib_llm)
    runner.calibrate([_make_scenario(sid=f"c{i}", correct="B") for i in range(10)])

    # At test time: low confidence (0.4) → many options pass threshold → ASK
    calib_llm.predict_json.return_value = {
        "prediction": "A",
        "confidence": 0.3,
        "reasoning": "unsure",
    }
    result = runner.run_scenario(_make_scenario())
    # With tau set tight and low confidence, prediction set should be large → ASK
    assert result.robot_decision == "ASK"


# ── run_evaluation ────────────────────────────────────────────────────────────

def test_run_evaluation_returns_standard_metrics_keys():
    runner = _runner()
    metrics = runner.run_evaluation([_make_scenario(sid=f"e{i}") for i in range(4)])
    for key in ["n_scenarios", "SR", "HR", "FPR", "NCR", "ESR"]:
        assert key in metrics


def test_run_evaluation_n_scenarios_matches_input():
    runner = _runner()
    n = 5
    metrics = runner.run_evaluation([_make_scenario(sid=f"e{i}") for i in range(n)])
    assert metrics["n_scenarios"] == n


def test_run_evaluation_metrics_in_unit_interval():
    runner = _runner()
    metrics = runner.run_evaluation([_make_scenario(sid=f"e{i}") for i in range(4)])
    for key in ["SR", "HR", "NCR", "ESR"]:
        assert 0.0 <= metrics[key] <= 1.0, f"{key} = {metrics[key]} out of [0,1]"
