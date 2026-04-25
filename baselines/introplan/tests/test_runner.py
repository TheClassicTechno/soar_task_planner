"""
Unit tests for runner.py — Claude API fully mocked. No real API calls.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from baselines.introplan.runner import (
    IntroPlanRunner,
    NavigationScenario,
    load_scenarios_from_json,
)
from baselines.introplan.metrics import ScenarioResult

CONFIG_PATH = str(Path(__file__).parents[1] / "config.yaml")
CALIBRATION_PATH = str(Path(__file__).parents[1] / "data" / "nav_calibration.json")


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_scenario(sid="s001", correct="B", utype=2):
    return NavigationScenario(
        scenario_id=sid,
        instruction="Take me to the library",
        terrain_description="cracked pavement 5m ahead",
        uncertainty_type=utype,
        correct_option=correct,
        options={
            "A": "Continue straight",
            "B": "Ask user about cracked pavement",
            "C": "Reroute automatically",
            "D": "Slow down",
        },
    )


def _llm_response(prediction="B", confidence=0.85):
    return {
        "reasoning": {
            "A": "Risky without user consent.",
            "B": "User preference unknown — ask.",
            "C": "Unnecessary if user is fine.",
            "D": "Doesn't address the uncertainty.",
        },
        "prediction": prediction,
        "confidence": confidence,
        "explanation": "Asking is correct for Type 2 uncertainty.",
    }


def _make_mock_llm(prediction="B", confidence=0.85):
    mock = MagicMock()
    mock.predict_json.return_value = _llm_response(prediction, confidence)
    return mock


# ── load_scenarios_from_json ───────────────────────────────────────────────────

def test_load_scenarios_from_calibration_file():
    scenarios = load_scenarios_from_json(CALIBRATION_PATH)
    assert len(scenarios) == 50
    assert all(isinstance(s, NavigationScenario) for s in scenarios)


def test_load_scenarios_all_have_correct_option():
    scenarios = load_scenarios_from_json(CALIBRATION_PATH)
    for s in scenarios:
        assert s.correct_option in ["A", "B", "C", "D"]


def test_load_scenarios_from_custom_json(tmp_path):
    data = [{
        "entry_id": "x001",
        "instruction": "Go to lab",
        "terrain_description": "gravel path",
        "uncertainty_type": 4,
        "correct_option": "B",
        "options": {"A": "Go", "B": "Ask", "C": "Reroute", "D": "Slow"},
    }]
    f = tmp_path / "test.json"
    f.write_text(json.dumps(data))
    scenarios = load_scenarios_from_json(str(f))
    assert len(scenarios) == 1
    assert scenarios[0].scenario_id == "x001"
    assert scenarios[0].instruction == "Go to lab"


# ── IntroPlanRunner ────────────────────────────────────────────────────────────

@pytest.fixture
def runner_with_mock_llm():
    mock_llm = _make_mock_llm(prediction="B", confidence=0.85)
    runner = IntroPlanRunner(CONFIG_PATH, llm=mock_llm)
    return runner, mock_llm


def test_run_scenario_returns_decision(runner_with_mock_llm):
    runner, _ = runner_with_mock_llm
    scenario = _make_scenario()
    decision = runner.run_scenario(scenario)
    assert decision.scenario_id == "s001"


def test_run_scenario_prediction_set_is_list(runner_with_mock_llm):
    runner, _ = runner_with_mock_llm
    decision = runner.run_scenario(_make_scenario())
    assert isinstance(decision.prediction_set, list)


def test_run_scenario_robot_decision_is_valid(runner_with_mock_llm):
    runner, _ = runner_with_mock_llm
    decision = runner.run_scenario(_make_scenario())
    assert decision.robot_decision in ["A", "B", "C", "D", "ASK"]


def test_run_scenario_option_confidences_sum_to_one(runner_with_mock_llm):
    runner, _ = runner_with_mock_llm
    decision = runner.run_scenario(_make_scenario())
    total = sum(decision.option_confidences.values())
    assert total == pytest.approx(1.0, abs=1e-6)


def test_run_scenario_retrieved_examples_are_listed(runner_with_mock_llm):
    runner, _ = runner_with_mock_llm
    decision = runner.run_scenario(_make_scenario())
    assert isinstance(decision.retrieved_example_ids, list)


def test_run_scenario_calls_llm_predict_json(runner_with_mock_llm):
    runner, mock_llm = runner_with_mock_llm
    runner.run_scenario(_make_scenario())
    assert mock_llm.predict_json.called


def test_run_scenario_uncalibrated_uses_direct_prediction(runner_with_mock_llm):
    # Without calibration, should fall back to highest-confidence option
    runner, _ = runner_with_mock_llm
    decision = runner.run_scenario(_make_scenario())
    # Direct prediction: pick best option, so |pred_set| == 1 and decision != "ASK"
    assert len(decision.prediction_set) == 1
    assert decision.robot_decision != "ASK"


def test_calibrate_sets_tau(runner_with_mock_llm):
    runner, _ = runner_with_mock_llm
    scenarios = [_make_scenario(sid=f"s{i:03d}") for i in range(5)]
    tau = runner.calibrate(scenarios)
    assert runner._predictor.tau is not None
    assert 0.0 <= tau <= 1.0


def test_run_evaluation_returns_metrics_dict(runner_with_mock_llm):
    runner, _ = runner_with_mock_llm
    scenarios = [_make_scenario(sid=f"s{i:03d}") for i in range(3)]
    result = runner.run_evaluation(scenarios)
    for key in ["n_scenarios", "SR", "HR", "FPR", "NCR", "ESR"]:
        assert key in result


def test_run_evaluation_n_scenarios_matches(runner_with_mock_llm):
    runner, _ = runner_with_mock_llm
    n = 4
    scenarios = [_make_scenario(sid=f"s{i:03d}") for i in range(n)]
    result = runner.run_evaluation(scenarios)
    assert result["n_scenarios"] == n


def test_run_scenario_without_options_calls_llm_twice(runner_with_mock_llm):
    """When no options provided, runner calls LLM to generate them (2 calls total)."""
    runner, mock_llm = runner_with_mock_llm

    # Make the first call (option generation) return valid options
    mock_llm.predict_json.side_effect = [
        {"A": "Go", "B": "Ask about pavement", "C": "Reroute", "D": "Slow down"},
        _llm_response("B", 0.85),
    ]

    scenario = NavigationScenario(
        scenario_id="no_opts",
        instruction="Go to library",
        terrain_description="cracked pavement",
        uncertainty_type=2,
        correct_option="B",
        options=None,  # No pre-set options
    )
    decision = runner.run_scenario(scenario)
    # Two LLM calls: one for options, one for prediction
    assert mock_llm.predict_json.call_count == 2
    assert "Ask about pavement" in decision.options["B"]
