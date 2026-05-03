"""
Unit tests for baselines/whentoask/runner.py — LLM fully mocked. No real API calls.

Tests:
  _resolve_strategy:
    - Singleton real option → EXECUTE + that option
    - Multiple real options → CLARIFY + ASK
    - "E" in set → INCAPABLE + ESCALATE
    - Empty set → CLARIFY + ASK

  WhenToAskDecision fields:
    - scenario_id, options, prediction_set, resolution_strategy, robot_decision all present
    - option_confidences A-D normalize to ~1.0 (E excluded from normalization)
    - robot_decision ∈ {"A","B","C","D","ASK","ESCALATE"}

  Calibration:
    - calibrate() sets tau
    - Records n_calibration samples

  Post-calibration behavior:
    - High-confidence on single option → EXECUTE (not ASK)
    - Uniform scores → CLARIFY (ASK)
    - High E score → INCAPABLE (ESCALATE)

  run_evaluation:
    - Returns standard metrics keys
    - Returns strategy_counts with EXECUTE/CLARIFY/INCAPABLE

  LLM call count:
    - With factorization (default): n_intents + 1 calls per scenario
    - Without factorization: 1 call per scenario
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from baselines.introplan.runner import NavigationScenario
from baselines.whentoask.runner import (
    STRATEGY_CLARIFY,
    STRATEGY_EXECUTE,
    STRATEGY_INCAPABLE,
    WhenToAskDecision,
    WhenToAskRunner,
    _resolve_strategy,
)

CONFIG_PATH = str(Path(__file__).parents[1] / "config.yaml")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_scenario(sid="w001", correct="B", utype=2):
    return NavigationScenario(
        scenario_id=sid,
        instruction="Continue through the park",
        terrain_description="Wet grass with exposed muddy patches",
        uncertainty_type=utype,
        correct_option=correct,
        options={
            "A": "Continue on the current path",
            "B": "Ask the user about the wet grass",
            "C": "Reroute automatically",
            "D": "Slow down and proceed cautiously",
        },
    )


def _intent_response(n=3):
    """Mock Stage 1 LLM response: n equal-probability intents."""
    intents = [
        {"description": f"Intent {i}", "probability": round(1.0 / n, 4)}
        for i in range(n)
    ]
    return {"intents": intents}


def _scores_response(a=0.1, b=0.85, c=0.1, d=0.1, e=0.0, reasoning="test"):
    """Mock Stage 2 LLM response: per-option scores."""
    return {"scores": {"A": a, "B": b, "C": c, "D": d, "E": e}, "reasoning": reasoning}


def _make_mock_llm_with_factorization(n_intents=3, scores=None):
    """
    Build a mock LLM that handles the factorized two-stage call sequence.
    First call returns intents; subsequent calls return option scores.
    """
    mock = MagicMock()
    score_resp = scores or _scores_response()
    # First call: intents. All remaining calls: scores per intent.
    mock.predict_json.side_effect = [_intent_response(n_intents)] + [score_resp] * n_intents
    return mock


def _make_mock_llm_direct(scores=None):
    """Mock LLM for direct scoring (no factorization)."""
    mock = MagicMock()
    mock.predict_json.return_value = scores or _scores_response()
    return mock


def _runner_no_fact(llm=None):
    """Build a WhenToAskRunner with factorization disabled (1 LLM call/scenario)."""
    import yaml
    config_path = CONFIG_PATH
    runner = WhenToAskRunner(config_path=config_path, llm=llm or _make_mock_llm_direct())
    runner._factorization_enabled = False
    return runner


# ── _resolve_strategy tests ───────────────────────────────────────────────────

def test_resolve_singleton_real_option_is_execute():
    strategy, decision = _resolve_strategy(["B"])
    assert strategy == STRATEGY_EXECUTE
    assert decision == "B"


def test_resolve_multiple_real_options_is_clarify():
    strategy, decision = _resolve_strategy(["A", "B"])
    assert strategy == STRATEGY_CLARIFY
    assert decision == "ASK"


def test_resolve_none_option_alone_is_incapable():
    strategy, decision = _resolve_strategy(["E"])
    assert strategy == STRATEGY_INCAPABLE
    assert decision == "ESCALATE"


def test_resolve_none_option_with_real_option_is_incapable():
    # E in set takes priority over real options
    strategy, decision = _resolve_strategy(["B", "E"])
    assert strategy == STRATEGY_INCAPABLE
    assert decision == "ESCALATE"


def test_resolve_empty_set_is_clarify():
    strategy, decision = _resolve_strategy([])
    assert strategy == STRATEGY_CLARIFY
    assert decision == "ASK"


def test_resolve_all_real_options_is_clarify():
    strategy, decision = _resolve_strategy(["A", "B", "C", "D"])
    assert strategy == STRATEGY_CLARIFY
    assert decision == "ASK"


# ── WhenToAskDecision fields ──────────────────────────────────────────────────

@pytest.fixture
def runner_direct():
    return _runner_no_fact()


def test_run_scenario_returns_decision(runner_direct):
    result = runner_direct.run_scenario(_make_scenario())
    assert isinstance(result, WhenToAskDecision)


def test_run_scenario_scenario_id_matches(runner_direct):
    result = runner_direct.run_scenario(_make_scenario(sid="w999"))
    assert result.scenario_id == "w999"


def test_run_scenario_robot_decision_is_valid(runner_direct):
    result = runner_direct.run_scenario(_make_scenario())
    assert result.robot_decision in {"A", "B", "C", "D", "ASK", "ESCALATE"}


def test_run_scenario_resolution_strategy_is_valid(runner_direct):
    result = runner_direct.run_scenario(_make_scenario())
    assert result.resolution_strategy in {STRATEGY_EXECUTE, STRATEGY_CLARIFY, STRATEGY_INCAPABLE}


def test_run_scenario_abcd_confidences_sum_to_one(runner_direct):
    result = runner_direct.run_scenario(_make_scenario())
    # A/B/C/D should sum to ~1.0 (E is not normalized with them)
    abcd_sum = sum(result.option_confidences[k] for k in ["A", "B", "C", "D"])
    assert abcd_sum == pytest.approx(1.0, abs=1e-6)


def test_run_scenario_options_has_four_keys(runner_direct):
    result = runner_direct.run_scenario(_make_scenario())
    assert set(result.options.keys()) == {"A", "B", "C", "D"}


# ── LLM call count ────────────────────────────────────────────────────────────

def test_direct_mode_makes_one_llm_call():
    mock_llm = _make_mock_llm_direct()
    runner = _runner_no_fact(llm=mock_llm)
    runner.run_scenario(_make_scenario())
    assert mock_llm.predict_json.call_count == 1


def test_factorized_mode_makes_n_intents_plus_one_calls():
    n_intents = 3
    mock_llm = _make_mock_llm_with_factorization(n_intents=n_intents)
    runner = WhenToAskRunner(CONFIG_PATH, llm=mock_llm)
    runner._factorization_enabled = True
    runner._n_intents = n_intents
    runner.run_scenario(_make_scenario())
    # 1 intent call + n_intents scoring calls
    assert mock_llm.predict_json.call_count == n_intents + 1


def test_decision_llm_calls_count_reflects_actual_calls():
    n_intents = 2
    mock_llm = _make_mock_llm_with_factorization(n_intents=n_intents)
    runner = WhenToAskRunner(CONFIG_PATH, llm=mock_llm)
    runner._factorization_enabled = True
    runner._n_intents = n_intents
    result = runner.run_scenario(_make_scenario())
    assert result.llm_calls == n_intents + 1


# ── Uncalibrated fallback ─────────────────────────────────────────────────────

def test_uncalibrated_picks_best_option_not_ask():
    # Without tau, runner falls back to argmax of real options — should not ASK
    mock_llm = _make_mock_llm_direct(scores=_scores_response(a=0.1, b=0.85, c=0.1, d=0.1))
    runner = _runner_no_fact(llm=mock_llm)
    result = runner.run_scenario(_make_scenario())
    assert result.robot_decision != "ASK"
    assert result.robot_decision != "ESCALATE"


# ── Calibration ───────────────────────────────────────────────────────────────

def test_calibrate_sets_tau():
    mock_llm = _make_mock_llm_direct()
    runner = _runner_no_fact(llm=mock_llm)
    assert runner._predictor.tau is None
    runner.calibrate([_make_scenario(sid=f"c{i}") for i in range(5)])
    assert runner._predictor.tau is not None


def test_calibrate_records_correct_n_samples():
    mock_llm = _make_mock_llm_direct()
    runner = _runner_no_fact(llm=mock_llm)
    n = 7
    runner.calibrate([_make_scenario(sid=f"c{i}") for i in range(n)])
    assert runner._predictor.n_calibration == n


# ── Post-calibration behavior ─────────────────────────────────────────────────

@pytest.fixture
def calibrated_runner():
    """Runner calibrated on 10 high-confidence-B scenarios."""
    mock_llm = _make_mock_llm_direct(scores=_scores_response(b=0.95, a=0.01, c=0.01, d=0.01, e=0.0))
    runner = _runner_no_fact(llm=mock_llm)
    runner.calibrate([_make_scenario(sid=f"c{i}") for i in range(10)])
    return runner, mock_llm


def test_calibrated_high_confidence_executes(calibrated_runner):
    runner, mock_llm = calibrated_runner
    mock_llm.predict_json.return_value = _scores_response(b=0.95, a=0.01, c=0.01, d=0.01)
    result = runner.run_scenario(_make_scenario())
    assert result.resolution_strategy == STRATEGY_EXECUTE
    assert result.robot_decision == "B"


def test_calibrated_uniform_scores_clarify(calibrated_runner):
    runner, mock_llm = calibrated_runner
    mock_llm.predict_json.return_value = _scores_response(a=0.25, b=0.25, c=0.25, d=0.25, e=0.0)
    result = runner.run_scenario(_make_scenario())
    assert result.resolution_strategy == STRATEGY_CLARIFY
    assert result.robot_decision == "ASK"


def test_calibrated_high_e_score_escalates(calibrated_runner):
    runner, mock_llm = calibrated_runner
    # High E score pushes it into prediction set → INCAPABLE
    mock_llm.predict_json.return_value = _scores_response(a=0.1, b=0.1, c=0.1, d=0.1, e=0.99)
    result = runner.run_scenario(_make_scenario())
    assert result.resolution_strategy == STRATEGY_INCAPABLE
    assert result.robot_decision == "ESCALATE"


# ── run_evaluation ────────────────────────────────────────────────────────────

def test_run_evaluation_returns_standard_metrics_keys():
    runner = _runner_no_fact()
    metrics = runner.run_evaluation([_make_scenario(sid=f"e{i}") for i in range(4)])
    for key in ["n_scenarios", "SR", "HR", "FPR", "NCR", "ESR"]:
        assert key in metrics


def test_run_evaluation_returns_strategy_counts():
    runner = _runner_no_fact()
    metrics = runner.run_evaluation([_make_scenario(sid=f"e{i}") for i in range(4)])
    assert "strategy_counts" in metrics
    for key in ["EXECUTE", "CLARIFY", "INCAPABLE"]:
        assert key in metrics["strategy_counts"]


def test_run_evaluation_n_scenarios_matches_input():
    runner = _runner_no_fact()
    n = 6
    metrics = runner.run_evaluation([_make_scenario(sid=f"e{i}") for i in range(n)])
    assert metrics["n_scenarios"] == n
