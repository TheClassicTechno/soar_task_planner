"""
Unit tests for metrics.py — no API calls, pure computation.
"""

import pytest
from baselines.introplan.metrics import (
    ScenarioResult,
    MetricsCalculator,
    compute_sr_hr_auc,
)


# ── ScenarioResult ────────────────────────────────────────────────────────────

def _make_result(correct, pred_set, decision=None):
    if decision is None:
        decision = "ASK" if len(pred_set) != 1 else pred_set[0]
    return ScenarioResult("test_id", correct, pred_set, decision)


def test_result_asked_human_when_decision_is_ask():
    r = _make_result("B", ["A", "B"], decision="ASK")
    assert r.asked_human is True


def test_result_not_asked_when_singleton():
    r = _make_result("A", ["A"], decision="A")
    assert r.asked_human is False


def test_result_correct_when_acted_right():
    r = _make_result("B", ["B"], decision="B")
    assert r.correct is True


def test_result_incorrect_when_acted_wrong():
    r = _make_result("B", ["A"], decision="A")
    assert r.correct is False


def test_result_correct_when_asked_because_human_resolves():
    r = _make_result("B", ["A", "B"], decision="ASK")
    assert r.correct is True


def test_result_exact_set_when_singleton_matches_correct():
    r = _make_result("B", ["B"], decision="B")
    assert r.exact_set is True


def test_result_not_exact_set_when_multiple():
    r = _make_result("B", ["A", "B"], decision="ASK")
    assert r.exact_set is False


def test_result_not_exact_set_when_singleton_wrong():
    r = _make_result("B", ["A"], decision="A")
    assert r.exact_set is False


def test_result_noncompliant_when_set_contains_wrong():
    r = _make_result("B", ["A", "B"])
    assert r.noncompliant is True


def test_result_not_noncompliant_when_set_is_correct_only():
    r = _make_result("B", ["B"])
    assert r.noncompliant is False


def test_result_false_positive_when_asked_and_correct_was_proceed():
    # Option A (proceed) is an autonomous act — asking instead is a false positive.
    r = _make_result("A", ["A", "B"], decision="ASK")
    assert r.false_positive is True


def test_result_false_positive_when_asked_and_correct_was_reroute():
    # Option C (reroute automatically) is an autonomous act — asking instead is a false positive.
    r = _make_result("C", ["B", "C"], decision="ASK")
    assert r.false_positive is True


def test_result_false_positive_when_asked_and_correct_was_slowdown():
    # Option D (slow down) is an autonomous act — asking instead is a false positive.
    r = _make_result("D", ["B", "D"], decision="ASK")
    assert r.false_positive is True


def test_result_not_false_positive_when_asked_and_correct_was_ask():
    # Option B means the robot should ask — asking is the right call, not a false positive.
    r = _make_result("B", ["A", "B"], decision="ASK")
    assert r.false_positive is False


def test_result_not_false_positive_when_did_not_ask():
    # Robot acted directly and was correct — no ask, no false positive.
    r = _make_result("A", ["A"], decision="A")
    assert r.false_positive is False


# ── MetricsCalculator ─────────────────────────────────────────────────────────

def _populated_calc():
    calc = MetricsCalculator()
    # 5 scenarios: 3 correct actions, 2 asks, 1 false positive ask
    calc.add(_make_result("A", ["A"], "A"))          # correct, acted
    calc.add(_make_result("B", ["B"], "B"))          # correct, acted
    calc.add(_make_result("C", ["C"], "C"))          # correct, acted
    calc.add(_make_result("B", ["A", "B"], "ASK"))   # asked, correct option is B
    calc.add(_make_result("A", ["A", "B"], "ASK"))   # asked, correct option is A (FP)
    return calc


def test_calc_len():
    calc = _populated_calc()
    assert len(calc) == 5


def test_calc_success_rate():
    calc = _populated_calc()
    # All 5 are correct (asking always counts as correct)
    assert calc.success_rate() == pytest.approx(1.0)


def test_calc_human_help_rate():
    calc = _populated_calc()
    # 2 out of 5 asked
    assert calc.human_help_rate() == pytest.approx(2 / 5)


def test_calc_false_positive_rate():
    calc = _populated_calc()
    # Of the 2 asks, 1 was a false positive (correct was A)
    assert calc.false_positive_rate() == pytest.approx(1 / 2)


def test_calc_exact_set_rate():
    calc = _populated_calc()
    # 3 exact singletons out of 5
    assert calc.exact_set_rate() == pytest.approx(3 / 5)


def test_calc_non_compliant_contamination():
    calc = _populated_calc()
    # 2 prediction sets contain wrong option (the multi-option sets)
    assert calc.non_compliant_contamination_rate() == pytest.approx(2 / 5)


def test_calc_empty_returns_zeros():
    calc = MetricsCalculator()
    assert calc.success_rate() == 0.0
    assert calc.human_help_rate() == 0.0
    assert calc.false_positive_rate() == 0.0


def test_calc_summary_has_all_keys():
    calc = _populated_calc()
    summary = calc.summary()
    for key in ["n_scenarios", "SR", "HR", "FPR", "NCR", "ESR"]:
        assert key in summary


def test_calc_summary_n_scenarios():
    calc = _populated_calc()
    assert calc.summary()["n_scenarios"] == 5


# ── compute_sr_hr_auc ─────────────────────────────────────────────────────────

def test_auc_monotone_curve():
    # Simulate a perfect monotone SR/HR trade-off curve
    # Higher tau → lower HR, lower SR
    results_by_tau = {}
    for tau_idx, (hr, sr) in enumerate([(0.0, 0.6), (0.3, 0.8), (0.6, 0.9), (1.0, 1.0)]):
        tau = tau_idx * 0.1
        # Generate results that produce the target HR and SR
        calc_results = []
        n = 10
        asks = int(round(hr * n))
        for i in range(n):
            if i < asks:
                calc_results.append(ScenarioResult(str(i), "B", ["A", "B"], "ASK"))
            else:
                calc_results.append(ScenarioResult(str(i), "A", ["A"], "A"))
        results_by_tau[tau] = calc_results

    auc = compute_sr_hr_auc(results_by_tau)
    # AUC should be in (0, 1) for a meaningful curve
    assert 0.0 < auc <= 1.0


def test_auc_single_point_returns_zero():
    results = {0.0: [ScenarioResult("x", "A", ["A"], "A")]}
    auc = compute_sr_hr_auc(results)
    assert auc == pytest.approx(0.0)


def test_auc_empty_returns_zero():
    auc = compute_sr_hr_auc({})
    assert auc == pytest.approx(0.0)
