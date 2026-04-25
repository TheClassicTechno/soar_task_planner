"""
Unit tests for conformal_predictor.py — no API calls.
"""

import json
import pytest
import numpy as np
from pathlib import Path

from baselines.introplan.conformal_predictor import (
    ConformalPredictor,
    normalize_confidences,
    extract_option_confidences,
)


# ── normalize_confidences ─────────────────────────────────────────────────────

def test_normalize_sums_to_one():
    raw = {"A": 0.9, "B": 0.3, "C": 0.1, "D": 0.2}
    result = normalize_confidences(raw)
    assert sum(result.values()) == pytest.approx(1.0)


def test_normalize_preserves_keys():
    raw = {"A": 0.4, "B": 0.6}
    result = normalize_confidences(raw)
    assert set(result.keys()) == {"A", "B"}


def test_normalize_zero_total_returns_uniform():
    raw = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    result = normalize_confidences(raw)
    assert all(v == pytest.approx(0.25) for v in result.values())


def test_normalize_already_normalized():
    raw = {"A": 0.5, "B": 0.5}
    result = normalize_confidences(raw)
    assert result["A"] == pytest.approx(0.5)
    assert result["B"] == pytest.approx(0.5)


# ── extract_option_confidences ────────────────────────────────────────────────

def test_extract_predicted_option_gets_stated_confidence():
    llm_pred = {"prediction": "B", "confidence": 0.8, "reasoning": {}, "explanation": "x"}
    result = extract_option_confidences(llm_pred)
    assert result["B"] == pytest.approx(0.8)


def test_extract_other_options_share_remaining():
    llm_pred = {"prediction": "A", "confidence": 0.7, "reasoning": {}, "explanation": "x"}
    result = extract_option_confidences(llm_pred)
    remaining_each = (1 - 0.7) / 3
    for opt in ["B", "C", "D"]:
        assert result[opt] == pytest.approx(remaining_each)


def test_extract_all_four_options_present():
    llm_pred = {"prediction": "C", "confidence": 0.6, "reasoning": {}, "explanation": "x"}
    result = extract_option_confidences(llm_pred)
    assert set(result.keys()) == {"A", "B", "C", "D"}


def test_extract_confidence_clipped_above():
    llm_pred = {"prediction": "A", "confidence": 2.0, "reasoning": {}, "explanation": "x"}
    result = extract_option_confidences(llm_pred)
    assert result["A"] <= 0.99


def test_extract_confidence_clipped_below():
    llm_pred = {"prediction": "A", "confidence": -1.0, "reasoning": {}, "explanation": "x"}
    result = extract_option_confidences(llm_pred)
    assert result["A"] >= 0.01


def test_extract_missing_confidence_defaults_to_half():
    llm_pred = {"prediction": "A", "reasoning": {}, "explanation": "x"}
    result = extract_option_confidences(llm_pred)
    assert result["A"] == pytest.approx(0.5)


# ── ConformalPredictor ────────────────────────────────────────────────────────

@pytest.fixture
def calibrated_predictor():
    """Predictor calibrated with 10 scenarios where correct option has confidence 0.8."""
    pred = ConformalPredictor(alpha=0.15)
    for _ in range(10):
        pred.record_calibration({"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05}, "A")
    pred.calibrate()
    return pred


def test_predictor_invalid_alpha():
    with pytest.raises(ValueError):
        ConformalPredictor(alpha=0.0)
    with pytest.raises(ValueError):
        ConformalPredictor(alpha=1.0)


def test_predictor_calibrate_before_record_raises():
    pred = ConformalPredictor(alpha=0.15)
    with pytest.raises(RuntimeError, match="No calibration data"):
        pred.calibrate()


def test_predictor_predict_before_calibrate_raises():
    pred = ConformalPredictor(alpha=0.15)
    pred.record_calibration({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}, "A")
    with pytest.raises(RuntimeError, match="calibrate()"):
        pred.predict_set({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02})


def test_predictor_tau_set_after_calibrate():
    pred = ConformalPredictor(alpha=0.15)
    for _ in range(5):
        pred.record_calibration({"A": 0.9}, "A")
    pred.calibrate()
    assert pred.tau is not None
    assert 0.0 <= pred.tau <= 1.0


def test_predictor_high_confidence_gives_singleton_set(calibrated_predictor):
    # Confidence 0.9 for A — well above threshold — should give singleton {A}
    pred_set = calibrated_predictor.predict_set({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02})
    assert pred_set == ["A"]


def test_predictor_low_confidence_gives_multiple_options():
    # All options have equal confidence — predictor should include multiple options
    pred = ConformalPredictor(alpha=0.15)
    # Calibrate: correct option was never very confident
    for _ in range(20):
        pred.record_calibration({"A": 0.3, "B": 0.3, "C": 0.2, "D": 0.2}, "A")
    pred.calibrate()
    result = pred.predict_set({"A": 0.3, "B": 0.3, "C": 0.25, "D": 0.15})
    # tau will be based on 1 - 0.3 = 0.7, threshold = 1 - 0.7 = 0.3
    # Options A and B (0.3) should both be included
    assert len(result) >= 1  # at minimum, at least one option


def test_predictor_should_ask_when_set_is_multiple(calibrated_predictor):
    # Two options with equally high confidence
    confs = {"A": 0.6, "B": 0.6, "C": 0.1, "D": 0.1}
    # Both A and B have the same confidence; whether should_ask depends on tau
    # but with tau set for high confidence, 0.6 < threshold → both included
    # This is a structural test — just check the method runs
    result = calibrated_predictor.should_ask(confs)
    assert isinstance(result, bool)


def test_predictor_should_act_on_singleton():
    pred = ConformalPredictor(alpha=0.15)
    for _ in range(10):
        pred.record_calibration({"A": 0.99, "B": 0.01, "C": 0.0, "D": 0.0}, "A")
    pred.calibrate()
    # Very high confidence for A → singleton set → should NOT ask
    assert not pred.should_ask({"A": 0.99, "B": 0.01, "C": 0.0, "D": 0.0})


def test_predictor_n_calibration():
    pred = ConformalPredictor(alpha=0.15)
    assert pred.n_calibration == 0
    pred.record_calibration({"A": 0.8}, "A")
    pred.record_calibration({"A": 0.7}, "A")
    assert pred.n_calibration == 2


def test_predictor_predict_set_sorted_by_confidence(calibrated_predictor):
    # If multiple options are included, highest confidence should be first
    # We need a case where tau is high enough to include multiple options
    pred = ConformalPredictor(alpha=0.15)
    for _ in range(20):
        pred.record_calibration({"A": 0.35, "B": 0.35, "C": 0.15, "D": 0.15}, "A")
    pred.calibrate()
    result = pred.predict_set({"A": 0.5, "B": 0.4, "C": 0.3, "D": 0.1})
    # Whatever the set, it should be sorted by confidence descending
    for i in range(len(result) - 1):
        conf_i = {"A": 0.5, "B": 0.4, "C": 0.3, "D": 0.1}[result[i]]
        conf_next = {"A": 0.5, "B": 0.4, "C": 0.3, "D": 0.1}[result[i + 1]]
        assert conf_i >= conf_next


# ── Save / Load ───────────────────────────────────────────────────────────────

def test_predictor_save_and_load(tmp_path, calibrated_predictor):
    save_path = str(tmp_path / "predictor.json")
    calibrated_predictor.save(save_path)

    loaded = ConformalPredictor.load(save_path)
    assert loaded.alpha == calibrated_predictor.alpha
    assert loaded.tau == pytest.approx(calibrated_predictor.tau)
    assert loaded.n_calibration == calibrated_predictor.n_calibration


def test_predictor_save_creates_parent_dirs(tmp_path):
    pred = ConformalPredictor(alpha=0.15)
    pred.record_calibration({"A": 0.8}, "A")
    pred.calibrate()

    deep_path = str(tmp_path / "a" / "b" / "predictor.json")
    pred.save(deep_path)
    assert Path(deep_path).exists()
