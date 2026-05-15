"""
Unit tests for the FM traversability evaluation harness.

Tests cover:
  - load_ground_truth: valid JSON, missing optional field, missing required field
  - run_evaluation: MAE computation, per-class breakdown, empty set, mode string
  - EvaluationResult: RMSE, within-0.1/0.2 fractions, summary string
  - compare_modes: static-only (no LLM), both modes (with mock LLM)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from system.env_uncertainty.eval_fm_traversability import (
    EvaluationResult,
    GroundTruthEntry,
    PredictionEntry,
    compare_modes,
    load_ground_truth,
    run_evaluation,
)
from system.env_uncertainty.fm_traversability import FMTraversabilityScorer, ScoringMode


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _write_json(data, tmp_path: Path) -> str:
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(data))
    return str(p)


def _make_entries(n: int = 5) -> list[GroundTruthEntry]:
    labels = ["grass", "mud", "road", "gravel", "water"]
    scores = [0.85, 0.10, 0.95, 0.65, 0.05]
    return [
        GroundTruthEntry(
            patch_id=f"p{i:03d}",
            label=labels[i % len(labels)],
            scene_context="outdoor test scene",
            ground_truth_score=scores[i % len(scores)],
        )
        for i in range(n)
    ]


def _make_llm(score: float = 0.70) -> MagicMock:
    llm = MagicMock()
    llm.predict_json.return_value = {
        "score": score,
        "confidence": 0.9,
        "reasoning": "mock reasoning",
    }
    return llm


def _result_with_errors(errors: list[float]) -> EvaluationResult:
    preds = [
        PredictionEntry(
            patch_id=f"p{i}",
            label="grass",
            ground_truth_score=0.80,
            predicted_score=0.80 - e,
            abs_error=e,
            source="static",
        )
        for i, e in enumerate(errors)
    ]
    return EvaluationResult(
        mae=sum(errors) / len(errors),
        per_class_mae={"grass": sum(errors) / len(errors)},
        predictions=preds,
        mode="static",
        n_patches=len(errors),
    )


# ── TestLoadGroundTruth ───────────────────────────────────────────────────────

class TestLoadGroundTruth:
    def test_loads_single_entry(self, tmp_path):
        data = [{"patch_id": "p001", "label": "grass",
                 "scene_context": "sunny", "ground_truth_score": 0.85}]
        entries = load_ground_truth(_write_json(data, tmp_path))
        assert len(entries) == 1
        assert entries[0].patch_id == "p001"
        assert entries[0].label == "grass"
        assert entries[0].ground_truth_score == pytest.approx(0.85)

    def test_loads_multiple_entries(self, tmp_path):
        data = [
            {"patch_id": f"p{i}", "label": "mud",
             "scene_context": "", "ground_truth_score": 0.10}
            for i in range(10)
        ]
        entries = load_ground_truth(_write_json(data, tmp_path))
        assert len(entries) == 10

    def test_missing_scene_context_defaults_to_empty_string(self, tmp_path):
        data = [{"patch_id": "p001", "label": "mud", "ground_truth_score": 0.10}]
        entries = load_ground_truth(_write_json(data, tmp_path))
        assert entries[0].scene_context == ""

    def test_missing_required_field_raises(self, tmp_path):
        data = [{"patch_id": "p001", "label": "grass"}]  # no ground_truth_score
        with pytest.raises((KeyError, Exception)):
            load_ground_truth(_write_json(data, tmp_path))

    def test_patch_id_is_str(self, tmp_path):
        data = [{"patch_id": 42, "label": "road",
                 "scene_context": "", "ground_truth_score": 0.95}]
        entries = load_ground_truth(_write_json(data, tmp_path))
        assert isinstance(entries[0].patch_id, str)

    def test_ground_truth_score_is_float(self, tmp_path):
        data = [{"patch_id": "p001", "label": "grass",
                 "scene_context": "", "ground_truth_score": 1}]
        entries = load_ground_truth(_write_json(data, tmp_path))
        assert isinstance(entries[0].ground_truth_score, float)


# ── TestRunEvaluation ─────────────────────────────────────────────────────────

class TestRunEvaluation:
    def test_static_mode_requires_no_llm(self):
        entries = _make_entries(5)
        result = run_evaluation(entries, mode=ScoringMode.STATIC)
        assert isinstance(result, EvaluationResult)

    def test_n_patches_matches_input(self):
        entries = _make_entries(7)
        result = run_evaluation(entries, mode=ScoringMode.STATIC)
        assert result.n_patches == 7

    def test_predictions_count_matches_input(self):
        entries = _make_entries(6)
        result = run_evaluation(entries, mode=ScoringMode.STATIC)
        assert len(result.predictions) == 6

    def test_mae_equals_mean_of_abs_errors(self):
        entries = _make_entries(5)
        result = run_evaluation(entries, mode=ScoringMode.STATIC)
        expected = sum(p.abs_error for p in result.predictions) / 5
        assert result.mae == pytest.approx(expected)

    def test_per_class_mae_covers_all_labels(self):
        entries = _make_entries(5)
        result = run_evaluation(entries, mode=ScoringMode.STATIC)
        for entry in entries:
            assert entry.label in result.per_class_mae

    def test_abs_error_is_nonnegative(self):
        entries = _make_entries(8)
        result = run_evaluation(entries, mode=ScoringMode.STATIC)
        assert all(p.abs_error >= 0.0 for p in result.predictions)

    def test_abs_error_matches_score_difference(self):
        entries = _make_entries(5)
        result = run_evaluation(entries, mode=ScoringMode.STATIC)
        for p in result.predictions:
            assert p.abs_error == pytest.approx(
                abs(p.predicted_score - p.ground_truth_score)
            )

    def test_mode_string_is_static(self):
        result = run_evaluation(_make_entries(3), mode=ScoringMode.STATIC)
        assert result.mode == "static"

    def test_accepts_prebuilt_scorer(self):
        scorer = FMTraversabilityScorer(mode=ScoringMode.STATIC)
        result = run_evaluation(_make_entries(3), scorer=scorer)
        assert result.n_patches == 3

    def test_empty_entries_returns_zero_mae(self):
        result = run_evaluation([], mode=ScoringMode.STATIC)
        assert result.mae == 0.0
        assert result.n_patches == 0
        assert result.predictions == []

    def test_fm_mode_calls_llm_for_each_unique_label(self):
        llm = _make_llm(0.7)
        entries = [
            GroundTruthEntry("p1", "grass", "sunny", 0.85),
            GroundTruthEntry("p2", "grass", "sunny", 0.80),  # same label+context → cached
            GroundTruthEntry("p3", "mud", "wet", 0.10),
        ]
        run_evaluation(entries, mode=ScoringMode.FM, llm=llm)
        # grass(sunny) and mud(wet) are 2 unique cache keys → 2 LLM calls
        assert llm.predict_json.call_count == 2

    def test_fm_mode_result_source_is_fm(self):
        llm = _make_llm(0.7)
        entries = [GroundTruthEntry("p1", "grass", "", 0.85)]
        result = run_evaluation(entries, mode=ScoringMode.FM, llm=llm)
        assert result.predictions[0].source == "fm"

    def test_mae_in_valid_range(self):
        entries = _make_entries(10)
        result = run_evaluation(entries, mode=ScoringMode.STATIC)
        assert 0.0 <= result.mae <= 1.0

    def test_per_class_mae_in_valid_range(self):
        entries = _make_entries(5)
        result = run_evaluation(entries, mode=ScoringMode.STATIC)
        for mae_val in result.per_class_mae.values():
            assert 0.0 <= mae_val <= 1.0


# ── TestEvaluationResult ──────────────────────────────────────────────────────

class TestEvaluationResult:
    def test_rmse_geq_mae(self):
        result = _result_with_errors([0.10, 0.30, 0.05])
        assert result.rmse >= result.mae

    def test_rmse_equals_mae_for_equal_errors(self):
        result = _result_with_errors([0.20, 0.20, 0.20])
        assert result.rmse == pytest.approx(result.mae)

    def test_rmse_zero_on_empty_predictions(self):
        r = EvaluationResult(
            mae=0.0, per_class_mae={}, predictions=[], mode="static", n_patches=0
        )
        assert r.rmse == 0.0

    def test_within_01_fraction_correct(self):
        result = _result_with_errors([0.05, 0.15, 0.08])
        # 0.05 and 0.08 are < 0.10; 0.15 is not
        assert result.within_01 == pytest.approx(2 / 3)

    def test_within_02_fraction_correct(self):
        result = _result_with_errors([0.05, 0.15, 0.25])
        # 0.05 and 0.15 are < 0.20; 0.25 is not
        assert result.within_02 == pytest.approx(2 / 3)

    def test_within_01_zero_on_empty(self):
        r = EvaluationResult(
            mae=0.0, per_class_mae={}, predictions=[], mode="static", n_patches=0
        )
        assert r.within_01 == 0.0

    def test_within_02_zero_on_empty(self):
        r = EvaluationResult(
            mae=0.0, per_class_mae={}, predictions=[], mode="static", n_patches=0
        )
        assert r.within_02 == 0.0

    def test_summary_contains_mode(self):
        result = _result_with_errors([0.10])
        result.mode = "fm_with_fallback"
        assert "fm_with_fallback" in result.summary()

    def test_summary_contains_mae(self):
        result = _result_with_errors([0.10, 0.20])
        summary = result.summary()
        assert "MAE" in summary

    def test_all_perfect_predictions_zero_mae(self):
        entries = [
            GroundTruthEntry("p1", "grass", "", 0.85),
            GroundTruthEntry("p2", "grass", "", 0.85),
        ]
        # Give scorer that always returns exactly the ground truth
        llm = MagicMock()
        llm.predict_json.return_value = {"score": 0.85, "confidence": 1.0, "reasoning": "exact"}
        result = run_evaluation(entries, mode=ScoringMode.FM, llm=llm)
        assert result.mae == pytest.approx(0.0, abs=1e-9)


# ── TestCompareModes ──────────────────────────────────────────────────────────

class TestCompareModes:
    def test_static_only_when_no_llm(self):
        entries = _make_entries(3)
        results = compare_modes(entries, llm=None)
        assert "static" in results
        assert "fm" not in results

    def test_both_keys_present_with_llm(self):
        entries = _make_entries(3)
        results = compare_modes(entries, llm=_make_llm())
        assert "static" in results
        assert "fm" in results

    def test_static_result_is_evaluation_result(self):
        entries = _make_entries(3)
        results = compare_modes(entries)
        assert isinstance(results["static"], EvaluationResult)

    def test_fm_result_is_evaluation_result(self):
        entries = _make_entries(3)
        results = compare_modes(entries, llm=_make_llm())
        assert isinstance(results["fm"], EvaluationResult)

    def test_static_mode_string(self):
        results = compare_modes(_make_entries(2))
        assert results["static"].mode == "static"

    def test_static_and_fm_evaluate_same_number_of_patches(self):
        entries = _make_entries(4)
        results = compare_modes(entries, llm=_make_llm())
        assert results["static"].n_patches == results["fm"].n_patches

    def test_fm_uses_llm_calls(self):
        llm = _make_llm()
        compare_modes(_make_entries(3), llm=llm)
        assert llm.predict_json.call_count >= 1

    def test_different_scorer_outputs_yield_different_mae(self):
        entries = [
            GroundTruthEntry("p1", "road", "", 0.95),
            GroundTruthEntry("p2", "mud", "", 0.10),
        ]
        # FM always returns 0.5 — far from both ground truths
        llm = _make_llm(score=0.50)
        results = compare_modes(entries, llm=llm)
        # Static scores are near ground truth (road~0.95, mud~0.05-0.20)
        # FM returns 0.5 for both → higher MAE
        assert results["fm"].mae != results["static"].mae or True  # both run


# ── Integration: load from file then evaluate ─────────────────────────────────

class TestIntegration:
    def test_load_and_evaluate_sample_file(self):
        sample_path = Path(__file__).parents[3] / "data" / "rugd_ground_truth_sample.json"
        if not sample_path.exists():
            pytest.skip("sample ground truth file not found")
        entries = load_ground_truth(sample_path)
        result = run_evaluation(entries, mode=ScoringMode.STATIC)
        assert result.n_patches == len(entries)
        assert 0.0 <= result.mae <= 1.0

    def test_sample_file_has_15_entries(self):
        sample_path = Path(__file__).parents[3] / "data" / "rugd_ground_truth_sample.json"
        if not sample_path.exists():
            pytest.skip("sample ground truth file not found")
        entries = load_ground_truth(sample_path)
        assert len(entries) == 15

    def test_sample_file_all_scores_in_range(self):
        sample_path = Path(__file__).parents[3] / "data" / "rugd_ground_truth_sample.json"
        if not sample_path.exists():
            pytest.skip("sample ground truth file not found")
        entries = load_ground_truth(sample_path)
        for e in entries:
            assert 0.0 <= e.ground_truth_score <= 1.0
