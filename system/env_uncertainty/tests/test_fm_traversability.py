"""
Unit tests for FMTraversabilityScorer.

All tests use a mock LLM so no network or GPU access is required.
The mock returns configurable JSON responses with controllable latency.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from system.env_uncertainty.detector import RegionInfo
from system.env_uncertainty.fm_traversability import (
    FMTraversabilityScorer,
    ScoringMode,
    TraversabilityJudgment,
)
from system.env_uncertainty.traversability import TRAVERSABILITY_SCORES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_llm(score: float = 0.8, confidence: float = 0.9,
              reasoning: str = "test", latency_s: float = 0.0) -> MagicMock:
    """Return a mock LLM whose predict_json sleeps latency_s then returns a score."""
    llm = MagicMock()
    def _predict(prompt, **kwargs):
        if latency_s > 0:
            time.sleep(latency_s)
        return {"score": score, "confidence": confidence, "reasoning": reasoning}
    llm.predict_json.side_effect = _predict
    return llm


def _make_region(label: str, pixel_fraction: float = 0.10) -> RegionInfo:
    mask = np.zeros((100, 100), dtype=bool)
    return RegionInfo(
        label=label,
        mask=mask,
        confidence=0.9,
        pixel_fraction=pixel_fraction,
        source="sam3",
        traversability=TRAVERSABILITY_SCORES.get(label, 0.0),
    )


# ── Static mode ───────────────────────────────────────────────────────────────

class TestStaticMode:
    def test_static_mode_requires_no_llm(self):
        scorer = FMTraversabilityScorer(mode=ScoringMode.STATIC)
        result = scorer.score_label("grass")
        assert result.source == "static"

    def test_static_mode_returns_table_value(self):
        scorer = FMTraversabilityScorer(mode=ScoringMode.STATIC)
        result = scorer.score_label("mud")
        assert result.score == pytest.approx(TRAVERSABILITY_SCORES["mud"])

    def test_static_mode_unknown_label_returns_zero(self):
        scorer = FMTraversabilityScorer(mode=ScoringMode.STATIC)
        result = scorer.score_label("lava_field")
        assert result.score == 0.0

    def test_static_mode_makes_no_llm_calls(self):
        llm = _make_llm()
        scorer = FMTraversabilityScorer(llm=llm, mode=ScoringMode.STATIC)
        scorer.score_label("grass")
        llm.predict_json.assert_not_called()

    def test_static_mode_score_in_valid_range(self):
        scorer = FMTraversabilityScorer(mode=ScoringMode.STATIC)
        for label in list(TRAVERSABILITY_SCORES.keys()):
            j = scorer.score_label(label)
            assert 0.0 <= j.score <= 1.0, f"score out of range for {label}: {j.score}"


# ── FM mode ───────────────────────────────────────────────────────────────────

class TestFMMode:
    def test_fm_mode_requires_llm(self):
        with pytest.raises(ValueError, match="requires an LLMInterface"):
            FMTraversabilityScorer(mode=ScoringMode.FM)

    def test_fm_mode_calls_llm(self):
        llm = _make_llm(score=0.75)
        scorer = FMTraversabilityScorer(llm=llm, mode=ScoringMode.FM)
        result = scorer.score_label("gravel")
        assert result.source == "fm"
        assert result.score == pytest.approx(0.75)
        llm.predict_json.assert_called_once()

    def test_fm_score_clamped_to_0_1(self):
        llm = _make_llm(score=1.5)
        scorer = FMTraversabilityScorer(llm=llm, mode=ScoringMode.FM)
        result = scorer.score_label("road")
        assert result.score <= 1.0

    def test_fm_score_clamped_lower_bound(self):
        llm = _make_llm(score=-0.3)
        scorer = FMTraversabilityScorer(llm=llm, mode=ScoringMode.FM)
        result = scorer.score_label("water")
        assert result.score >= 0.0

    def test_fm_returns_reasoning_string(self):
        llm = _make_llm(score=0.9, reasoning="Dry flat road, safe for wheeled robots.")
        scorer = FMTraversabilityScorer(llm=llm, mode=ScoringMode.FM)
        result = scorer.score_label("road")
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    def test_fm_prompt_includes_label(self):
        llm = _make_llm()
        scorer = FMTraversabilityScorer(llm=llm, mode=ScoringMode.FM)
        scorer.score_label("cracked pavement", context="evening rain")
        call_args = llm.predict_json.call_args
        prompt_text = call_args[0][0]
        assert "cracked pavement" in prompt_text
        assert "evening rain" in prompt_text


# ── Cache behavior ────────────────────────────────────────────────────────────

class TestCache:
    def test_repeated_label_uses_cache(self):
        llm = _make_llm(score=0.7)
        scorer = FMTraversabilityScorer(llm=llm, mode=ScoringMode.FM)
        scorer.score_label("dirt")
        scorer.score_label("dirt")
        assert llm.predict_json.call_count == 1

    def test_cache_hit_source_is_cache(self):
        llm = _make_llm(score=0.7)
        scorer = FMTraversabilityScorer(llm=llm, mode=ScoringMode.FM)
        scorer.score_label("dirt")
        result2 = scorer.score_label("dirt")
        assert result2.source == "cache"

    def test_different_context_different_cache_entry(self):
        llm = _make_llm(score=0.7)
        scorer = FMTraversabilityScorer(llm=llm, mode=ScoringMode.FM)
        scorer.score_label("grass", context="dry morning")
        scorer.score_label("grass", context="wet evening")
        assert llm.predict_json.call_count == 2

    def test_clear_cache_forces_re_query(self):
        llm = _make_llm(score=0.7)
        scorer = FMTraversabilityScorer(llm=llm, mode=ScoringMode.FM)
        scorer.score_label("grass")
        scorer.clear_cache()
        scorer.score_label("grass")
        assert llm.predict_json.call_count == 2

    def test_cache_size_tracks_entries(self):
        llm = _make_llm(score=0.7)
        scorer = FMTraversabilityScorer(llm=llm, mode=ScoringMode.FM)
        scorer.score_label("grass")
        scorer.score_label("mud")
        assert scorer.cache_size == 2


# ── FM_WITH_FALLBACK mode ─────────────────────────────────────────────────────

class TestFMWithFallback:
    def test_fallback_on_llm_exception(self):
        llm = MagicMock()
        llm.predict_json.side_effect = RuntimeError("API error")
        scorer = FMTraversabilityScorer(llm=llm, mode=ScoringMode.FM_WITH_FALLBACK)
        result = scorer.score_label("grass")
        assert result.source == "static"
        assert result.score == pytest.approx(TRAVERSABILITY_SCORES["grass"])

    def test_fallback_on_latency_exceeded(self):
        llm = _make_llm(score=0.8, latency_s=0.05)
        scorer = FMTraversabilityScorer(
            llm=llm,
            mode=ScoringMode.FM_WITH_FALLBACK,
            latency_budget_ms=1.0,  # 1 ms budget, 50 ms actual
        )
        result = scorer.score_label("gravel")
        assert result.source == "static"

    def test_no_fallback_within_budget(self):
        llm = _make_llm(score=0.85)  # effectively instant
        scorer = FMTraversabilityScorer(
            llm=llm,
            mode=ScoringMode.FM_WITH_FALLBACK,
            latency_budget_ms=5000.0,
        )
        result = scorer.score_label("concrete")
        assert result.source == "fm"
        assert result.score == pytest.approx(0.85)


# ── score_region ──────────────────────────────────────────────────────────────

class TestScoreRegion:
    def test_score_region_uses_region_label(self):
        llm = _make_llm(score=0.6)
        scorer = FMTraversabilityScorer(llm=llm, mode=ScoringMode.FM)
        region = _make_region("vegetation", pixel_fraction=0.15)
        result = scorer.score_region(region)
        assert result.score == pytest.approx(0.6)

    def test_score_region_includes_pixel_fraction_in_context(self):
        llm = _make_llm()
        scorer = FMTraversabilityScorer(llm=llm, mode=ScoringMode.FM)
        region = _make_region("sand", pixel_fraction=0.30)
        scorer.score_region(region)
        prompt = llm.predict_json.call_args[0][0]
        assert "30.0%" in prompt


# ── score_batch ───────────────────────────────────────────────────────────────

class TestScoreBatch:
    def test_batch_deduplicates_labels(self):
        llm = _make_llm(score=0.7)
        scorer = FMTraversabilityScorer(llm=llm, mode=ScoringMode.FM)
        regions = [
            _make_region("grass"),
            _make_region("grass"),
            _make_region("mud"),
        ]
        results = scorer.score_batch(regions)
        assert len(results) == 3
        # Only 2 unique labels → at most 2 LLM calls (grass + mud)
        assert llm.predict_json.call_count <= 2

    def test_batch_returns_one_result_per_region(self):
        scorer = FMTraversabilityScorer(mode=ScoringMode.STATIC)
        regions = [_make_region("road"), _make_region("dirt"), _make_region("water")]
        results = scorer.score_batch(regions)
        assert len(results) == 3

    def test_batch_scores_match_individual_scores(self):
        scorer = FMTraversabilityScorer(mode=ScoringMode.STATIC)
        labels = ["road", "grass", "mud"]
        regions = [_make_region(l) for l in labels]
        batch = scorer.score_batch(regions)
        individual = [scorer.score_label(l) for l in labels]
        for b, ind in zip(batch, individual):
            assert b.score == pytest.approx(ind.score)


# ── Known-class sanity checks ─────────────────────────────────────────────────

class TestKnownClassSanity:
    def test_safe_terrain_static_score_above_threshold(self):
        scorer = FMTraversabilityScorer(mode=ScoringMode.STATIC)
        for label in ("road", "concrete", "sidewalk", "grass"):
            j = scorer.score_label(label)
            assert j.score >= 0.70, f"{label}: expected score >= 0.70, got {j.score}"

    def test_dangerous_terrain_static_score_below_threshold(self):
        scorer = FMTraversabilityScorer(mode=ScoringMode.STATIC)
        for label in ("water", "puddle", "mud", "person"):
            j = scorer.score_label(label)
            assert j.score <= 0.20, f"{label}: expected score <= 0.20, got {j.score}"

    def test_unknown_label_always_returns_zero(self):
        scorer = FMTraversabilityScorer(mode=ScoringMode.STATIC)
        result = scorer.score_label("unknown")
        assert result.score == 0.0
