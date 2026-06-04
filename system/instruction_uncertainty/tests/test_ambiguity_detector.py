"""
Tests for AmbiguityDetector and supporting utilities.

Covers:
  - Constructor validation (FM/FM_WITH_FALLBACK require LLM; RULE does not)
  - RULE mode: each of 6 ambiguity types + no_uncertainty from crafted instructions
  - RULE mode result structure: source, latency, slots, p_ambiguous, score
  - FM mode: LLM called, response parsed, fields set correctly
  - FM mode: p_ambiguous clamped to [0, 1]; unknown type mapped to no_uncertainty
  - FM mode: prompt includes instruction and scene_context text
  - Cache: repeated call returns cache hit; different context → new entry
  - Cache: clear_cache forces re-query; cache_size tracks entries
  - FM_WITH_FALLBACK: exception falls back to rule; no exception returns FM result
  - _compute_nonconformity: severity * p for known types; 0.0 for no_uncertainty
  - nonconformity_score matches manual ambiguity_score calculation
"""

import pytest
from unittest.mock import MagicMock

from system.instruction_uncertainty.ambiguity_detector import (
    AmbiguityDetection,
    AmbiguityDetector,
    DetectionMode,
    _compute_nonconformity,
    _rule_detect,
)
from system.instruction_uncertainty.intent_memory import (
    AMBIGUITY_TYPES,
    SEVERITY_WEIGHTS,
    ambiguity_score,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_llm(
    ambiguity_type: str = "no_uncertainty",
    p_ambiguous: float = 0.0,
    missing_slots=None,
    reasoning: str = "Test reasoning.",
) -> MagicMock:
    llm = MagicMock()
    llm.predict_json.return_value = {
        "ambiguity_type": ambiguity_type,
        "p_ambiguous": p_ambiguous,
        "missing_slots": missing_slots or [],
        "reasoning": reasoning,
    }
    return llm


# ── Constructor validation ─────────────────────────────────────────────────────

class TestConstructor:
    def test_rule_mode_requires_no_llm(self):
        d = AmbiguityDetector(mode=DetectionMode.RULE)
        assert d is not None

    def test_fm_mode_requires_llm(self):
        with pytest.raises(ValueError, match="requires an LLMInterface"):
            AmbiguityDetector(mode=DetectionMode.FM)

    def test_fm_with_fallback_requires_llm(self):
        with pytest.raises(ValueError, match="requires an LLMInterface"):
            AmbiguityDetector(mode=DetectionMode.FM_WITH_FALLBACK)

    def test_fm_mode_with_llm_succeeds(self):
        d = AmbiguityDetector(llm=_make_llm(), mode=DetectionMode.FM)
        assert d is not None

    def test_cache_starts_empty(self):
        d = AmbiguityDetector(mode=DetectionMode.RULE)
        assert d.cache_size == 0


# ── RULE mode: one clear instruction per ambiguity type ───────────────────────

class TestRuleMode:
    def setup_method(self):
        self.d = AmbiguityDetector(mode=DetectionMode.RULE)

    def test_missing_action_no_verb(self):
        result = self.d.detect("The library please")
        assert result.ambiguity_type == "missing_action"

    def test_ambiguous_target_pronoun_destination(self):
        result = self.d.detect("Go there")
        assert result.ambiguity_type == "ambiguous_target"

    def test_ambiguous_target_that_way(self):
        result = self.d.detect("Head that way")
        assert result.ambiguity_type == "ambiguous_target"

    def test_missing_object_pronoun_object(self):
        result = self.d.detect("Pick it up")
        assert result.ambiguity_type == "missing_object"

    def test_ambiguous_action_vague_verb(self):
        result = self.d.detect("Handle the obstacle in the path")
        assert result.ambiguity_type == "ambiguous_action"

    def test_missing_direction_movement_no_destination(self):
        result = self.d.detect("Go now")
        assert result.ambiguity_type == "missing_direction"

    def test_missing_distance_vague_quantity(self):
        result = self.d.detect("Move a bit to the right")
        assert result.ambiguity_type == "missing_distance"

    def test_no_uncertainty_complete_instruction(self):
        result = self.d.detect(
            "Turn left at the intersection and walk 50 meters to the library"
        )
        assert result.ambiguity_type == "no_uncertainty"

    def test_no_uncertainty_specific_destination(self):
        result = self.d.detect("Go to the sports field")
        assert result.ambiguity_type == "no_uncertainty"

    def test_specific_distance_suppresses_missing_distance(self):
        # "a bit" AND "5 meters" — specific wins, should not be missing_distance
        result = self.d.detect("Move a bit, maybe 5 meters forward")
        assert result.ambiguity_type != "missing_distance"

    def test_named_location_suppresses_ambiguous_target(self):
        # "there" present but a named location also present
        result = self.d.detect("Go there toward the library entrance")
        assert result.ambiguity_type != "ambiguous_target"


# ── RULE mode: result structure ───────────────────────────────────────────────

class TestRuleModeStructure:
    def setup_method(self):
        self.d = AmbiguityDetector(mode=DetectionMode.RULE)

    def test_source_is_rule(self):
        result = self.d.detect("Turn left at the crosswalk")
        assert result.source == "rule"

    def test_latency_is_zero(self):
        result = self.d.detect("Go to the park")
        assert result.latency_ms == 0.0

    def test_missing_slots_is_list(self):
        result = self.d.detect("Go there")
        assert isinstance(result.missing_slots, list)

    def test_p_ambiguous_in_range(self):
        for instr in [
            "The library", "Go there", "Move a bit forward",
            "Handle it", "Go now", "Turn left at the park",
        ]:
            r = self.d.detect(instr)
            assert 0.0 <= r.p_ambiguous <= 1.0, f"p_ambiguous out of range for: {instr}"

    def test_no_uncertainty_p_ambiguous_is_zero(self):
        result = self.d.detect("Go to the library")
        assert result.p_ambiguous == 0.0

    def test_reasoning_is_nonempty_string(self):
        result = self.d.detect("Go there")
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    def test_result_is_ambiguity_detection_instance(self):
        result = self.d.detect("Walk forward")
        assert isinstance(result, AmbiguityDetection)


# ── RULE mode: nonconformity score ────────────────────────────────────────────

class TestNonconformityScore:
    def setup_method(self):
        self.d = AmbiguityDetector(mode=DetectionMode.RULE)

    def test_no_uncertainty_score_is_zero(self):
        result = self.d.detect("Turn right and go to the cafeteria")
        assert result.nonconformity_score == 0.0

    def test_score_equals_severity_times_p_ambiguous(self):
        result = self.d.detect("Go there")
        expected = SEVERITY_WEIGHTS[result.ambiguity_type] * result.p_ambiguous
        assert result.nonconformity_score == pytest.approx(expected)

    def test_missing_action_has_highest_possible_score(self):
        result_ma = self.d.detect("The library")       # missing_action, w=1.0
        result_md = self.d.detect("Move a bit left")   # missing_distance, w=0.25
        assert result_ma.nonconformity_score > result_md.nonconformity_score

    def test_score_in_01_range(self):
        for instr in ["The park", "Go there", "Handle it", "Move a bit"]:
            r = self.d.detect(instr)
            assert 0.0 <= r.nonconformity_score <= 1.0

    def test_cooccurrence_boost_raises_p_ambiguous(self):
        # "Go there" fires both ambiguous_target (p=0.80) and missing_direction.
        # Co-occurrence boost: +0.05 → p_ambiguous = 0.85 > 0.80 (single-rule baseline).
        result = self.d.detect("Go there")
        assert result.ambiguity_type == "ambiguous_target"
        assert result.p_ambiguous > 0.80

    def test_single_fired_rule_has_no_boost(self):
        # "Pick it up" fires only missing_object — no co-occurrence boost.
        result = self.d.detect("Pick it up")
        assert result.ambiguity_type == "missing_object"
        assert result.p_ambiguous == pytest.approx(0.75)

    def test_cooccurrence_boost_capped_at_095(self):
        # Even with many co-occurring signals, p_ambiguous must not exceed 0.95.
        for instr in ["Go there a bit further", "Go there"]:
            r = self.d.detect(instr)
            assert r.p_ambiguous <= 0.95

    def test_existential_vague_boosts_above_directional_pronoun(self):
        # "somewhere" triggers the _EXISTENTIAL_VAGUE intra-type booster;
        # "there" does not — so "Head somewhere" should score higher than "Head there".
        r_somewhere = self.d.detect("Head somewhere")
        r_there = self.d.detect("Head there")
        assert r_somewhere.ambiguity_type == "ambiguous_target"
        assert r_there.ambiguity_type == "ambiguous_target"
        assert r_somewhere.p_ambiguous > r_there.p_ambiguous

    def test_multiple_vague_distance_phrases_boost_score(self):
        # Two distinct vague distance matches trigger the intra-type booster → p > 0.70.
        result = self.d.detect("Move a bit further, just a little ways ahead")
        assert result.ambiguity_type == "missing_distance"
        assert result.p_ambiguous > 0.70

    def test_bare_movement_command_boosts_missing_direction(self):
        # Single-token movement command is maximally underdetermined;
        # terse-command intra-type booster should raise p above the base 0.70.
        result = self.d.detect("Go")
        assert result.ambiguity_type == "missing_direction"
        assert result.p_ambiguous > 0.70

    def test_per_type_scores_computed_independently(self):
        # "Handle it a bit further" fires ambiguous_action (w=0.50, p=0.75, κ=0.375)
        # and missing_distance (w=0.25, p=0.70, κ=0.175); ambiguous_action wins.
        result = self.d.detect("Handle it a bit further")
        assert result.ambiguity_type == "ambiguous_action"


# ── _compute_nonconformity module-level helper ────────────────────────────────

class TestComputeNonconformity:
    def test_no_uncertainty_returns_zero(self):
        assert _compute_nonconformity("no_uncertainty", 0.9) == 0.0

    def test_missing_action_full_probability(self):
        assert _compute_nonconformity("missing_action", 1.0) == pytest.approx(1.0)

    def test_ambiguous_target_half_probability(self):
        # w=0.75, p=0.5 → 0.375
        assert _compute_nonconformity("ambiguous_target", 0.5) == pytest.approx(0.375)

    def test_missing_distance_low_weight(self):
        # w=0.25, p=1.0 → 0.25
        assert _compute_nonconformity("missing_distance", 1.0) == pytest.approx(0.25)

    def test_all_ambiguity_types_produce_nonzero_score(self):
        for atype in AMBIGUITY_TYPES:
            assert _compute_nonconformity(atype, 1.0) > 0.0


# ── FM mode ───────────────────────────────────────────────────────────────────

class TestFMMode:
    def test_fm_mode_calls_llm(self):
        llm = _make_llm("ambiguous_target", 0.9, ["target"], "Pronoun with no referent.")
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM)
        d.detect("Go there", "bench and gate visible")
        llm.predict_json.assert_called_once()

    def test_fm_returns_type_from_llm(self):
        llm = _make_llm("missing_direction", 0.85, ["direction"])
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM)
        result = d.detect("Go now")
        assert result.ambiguity_type == "missing_direction"

    def test_fm_returns_p_ambiguous_from_llm(self):
        llm = _make_llm("missing_action", 0.95)
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM)
        result = d.detect("The library")
        assert result.p_ambiguous == pytest.approx(0.95)

    def test_fm_p_ambiguous_clamped_above_one(self):
        llm = _make_llm("missing_action", 1.5)
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM)
        result = d.detect("Some instruction")
        assert result.p_ambiguous <= 1.0

    def test_fm_p_ambiguous_clamped_below_zero(self):
        llm = _make_llm("no_uncertainty", -0.3)
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM)
        result = d.detect("Go to the library")
        assert result.p_ambiguous >= 0.0

    def test_fm_unknown_type_mapped_to_no_uncertainty(self):
        llm = _make_llm("totally_invented_type", 0.9)
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM)
        result = d.detect("Some instruction")
        assert result.ambiguity_type == "no_uncertainty"

    def test_fm_missing_slots_parsed(self):
        llm = _make_llm("missing_object", 0.8, ["object", "destination"])
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM)
        result = d.detect("Pick it up")
        assert "object" in result.missing_slots

    def test_fm_reasoning_parsed(self):
        llm = _make_llm("missing_action", 0.9, [], "No verb found in instruction.")
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM)
        result = d.detect("The park")
        assert "No verb" in result.reasoning

    def test_fm_source_is_fm(self):
        llm = _make_llm()
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM)
        result = d.detect("Go to the cafeteria")
        assert result.source == "fm"

    def test_fm_prompt_includes_instruction(self):
        llm = _make_llm()
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM)
        d.detect("Navigate to the fountain", "sunny park environment")
        call_prompt = llm.predict_json.call_args[0][0]
        assert "Navigate to the fountain" in call_prompt
        assert "sunny park environment" in call_prompt

    def test_fm_nonconformity_score_computed_from_parsed_values(self):
        llm = _make_llm("ambiguous_action", 0.6)
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM)
        result = d.detect("Handle it")
        expected = SEVERITY_WEIGHTS["ambiguous_action"] * 0.6
        assert result.nonconformity_score == pytest.approx(expected)


# ── Cache behavior ────────────────────────────────────────────────────────────

class TestCache:
    def test_repeated_call_uses_cache(self):
        llm = _make_llm("missing_action", 0.9)
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM)
        d.detect("The library")
        d.detect("The library")
        assert llm.predict_json.call_count == 1

    def test_cache_hit_source_is_cache(self):
        llm = _make_llm("missing_action", 0.9)
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM)
        d.detect("The library")
        result2 = d.detect("The library")
        assert result2.source == "cache"

    def test_different_context_new_cache_entry(self):
        llm = _make_llm()
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM)
        d.detect("Go to the library", "morning")
        d.detect("Go to the library", "evening")
        assert llm.predict_json.call_count == 2

    def test_clear_cache_forces_re_query(self):
        llm = _make_llm()
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM)
        d.detect("Go to the park")
        d.clear_cache()
        d.detect("Go to the park")
        assert llm.predict_json.call_count == 2

    def test_cache_size_tracks_unique_queries(self):
        llm = _make_llm()
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM)
        d.detect("Go to the library", "context A")
        d.detect("Turn left at the crosswalk", "context A")
        assert d.cache_size == 2

    def test_cache_hit_preserves_original_values(self):
        llm = _make_llm("missing_direction", 0.77, ["direction"])
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM)
        first = d.detect("Go now")
        second = d.detect("Go now")
        assert second.ambiguity_type == first.ambiguity_type
        assert second.p_ambiguous == pytest.approx(first.p_ambiguous)
        assert second.nonconformity_score == pytest.approx(first.nonconformity_score)

    def test_rule_mode_does_not_use_cache(self):
        d = AmbiguityDetector(mode=DetectionMode.RULE)
        d.detect("Go there")
        d.detect("Go there")
        assert d.cache_size == 0


# ── FM_WITH_FALLBACK ──────────────────────────────────────────────────────────

class TestFMWithFallback:
    def test_exception_falls_back_to_rule(self):
        llm = MagicMock()
        llm.predict_json.side_effect = RuntimeError("API error")
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM_WITH_FALLBACK)
        result = d.detect("Go there")
        assert result.source == "rule"

    def test_no_exception_returns_fm_result(self):
        llm = _make_llm("ambiguous_target", 0.88)
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM_WITH_FALLBACK)
        result = d.detect("Go there")
        assert result.source == "fm"

    def test_fallback_result_has_valid_type(self):
        llm = MagicMock()
        llm.predict_json.side_effect = ValueError("timeout")
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM_WITH_FALLBACK)
        result = d.detect("Go now")
        assert result.ambiguity_type in AMBIGUITY_TYPES or result.ambiguity_type == "no_uncertainty"

    def test_fm_exception_does_not_populate_cache(self):
        llm = MagicMock()
        llm.predict_json.side_effect = RuntimeError("API error")
        d = AmbiguityDetector(llm=llm, mode=DetectionMode.FM_WITH_FALLBACK)
        d.detect("Go there")
        assert d.cache_size == 0
