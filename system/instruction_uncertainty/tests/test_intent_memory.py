"""
Tests for IntentMemory and supporting utilities.

Covers:
  - First observation stores entry with initial_confidence
  - Second confirmation raises confidence above reuse_threshold (0.85)
  - Contradicting answer lowers confidence on old entry, creates new one
  - should_skip_asking returns False below threshold, True above it
  - recall returns None for unknown context
  - Stale entries are excluded when max_age_seconds is set
  - purge_stale removes old entries and returns correct count
  - clear removes all entries
  - total_entries counts correctly across types
  - ambiguity_score formula: u_I = w_type * p_ambiguous
  - ambiguity_score raises on unknown type
  - _bayesian_update numerical accuracy
  - Invalid constructor arguments raise ValueError
"""

import time

import pytest

from system.instruction_uncertainty.intent_memory import (
    AMBIGUITY_TYPES,
    LIKELIHOOD_CONFIRM,
    LIKELIHOOD_MISMATCH,
    SEVERITY_WEIGHTS,
    IntentEntry,
    IntentMemory,
    _bayesian_update,
    _context_hash,
    ambiguity_score,
)


# ── _bayesian_update numerical checks ────────────────────────────────────────

class TestBayesianUpdate:
    def test_confirmation_raises_confidence_from_half(self):
        # τ_0=0.5, confirmed → τ_1 = 0.95*0.5 / (0.95*0.5 + 0.10*0.5) = 0.9048...
        result = _bayesian_update(0.5, confirmed=True)
        assert abs(result - (LIKELIHOOD_CONFIRM * 0.5) / (
            LIKELIHOOD_CONFIRM * 0.5 + LIKELIHOOD_MISMATCH * 0.5
        )) < 1e-6

    def test_contradiction_lowers_confidence_from_half(self):
        result = _bayesian_update(0.5, confirmed=False)
        expected = (LIKELIHOOD_MISMATCH * 0.5) / (
            LIKELIHOOD_MISMATCH * 0.5 + LIKELIHOOD_CONFIRM * 0.5
        )
        assert abs(result - expected) < 1e-6

    def test_two_confirmations_from_half_exceed_reuse_threshold(self):
        tau = _bayesian_update(0.5, confirmed=True)
        tau = _bayesian_update(tau, confirmed=True)
        assert tau >= 0.85

    def test_result_clamped_to_01(self):
        assert _bayesian_update(0.0, confirmed=True) >= 0.01
        assert _bayesian_update(1.0, confirmed=False) <= 0.99

    def test_high_prior_further_confirmed_stays_below_1(self):
        result = _bayesian_update(0.98, confirmed=True)
        assert result < 1.0


# ── _context_hash ─────────────────────────────────────────────────────────────

class TestContextHash:
    def test_same_inputs_produce_same_hash(self):
        h1 = _context_hash("ambiguous_target", "bench gate sign")
        h2 = _context_hash("ambiguous_target", "bench gate sign")
        assert h1 == h2

    def test_different_types_produce_different_hashes(self):
        h1 = _context_hash("missing_direction", "bench gate sign")
        h2 = _context_hash("ambiguous_target", "bench gate sign")
        assert h1 != h2

    def test_case_insensitive(self):
        h1 = _context_hash("AMBIGUOUS_TARGET", "BENCH GATE SIGN")
        h2 = _context_hash("ambiguous_target", "bench gate sign")
        assert h1 == h2

    def test_hash_length_is_16(self):
        h = _context_hash("missing_action", "clear path")
        assert len(h) == 16


# ── ambiguity_score ───────────────────────────────────────────────────────────

class TestAmbiguityScore:
    def test_missing_action_full_probability(self):
        score = ambiguity_score("missing_action", 1.0)
        assert score == pytest.approx(1.00)

    def test_ambiguous_target_half_probability(self):
        score = ambiguity_score("ambiguous_target", 0.5)
        assert score == pytest.approx(0.375)

    def test_missing_distance_weight(self):
        score = ambiguity_score("missing_distance", 1.0)
        assert score == pytest.approx(0.25)

    def test_all_types_produce_valid_scores(self):
        for t in AMBIGUITY_TYPES:
            s = ambiguity_score(t, 0.8)
            assert 0.0 <= s <= 1.0

    def test_zero_probability_gives_zero_score(self):
        assert ambiguity_score("missing_action", 0.0) == 0.0

    def test_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown ambiguity type"):
            ambiguity_score("nonsense_type", 0.5)

    def test_p_clamped_above_1(self):
        score = ambiguity_score("missing_action", 1.5)
        assert score == pytest.approx(SEVERITY_WEIGHTS["missing_action"])


# ── IntentMemory construction ─────────────────────────────────────────────────

class TestIntentMemoryConstruction:
    def test_default_construction(self):
        mem = IntentMemory()
        assert mem.total_entries == 0

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            IntentMemory(reuse_threshold=0.0)

    def test_invalid_initial_confidence_raises(self):
        with pytest.raises(ValueError):
            IntentMemory(initial_confidence=0.0)


# ── update — first observation ────────────────────────────────────────────────

class TestIntentMemoryFirstObservation:
    def test_first_update_creates_entry(self):
        mem = IntentMemory(initial_confidence=0.75)
        entry = mem.update("ambiguous_target", "bench gate sign", "go to the bench")
        assert isinstance(entry, IntentEntry)
        assert entry.resolved_answer == "go to the bench"
        assert entry.n_observations == 1

    def test_first_confidence_is_initial(self):
        mem = IntentMemory(initial_confidence=0.75)
        entry = mem.update("ambiguous_target", "bench gate sign", "go to the bench")
        assert entry.confidence == pytest.approx(0.75)

    def test_first_observation_below_reuse_threshold(self):
        mem = IntentMemory(reuse_threshold=0.85, initial_confidence=0.75)
        mem.update("ambiguous_target", "bench gate sign", "go to the bench")
        skip, _ = mem.should_skip_asking("ambiguous_target", "bench gate sign")
        assert skip is False

    def test_total_entries_increases(self):
        mem = IntentMemory()
        mem.update("missing_direction", "junction A B C", "go left")
        assert mem.total_entries == 1


# ── update — confirmation (same answer) ──────────────────────────────────────

class TestIntentMemoryConfirmation:
    def test_second_confirmation_raises_confidence(self):
        mem = IntentMemory(initial_confidence=0.75)
        mem.update("ambiguous_target", "bench gate", "bench")
        entry = mem.update("ambiguous_target", "bench gate", "bench")
        assert entry.confidence > 0.75

    def test_two_confirmations_from_default_exceed_threshold(self):
        mem = IntentMemory(reuse_threshold=0.85, initial_confidence=0.75)
        mem.update("ambiguous_target", "bench gate", "bench")
        mem.update("ambiguous_target", "bench gate", "bench")
        skip, answer = mem.should_skip_asking("ambiguous_target", "bench gate")
        assert skip is True
        assert answer == "bench"

    def test_n_observations_increments(self):
        mem = IntentMemory()
        mem.update("missing_direction", "junction", "left")
        entry = mem.update("missing_direction", "junction", "left")
        assert entry.n_observations == 2

    def test_no_duplicate_entry_on_confirmation(self):
        mem = IntentMemory()
        mem.update("missing_direction", "junction", "left")
        mem.update("missing_direction", "junction", "left")
        assert mem.total_entries == 1


# ── update — contradiction (different answer) ─────────────────────────────────

class TestIntentMemoryContradiction:
    def test_different_answer_creates_new_entry(self):
        mem = IntentMemory()
        mem.update("ambiguous_target", "bench gate", "bench")
        mem.update("ambiguous_target", "bench gate", "gate")
        assert mem.total_entries == 2

    def test_contradiction_lowers_old_entry_confidence(self):
        mem = IntentMemory(initial_confidence=0.75)
        old = mem.update("ambiguous_target", "bench gate", "bench")
        old_conf = old.confidence
        mem.update("ambiguous_target", "bench gate", "gate")
        assert old.confidence < old_conf

    def test_new_entry_after_contradiction_has_initial_confidence(self):
        mem = IntentMemory(initial_confidence=0.75)
        mem.update("ambiguous_target", "bench gate", "bench")
        new = mem.update("ambiguous_target", "bench gate", "gate")
        assert new.confidence == pytest.approx(0.75)


# ── recall ────────────────────────────────────────────────────────────────────

class TestIntentMemoryRecall:
    def test_recall_returns_none_for_unknown_context(self):
        mem = IntentMemory()
        assert mem.recall("ambiguous_target", "some scene") is None

    def test_recall_returns_answer_and_confidence(self):
        mem = IntentMemory()
        mem.update("missing_direction", "junction A B", "left")
        result = mem.recall("missing_direction", "junction A B")
        assert result is not None
        answer, confidence = result
        assert answer == "left"
        assert 0.0 < confidence <= 1.0

    def test_recall_returns_highest_confidence_after_contradiction(self):
        mem = IntentMemory(initial_confidence=0.75)
        mem.update("ambiguous_target", "bench gate", "bench")
        mem.update("ambiguous_target", "bench gate", "bench")  # confirm bench
        mem.update("ambiguous_target", "bench gate", "gate")   # new answer
        answer, confidence = mem.recall("ambiguous_target", "bench gate")
        # bench has been confirmed twice then contradicted; gate is new
        # bench confidence after 2 confirms then contradiction:
        # ~0.957 → contradiction → ~0.167
        # gate was just created at 0.75
        assert answer == "gate"

    def test_recall_ignores_different_instruction_type(self):
        mem = IntentMemory()
        mem.update("missing_direction", "junction", "left")
        result = mem.recall("ambiguous_target", "junction")
        assert result is None


# ── should_skip_asking ────────────────────────────────────────────────────────

class TestShouldSkipAsking:
    def test_no_memory_returns_false_none(self):
        mem = IntentMemory()
        skip, answer = mem.should_skip_asking("missing_action", "scene")
        assert skip is False
        assert answer is None

    def test_below_threshold_returns_false(self):
        mem = IntentMemory(reuse_threshold=0.90, initial_confidence=0.75)
        mem.update("missing_action", "scene", "go forward")
        skip, _ = mem.should_skip_asking("missing_action", "scene")
        assert skip is False

    def test_above_threshold_returns_true_with_answer(self):
        mem = IntentMemory(reuse_threshold=0.85, initial_confidence=0.75)
        mem.update("ambiguous_target", "bench gate", "bench")
        mem.update("ambiguous_target", "bench gate", "bench")
        skip, answer = mem.should_skip_asking("ambiguous_target", "bench gate")
        assert skip is True
        assert answer == "bench"

    def test_custom_high_threshold_requires_more_confirmations(self):
        mem = IntentMemory(reuse_threshold=0.99, initial_confidence=0.75)
        mem.update("ambiguous_target", "scene", "bench")
        mem.update("ambiguous_target", "scene", "bench")
        skip, _ = mem.should_skip_asking("ambiguous_target", "scene")
        # Two confirmations from 0.75 → ~0.957 < 0.99
        assert skip is False


# ── staleness and expiry ──────────────────────────────────────────────────────

class TestIntentMemoryStaleness:
    def test_stale_entry_not_returned_by_recall(self, monkeypatch):
        mem = IntentMemory(max_age_seconds=1.0)
        mem.update("missing_direction", "junction", "left")
        future = time.time() + 10.0
        monkeypatch.setattr(
            "system.instruction_uncertainty.intent_memory.time.time",
            lambda: future,
        )
        result = mem.recall("missing_direction", "junction")
        assert result is None

    def test_stale_entry_skipped_by_should_skip(self, monkeypatch):
        mem = IntentMemory(max_age_seconds=1.0, initial_confidence=0.75)
        mem.update("ambiguous_target", "bench gate", "bench")
        mem.update("ambiguous_target", "bench gate", "bench")
        future = time.time() + 10.0
        monkeypatch.setattr(
            "system.instruction_uncertainty.intent_memory.time.time",
            lambda: future,
        )
        skip, _ = mem.should_skip_asking("ambiguous_target", "bench gate")
        assert skip is False

    def test_purge_stale_removes_old_entries(self, monkeypatch):
        mem = IntentMemory(max_age_seconds=1.0)
        mem.update("missing_direction", "junction", "left")
        assert mem.total_entries == 1
        future = time.time() + 10.0
        monkeypatch.setattr(
            "system.instruction_uncertainty.intent_memory.time.time",
            lambda: future,
        )
        removed = mem.purge_stale()
        assert removed == 1
        assert mem.total_entries == 0

    def test_purge_stale_no_max_age_removes_nothing(self):
        mem = IntentMemory(max_age_seconds=None)
        mem.update("missing_direction", "junction", "left")
        removed = mem.purge_stale()
        assert removed == 0


# ── clear and counts ──────────────────────────────────────────────────────────

class TestIntentMemoryClear:
    def test_clear_removes_all_entries(self):
        mem = IntentMemory()
        mem.update("missing_direction", "junction A", "left")
        mem.update("ambiguous_target", "bench gate", "bench")
        mem.clear()
        assert mem.total_entries == 0

    def test_total_entries_counts_across_types(self):
        mem = IntentMemory()
        mem.update("missing_direction", "scene1", "left")
        mem.update("ambiguous_target", "scene2", "bench")
        mem.update("missing_distance", "scene3", "10 meters")
        assert mem.total_entries == 3

    def test_entries_independent_across_instruction_types(self):
        mem = IntentMemory(initial_confidence=0.75)
        mem.update("missing_direction", "same scene", "left")
        mem.update("ambiguous_target", "same scene", "bench")
        # Each type's entries are independent
        skip_dir, _ = mem.should_skip_asking("missing_direction", "same scene")
        skip_tgt, _ = mem.should_skip_asking("ambiguous_target", "same scene")
        assert skip_dir is False  # only one observation
        assert skip_tgt is False


# ── AMBIGUITY_TYPES and SEVERITY_WEIGHTS completeness ─────────────────────────

class TestTypeRegistry:
    def test_all_six_types_present(self):
        assert len(AMBIGUITY_TYPES) == 6

    def test_severity_weights_cover_all_types(self):
        assert set(SEVERITY_WEIGHTS.keys()) == AMBIGUITY_TYPES

    def test_severity_weights_in_valid_range(self):
        for t, w in SEVERITY_WEIGHTS.items():
            assert 0.0 < w <= 1.0, f"Weight for '{t}' out of range: {w}"
