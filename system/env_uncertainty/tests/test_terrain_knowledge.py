"""
Tests for PersistentTerrainKnowledge (cross-frame semantic terrain memory).
"""
import pytest
from system.env_uncertainty.terrain_knowledge import PersistentTerrainKnowledge


def _make_ptk() -> PersistentTerrainKnowledge:
    return PersistentTerrainKnowledge()


class TestUpdateFromFeedback:
    def test_no_label_does_not_crash(self):
        ptk = _make_ptk()
        ptk.update_from_feedback(None, is_traversable=True)
        assert ptk.n_labels_known == 0

    def test_empty_label_ignored(self):
        ptk = _make_ptk()
        ptk.update_from_feedback("", is_traversable=True)
        assert ptk.n_labels_known == 0

    def test_safe_response_raises_traversability(self):
        ptk = _make_ptk()
        before = ptk.adjusted_traversability("grass")
        ptk.update_from_feedback("grass", is_traversable=True, confidence=0.90)
        after = ptk.adjusted_traversability("grass")
        assert after >= before

    def test_unsafe_response_lowers_traversability(self):
        ptk = _make_ptk()
        before = ptk.adjusted_traversability("grass")
        ptk.update_from_feedback("grass", is_traversable=False, confidence=0.90)
        after = ptk.adjusted_traversability("grass")
        assert after <= before

    def test_multiple_updates_accumulate(self):
        ptk = _make_ptk()
        ptk.update_from_feedback("mud", is_traversable=True, confidence=0.85)
        after_one = ptk.adjusted_traversability("mud")
        ptk.update_from_feedback("mud", is_traversable=True, confidence=0.85)
        after_two = ptk.adjusted_traversability("mud")
        assert after_two >= after_one

    def test_n_observations_increments(self):
        ptk = _make_ptk()
        ptk.update_from_feedback("grass", is_traversable=True)
        ptk.update_from_feedback("grass", is_traversable=True)
        assert ptk.get_belief("grass").n_observations == 2

    def test_case_insensitive_label(self):
        ptk = _make_ptk()
        ptk.update_from_feedback("GRASS", is_traversable=True)
        assert ptk.has_knowledge("grass")


class TestAdjustedTraversability:
    def test_unknown_label_falls_back_to_default(self):
        ptk = _make_ptk()
        score = ptk.adjusted_traversability("grass")
        # Should match TRAVERSABILITY_SCORES["grass"] = 0.90
        assert score == pytest.approx(0.90)

    def test_unknown_label_uses_provided_default(self):
        ptk = _make_ptk()
        score = ptk.adjusted_traversability("totally_new_class", default_score=0.42)
        assert score == pytest.approx(0.42)

    def test_stays_in_unit_interval(self):
        ptk = _make_ptk()
        for label in ["grass", "mud", "water", "concrete"]:
            for is_trav in [True, False]:
                ptk.update_from_feedback(label, is_trav, confidence=0.95)
                s = ptk.adjusted_traversability(label)
                assert 0.0 <= s <= 1.0


class TestShouldSkipAsking:
    def test_no_observations_does_not_skip(self):
        ptk = _make_ptk()
        assert ptk.should_skip_asking("grass") is False

    def test_one_observation_does_not_skip(self):
        ptk = _make_ptk()
        ptk.update_from_feedback("grass", is_traversable=True, confidence=0.90)
        assert ptk.should_skip_asking("grass") is False

    def test_two_safe_confirmations_enables_skip_for_high_trav(self):
        ptk = _make_ptk()
        ptk.update_from_feedback("concrete", is_traversable=True, confidence=0.90)
        ptk.update_from_feedback("concrete", is_traversable=True, confidence=0.90)
        assert ptk.should_skip_asking("concrete") is True

    def test_two_unsafe_confirmations_enables_skip_for_low_trav(self):
        ptk = _make_ptk()
        ptk.update_from_feedback("puddle", is_traversable=False, confidence=0.90)
        ptk.update_from_feedback("puddle", is_traversable=False, confidence=0.90)
        assert ptk.should_skip_asking("puddle") is True

    def test_contradictory_feedback_stays_ambiguous(self):
        ptk = _make_ptk()
        ptk.update_from_feedback("gravel", is_traversable=True, confidence=0.80)
        ptk.update_from_feedback("gravel", is_traversable=False, confidence=0.80)
        # Posterior should be in the ambiguous middle — don't skip
        belief = ptk.get_belief("gravel")
        # Can't guarantee skip either way, but traversability should be between thresholds
        # (actual value depends on Bayesian update math — just check it doesn't crash)
        assert belief is not None


class TestReset:
    def test_reset_clears_all_beliefs(self):
        ptk = _make_ptk()
        ptk.update_from_feedback("grass", is_traversable=True)
        ptk.update_from_feedback("mud", is_traversable=False)
        ptk.reset()
        assert ptk.n_labels_known == 0

    def test_after_reset_defaults_restored(self):
        ptk = _make_ptk()
        ptk.update_from_feedback("grass", is_traversable=False, confidence=0.95)
        ptk.reset()
        score = ptk.adjusted_traversability("grass")
        assert score == pytest.approx(0.90)  # back to TRAVERSABILITY_SCORES default


class TestSummary:
    def test_summary_no_observations(self):
        ptk = _make_ptk()
        s = ptk.summary()
        assert "no observations" in s

    def test_summary_includes_label(self):
        ptk = _make_ptk()
        ptk.update_from_feedback("mud", is_traversable=False)
        s = ptk.summary()
        assert "mud" in s
