"""
Unit tests for user-profile-aware question generation.

Tests cover:
  - All verbosity levels produce distinct output
  - Expertise level affects prompt when using LLM mode
  - option_list format produces numbered choices
  - UserProfile validation rejects bad values
  - UserProfileStore returns DEFAULT_PROFILE for unknown IDs
  - describe_profile_for_prompt includes all profile fields
  - LLM prompt includes the user profile section
  - Scenario context propagates into the LLM prompt
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.question_generator import (
    QuestionGenerator,
    generate_question_template,
)
from system.env_uncertainty.traversability import TraversabilityMap
from system.env_uncertainty.user_profile import (
    DEFAULT_PROFILE,
    UserProfile,
    UserProfileStore,
    describe_profile_for_prompt,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_mask(h: int = 100, w: int = 100, fill: bool = True) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    if fill:
        m[40:60, 40:60] = True
    return m


def _make_region(label: str = "unknown", frac: float = 0.10) -> RegionInfo:
    return RegionInfo(
        label=label,
        mask=_make_mask(),
        confidence=0.85,
        pixel_fraction=frac,
        source="sam2",
        traversability=0.0,
    )


def _make_result(
    n_unknown: int = 1,
    unknown_coverage: float = 0.15,
) -> DetectionResult:
    h, w = 100, 100
    tmap = TraversabilityMap.create(h, w)
    unknown_regions = [_make_region() for _ in range(n_unknown)]
    return DetectionResult(
        known_regions=[],
        unknown_regions=unknown_regions,
        image_shape=(h, w),
        sam3_coverage=0.5,
        unknown_coverage=unknown_coverage,
        has_unknown=True,
        traversability_map=tmap,
    )


def _make_llm(question: str = "Is it safe to proceed?") -> MagicMock:
    llm = MagicMock()
    llm.predict_json.return_value = {"question": question}
    return llm


# ── UserProfile validation ────────────────────────────────────────────────────

class TestUserProfileValidation:
    def test_valid_profile_creates_successfully(self):
        p = UserProfile("u1", verbosity="terse", expertise="expert",
                        preferred_format="question")
        assert p.verbosity == "terse"

    def test_invalid_verbosity_raises(self):
        with pytest.raises(ValueError, match="verbosity"):
            UserProfile("u1", verbosity="whisper")

    def test_invalid_expertise_raises(self):
        with pytest.raises(ValueError, match="expertise"):
            UserProfile("u1", expertise="genius")

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="preferred_format"):
            UserProfile("u1", preferred_format="emoji")

    def test_profile_is_immutable(self):
        p = UserProfile("u1")
        with pytest.raises((AttributeError, TypeError)):
            p.verbosity = "verbose"  # type: ignore[misc]


# ── UserProfileStore ──────────────────────────────────────────────────────────

class TestUserProfileStore:
    def test_registered_profile_is_retrievable(self):
        store = UserProfileStore()
        p = UserProfile("alice", verbosity="verbose", expertise="expert")
        store.register(p)
        assert store.get("alice") is p

    def test_unknown_id_returns_default(self):
        store = UserProfileStore()
        result = store.get("nobody")
        assert result is DEFAULT_PROFILE

    def test_register_overwrites_existing(self):
        store = UserProfileStore()
        store.register(UserProfile("bob", verbosity="terse"))
        store.register(UserProfile("bob", verbosity="verbose"))
        assert store.get("bob").verbosity == "verbose"

    def test_remove_makes_id_return_default(self):
        store = UserProfileStore()
        store.register(UserProfile("carol"))
        store.remove("carol")
        assert store.get("carol") is DEFAULT_PROFILE

    def test_len_counts_registered_profiles(self):
        store = UserProfileStore()
        store.register(UserProfile("a"))
        store.register(UserProfile("b"))
        assert len(store) == 2


# ── describe_profile_for_prompt ───────────────────────────────────────────────

class TestDescribeProfileForPrompt:
    def test_includes_verbosity_description(self):
        p = UserProfile("u1", verbosity="terse")
        desc = describe_profile_for_prompt(p)
        assert "terse" in desc.lower() or "one sentence" in desc.lower()

    def test_includes_expertise_description(self):
        p = UserProfile("u1", expertise="novice")
        desc = describe_profile_for_prompt(p)
        assert "novice" in desc.lower() or "jargon" in desc.lower()

    def test_includes_format_description(self):
        p = UserProfile("u1", preferred_format="option_list")
        desc = describe_profile_for_prompt(p)
        assert "option" in desc.lower() or "numbered" in desc.lower()

    def test_includes_name_when_present(self):
        p = UserProfile("u1", name="Dr. Smith")
        desc = describe_profile_for_prompt(p)
        assert "Dr. Smith" in desc

    def test_no_name_no_name_line(self):
        p = UserProfile("u1")
        desc = describe_profile_for_prompt(p)
        assert "Name:" not in desc


# ── Template mode — verbosity ─────────────────────────────────────────────────

class TestTemplateVerbosity:
    def test_terse_shorter_than_standard(self):
        result = _make_result(n_unknown=1, unknown_coverage=0.20)
        terse = UserProfile("t", verbosity="terse")
        standard = UserProfile("s", verbosity="standard")
        q_terse = generate_question_template(result, profile=terse)
        q_standard = generate_question_template(result, profile=standard)
        assert len(q_terse) < len(q_standard)

    def test_verbose_longer_than_standard(self):
        result = _make_result(n_unknown=1, unknown_coverage=0.20)
        verbose = UserProfile("v", verbosity="verbose")
        standard = UserProfile("s", verbosity="standard")
        q_verbose = generate_question_template(result, profile=verbose)
        q_standard = generate_question_template(result, profile=standard)
        assert len(q_verbose) > len(q_standard)

    def test_all_three_verbosities_produce_different_output(self):
        result = _make_result(n_unknown=1, unknown_coverage=0.20)
        questions = {
            v: generate_question_template(
                result, profile=UserProfile("u", verbosity=v)
            )
            for v in ("terse", "standard", "verbose")
        }
        assert len(set(questions.values())) == 3

    def test_none_profile_uses_standard(self):
        result = _make_result()
        q_none = generate_question_template(result, profile=None)
        q_standard = generate_question_template(
            result, profile=UserProfile("s", verbosity="standard")
        )
        assert q_none == q_standard


# ── Template mode — format ────────────────────────────────────────────────────

class TestTemplateFormat:
    def test_option_list_contains_numbered_choices(self):
        result = _make_result()
        p = UserProfile("u", preferred_format="option_list")
        q = generate_question_template(result, profile=p)
        assert "1." in q or "1)" in q

    def test_option_list_overrides_verbosity(self):
        result = _make_result()
        p_terse = UserProfile("t", verbosity="terse", preferred_format="option_list")
        p_verbose = UserProfile("v", verbosity="verbose", preferred_format="option_list")
        q_terse = generate_question_template(result, profile=p_terse)
        q_verbose = generate_question_template(result, profile=p_verbose)
        # Both should produce option-list format, not verbosity-based templates
        assert "1." in q_terse or "1)" in q_terse
        assert "1." in q_verbose or "1)" in q_verbose


# ── Template mode — situation ─────────────────────────────────────────────────

class TestTemplateSituations:
    def test_large_unknown_coverage_triggers_stop_question(self):
        result = _make_result(unknown_coverage=0.60)
        q = generate_question_template(result)
        assert "stop" in q.lower() or "Stop" in q

    def test_multiple_unknown_regions(self):
        result = _make_result(n_unknown=4, unknown_coverage=0.30)
        q = generate_question_template(result)
        assert "multiple" in q.lower() or "path" in q.lower()


# ── QuestionGenerator — profile parameter ────────────────────────────────────

class TestQuestionGeneratorProfile:
    def test_template_mode_respects_verbosity(self):
        gen = QuestionGenerator(mode="template")
        result = _make_result()
        p_terse = UserProfile("t", verbosity="terse")
        p_verbose = UserProfile("v", verbosity="verbose")
        q_terse = gen.generate(result, user_profile=p_terse)
        q_verbose = gen.generate(result, user_profile=p_verbose)
        assert len(q_terse) < len(q_verbose)

    def test_default_profile_used_when_none_provided(self):
        gen = QuestionGenerator(mode="template")
        result = _make_result()
        q_none = gen.generate(result)
        q_default = gen.generate(result, user_profile=DEFAULT_PROFILE)
        assert q_none == q_default


# ── QuestionGenerator — LLM mode with profile ────────────────────────────────

class TestQuestionGeneratorLLMMode:
    def test_llm_prompt_includes_user_profile_section(self):
        llm = _make_llm()
        gen = QuestionGenerator(mode="llm", llm=llm)
        result = _make_result()
        p = UserProfile("u", verbosity="verbose", expertise="expert")
        gen.generate(result, user_profile=p)
        prompt = llm.predict_json.call_args[0][0]
        assert "verbose" in prompt.lower() or "full explanation" in prompt.lower()

    def test_llm_prompt_includes_scenario_context(self):
        llm = _make_llm()
        gen = QuestionGenerator(mode="llm", llm=llm)
        result = _make_result()
        gen.generate(result, scenario_context="night operation")
        prompt = llm.predict_json.call_args[0][0]
        assert "night operation" in prompt

    def test_llm_mode_falls_back_to_template_on_error(self):
        llm = MagicMock()
        llm.predict_json.side_effect = RuntimeError("API down")
        gen = QuestionGenerator(mode="llm", llm=llm)
        result = _make_result()
        q = gen.generate(result)
        assert isinstance(q, str) and len(q) > 0

    def test_llm_mode_falls_back_to_template_on_empty_response(self):
        llm = MagicMock()
        llm.predict_json.return_value = {"question": ""}
        gen = QuestionGenerator(mode="llm", llm=llm)
        result = _make_result()
        q = gen.generate(result)
        assert isinstance(q, str) and len(q) > 0

    def test_expert_profile_prompt_includes_technical_language_hint(self):
        llm = _make_llm()
        gen = QuestionGenerator(mode="llm", llm=llm)
        result = _make_result()
        p = UserProfile("u", expertise="expert")
        gen.generate(result, user_profile=p)
        prompt = llm.predict_json.call_args[0][0]
        assert "expert" in prompt.lower() or "technical" in prompt.lower() \
               or "traversability" in prompt.lower()

    def test_novice_profile_prompt_includes_plain_language_hint(self):
        llm = _make_llm()
        gen = QuestionGenerator(mode="llm", llm=llm)
        result = _make_result()
        p = UserProfile("u", expertise="novice")
        gen.generate(result, user_profile=p)
        prompt = llm.predict_json.call_args[0][0]
        assert "novice" in prompt.lower() or "jargon" in prompt.lower() \
               or "plain" in prompt.lower()
