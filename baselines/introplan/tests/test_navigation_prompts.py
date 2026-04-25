"""
Unit tests for navigation_prompts.py — no API calls, no GPU.
Tests prompt formatting, completeness, and structural correctness.
"""

import pytest
from baselines.introplan.navigation_prompts import (
    OPTION_LABELS,
    OPTION_DESCRIPTIONS,
    UNCERTAINTY_TYPE_LABELS,
    format_candidate_options_prompt,
    format_kb_entry_prompt,
    format_retrieval_examples,
    format_introspective_predict_prompt,
    SCENE_DESCRIPTION_PROMPT,
)


# ── Constants ─────────────────────────────────────────────────────────────────

def test_option_labels_are_abcd():
    assert OPTION_LABELS == ["A", "B", "C", "D"]


def test_option_descriptions_all_present():
    for label in OPTION_LABELS:
        assert label in OPTION_DESCRIPTIONS
        assert isinstance(OPTION_DESCRIPTIONS[label], str)
        assert len(OPTION_DESCRIPTIONS[label]) > 0


def test_uncertainty_type_labels_cover_types_1_to_4():
    for t in [1, 2, 3, 4]:
        assert t in UNCERTAINTY_TYPE_LABELS
        assert f"Type {t}" in UNCERTAINTY_TYPE_LABELS[t]


def test_scene_description_prompt_is_non_empty():
    assert isinstance(SCENE_DESCRIPTION_PROMPT, str)
    assert len(SCENE_DESCRIPTION_PROMPT) > 50


# ── format_candidate_options_prompt ──────────────────────────────────────────

def test_candidate_options_prompt_contains_instruction():
    instruction = "Take me to the library"
    result = format_candidate_options_prompt(instruction, "gravel path ahead", 2)
    assert instruction in result


def test_candidate_options_prompt_contains_terrain():
    terrain = "cracked pavement 5m ahead, confidence high"
    result = format_candidate_options_prompt("Go forward", terrain, 2)
    assert terrain in result


def test_candidate_options_prompt_contains_uncertainty_label():
    result = format_candidate_options_prompt("Go forward", "puddle", 2)
    assert "Type 2" in result


def test_candidate_options_prompt_contains_all_options():
    result = format_candidate_options_prompt("Go forward", "puddle", 2)
    for label in OPTION_LABELS:
        assert f"{label}:" in result


def test_candidate_options_prompt_unknown_type_doesnt_crash():
    result = format_candidate_options_prompt("Go forward", "puddle", 99)
    assert "Type 99" in result


# ── format_kb_entry_prompt ────────────────────────────────────────────────────

def test_kb_entry_prompt_contains_instruction():
    instruction = "Take me to the parking lot"
    result = format_kb_entry_prompt(instruction, "gravel", 4, "B", "Ask user about gravel")
    assert instruction in result


def test_kb_entry_prompt_contains_correct_option():
    result = format_kb_entry_prompt("Go", "wet grass", 3, "D", "Slow down")
    assert "D" in result
    assert "Slow down" in result


def test_kb_entry_prompt_contains_terrain():
    terrain = "steep slope ahead"
    result = format_kb_entry_prompt("Continue", terrain, 3, "B", "Ask about slope")
    assert terrain in result


# ── format_retrieval_examples ─────────────────────────────────────────────────

def test_retrieval_examples_empty_returns_fallback():
    result = format_retrieval_examples([])
    assert "No similar" in result


def test_retrieval_examples_single_entry():
    examples = [{
        "instruction": "Take me to the cafeteria",
        "terrain_description": "gravel path, confidence high",
        "correct_option": "B",
        "reasoning": "User has bad knees so gravel is risky.",
    }]
    result = format_retrieval_examples(examples)
    assert "Example 1" in result
    assert "Take me to the cafeteria" in result
    assert "gravel" in result
    assert "Option B" in result
    assert "bad knees" in result


def test_retrieval_examples_multiple_entries_numbered():
    examples = [
        {"instruction": "A", "terrain_description": "T1", "correct_option": "B", "reasoning": "R1"},
        {"instruction": "B", "terrain_description": "T2", "correct_option": "A", "reasoning": "R2"},
    ]
    result = format_retrieval_examples(examples)
    assert "Example 1" in result
    assert "Example 2" in result


def test_retrieval_examples_missing_keys_doesnt_crash():
    examples = [{}]  # Empty dict — all keys missing
    result = format_retrieval_examples(examples)
    assert "Example 1" in result  # Should not raise


# ── format_introspective_predict_prompt ───────────────────────────────────────

def test_introspective_predict_prompt_contains_instruction():
    instruction = "Take me to the library"
    prompt = format_introspective_predict_prompt(
        instruction=instruction,
        terrain_description="puddle ahead",
        options=OPTION_DESCRIPTIONS,
        retrieved_examples=[],
    )
    assert instruction in prompt


def test_introspective_predict_prompt_contains_terrain():
    terrain = "muddy trail"
    prompt = format_introspective_predict_prompt(
        instruction="Go to the lab",
        terrain_description=terrain,
        options=OPTION_DESCRIPTIONS,
        retrieved_examples=[],
    )
    assert terrain in prompt


def test_introspective_predict_prompt_contains_all_option_letters():
    prompt = format_introspective_predict_prompt(
        instruction="Go forward",
        terrain_description="cracked pavement",
        options=OPTION_DESCRIPTIONS,
        retrieved_examples=[],
    )
    for label in OPTION_LABELS:
        assert f"{label}:" in prompt


def test_introspective_predict_prompt_contains_json_format_spec():
    prompt = format_introspective_predict_prompt(
        instruction="Go",
        terrain_description="gravel",
        options=OPTION_DESCRIPTIONS,
        retrieved_examples=[],
    )
    # The prompt should specify the expected JSON keys
    assert '"prediction"' in prompt
    assert '"confidence"' in prompt
    assert '"reasoning"' in prompt


def test_introspective_predict_prompt_includes_retrieved_examples():
    examples = [{
        "instruction": "Take me to building A",
        "terrain_description": "gravel path",
        "correct_option": "B",
        "reasoning": "User prefers smooth surfaces.",
    }]
    prompt = format_introspective_predict_prompt(
        instruction="Go",
        terrain_description="gravel",
        options=OPTION_DESCRIPTIONS,
        retrieved_examples=examples,
    )
    assert "building A" in prompt
    assert "smooth surfaces" in prompt
