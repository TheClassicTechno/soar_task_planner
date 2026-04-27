"""
Unit tests for knowledge_base.py — no API calls, no GPU.
"""

import json
import pytest
from pathlib import Path

from baselines.introplan.knowledge_base import (
    KnowledgeBaseEntry,
    NavigationKnowledgeBase,
    _tokenize,
    _similarity,
)


CALIBRATION_DATA_PATH = str(
    Path(__file__).parents[1] / "data" / "nav_calibration.json"
)


# ── _tokenize ─────────────────────────────────────────────────────────────────

def test_tokenize_basic():
    tokens = _tokenize("Take me to the library")
    assert "take" in tokens
    assert "library" in tokens
    # stopword "the" should be removed
    assert "the" not in tokens


def test_tokenize_removes_stopwords():
    tokens = _tokenize("is the a an and or but")
    assert tokens == []


def test_tokenize_empty_string():
    assert _tokenize("") == []


def test_tokenize_numbers():
    tokens = _tokenize("5m ahead confidence 0.8")
    assert "5m" in tokens or "5" in tokens


# ── _similarity ───────────────────────────────────────────────────────────────

def test_similarity_identical_texts():
    t = _tokenize("cracked pavement library")
    assert _similarity(t, t) == pytest.approx(1.0)


def test_similarity_no_overlap():
    a = _tokenize("cracked pavement")
    b = _tokenize("steep slope")
    assert _similarity(a, b) == pytest.approx(0.0)


def test_similarity_partial_overlap():
    a = _tokenize("cracked pavement library")
    b = _tokenize("cracked road cafeteria")
    # shared: "cracked"; union: cracked, pavement, library, road, cafeteria = 5
    assert 0.0 < _similarity(a, b) < 1.0


def test_similarity_both_empty():
    assert _similarity([], []) == pytest.approx(0.0)


# ── KnowledgeBaseEntry ────────────────────────────────────────────────────────

@pytest.fixture
def sample_entry():
    return KnowledgeBaseEntry(
        entry_id="test_001",
        instruction="Take me to the library",
        terrain_description="cracked pavement 5m ahead",
        uncertainty_type=2,
        options={"A": "Go", "B": "Ask", "C": "Reroute", "D": "Slow"},
        correct_option="B",
        reasoning="User preference for cracked pavement is unknown.",
    )


def test_entry_to_dict_has_required_keys(sample_entry):
    d = sample_entry.to_dict()
    for key in ["entry_id", "instruction", "terrain_description",
                "uncertainty_type", "options", "correct_option", "reasoning"]:
        assert key in d


def test_entry_roundtrip_from_dict(sample_entry):
    d = sample_entry.to_dict()
    restored = KnowledgeBaseEntry.from_dict(d)
    assert restored.entry_id == sample_entry.entry_id
    assert restored.instruction == sample_entry.instruction
    assert restored.correct_option == sample_entry.correct_option
    assert restored.uncertainty_type == sample_entry.uncertainty_type


def test_entry_retrieval_text_contains_instruction_and_terrain(sample_entry):
    text = sample_entry.retrieval_text()
    assert "library" in text
    assert "cracked" in text


# ── NavigationKnowledgeBase ───────────────────────────────────────────────────

def _make_kb(n: int) -> NavigationKnowledgeBase:
    kb = NavigationKnowledgeBase()
    for i in range(n):
        kb.add(KnowledgeBaseEntry(
            entry_id=f"e{i:03d}",
            instruction=f"Instruction {i}",
            terrain_description=f"terrain {i} ahead gravel",
            uncertainty_type=(i % 4) + 1,
            options={"A": "Go", "B": "Ask", "C": "Reroute", "D": "Slow"},
            correct_option="B",
            reasoning=f"Reasoning {i}",
        ))
    return kb


def test_kb_len():
    kb = _make_kb(5)
    assert len(kb) == 5


def test_kb_retrieve_returns_at_most_top_k():
    kb = _make_kb(10)
    results = kb.retrieve("Go to library", "cracked pavement", top_k=3)
    assert len(results) <= 3


def test_kb_retrieve_empty_kb_returns_empty():
    kb = NavigationKnowledgeBase()
    results = kb.retrieve("Go to library", "cracked pavement", top_k=3)
    assert results == []


def test_kb_retrieve_filters_by_uncertainty_type():
    kb = _make_kb(12)  # types cycle 1,2,3,4,1,2,3,4,...
    results = kb.retrieve("Go", "terrain", top_k=10, same_uncertainty_type=2)
    for entry in results:
        assert entry.uncertainty_type == 2


def test_kb_retrieve_similar_terrain_ranked_first():
    kb = NavigationKnowledgeBase()
    # Add a very similar entry and a dissimilar entry
    kb.add(KnowledgeBaseEntry(
        entry_id="sim",
        instruction="Take me to the library",
        terrain_description="cracked pavement library route",
        uncertainty_type=2,
        options={}, correct_option="B", reasoning="Similar",
    ))
    kb.add(KnowledgeBaseEntry(
        entry_id="dis",
        instruction="Lead me to the sports field",
        terrain_description="steep slope muddy trail",
        uncertainty_type=3,
        options={}, correct_option="D", reasoning="Dissimilar",
    ))
    results = kb.retrieve("Take me to library", "cracked pavement", top_k=2)
    assert results[0].entry_id == "sim"


def test_kb_save_and_load(tmp_path):
    kb = _make_kb(3)
    save_path = str(tmp_path / "kb.json")
    kb.save(save_path)

    loaded = NavigationKnowledgeBase.from_json(save_path)
    assert len(loaded) == 3
    assert loaded._entries[0].entry_id == kb._entries[0].entry_id


def test_kb_retrieve_as_dicts_returns_dicts():
    kb = _make_kb(5)
    results = kb.retrieve_as_dicts("Go", "gravel", top_k=2)
    assert all(isinstance(r, dict) for r in results)
    assert all("instruction" in r for r in results)


# ── Loading real calibration data ─────────────────────────────────────────────

def test_calibration_data_loads_correctly():
    kb = NavigationKnowledgeBase.from_json(CALIBRATION_DATA_PATH)
    assert len(kb) == 57


def test_calibration_data_all_entries_valid():
    kb = NavigationKnowledgeBase.from_json(CALIBRATION_DATA_PATH)
    for entry in kb._entries:
        assert entry.correct_option in ["A", "B", "C", "D"]
        assert entry.uncertainty_type in [1, 2, 3, 4]
        assert len(entry.reasoning) > 20
        assert len(entry.instruction) > 3


def test_calibration_data_retrieval_works():
    kb = NavigationKnowledgeBase.from_json(CALIBRATION_DATA_PATH)
    results = kb.retrieve(
        instruction="Take me to the library",
        terrain_description="cracked pavement ahead",
        top_k=3,
    )
    assert len(results) > 0
    # The most similar entry should be nav_001 (library + cracked pavement)
    assert results[0].entry_id == "nav_001"
