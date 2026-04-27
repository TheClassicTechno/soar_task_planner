"""
Navigation Knowledge Base for the IntroPlan baseline.

Stores introspective reasoning examples for terrain navigation decisions.
At inference, retrieves the K most similar examples for the current scenario
so the LLM can use them as in-context few-shot guidance.

Similarity is computed using a simple TF-IDF-style keyword overlap on the
terrain description + instruction pair. This is intentionally lightweight —
no vector DB needed for the ~50-entry calibration set we're using.

For a production system, replace _similarity() with a sentence-transformer
embedding lookup as described in the IntroPlan paper.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class KnowledgeBaseEntry:
    """One entry in the navigation KB."""

    def __init__(
        self,
        entry_id: str,
        instruction: str,
        terrain_description: str,
        uncertainty_type: int,
        options: Dict[str, str],
        correct_option: str,
        reasoning: str,
        source_image: Optional[str] = None,
    ):
        self.entry_id = entry_id
        self.instruction = instruction
        self.terrain_description = terrain_description
        self.uncertainty_type = uncertainty_type
        self.options = options
        self.correct_option = correct_option
        self.reasoning = reasoning
        # None = synthetic hand-crafted entry; filename = RUGD-derived entry
        self.source_image: Optional[str] = source_image

    def is_rugd_grounded(self) -> bool:
        """Returns True if this entry was generated from a real RUGD image."""
        return self.source_image is not None

    def to_dict(self) -> Dict:
        return {
            "entry_id": self.entry_id,
            "instruction": self.instruction,
            "terrain_description": self.terrain_description,
            "uncertainty_type": self.uncertainty_type,
            "options": self.options,
            "correct_option": self.correct_option,
            "reasoning": self.reasoning,
            "source_image": self.source_image,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "KnowledgeBaseEntry":
        return cls(
            entry_id=d["entry_id"],
            instruction=d["instruction"],
            terrain_description=d["terrain_description"],
            uncertainty_type=d["uncertainty_type"],
            options=d["options"],
            correct_option=d["correct_option"],
            reasoning=d["reasoning"],
            source_image=d.get("source_image"),  # graceful: absent key → None
        )

    def retrieval_text(self) -> str:
        """Combined text used for similarity comparison."""
        return f"{self.instruction} {self.terrain_description}".lower()


class NavigationKnowledgeBase:
    """
    In-memory knowledge base of navigation introspective reasoning examples.

    Usage:
        kb = NavigationKnowledgeBase.from_json("baselines/introplan/data/nav_calibration.json")
        similar = kb.retrieve(instruction="Take me to the library",
                              terrain_description="cracked pavement 5m ahead",
                              top_k=3)
        for ex in similar:
            print(ex.reasoning)
    """

    def __init__(self):
        self._entries: List[KnowledgeBaseEntry] = []

    def add(self, entry: KnowledgeBaseEntry) -> None:
        self._entries.append(entry)

    def __len__(self) -> int:
        return len(self._entries)

    def retrieve(
        self,
        instruction: str,
        terrain_description: str,
        top_k: int = 3,
        same_uncertainty_type: Optional[int] = None,
    ) -> List[KnowledgeBaseEntry]:
        """
        Return the top_k most similar KB entries to the query.

        Args:
            instruction:           Current user instruction.
            terrain_description:   Current terrain observation.
            top_k:                 Number of examples to return.
            same_uncertainty_type: If set, only retrieve entries of this type.

        Returns:
            List of KnowledgeBaseEntry ordered by similarity (most similar first).
        """
        query_text = f"{instruction} {terrain_description}".lower()
        query_tokens = _tokenize(query_text)

        candidates = self._entries
        if same_uncertainty_type is not None:
            candidates = [e for e in candidates if e.uncertainty_type == same_uncertainty_type]

        scored: List[Tuple[float, KnowledgeBaseEntry]] = [
            (_similarity(query_tokens, _tokenize(e.retrieval_text())), e)
            for e in candidates
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def retrieve_as_dicts(
        self,
        instruction: str,
        terrain_description: str,
        top_k: int = 3,
        same_uncertainty_type: Optional[int] = None,
    ) -> List[Dict]:
        """Convenience wrapper returning dicts (for prompt formatting)."""
        entries = self.retrieve(instruction, terrain_description, top_k, same_uncertainty_type)
        return [e.to_dict() for e in entries]

    def rugd_grounded_count(self) -> int:
        """Number of entries backed by a real RUGD image."""
        return sum(1 for e in self._entries if e.is_rugd_grounded())

    def synthetic_count(self) -> int:
        """Number of hand-crafted synthetic entries (no RUGD image)."""
        return sum(1 for e in self._entries if not e.is_rugd_grounded())

    def save(self, path: str) -> None:
        """Serialize the KB to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([e.to_dict() for e in self._entries], f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "NavigationKnowledgeBase":
        """Load a KB from a JSON file (list of entry dicts)."""
        kb = cls()
        with open(path, "r") as f:
            entries = json.load(f)
        for d in entries:
            kb.add(KnowledgeBaseEntry.from_dict(d))
        return kb


# ── Similarity helpers ────────────────────────────────────────────────────────

_STOPWORDS = frozenset(
    "the a an and or but in on at to for of with is are was were be been".split()
)


def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """
    Jaccard similarity between two token lists.
    Returns intersection / union of unique token sets.
    Used as a lightweight proxy for semantic similarity.
    """
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)
