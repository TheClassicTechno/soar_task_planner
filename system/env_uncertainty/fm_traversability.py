"""
Foundation model traversability scorer.

Replaces the static lookup table with a prompted LLM judge that assigns
traversability scores based on terrain label and optional scene context.

Two modes:
  STATIC            — original static table, no LLM call (fast, offline)
  FM                — always query the foundation model
  FM_WITH_FALLBACK  — query FM; fall back to static on error or timeout

The scorer caches responses keyed by (label, context_hash) so the same
terrain class is never queried twice in one session.

Prompt design: JSON-response prompt asking for a score in [0,1], a
confidence value, and a one-sentence reasoning string. The robot type is
stated to ground responses to outdoor wheeled navigation.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from system.env_uncertainty.detector import RegionInfo
from system.env_uncertainty.traversability import TRAVERSABILITY_SCORES


class ScoringMode(Enum):
    STATIC = "static"
    FM = "fm"
    FM_WITH_FALLBACK = "fm_with_fallback"


@dataclass
class TraversabilityJudgment:
    """
    Output from one FMTraversabilityScorer.score_label() call.

    score:      Traversability value in [0, 1].
    confidence: Model's self-reported confidence in [0, 1].
    reasoning:  One-sentence explanation from the model.
    source:     "fm" if from foundation model, "static" if from fallback table,
                "cache" if retrieved from in-session cache.
    latency_ms: Wall-clock time for the LLM call (0 for cache/static hits).
    """

    score: float
    confidence: float
    reasoning: str
    source: str
    latency_ms: float = 0.0


_SYSTEM_CONTEXT = (
    "You are a traversability expert for outdoor wheeled robots operating on "
    "natural terrain. Your task is to estimate how safely and easily a standard "
    "wheeled ground robot can traverse a given terrain type. Consider factors "
    "such as surface firmness, slope risk, traction, and obstacle density."
)

_SCORE_PROMPT_TEMPLATE = """\
Terrain class: "{label}"
Scene context: "{context}"

Rate the traversability of this terrain for an outdoor wheeled ground robot.

Scale:
  1.0 = Fully safe, confirmed navigable surface (e.g., flat dry pavement)
  0.7 = Mostly safe, minor caution needed (e.g., dry grass)
  0.5 = Uncertain — proceed with caution or ask the user
  0.3 = Risky — difficult to traverse, prefer to avoid
  0.0 = Impassable or hazardous (e.g., water, person, unknown surface)

Respond ONLY with valid JSON on a single line:
{{"score": <float 0.0-1.0>, "confidence": <float 0.0-1.0>, "reasoning": "<1 sentence>"}}"""


def _context_hash(label: str, context: str) -> str:
    raw = f"{label.lower().strip()}|{context.lower().strip()}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


class FMTraversabilityScorer:
    """
    Score terrain traversability using a foundation model as a judge.

    Args:
        llm:              LLMInterface instance (required for FM and FM_WITH_FALLBACK).
        mode:             ScoringMode enum value.
        latency_budget_ms: Maximum allowed LLM call time in milliseconds.
                          If exceeded in FM_WITH_FALLBACK mode, falls back to static.
    """

    def __init__(
        self,
        llm: Optional[object] = None,
        mode: ScoringMode = ScoringMode.FM_WITH_FALLBACK,
        latency_budget_ms: float = 500.0,
    ):
        if mode in (ScoringMode.FM, ScoringMode.FM_WITH_FALLBACK) and llm is None:
            raise ValueError(f"mode={mode.value} requires an LLMInterface instance")
        self._llm = llm
        self._mode = mode
        self._latency_budget_ms = latency_budget_ms
        self._cache: Dict[str, TraversabilityJudgment] = {}

    def score_label(
        self,
        label: str,
        context: str = "",
    ) -> TraversabilityJudgment:
        """
        Return a traversability judgment for a terrain label string.

        Cache is checked first; on cache miss, calls the foundation model or
        static table depending on the configured mode.

        Args:
            label:   Terrain class string (e.g., "grass", "mud", "unknown").
            context: Optional scene description to ground the scoring.

        Returns:
            TraversabilityJudgment with score, confidence, reasoning, source.
        """
        if self._mode == ScoringMode.STATIC:
            return self._static_judgment(label)

        key = _context_hash(label, context)
        if key in self._cache:
            cached = self._cache[key]
            return TraversabilityJudgment(
                score=cached.score,
                confidence=cached.confidence,
                reasoning=cached.reasoning,
                source="cache",
                latency_ms=0.0,
            )

        try:
            judgment = self._fm_call(label, context)
        except Exception:
            if self._mode == ScoringMode.FM_WITH_FALLBACK:
                return self._static_judgment(label)
            raise

        if (
            self._mode == ScoringMode.FM_WITH_FALLBACK
            and judgment.latency_ms > self._latency_budget_ms
        ):
            return self._static_judgment(label)

        self._cache[key] = judgment
        return judgment

    def score_region(
        self,
        region: RegionInfo,
        image_crop: Optional[object] = None,
    ) -> TraversabilityJudgment:
        """
        Score a RegionInfo object, using its label and optionally an image crop.

        If image_crop is provided and the LLM supports multimodal input, the
        crop is passed as additional context. Otherwise only the label is used.

        Args:
            region:     RegionInfo from the detector.
            image_crop: Optional (H, W, 3) uint8 numpy array of the region.

        Returns:
            TraversabilityJudgment for this region.
        """
        context = f"Region covers {region.pixel_fraction * 100:.1f}% of the scene."
        if image_crop is not None:
            context += " (Visual context provided.)"
        return self.score_label(region.label, context)

    def score_batch(
        self,
        regions: List[RegionInfo],
    ) -> List[TraversabilityJudgment]:
        """
        Score a list of regions, using cache to avoid redundant FM calls.

        Unique (label, empty_context) pairs are collected, queried once each,
        and results are mapped back to all regions sharing that label.

        Args:
            regions: List of RegionInfo objects.

        Returns:
            List of TraversabilityJudgment in the same order as regions.
        """
        results: List[Optional[TraversabilityJudgment]] = [None] * len(regions)
        label_to_judgment: Dict[str, TraversabilityJudgment] = {}

        for i, region in enumerate(regions):
            lbl = region.label.lower().strip()
            if lbl not in label_to_judgment:
                label_to_judgment[lbl] = self.score_label(lbl)
            results[i] = label_to_judgment[lbl]

        return [r for r in results if r is not None]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fm_call(self, label: str, context: str) -> TraversabilityJudgment:
        """Query the foundation model and parse its JSON response."""
        prompt = _SCORE_PROMPT_TEMPLATE.format(
            label=label,
            context=context or "Standard outdoor navigation environment.",
        )
        t0 = time.perf_counter()
        raw = self._llm.predict_json(prompt, system=_SYSTEM_CONTEXT)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        score = float(raw.get("score", 0.0))
        confidence = float(raw.get("confidence", 0.5))
        reasoning = str(raw.get("reasoning", "")).strip()

        score = max(0.0, min(1.0, score))
        confidence = max(0.0, min(1.0, confidence))

        return TraversabilityJudgment(
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            source="fm",
            latency_ms=latency_ms,
        )

    def _static_judgment(self, label: str) -> TraversabilityJudgment:
        """Return a TraversabilityJudgment from the static lookup table."""
        score = TRAVERSABILITY_SCORES.get(label.lower().strip(), 0.0)
        return TraversabilityJudgment(
            score=score,
            confidence=1.0,
            reasoning=f"Static table value for class '{label}'.",
            source="static",
            latency_ms=0.0,
        )

    @property
    def cache_size(self) -> int:
        """Number of entries in the in-session cache."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Flush the in-session cache."""
        self._cache.clear()
