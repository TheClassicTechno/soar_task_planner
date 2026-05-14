"""
LLM consistency check for instruction ambiguity detection.

Runs AmbiguityDetector n_runs times with independent LLM calls (each instance
has a fresh cache). If the detected ambiguity_type disagrees across runs the
instruction is flagged as unstable and κ_I is forced to 1.0 (always ASK).
If all runs agree, the average p_ambiguous is used to recompute κ_I normally.

This is a zero-cost improvement: no new data, no new model, just three forward
passes whose disagreement reveals prompt sensitivity in the underlying LLM.

Non-conformity score semantics
-------------------------------
  consistent   → κ_I = _compute_nonconformity(agreed_type, avg_p_ambiguous)
  inconsistent → κ_I = 1.0  (worst case; always triggers the ASK branch)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from system.instruction_uncertainty.ambiguity_detector import (
    AmbiguityDetection,
    AmbiguityDetector,
    DetectionMode,
    _compute_nonconformity,
)


@dataclass
class ConsistencyResult:
    """
    Output from one ConsistencyChecker.check() call.

    final_detection:  AmbiguityDetection to act on (modified nonconformity_score
                      and source when inconsistent).
    is_consistent:    True if all n_runs agreed on ambiguity_type.
    all_types:        List of ambiguity_type from each run, length == n_runs.
    avg_p_ambiguous:  Mean p_ambiguous across runs.
    source:           "consistent" or "conservative_ask".
    """

    final_detection: AmbiguityDetection
    is_consistent: bool
    all_types: List[str]
    avg_p_ambiguous: float
    source: str


class ConsistencyChecker:
    """
    Run AmbiguityDetector n_runs times and flag type disagreement.

    Each run uses a freshly constructed AmbiguityDetector so that the
    in-session cache never short-circuits independent LLM calls.

    Args:
        llm:    LLMInterface instance (passed through to each AmbiguityDetector).
        mode:   DetectionMode (default FM_WITH_FALLBACK).
        n_runs: Number of independent detections to compare (default 3).

    Example::

        checker = ConsistencyChecker(llm=my_llm)
        result = checker.check("Go there", "bench visible")
        if not result.is_consistent:
            # κ_I == 1.0 → always ask
            ...
    """

    def __init__(
        self,
        llm: object,
        mode: DetectionMode = DetectionMode.FM_WITH_FALLBACK,
        n_runs: int = 3,
    ) -> None:
        if n_runs < 2:
            raise ValueError(f"n_runs must be >= 2, got {n_runs}")
        self._llm = llm
        self._mode = mode
        self._n_runs = n_runs

    def check(
        self,
        instruction: str,
        scene_context: str = "",
    ) -> ConsistencyResult:
        """
        Run n_runs independent detections and check for type agreement.

        Args:
            instruction:   Natural-language instruction to analyze.
            scene_context: Optional scene description.

        Returns:
            ConsistencyResult with final_detection, consistency flag, and κ_I.
        """
        detections: List[AmbiguityDetection] = []
        for _ in range(self._n_runs):
            detector = AmbiguityDetector(llm=self._llm, mode=self._mode)
            detections.append(detector.detect(instruction, scene_context))

        all_types = [d.ambiguity_type for d in detections]
        avg_p = sum(d.p_ambiguous for d in detections) / len(detections)
        is_consistent = len(set(all_types)) == 1

        if is_consistent:
            agreed_type = all_types[0]
            score = _compute_nonconformity(agreed_type, avg_p)
            base = detections[0]
            final = AmbiguityDetection(
                ambiguity_type=agreed_type,
                p_ambiguous=avg_p,
                nonconformity_score=score,
                missing_slots=list(base.missing_slots),
                reasoning=base.reasoning,
                source=base.source,
                latency_ms=sum(d.latency_ms for d in detections),
            )
            return ConsistencyResult(
                final_detection=final,
                is_consistent=True,
                all_types=all_types,
                avg_p_ambiguous=avg_p,
                source="consistent",
            )

        # Inconsistent → conservative: always ASK (κ_I = 1.0)
        # Use first detection's metadata but override score and source.
        base = detections[0]
        final = AmbiguityDetection(
            ambiguity_type=base.ambiguity_type,
            p_ambiguous=avg_p,
            nonconformity_score=1.0,
            missing_slots=list(base.missing_slots),
            reasoning=base.reasoning,
            source="conservative_ask",
            latency_ms=sum(d.latency_ms for d in detections),
        )
        return ConsistencyResult(
            final_detection=final,
            is_consistent=False,
            all_types=all_types,
            avg_p_ambiguous=avg_p,
            source="conservative_ask",
        )
