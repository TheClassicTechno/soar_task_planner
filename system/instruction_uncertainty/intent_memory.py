"""
Intent memory for instruction uncertainty.

When the robot resolves an ambiguous instruction (e.g., user clarifies that
"go there" means "go to the bench"), it stores that resolved intent so that
identical or structurally similar instructions can be acted on without asking
again in future interactions.

Bayesian update rule
--------------------
After each user response r, the confidence in a stored answer a is updated via:

    τ_{k+1}(a) = P(r | intent=a) · τ_k(a)
                 ─────────────────────────────────────────────────────
                 P(r | intent=a) · τ_k(a) + P(r | intent=≠a) · (1 − τ_k(a))

where:
    P(r="confirmed" | intent=a)     = LIKELIHOOD_CONFIRM  = 0.95
    P(r="confirmed" | intent=≠a)    = LIKELIHOOD_MISMATCH = 0.10

Reuse condition
---------------
If τ(a) ≥ reuse_threshold (default 0.85), the robot acts on a without asking.
Numerically: two confirmations from τ_0=0.5 → τ_2 ≈ 0.994; one → τ_1 ≈ 0.905.

Context matching
----------------
Intent entries are indexed by (instruction_type, context_hash).  The context
hash is an MD5 of the instruction_type and a normalised scene description so
that "go there" at a junction with [bench, gate] hits the same entry on the
next visit to the same junction.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Likelihood parameters (from theoretical_innovations.md) ──────────────────

LIKELIHOOD_CONFIRM = 0.95   # P(user confirms | stored answer is correct)
LIKELIHOOD_MISMATCH = 0.10  # P(user confirms | stored answer is wrong)


# ── Ambiguity type registry ───────────────────────────────────────────────────

AMBIGUITY_TYPES = frozenset({
    "missing_action",
    "missing_object",
    "missing_direction",
    "missing_distance",
    "ambiguous_target",
    "ambiguous_action",
})

SEVERITY_WEIGHTS: Dict[str, float] = {
    "missing_action":    1.00,
    "ambiguous_target":  0.75,
    "missing_object":    0.75,
    "ambiguous_action":  0.50,
    "missing_direction": 0.50,
    "missing_distance":  0.25,
}


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class IntentEntry:
    """
    A single resolved instruction intent stored in memory.

    instruction_type: One of the 6 AMBIGUITY_TYPES.
    context_hash:     MD5 of (instruction_type, normalised scene context).
    resolved_answer:  The clarified slot value (e.g. "go to the bench").
    confidence:       Posterior P(answer is correct) in [0, 1].
    n_observations:   Total confirmations / contradictions seen.
    last_seen:        Unix timestamp of most recent update.
    """

    instruction_type: str
    context_hash: str
    resolved_answer: str
    confidence: float
    n_observations: int = 1
    last_seen: float = field(default_factory=time.time)


def _context_hash(instruction_type: str, scene_context: str) -> str:
    """Return a 16-character hex digest for (instruction_type, scene_context)."""
    raw = f"{instruction_type.lower().strip()}|{scene_context.lower().strip()}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ── Core class ────────────────────────────────────────────────────────────────

class IntentMemory:
    """
    Bayesian intent memory for instruction uncertainty.

    Stores resolved ambiguities across interactions and provides a reuse
    decision: if confidence in a stored answer exceeds reuse_threshold,
    the robot skips asking and acts on the stored intent directly.

    Args:
        reuse_threshold:  Confidence above which asking is skipped (default 0.85).
        initial_confidence: Prior confidence assigned to the first observation
                            (default 0.75 — one observation is not yet enough
                            to reuse; requires at least one more confirmation).
        max_age_seconds:  Entries older than this are treated as stale and
                          ignored. None means no expiry (default).

    Example::

        mem = IntentMemory()
        mem.update("ambiguous_target", "bench gate sign", "go to the bench")
        skip, answer = mem.should_skip_asking("ambiguous_target", "bench gate sign")
        # skip=False; first observation only → confidence=0.75 < 0.85

        mem.update("ambiguous_target", "bench gate sign", "go to the bench")
        skip, answer = mem.should_skip_asking("ambiguous_target", "bench gate sign")
        # skip=True; after second confirmation → confidence≈0.957 ≥ 0.85
    """

    def __init__(
        self,
        reuse_threshold: float = 0.85,
        initial_confidence: float = 0.75,
        max_age_seconds: Optional[float] = None,
    ) -> None:
        if not 0.0 < reuse_threshold <= 1.0:
            raise ValueError(f"reuse_threshold must be in (0, 1], got {reuse_threshold}")
        if not 0.0 < initial_confidence <= 1.0:
            raise ValueError(f"initial_confidence must be in (0, 1], got {initial_confidence}")
        self._threshold = reuse_threshold
        self._initial = initial_confidence
        self._max_age = max_age_seconds
        self._entries: Dict[str, List[IntentEntry]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        instruction_type: str,
        scene_context: str,
        resolved_answer: str,
    ) -> IntentEntry:
        """
        Record a resolved instruction intent after receiving a user response.

        If an entry for (instruction_type, context_hash) already exists:
          - Same answer: apply Bayesian confirmation update (confidence rises).
          - Different answer: apply Bayesian contradiction update (confidence
            falls on old entry); create a new entry for the new answer.

        If no entry exists: create a new entry with initial_confidence.

        Args:
            instruction_type: One of the 6 AMBIGUITY_TYPES.
            scene_context:    Normalised scene description (object list, env).
            resolved_answer:  The user's clarified intent.

        Returns:
            The IntentEntry that was created or updated.
        """
        key = _context_hash(instruction_type, scene_context)
        bucket: List[IntentEntry] = self._entries.setdefault(instruction_type, [])
        matches = [e for e in bucket if e.context_hash == key]

        if not matches:
            entry = IntentEntry(
                instruction_type=instruction_type,
                context_hash=key,
                resolved_answer=resolved_answer,
                confidence=self._initial,
            )
            bucket.append(entry)
            return entry

        # Find best-matching existing entry for this context
        same = [e for e in matches if e.resolved_answer == resolved_answer]
        diff = [e for e in matches if e.resolved_answer != resolved_answer]

        if same:
            target = max(same, key=lambda e: e.confidence)
            target.confidence = _bayesian_update(
                target.confidence, confirmed=True
            )
            target.n_observations += 1
            target.last_seen = time.time()
            return target

        # User gave a different answer — contradict existing entries and add new
        for e in diff:
            e.confidence = _bayesian_update(e.confidence, confirmed=False)
            e.n_observations += 1
            e.last_seen = time.time()

        new_entry = IntentEntry(
            instruction_type=instruction_type,
            context_hash=key,
            resolved_answer=resolved_answer,
            confidence=self._initial,
        )
        bucket.append(new_entry)
        return new_entry

    def recall(
        self,
        instruction_type: str,
        scene_context: str,
    ) -> Optional[Tuple[str, float]]:
        """
        Return the highest-confidence stored answer for this context, or None.

        Stale entries (older than max_age_seconds) are excluded.
        Does not check reuse_threshold; caller decides what to do with the score.

        Args:
            instruction_type: One of the 6 AMBIGUITY_TYPES.
            scene_context:    Current scene context.

        Returns:
            (resolved_answer, confidence) or None if no relevant entry found.
        """
        key = _context_hash(instruction_type, scene_context)
        bucket = self._entries.get(instruction_type, [])
        valid = self._live_entries(bucket, key)
        if not valid:
            return None
        best = max(valid, key=lambda e: e.confidence)
        return best.resolved_answer, best.confidence

    def should_skip_asking(
        self,
        instruction_type: str,
        scene_context: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Decide whether the robot should act on stored intent without asking.

        Returns:
            (should_skip, resolved_answer):
              - should_skip=True  → act on resolved_answer; confidence ≥ threshold.
              - should_skip=False → ask the user; no sufficient memory found.
        """
        result = self.recall(instruction_type, scene_context)
        if result is None:
            return False, None
        answer, confidence = result
        if confidence >= self._threshold:
            return True, answer
        return False, None

    def clear(self) -> None:
        """Remove all stored entries."""
        self._entries.clear()

    def purge_stale(self) -> int:
        """
        Remove entries older than max_age_seconds.

        Returns:
            Number of entries removed.
        """
        if self._max_age is None:
            return 0
        cutoff = time.time() - self._max_age
        removed = 0
        for bucket in self._entries.values():
            before = len(bucket)
            bucket[:] = [e for e in bucket if e.last_seen >= cutoff]
            removed += before - len(bucket)
        return removed

    @property
    def total_entries(self) -> int:
        """Total number of stored intent entries across all ambiguity types."""
        return sum(len(v) for v in self._entries.values())

    # ── Private helpers ───────────────────────────────────────────────────────

    def _live_entries(
        self,
        bucket: List[IntentEntry],
        key: str,
    ) -> List[IntentEntry]:
        """Return entries matching key that are not stale."""
        now = time.time()
        return [
            e for e in bucket
            if e.context_hash == key
            and (self._max_age is None or now - e.last_seen < self._max_age)
        ]


# ── Module-level helper ───────────────────────────────────────────────────────

def _bayesian_update(prior: float, confirmed: bool) -> float:
    """
    Apply one step of the binary noisy-channel Bayesian update.

    If confirmed=True:  posterior = p_tp * prior / (p_tp*prior + p_fp*(1-prior))
    If confirmed=False: posterior = p_fp * prior / (p_fp*prior + p_tp*(1-prior))

    Result is clamped to [0.01, 0.99] to avoid degenerate beliefs.
    """
    if confirmed:
        p_match, p_other = LIKELIHOOD_CONFIRM, LIKELIHOOD_MISMATCH
    else:
        p_match, p_other = LIKELIHOOD_MISMATCH, LIKELIHOOD_CONFIRM

    numerator = p_match * prior
    denominator = numerator + p_other * (1.0 - prior)
    if denominator == 0.0:
        return prior
    return max(0.01, min(0.99, numerator / denominator))


def ambiguity_score(
    ambiguity_type: str,
    p_ambiguous: float,
) -> float:
    """
    Compute the scalar ambiguity score u_I for a detected instruction ambiguity.

    u_I = w_type(t*) · P(ambiguous | instruction, scene)

    Args:
        ambiguity_type: One of the 6 AMBIGUITY_TYPES.
        p_ambiguous:    LLM probability that the instruction is ambiguous, in [0, 1].

    Returns:
        Ambiguity score u_I in [0, 1].
    """
    if ambiguity_type not in SEVERITY_WEIGHTS:
        raise ValueError(
            f"Unknown ambiguity type '{ambiguity_type}'. "
            f"Must be one of: {sorted(SEVERITY_WEIGHTS)}"
        )
    return SEVERITY_WEIGHTS[ambiguity_type] * max(0.0, min(1.0, p_ambiguous))
