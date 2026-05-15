"""
Instruction ambiguity detector — LLM slot-fill parser.

Detects which of 6 ambiguity types is present in a natural-language robot
instruction, and outputs a non-conformity score κ_I for use in conformal
prediction.

Detection modes
---------------
  RULE               — fast, offline, keyword/regex heuristics (deterministic)
  FM                 — always query the foundation model (most accurate)
  FM_WITH_FALLBACK   — query FM; fall back to RULE on error

Non-conformity score
--------------------
κ_I = u_I = w_type(t*) · P(ambiguous | instruction, scene)

where w_type is the severity weight for the detected ambiguity type and
P(ambiguous) is the LLM's estimated probability that the instruction is
ambiguous.  This is the κ_I used in the joint CP predictor:

    κ_joint = max(κ_I, κ_E)

Six ambiguity types (see SEVERITY_WEIGHTS in intent_memory.py)
--------------------------------------------------------------
  missing_action    (w=1.00): No action verb — robot cannot determine what to do
  ambiguous_target  (w=0.75): Destination is a pronoun with no clear referent
  missing_object    (w=0.75): Action stated but the object is unspecified
  ambiguous_action  (w=0.50): Multiple interpretations of what action to take
  missing_direction (w=0.50): Movement intended but no direction given
  missing_distance  (w=0.25): Distance reference exists but is not quantified

Caching
-------
Responses are cached per (instruction, scene_context) MD5 pair to avoid
redundant LLM calls within a session.  Cache is not persisted across sessions.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from system.instruction_uncertainty.intent_memory import (
    AMBIGUITY_TYPES,
    ambiguity_score,
)


class DetectionMode(Enum):
    RULE = "rule"
    FM = "fm"
    FM_WITH_FALLBACK = "fm_with_fallback"


@dataclass
class AmbiguityDetection:
    """
    Output from one AmbiguityDetector.detect() call.

    ambiguity_type:      One of the 6 AMBIGUITY_TYPES or "no_uncertainty".
    p_ambiguous:         P(ambiguous | instruction, scene) in [0, 1].
    nonconformity_score: κ_I = u_I = severity_weight * p_ambiguous.
    missing_slots:       Human-readable slot names that are missing or unclear.
    reasoning:           One-sentence explanation of the detected ambiguity.
    source:              "rule", "fm", or "cache".
    latency_ms:          Wall-clock LLM call time (0 for rule/cache hits).
    """

    ambiguity_type: str
    p_ambiguous: float
    nonconformity_score: float
    missing_slots: List[str]
    reasoning: str
    source: str
    latency_ms: float = 0.0


# ── LLM prompt ────────────────────────────────────────────────────────────────

_SYSTEM_CONTEXT = (
    "You are an instruction clarity analyzer for an outdoor wheeled navigation robot. "
    "Your task is to identify ambiguities in user instructions that would prevent the "
    "robot from executing a navigation command correctly."
)

_DETECT_PROMPT_TEMPLATE = """\
Instruction: "{instruction}"
Scene context: "{context}"

Classify the instruction into exactly one of these ambiguity types:
  missing_action    — No clear action verb (robot cannot determine what to DO)
  ambiguous_target  — Destination or object is an unclear pronoun ("there", "it", "that way")
  missing_object    — Action is clear but the object or destination is unspecified
  ambiguous_action  — Multiple interpretations of what action to perform
  missing_direction — Movement intended but no direction, turn, or named destination given
  missing_distance  — A distance reference exists but is not quantified (e.g., "a bit further")
  no_uncertainty    — Instruction is complete and unambiguous

Also estimate P(ambiguous), the probability [0.0-1.0] that the instruction is ambiguous.
List the missing or unclear slots (e.g., ["target", "distance"]), or [] if none.

Respond ONLY with valid JSON on a single line:
{{"ambiguity_type": "<type>", "p_ambiguous": <float>, "missing_slots": [<slots>], "reasoning": "<1 sentence>"}}"""


def _instruction_hash(instruction: str, scene_context: str) -> str:
    raw = f"{instruction.lower().strip()}|{scene_context.lower().strip()}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ── Main class ────────────────────────────────────────────────────────────────

class AmbiguityDetector:
    """
    Detect instruction ambiguity type and compute the CP non-conformity score κ_I.

    Args:
        llm:  LLMInterface instance (required for FM and FM_WITH_FALLBACK modes).
        mode: DetectionMode (default FM_WITH_FALLBACK).

    Example::

        detector = AmbiguityDetector(mode=DetectionMode.RULE)
        result = detector.detect("Go there", "bench and gate visible ahead")
        # result.ambiguity_type == "ambiguous_target"
        # result.nonconformity_score == 0.75 * 0.80 == 0.60
    """

    def __init__(
        self,
        llm: Optional[object] = None,
        mode: DetectionMode = DetectionMode.FM_WITH_FALLBACK,
    ) -> None:
        if mode in (DetectionMode.FM, DetectionMode.FM_WITH_FALLBACK) and llm is None:
            raise ValueError(f"mode={mode.value} requires an LLMInterface instance")
        self._llm = llm
        self._mode = mode
        self._cache: Dict[str, AmbiguityDetection] = {}

    def detect(
        self,
        instruction: str,
        scene_context: str = "",
    ) -> AmbiguityDetection:
        """
        Detect ambiguity in a natural-language robot instruction.

        Args:
            instruction:   Natural-language instruction to analyze.
            scene_context: Optional scene description (visible objects, terrain).

        Returns:
            AmbiguityDetection with type, probability, κ_I score, and explanation.
        """
        if self._mode == DetectionMode.RULE:
            return _rule_detect(instruction, scene_context)

        key = _instruction_hash(instruction, scene_context)
        if key in self._cache:
            cached = self._cache[key]
            return AmbiguityDetection(
                ambiguity_type=cached.ambiguity_type,
                p_ambiguous=cached.p_ambiguous,
                nonconformity_score=cached.nonconformity_score,
                missing_slots=list(cached.missing_slots),
                reasoning=cached.reasoning,
                source="cache",
                latency_ms=0.0,
            )

        try:
            detection = self._fm_detect(instruction, scene_context)
        except Exception:
            if self._mode == DetectionMode.FM_WITH_FALLBACK:
                return _rule_detect(instruction, scene_context)
            raise

        self._cache[key] = detection
        return detection

    @property
    def cache_size(self) -> int:
        """Number of entries in the in-session cache."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Flush the in-session cache."""
        self._cache.clear()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fm_detect(self, instruction: str, scene_context: str) -> AmbiguityDetection:
        """Query the foundation model and parse its JSON response."""
        assert self._llm is not None  # enforced by constructor
        prompt = _DETECT_PROMPT_TEMPLATE.format(
            instruction=instruction,
            context=scene_context or "General outdoor navigation environment.",
        )
        t0 = time.perf_counter()
        raw = self._llm.predict_json(prompt, system=_SYSTEM_CONTEXT)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        atype = str(raw.get("ambiguity_type", "no_uncertainty")).strip()
        if atype not in AMBIGUITY_TYPES and atype != "no_uncertainty":
            atype = "no_uncertainty"

        p_amb = float(raw.get("p_ambiguous", 0.0))
        p_amb = max(0.0, min(1.0, p_amb))

        slots = [str(s) for s in raw.get("missing_slots", [])]
        reasoning = str(raw.get("reasoning", "")).strip()
        score = _compute_nonconformity(atype, p_amb)

        return AmbiguityDetection(
            ambiguity_type=atype,
            p_ambiguous=p_amb,
            nonconformity_score=score,
            missing_slots=slots,
            reasoning=reasoning,
            source="fm",
            latency_ms=latency_ms,
        )


# ── Non-conformity score ──────────────────────────────────────────────────────

def _compute_nonconformity(ambiguity_type: str, p_ambiguous: float) -> float:
    """κ_I = u_I = severity_weight(type) * p_ambiguous; 0.0 for no_uncertainty."""
    if ambiguity_type == "no_uncertainty":
        return 0.0
    return ambiguity_score(ambiguity_type, p_ambiguous)


# ── Rule-based detector ───────────────────────────────────────────────────────

_ACTION_VERBS = frozenset({
    "go", "take", "navigate", "turn", "walk", "move", "stop", "cross",
    "follow", "reach", "head", "proceed", "get", "bring", "carry", "deliver",
    "come", "approach", "enter", "exit", "avoid", "pass", "continue",
    "advance", "return", "find", "locate", "travel", "run", "push", "roll",
    # object-handling verbs
    "pick", "grab", "fetch", "collect", "retrieve", "drop", "place", "put",
    # vague-but-present action verbs (classified as ambiguous_action, not missing_action)
    "handle", "deal", "fix", "manage", "check", "inspect", "address",
})

_MOVEMENT_VERBS = frozenset({
    "go", "move", "walk", "head", "travel", "advance", "proceed", "navigate",
})

_DIRECTION_WORDS = frozenset({
    "left", "right", "straight", "forward", "backward", "north", "south",
    "east", "west", "toward", "towards", "away", "along", "through",
    "beside", "behind", "past", "up", "down", "uphill", "downhill", "ahead",
    "back", "northwest", "northeast", "southwest", "southeast",
})

_VAGUE_DISTANCE = re.compile(
    r"\b(a bit|a little|a few steps?|nearby|close by|not far|some distance|"
    r"a ways?|a short distance|a little further|a bit further|closer|"
    r"a while longer|further along|just ahead)\b",
    re.IGNORECASE,
)

_SPECIFIC_DISTANCE = re.compile(
    r"\b\d+\s*(m|meter|meters|metre|metres|ft|feet|foot|km|kilometer|kilometers|"
    r"miles?|block|blocks|step|steps|yard|yards)\b",
    re.IGNORECASE,
)

_AMBIGUOUS_TARGET_PRONOUNS = re.compile(
    r"\b(there|over there|that way|this way|that place|this place|"
    r"that area|this area|that spot|this spot|that location|this location)\b",
    re.IGNORECASE,
)

_NAMED_LOCATION = re.compile(
    r"\b(library|cafeteria|lab|office|park|building|hospital|school|"
    r"station|entrance|exit|gate|bench|fountain|intersection|crosswalk|"
    r"corner|street|road|path|trail|parking|lot|room|floor|door|"
    r"hallway|lobby|plaza|garden|court|field|center|centre|restroom|"
    r"bathroom|cafe|restaurant|store|shop|market|gym|arena|stadium|"
    r"terminal|platform|bridge|underpass|overpass)\b",
    re.IGNORECASE,
)

_VAGUE_ACTION_VERBS = re.compile(
    r"\b(handle|deal with|take care of|do something|fix|manage|sort out|"
    r"address|resolve|figure out)\b",
    re.IGNORECASE,
)

_OBJECT_HANDLING_VERBS = re.compile(
    r"\b(pick(?: up)?|grab|bring|carry|deliver|fetch|collect|retrieve|drop(?: off)?)\b",
    re.IGNORECASE,
)

_OBJECT_PRONOUNS = re.compile(
    r"\b(it|that|this|the thing|the item|the object|them|those)\b",
    re.IGNORECASE,
)


def _rule_detect(instruction: str, scene_context: str) -> AmbiguityDetection:
    """
    Apply keyword/regex heuristics to classify instruction ambiguity.

    Rules fire in severity order (highest first); first match wins.
    Returns no_uncertainty if no rule fires.
    """
    tokens = set(instruction.lower().split())

    # 1. missing_action — no recognisable action verb in the instruction
    if not (tokens & _ACTION_VERBS):
        return _rule_result(
            "missing_action", 0.80, ["action"],
            "No action verb found — robot cannot determine what to do.",
        )

    # 2. ambiguous_target — pronoun destination with no named location
    if _AMBIGUOUS_TARGET_PRONOUNS.search(instruction):
        if not _NAMED_LOCATION.search(instruction):
            return _rule_result(
                "ambiguous_target", 0.80, ["target"],
                "Destination is an unresolved pronoun with no named location.",
            )

    # 3. missing_object — object-handling verb with only a pronoun as object
    if _OBJECT_HANDLING_VERBS.search(instruction) and _OBJECT_PRONOUNS.search(instruction):
        if not _NAMED_LOCATION.search(instruction):
            return _rule_result(
                "missing_object", 0.75, ["object"],
                "Object of the action is an unspecified pronoun.",
            )

    # 4. ambiguous_action — vague action verb that does not specify a robot action
    if _VAGUE_ACTION_VERBS.search(instruction):
        return _rule_result(
            "ambiguous_action", 0.75, ["action"],
            "Vague action verb does not specify a concrete robot action.",
        )

    # 5. missing_direction — movement verb with no direction or named destination
    if tokens & _MOVEMENT_VERBS:
        has_direction = bool(tokens & _DIRECTION_WORDS)
        has_destination = bool(_NAMED_LOCATION.search(instruction))
        if not has_direction and not has_destination:
            return _rule_result(
                "missing_direction", 0.70, ["direction"],
                "Movement verb present but no direction or destination specified.",
            )

    # 6. missing_distance — vague distance without a quantified value
    if _VAGUE_DISTANCE.search(instruction) and not _SPECIFIC_DISTANCE.search(instruction):
        return _rule_result(
            "missing_distance", 0.70, ["distance"],
            "Vague distance reference found with no quantified value.",
        )

    return _rule_result(
        "no_uncertainty", 0.0, [],
        "Instruction is complete and unambiguous.",
    )


def _rule_result(
    ambiguity_type: str,
    p_ambiguous: float,
    missing_slots: List[str],
    reasoning: str,
) -> AmbiguityDetection:
    return AmbiguityDetection(
        ambiguity_type=ambiguity_type,
        p_ambiguous=p_ambiguous,
        nonconformity_score=_compute_nonconformity(ambiguity_type, p_ambiguous),
        missing_slots=missing_slots,
        reasoning=reasoning,
        source="rule",
        latency_ms=0.0,
    )
