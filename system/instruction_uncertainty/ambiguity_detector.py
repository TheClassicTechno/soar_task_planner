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

Short-distance instruction examples (single-image, ≤5 m depth range)
----------------------------------------------------------------------
Because the system operates on one RGB frame and the RealSense D435i provides
reliable depth up to ~5 m, instructions must refer to visible, close objects.

  Unambiguous (clear target, no missing info):
    "Go to that fire hydrant"          → target visible in image, ~3-5 m
    "Navigate to the bench ahead"      → specific near object
    "Move to the sidewalk on the right" → clear direction + target
    "Head toward the trash can"        → close visible object

  Ambiguous target (w=0.75 — multiple matching objects visible):
    "Go to that sign"                  → which sign? (multiple in view)
    "Move to that object"              → no distinguishing feature given

  Missing distance (w=0.25 — distance qualifier unquantified):
    "Move a bit closer to the wall"    → "a bit" not quantifiable
    "Go forward a little"              → how far is "a little"?
    "Back up slightly"                 → "slightly" undefined

  Missing direction (w=0.50):
    "Go to the door"                   → which side? front/back unclear
    "Move forward"                     → which direction is "forward"?

  Missing action (w=1.00):
    "The fire hydrant"                 → no verb, robot cannot act
    "That bench over there"            → no action specified

  Ambiguous action (w=0.50):
    "Handle the puddle ahead"          → go around? stop? ask?
    "Deal with the obstacle"           → underspecified action
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
    # continuation/manner verbs — present but non-directional; not missing_action
    "keep", "maintain", "stay", "resume",
    # object-handling verbs
    "pick", "grab", "fetch", "collect", "retrieve", "drop", "place", "put",
    "lead", "guide", "escort", "show",
    # vague-but-present action verbs (classified as ambiguous_action, not missing_action)
    "handle", "deal", "fix", "manage", "check", "inspect", "address",
})

_MOVEMENT_VERBS = frozenset({
    "go", "move", "walk", "head", "travel", "advance", "proceed", "navigate",
    # turn-class verbs: have a destination but still need an explicit turn direction
    "turn", "veer", "rotate", "swing", "bank",
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
    r"that area|this area|that spot|this spot|that location|this location|"
    r"somewhere|anywhere|someplace|some place)\b",
    re.IGNORECASE,
)

# "that building" / "those benches" — far deictic + class noun is still ambiguous
# (which building/bench?) even though the class noun would suppress the normal rule.
# "this path" / "this road" are excluded (proximal "this" refers to the current one).
_DEICTIC_NAMED_LOCATION = re.compile(
    r"\b(that|those)\s+\w*\s*"
    r"(library|cafeteria|lab|office|park|building|hospital|school|"
    r"station|entrance|exit|gate|bench|fountain|intersection|crosswalk|"
    r"corner|street|road|path|trail|parking|lot|room|floor|door|"
    r"hallway|lobby|plaza|garden|court|field|center|centre|restroom|"
    r"bathroom|cafe|restaurant|store|shop|market|gym|arena|stadium|"
    r"terminal|platform|bridge|underpass|overpass)\b",
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
    r"\b(pick(?: up)?|grab|bring|carry|deliver|fetch|collect|retrieve|drop(?: off)?|"
    r"take|find|locate|lead|guide|escort|show)\b",
    re.IGNORECASE,
)

_OBJECT_PRONOUNS = re.compile(
    r"\b(it|that|this|the thing|the item|the object|them|those)\b",
    re.IGNORECASE,
)

# "take me home", "find me a shaded area" — object-handling verb targeting the
# user (me/us) with a vague or unnamed destination.
_ME_OBJECT = re.compile(r"\bme\b", re.IGNORECASE)

# "Move faster", "go slowly" — manner modifiers indicate intent to adjust current
# motion, not to initiate new navigation; missing_direction should not fire.
_MANNER_MODIFIER = re.compile(
    r"\b(faster|slower|quickly|carefully|slowly|quicker|hurriedly|hurry|"
    r"cautiously|gently|steadily)\b",
    re.IGNORECASE,
)

# "let's go", "let us proceed" — continuation commands; no destination needed.
_LETS_CONTINUE = re.compile(r"\blet'?s?\b", re.IGNORECASE)

# Turn verbs that need an explicit left/right direction even when a destination is given.
_TURN_VERBS = re.compile(r"\b(turn|veer|hang|swing|bank|rotate)\b", re.IGNORECASE)

# Existential vague pronouns are a strict subset of _AMBIGUOUS_TARGET_PRONOUNS but
# signal even less specificity than directional pronouns ("there", "that way"):
# "go anywhere" is more underdetermined than "go there".
_EXISTENTIAL_VAGUE = re.compile(
    r"\b(anywhere|somewhere|someplace|any place|some place)\b",
    re.IGNORECASE,
)

# Minimum per-type score to include a type in the candidate set.
_MIN_TYPE_THRESHOLD = 0.50


# ── Per-type word-bank scorers ────────────────────────────────────────────────
# Each function returns (p_ambiguous, missing_slots, reasoning).
# Returns (0.0, [], "") when the type does not apply.
# Intra-type boosters raise p within the type when multiple indicator patterns
# fire; inter-type (co-occurrence) boost is applied in _rule_detect.

def _score_ambiguous_target(
    instruction: str,
    has_named_loc: bool,
) -> tuple:
    # Full suppressor: pronoun present AND named location resolves the reference.
    if _AMBIGUOUS_TARGET_PRONOUNS.search(instruction) and has_named_loc:
        return 0.0, [], ""
    if _DEICTIC_NAMED_LOCATION.search(instruction):
        p = 0.80
        reasoning = "Deictic reference to a location class is ambiguous (which one?)."
    elif _AMBIGUOUS_TARGET_PRONOUNS.search(instruction):
        p = 0.80
        reasoning = "Destination is an unresolved pronoun with no named location."
    else:
        return 0.0, [], ""
    # Intra-type boosters: existential vague words ("anywhere") are more
    # underdetermined than directional pronouns ("there").
    if _EXISTENTIAL_VAGUE.search(instruction):
        p = min(0.90, p + 0.05)
    if len(_AMBIGUOUS_TARGET_PRONOUNS.findall(instruction)) > 1:
        p = min(0.90, p + 0.05)
    return p, ["target"], reasoning


def _score_missing_object(
    instruction: str,
    has_named_loc: bool,
) -> tuple:
    if not _OBJECT_HANDLING_VERBS.search(instruction):
        return 0.0, [], ""
    has_pronoun_obj = bool(_OBJECT_PRONOUNS.search(instruction))
    has_me = bool(_ME_OBJECT.search(instruction))
    if has_pronoun_obj and not has_named_loc:
        p = 0.75
        slots = ["object"]
        reasoning = "Object of the action is an unspecified pronoun."
    elif has_me and not has_named_loc:
        p = 0.75
        slots = ["destination"]
        reasoning = "Robot is directed to take the user somewhere but destination is not named."
    else:
        return 0.0, [], ""
    # Intra-type booster: doubly underdetermined (both object AND destination unknown).
    if has_pronoun_obj and has_me:
        p = min(0.90, p + 0.05)
    return p, slots, reasoning


def _score_ambiguous_action(instruction: str) -> tuple:
    if not _VAGUE_ACTION_VERBS.search(instruction):
        return 0.0, [], ""
    p = 0.75
    # Intra-type booster: multiple vague verbs compound the ambiguity.
    if len(_VAGUE_ACTION_VERBS.findall(instruction)) > 1:
        p = min(0.90, p + 0.05)
    return p, ["action"], "Vague action verb does not specify a concrete robot action."


def _score_missing_direction(
    instruction: str,
    tokens: frozenset,
    has_named_loc: bool,
) -> tuple:
    if not (tokens & _MOVEMENT_VERBS):
        return 0.0, [], ""
    has_direction = bool(tokens & _DIRECTION_WORDS)
    is_continuation = bool(_LETS_CONTINUE.search(instruction))
    is_manner_only = (
        bool(_MANNER_MODIFIER.search(instruction))
        and not has_direction
        and not has_named_loc
    )
    needs_turn_direction = (
        bool(_TURN_VERBS.search(instruction))
        and has_named_loc
        and not has_direction
    )
    if needs_turn_direction:
        p = 0.70
        reasoning = "Turn instruction has a destination but no left/right direction specified."
    elif not has_direction and not has_named_loc and not is_continuation and not is_manner_only:
        p = 0.70
        reasoning = "Movement verb present but no direction or destination specified."
    else:
        return 0.0, [], ""
    # Intra-type booster: very terse instruction is maximally underdetermined
    # (e.g. bare "Go" or "Move" with no other words).
    if len(tokens) <= 2:
        p = min(0.85, p + 0.05)
    return p, ["direction"], reasoning


def _score_missing_distance(instruction: str) -> tuple:
    if not _VAGUE_DISTANCE.search(instruction):
        return 0.0, [], ""
    if _SPECIFIC_DISTANCE.search(instruction):
        return 0.0, [], ""
    p = 0.70
    # Intra-type booster: multiple vague distance phrases in one instruction.
    if len(_VAGUE_DISTANCE.findall(instruction)) > 1:
        p = min(0.85, p + 0.05)
    return p, ["distance"], "Vague distance reference found with no quantified value."


def _rule_detect(instruction: str, scene_context: str) -> AmbiguityDetection:
    """
    Apply keyword/regex heuristics to classify instruction ambiguity.

    Word-bank + ranking: each ambiguity type (2–6) is scored independently by
    its own ``_score_*`` function.  Each scorer returns a continuous p in [0, 1]
    based on how many indicator patterns fire for that type (primary trigger sets
    the base p; intra-type boosters can raise it further).  Types with p below
    ``_MIN_TYPE_THRESHOLD`` are discarded.  Remaining candidates are ranked by
    severity_weight × p (descending).  When multiple types co-occur, the winner's
    p is boosted by +0.05 per additional signal, capped at 0.95.

    Rule 1 (missing_action) remains an exclusive early return — without an
    action verb the other rules are not meaningful.
    """
    tokens = frozenset(instruction.lower().split())

    # 1. missing_action — exclusive: no action verb means no actionable plan.
    if not (tokens & _ACTION_VERBS):
        return _rule_result(
            "missing_action", 0.80, ["action"],
            "No action verb found — robot cannot determine what to do.",
        )

    has_named_loc = bool(_NAMED_LOCATION.search(instruction))

    candidates: List[tuple] = []
    for atype, score in [
        ("ambiguous_target",  _score_ambiguous_target(instruction, has_named_loc)),
        ("missing_object",    _score_missing_object(instruction, has_named_loc)),
        ("ambiguous_action",  _score_ambiguous_action(instruction)),
        ("missing_direction", _score_missing_direction(instruction, tokens, has_named_loc)),
        ("missing_distance",  _score_missing_distance(instruction)),
    ]:
        p, slots, reasoning = score
        if p >= _MIN_TYPE_THRESHOLD:
            candidates.append((atype, p, slots, reasoning))

    if not candidates:
        return _rule_result(
            "no_uncertainty", 0.0, [],
            "Instruction is complete and unambiguous.",
        )

    # Rank by severity_weight × p_ambiguous (descending); deterministic tie-break by type name.
    candidates.sort(key=lambda x: (-_compute_nonconformity(x[0], x[1]), x[0]))

    best_type, best_p, best_slots, best_reasoning = candidates[0]

    # Inter-type co-occurrence boost: each additional fired type adds +0.05
    # confidence to the winner, capped at 0.95.
    n_extra = len(candidates) - 1
    if n_extra > 0:
        best_p = min(0.95, best_p + 0.05 * n_extra)

    return _rule_result(best_type, best_p, best_slots, best_reasoning)


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
