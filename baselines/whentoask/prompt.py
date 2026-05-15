"""
When-to-Ask (UPS) prompt templates.

Two-stage Bayesian factorization from UPS §IV-B (Eq. 5):
  p(y | options, L) = Σ_θ P(y | options, θ) * P(θ | L)

Stage 1 — Intent hypotheses: Ask the LLM what possible user intents θ ∈ Θ
  could be behind the instruction L, each with a probability P(θ | L).

Stage 2 — Option scoring per intent: For each hypothesized intent θ,
  ask the LLM to score each option (A/B/C/D/E) given that specific intent.
  E = "none of the above" (incapability signal).

The final score is the weighted sum: score[y] = Σ_θ P(θ|L) * P(y|options, θ).

Why factorize? Raw VLM scoring is overconfident — it assigns near-1.0 to one
option even when the instruction is ambiguous. By explicitly marginalizing over
user intents, the model is forced to split probability mass, which calibrates
p_VLM before conformal prediction even runs. This produces tighter, more
informative prediction sets (UPS empirically shows ~30% improvement over
uncalibrated baselines on ambiguous scenarios).

None option (option E):
  E represents "none of the above — the robot cannot safely navigate in this
  situation at all." When E appears in the prediction set, the system
  switches to INCAPABLE resolution strategy instead of CLARIFY.
"""

from typing import Dict, List, Tuple

# Option E label — kept separate from nav options A/B/C/D
NONE_OPTION_LABEL = "E"
NONE_OPTION_TEXT = (
    "None of the above — the robot cannot safely navigate in this "
    "situation without additional capabilities or human takeover."
)

# ── Stage 1: Intent hypotheses ────────────────────────────────────────────────

INTENT_HYPOTHESES_PROMPT = """You are a navigation assistant analyzing an outdoor robot's situation.

## Situation
Instruction: {instruction}
Terrain: {terrain_description}

## Task
The robot received the instruction above. What are the {n_intents} most likely
distinct user intents behind this instruction? Consider what the user might
want the robot to accomplish, including possible ambiguities.

Return ONLY a JSON object with this structure:
{{
  "intents": [
    {{"description": "<one-sentence intent>", "probability": <float 0.0-1.0>}},
    ...
  ]
}}

The probabilities must sum to 1.0. List the most probable intent first."""


def format_intent_prompt(
    instruction: str,
    terrain_description: str,
    n_intents: int = 3,
) -> str:
    """
    Format Stage 1: ask LLM to hypothesize possible user intents.

    Args:
        instruction:         The user's navigation command.
        terrain_description: Description of the current terrain.
        n_intents:           Number of intents to hypothesize (default 3).

    Returns:
        Formatted prompt string.
    """
    return INTENT_HYPOTHESES_PROMPT.format(
        instruction=instruction,
        terrain_description=terrain_description,
        n_intents=n_intents,
    )


# ── Stage 2: Option scoring given a specific intent ───────────────────────────

OPTION_SCORING_PROMPT = """You are a navigation decision system for an outdoor mobile robot.

## Situation
Instruction: {instruction}
Terrain: {terrain_description}

## Assumed User Intent
{intent_description}

## Options (score each given the above intent)
{options_block}
  E: {none_option_text}

## Task
Given the assumed user intent, score each option (A through E) independently
from 0.0 (completely inappropriate) to 1.0 (clearly correct).

Scores do NOT need to sum to 1.0 — each is an independent appropriateness rating.
Score option E highly only if the robot genuinely cannot handle this safely at all.

Return ONLY a JSON object:
{{
  "scores": {{
    "A": <float 0.0-1.0>,
    "B": <float 0.0-1.0>,
    "C": <float 0.0-1.0>,
    "D": <float 0.0-1.0>,
    "E": <float 0.0-1.0>
  }},
  "reasoning": "<one sentence>"
}}"""


def format_scoring_prompt(
    instruction: str,
    terrain_description: str,
    intent_description: str,
    options: Dict[str, str],
) -> str:
    """
    Format Stage 2: ask LLM to score all options given a specific user intent.

    Args:
        instruction:         The user's navigation command.
        terrain_description: Description of the current terrain.
        intent_description:  One specific hypothesized user intent.
        options:             {"A": "...", "B": "...", "C": "...", "D": "..."}

    Returns:
        Formatted prompt string.
    """
    options_block = "\n".join(
        f"  {label}: {description}"
        for label, description in sorted(options.items())
        if label != NONE_OPTION_LABEL
    )
    return OPTION_SCORING_PROMPT.format(
        instruction=instruction,
        terrain_description=terrain_description,
        intent_description=intent_description,
        options_block=options_block,
        none_option_text=NONE_OPTION_TEXT,
    )


# ── Fallback: single-call scoring (factorization disabled) ────────────────────

DIRECT_SCORING_PROMPT = """You are a navigation decision system for an outdoor mobile robot.

## Situation
Instruction: {instruction}
Terrain: {terrain_description}

## Options
{options_block}
  E: {none_option_text}

## Task
Score each option (A through E) independently from 0.0 to 1.0.
Score option E highly only if the robot genuinely cannot handle this situation safely at all.

Return ONLY a JSON object:
{{
  "scores": {{
    "A": <float 0.0-1.0>,
    "B": <float 0.0-1.0>,
    "C": <float 0.0-1.0>,
    "D": <float 0.0-1.0>,
    "E": <float 0.0-1.0>
  }},
  "reasoning": "<one sentence>"
}}"""


def format_direct_scoring_prompt(
    instruction: str,
    terrain_description: str,
    options: Dict[str, str],
) -> str:
    """Format single-call scoring prompt (no factorization)."""
    options_block = "\n".join(
        f"  {label}: {description}"
        for label, description in sorted(options.items())
        if label != NONE_OPTION_LABEL
    )
    return DIRECT_SCORING_PROMPT.format(
        instruction=instruction,
        terrain_description=terrain_description,
        options_block=options_block,
        none_option_text=NONE_OPTION_TEXT,
    )


# ── Parsing helpers ───────────────────────────────────────────────────────────

def parse_intent_response(llm_response: Dict, n_intents: int = 3) -> List[Tuple[str, float]]:
    """
    Parse the Stage 1 intent hypotheses response.

    Args:
        llm_response: Parsed JSON from format_intent_prompt call.
        n_intents:    Expected number of intents (used for normalization fallback).

    Returns:
        List of (intent_description, probability) tuples, probabilities sum to 1.0.
    """
    raw = llm_response.get("intents", [])
    if not raw:
        return [("Navigate as instructed", 1.0)]

    intents: List[Tuple[str, float]] = []
    for item in raw[:n_intents]:
        desc = str(item.get("description", "Navigate as instructed"))
        prob = float(item.get("probability", 1.0 / n_intents))
        intents.append((desc, max(0.0, min(1.0, prob))))

    if not intents:
        return [("Navigate as instructed", 1.0)]

    # Normalize probabilities to sum to 1.0
    total = sum(p for _, p in intents)
    if total <= 0:
        n = len(intents)
        return [(d, 1.0 / n) for d, _ in intents]
    return [(d, p / total) for d, p in intents]


def parse_option_scores(llm_response: Dict) -> Dict[str, float]:
    """
    Parse per-option scores from a Stage 2 (or direct) scoring response.

    Args:
        llm_response: Parsed JSON from format_scoring_prompt call.

    Returns:
        {"A": float, "B": float, "C": float, "D": float, "E": float}
        Missing keys default to 0.25 (for A-D) or 0.0 (for E, conservative).
    """
    raw = llm_response.get("scores", {})
    result: Dict[str, float] = {}

    for label in ["A", "B", "C", "D"]:
        val = raw.get(label, 0.25)
        try:
            result[label] = float(max(0.0, min(1.0, val)))
        except (TypeError, ValueError):
            result[label] = 0.25

    # E (none-of-above) defaults to 0 — incapability should be rare
    val_e = raw.get("E", 0.0)
    try:
        result["E"] = float(max(0.0, min(1.0, val_e)))
    except (TypeError, ValueError):
        result["E"] = 0.0

    return result


def marginalize_scores(
    intent_scores: List[Tuple[Dict[str, float], float]]
) -> Dict[str, float]:
    """
    Marginalize per-option scores over multiple intent hypotheses.

    Implements Eq. 5 from UPS:
      p(y | options, L) = Σ_θ P(y | options, θ) * P(θ | L)

    Args:
        intent_scores: List of (scores_dict, intent_probability) pairs.
                       scores_dict has keys A/B/C/D/E.

    Returns:
        Weighted sum of scores, normalized to reflect the marginal distribution.
    """
    labels = ["A", "B", "C", "D", "E"]
    result = {label: 0.0 for label in labels}

    for scores, prob in intent_scores:
        for label in labels:
            result[label] += scores.get(label, 0.0) * prob

    return result
