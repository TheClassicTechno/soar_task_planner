"""
Natural-language question generation for environmental uncertainty.

When the robot detects an unknown terrain region, it needs to ask the user
a clear, specific question. This module supports two modes:

  template mode: fast, no LLM call, uses pre-written templates
  llm mode:      calls an LLM to generate a more contextual question

Template mode is the default and is always available. LLM mode requires
a pre-built LLMInterface instance.
"""

from typing import List, Optional

from system.env_uncertainty.detector import DetectionResult
from system.env_uncertainty.trajectory import Trajectory


# ── Question templates ────────────────────────────────────────────────────────
# Indexed by (number_of_unknown_regions, best_alternative_available).
# Best_alternative: True if a safe trajectory exists that avoids unknown areas.

_TEMPLATE_HAS_ALTERNATIVE = (
    "I see an unrecognized area ahead on my current path. "
    "I can take a longer alternative route to avoid it. "
    "Should I take the alternative route, or is the area ahead safe to cross?"
)

_TEMPLATE_NO_ALTERNATIVE = (
    "There is an unrecognized area directly in my path. "
    "I cannot determine if it is safe to cross. "
    "Should I stop and wait for you to assess it?"
)

_TEMPLATE_LARGE_UNKNOWN = (
    "Most of the area ahead is unrecognizable to me — "
    "I cannot determine if the terrain is safe. "
    "Should I stop here?"
)

_TEMPLATE_MULTIPLE_UNKNOWNS = (
    "I see multiple unrecognized areas ahead. "
    "I am not sure which path is safe. "
    "Can you tell me where it is safe to go?"
)

# Threshold for "large unknown" — if unknown_coverage exceeds this fraction,
# use the large-unknown template.
_LARGE_UNKNOWN_THRESHOLD = 0.50


def generate_question_template(
    result: DetectionResult,
    trajectories: Optional[List[Trajectory]] = None,
) -> str:
    """
    Generate a clarification question using pre-written templates.

    Selects the most appropriate template based on:
      - How much of the scene is unknown (large_unknown threshold)
      - How many distinct unknown regions exist
      - Whether any safe alternative trajectory is available

    Args:
        result:      DetectionResult from EnvironmentalUncertaintyDetector.
        trajectories: Scored trajectories (used to check for safe alternatives).

    Returns:
        A natural-language question string.
    """
    if result.unknown_coverage >= _LARGE_UNKNOWN_THRESHOLD:
        return _TEMPLATE_LARGE_UNKNOWN

    if len(result.unknown_regions) >= 3:
        return _TEMPLATE_MULTIPLE_UNKNOWNS

    # Check if any trajectory avoids all unknown regions
    has_safe_alternative = False
    if trajectories:
        has_safe_alternative = any(
            not t.passes_through_unknown for t in trajectories
        )

    if has_safe_alternative:
        return _TEMPLATE_HAS_ALTERNATIVE
    return _TEMPLATE_NO_ALTERNATIVE


class QuestionGenerator:
    """
    Generate clarification questions for unknown terrain regions.

    Supports two modes:
      - "template": fast, deterministic, no LLM call (default)
      - "llm":      uses an LLMInterface to produce context-aware questions

    In LLM mode, falls back to template mode if the LLM call fails.
    """

    def __init__(self, mode: str = "template", llm: Optional[object] = None):
        """
        Args:
            mode: "template" or "llm".
            llm:  LLMInterface instance (required for mode="llm").
        """
        if mode not in ("template", "llm"):
            raise ValueError(f"mode must be 'template' or 'llm', got '{mode}'")
        if mode == "llm" and llm is None:
            raise ValueError("mode='llm' requires an LLMInterface instance")
        self._mode = mode
        self._llm = llm

    def generate(
        self,
        result: DetectionResult,
        trajectories: Optional[List[Trajectory]] = None,
    ) -> str:
        """
        Generate a clarification question for the detected unknown regions.

        Args:
            result:      DetectionResult from EnvironmentalUncertaintyDetector.
            trajectories: Scored trajectories (used to detect safe alternatives).

        Returns:
            A natural-language question string.
        """
        if self._mode == "llm":
            try:
                return self._llm_question(result, trajectories)
            except Exception:
                # LLM failure → fall back to template
                pass
        return generate_question_template(result, trajectories)

    def _llm_question(
        self,
        result: DetectionResult,
        trajectories: Optional[List[Trajectory]],
    ) -> str:
        """
        Use an LLM to generate a context-rich question.

        Builds a prompt summarizing the detection result and trajectory
        options, then calls llm.predict_json() for a JSON response with
        a "question" field.
        """
        prompt = _build_llm_prompt(result, trajectories)
        response = self._llm.predict_json(prompt)
        question = response.get("question", "").strip()
        if not question:
            raise ValueError("LLM returned empty question")
        return question


# ── LLM prompt builder ────────────────────────────────────────────────────────

def _build_llm_prompt(
    result: DetectionResult,
    trajectories: Optional[List[Trajectory]],
) -> str:
    """
    Build the prompt sent to the LLM for question generation.

    Summarizes: unknown region count and size, safe trajectory availability,
    and asks the LLM to produce a one-sentence user-facing question.
    """
    n_unknown = len(result.unknown_regions)
    pct_unknown = int(result.unknown_coverage * 100)

    has_safe = False
    if trajectories:
        has_safe = any(not t.passes_through_unknown for t in trajectories)

    traj_note = (
        "A safe alternative route exists that avoids the unknown area."
        if has_safe
        else "No safe alternative route was found — all paths lead through the unknown area."
    )

    prompt = f"""You are an outdoor navigation robot. Your terrain perception system has detected an area it cannot classify.

Unknown region summary:
- Number of unknown regions: {n_unknown}
- Unknown area: approximately {pct_unknown}% of the scene
- {traj_note}

Generate a single short clarification question to ask the human user. The question should:
1. Briefly describe what the robot observes
2. Ask whether it is safe to proceed
3. Mention the alternative route if one exists
4. Be polite and concise (1-2 sentences)

Respond in JSON:
{{"question": "<your question here>"}}"""
    return prompt
