"""
Natural-language question generation for environmental uncertainty.

When the robot detects an unknown terrain region, it needs to ask the user
a clear, specific question. This module supports two modes:

  template mode: fast, no LLM call, uses pre-written templates
  llm mode:      calls an LLM to generate a more contextual question

Template mode is the default and is always available. LLM mode requires
a pre-built LLMInterface instance.

Both modes support an optional UserProfile that adapts question style
(verbosity, expertise level, output format) to the specific operator.
"""

from typing import Any, List, Optional

from system.env_uncertainty.detector import DetectionResult
from system.env_uncertainty.trajectory import Trajectory
from system.env_uncertainty.user_profile import (
    DEFAULT_PROFILE,
    UserProfile,
    describe_profile_for_prompt,
)


# ── Question templates — standard verbosity ───────────────────────────────────

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

# ── Question templates — terse verbosity ─────────────────────────────────────

_TEMPLATE_TERSE_HAS_ALTERNATIVE = "Unknown terrain ahead. Take alternate route?"
_TEMPLATE_TERSE_NO_ALTERNATIVE = "Unknown terrain in path. Stop?"
_TEMPLATE_TERSE_LARGE_UNKNOWN = "Unknown terrain ahead. Stop?"
_TEMPLATE_TERSE_MULTIPLE_UNKNOWNS = "Multiple unknown areas. Which path is safe?"

# ── Question templates — verbose verbosity ────────────────────────────────────

_TEMPLATE_VERBOSE_HAS_ALTERNATIVE = (
    "My terrain perception system has identified an unrecognized surface region "
    "that does not match any class in my terrain vocabulary. "
    "I cannot estimate traversability for this area. "
    "A longer alternative route is available that avoids the unknown region. "
    "Shall I take the alternative route, or do you confirm that the area ahead is safe to cross?"
)

_TEMPLATE_VERBOSE_NO_ALTERNATIVE = (
    "My terrain perception system has identified an unrecognized surface region "
    "that does not match any class in my terrain vocabulary (traversability: unknown). "
    "All candidate trajectories pass through this region and I cannot determine safety. "
    "I recommend stopping. Do you want me to hold position while you assess the area?"
)

_TEMPLATE_VERBOSE_LARGE_UNKNOWN = (
    "My terrain classifier cannot label the majority of the scene ahead "
    "(approximately {pct}% of the forward view is unclassified). "
    "No traversability estimate is available. "
    "I recommend stopping until you can assess conditions. "
    "Shall I hold position here?"
)

_TEMPLATE_VERBOSE_MULTIPLE_UNKNOWNS = (
    "I have detected {n} distinct unrecognized surface regions ahead. "
    "I cannot determine which, if any, are safe to cross. "
    "All candidate trajectories encounter at least one unknown region. "
    "Can you guide me to a safe path, or shall I stop here?"
)

# ── Option-list templates ──────────────────────────────────────────────────────

_TEMPLATE_OPTIONS_HAS_ALTERNATIVE = (
    "I see something I don't recognize in my path. What should I do?\n"
    "  1. Take the longer alternative route around it\n"
    "  2. Cross through it (you confirm it is safe)\n"
    "  3. Stop here and wait"
)

_TEMPLATE_OPTIONS_NO_ALTERNATIVE = (
    "There is an unknown area in my path and no way around it. What should I do?\n"
    "  1. Stop here and wait for your assessment\n"
    "  2. Proceed through it carefully"
)

# Threshold for "large unknown" — if unknown_coverage exceeds this fraction,
# use the large-unknown template.
_LARGE_UNKNOWN_THRESHOLD = 0.50


def generate_question_template(
    result: DetectionResult,
    trajectories: Optional[List[Trajectory]] = None,
    profile: Optional[UserProfile] = None,
) -> str:
    """
    Generate a clarification question using pre-written templates.

    Selects the most appropriate template based on:
      - How much of the scene is unknown (large_unknown threshold)
      - How many distinct unknown regions exist
      - Whether any safe alternative trajectory is available
      - The user profile's verbosity and preferred_format settings

    Args:
        result:       DetectionResult from EnvironmentalUncertaintyDetector.
        trajectories: Scored trajectories (used to check for safe alternatives).
        profile:      UserProfile controlling verbosity and format. Uses
                      DEFAULT_PROFILE if None.

    Returns:
        A natural-language question string.
    """
    p = profile or DEFAULT_PROFILE
    verbosity = p.verbosity
    fmt = p.preferred_format

    n_unknown = len(result.unknown_regions)
    pct = int(result.unknown_coverage * 100)
    has_safe = bool(trajectories and any(
        not t.passes_through_unknown for t in trajectories
    ))

    # Option-list format overrides verbosity for template selection
    if fmt == "option_list":
        if has_safe:
            return _TEMPLATE_OPTIONS_HAS_ALTERNATIVE
        return _TEMPLATE_OPTIONS_NO_ALTERNATIVE

    if verbosity == "terse":
        if result.unknown_coverage >= _LARGE_UNKNOWN_THRESHOLD:
            return _TEMPLATE_TERSE_LARGE_UNKNOWN
        if n_unknown >= 3:
            return _TEMPLATE_TERSE_MULTIPLE_UNKNOWNS
        if has_safe:
            return _TEMPLATE_TERSE_HAS_ALTERNATIVE
        return _TEMPLATE_TERSE_NO_ALTERNATIVE

    if verbosity == "verbose":
        if result.unknown_coverage >= _LARGE_UNKNOWN_THRESHOLD:
            return _TEMPLATE_VERBOSE_LARGE_UNKNOWN.format(pct=pct)
        if n_unknown >= 3:
            return _TEMPLATE_VERBOSE_MULTIPLE_UNKNOWNS.format(n=n_unknown)
        if has_safe:
            return _TEMPLATE_VERBOSE_HAS_ALTERNATIVE
        return _TEMPLATE_VERBOSE_NO_ALTERNATIVE

    # standard verbosity (default)
    if result.unknown_coverage >= _LARGE_UNKNOWN_THRESHOLD:
        return _TEMPLATE_LARGE_UNKNOWN
    if n_unknown >= 3:
        return _TEMPLATE_MULTIPLE_UNKNOWNS
    if has_safe:
        return _TEMPLATE_HAS_ALTERNATIVE
    return _TEMPLATE_NO_ALTERNATIVE


class QuestionGenerator:
    """
    Generate clarification questions for unknown terrain regions.

    Supports two modes:
      - "template": fast, deterministic, no LLM call (default)
      - "llm":      uses an LLMInterface to produce context-aware questions

    Both modes accept an optional UserProfile to adapt question style.
    In LLM mode, the profile is injected into the prompt so the model
    produces output matching the user's verbosity and expertise level.
    In LLM mode, falls back to template mode if the LLM call fails.
    """

    def __init__(self, mode: str = "template", llm: Optional[Any] = None):
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
        user_profile: Optional[UserProfile] = None,
        scenario_context: Optional[str] = None,
    ) -> str:
        """
        Generate a clarification question for the detected unknown regions.

        Args:
            result:           DetectionResult from EnvironmentalUncertaintyDetector.
            trajectories:     Scored trajectories (used to detect safe alternatives).
            user_profile:     UserProfile controlling verbosity, expertise, format.
                              Uses DEFAULT_PROFILE if None.
            scenario_context: Optional deployment context (e.g., "construction zone",
                              "night operation") for further question tailoring.

        Returns:
            A natural-language question string.
        """
        profile = user_profile or DEFAULT_PROFILE

        if self._mode == "llm":
            try:
                return self._llm_question(result, trajectories, profile, scenario_context)
            except Exception:
                pass
        return generate_question_template(result, trajectories, profile)

    def _llm_question(
        self,
        result: DetectionResult,
        trajectories: Optional[List[Trajectory]],
        profile: UserProfile,
        scenario_context: Optional[str],
    ) -> str:
        """
        Use an LLM to generate a context-rich, profile-aware question.

        Builds a prompt summarizing the detection result, trajectory options,
        user profile, and optional scenario context. Calls llm.predict_json()
        for a JSON response with a "question" field.
        """
        prompt = _build_llm_prompt(result, trajectories, profile, scenario_context)
        response = self._llm.predict_json(prompt)
        question = response.get("question", "").strip()
        if not question:
            raise ValueError("LLM returned empty question")
        return question


# ── LLM prompt builder ────────────────────────────────────────────────────────

def _build_llm_prompt(
    result: DetectionResult,
    trajectories: Optional[List[Trajectory]],
    profile: UserProfile,
    scenario_context: Optional[str],
) -> str:
    """
    Build the prompt sent to the LLM for question generation.

    Includes: unknown region summary, trajectory safety, user profile
    description, and optional scenario context.
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

    context_line = (
        f"Deployment context: {scenario_context}\n"
        if scenario_context
        else ""
    )

    profile_section = describe_profile_for_prompt(profile)

    prompt = f"""You are an outdoor navigation robot. Your terrain perception system has detected an area it cannot classify.

Unknown region summary:
- Number of unknown regions: {n_unknown}
- Unknown area: approximately {pct_unknown}% of the scene
- {traj_note}
{context_line}
{profile_section}

Generate a single clarification question to ask the human user. Strictly follow the user profile above for verbosity, expertise level, and format.

Respond in JSON:
{{"question": "<your question here>"}}"""
    return prompt
