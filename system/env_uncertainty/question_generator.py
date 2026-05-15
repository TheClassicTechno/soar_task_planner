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

from typing import Any, List, Optional, Tuple

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


# ── Scene-graph-grounded question formatters ──────────────────────────────────
# Used when the Dirichlet posterior gives us candidate terrain classes.
# These replace the generic "unrecognized area" phrasing with the actual
# top-k class names the robot currently believes it might be looking at.

def _grounded_standard(classes: List[Tuple[str, float]]) -> str:
    names = [c[0] for c in classes[:3]]
    if len(names) == 1:
        candidates = names[0]
    elif len(names) == 2:
        candidates = f"{names[0]} or {names[1]}"
    else:
        candidates = f"{names[0]}, {names[1]}, or {names[2]}"
    return (
        f"I see something ahead that I'm not sure about — "
        f"it could be {candidates}. "
        "Is it safe to cross?"
    )


def _grounded_terse(classes: List[Tuple[str, float]]) -> str:
    names = ", ".join(c[0] for c in classes[:3])
    return f"Unknown terrain — possibly {names}. Safe to cross?"


def _grounded_verbose(classes: List[Tuple[str, float]]) -> str:
    lines = "\n".join(
        f"  {name} ({int(prob * 100)}%)" for name, prob in classes[:3]
    )
    return (
        "My terrain classifier is uncertain about the surface ahead. "
        "Based on current observations the most likely terrain classes are:\n"
        f"{lines}\n"
        "I cannot confidently estimate traversability for any of these. "
        "Is this area safe to cross?"
    )


def generate_question_template(
    result: DetectionResult,
    trajectories: Optional[List[Trajectory]] = None,
    profile: Optional[UserProfile] = None,
    top_k_classes: Optional[List[Tuple[str, float]]] = None,
) -> str:
    """
    Generate a clarification question using pre-written templates.

    Selects the most appropriate template based on:
      - top_k_classes: if provided (from scene graph Dirichlet posterior),
        generates a grounded question naming the candidate terrain classes.
        This is the primary case when the robot has semantic uncertainty.
      - Otherwise selects by unknown coverage, region count, and safe alternatives.
      - The user profile's verbosity and preferred_format settings always apply.

    Args:
        result:         DetectionResult from EnvironmentalUncertaintyDetector.
        trajectories:   Scored trajectories (used to check for safe alternatives).
        profile:        UserProfile controlling verbosity and format. Uses
                        DEFAULT_PROFILE if None.
        top_k_classes:  Top-k terrain candidates from TerrainNode.top_k_classes().
                        When provided, overrides generic templates with a grounded
                        question that names what the robot thinks it might be seeing.

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

    # Grounded path: scene graph gave us Dirichlet top-k candidates.
    # Use terrain-specific question instead of generic "unrecognized area" wording.
    # Option-list format is excluded — grounded questions don't map cleanly to numbered options.
    if top_k_classes and fmt != "option_list":
        if verbosity == "terse":
            return _grounded_terse(top_k_classes)
        if verbosity == "verbose":
            return _grounded_verbose(top_k_classes)
        return _grounded_standard(top_k_classes)

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
        top_k_classes: Optional[List[Tuple[str, float]]] = None,
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
            top_k_classes:    Top-k terrain candidates from TerrainNode.top_k_classes().
                              When provided, the question names the candidates so the
                              user knows what the robot thinks it might be seeing.

        Returns:
            A natural-language question string.
        """
        profile = user_profile or DEFAULT_PROFILE

        if self._mode == "llm":
            try:
                return self._llm_question(
                    result, trajectories, profile, scenario_context, top_k_classes
                )
            except Exception:
                pass
        return generate_question_template(result, trajectories, profile, top_k_classes)

    def _llm_question(
        self,
        result: DetectionResult,
        trajectories: Optional[List[Trajectory]],
        profile: UserProfile,
        scenario_context: Optional[str],
        top_k_classes: Optional[List[Tuple[str, float]]] = None,
    ) -> str:
        """
        Use an LLM to generate a context-rich, profile-aware question.

        Builds a prompt summarizing the detection result, trajectory options,
        user profile, optional scenario context, and Dirichlet top-k terrain
        candidates when available.
        """
        prompt = _build_llm_prompt(
            result, trajectories, profile, scenario_context, top_k_classes
        )
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
    top_k_classes: Optional[List[Tuple[str, float]]] = None,
) -> str:
    """
    Build the prompt sent to the LLM for question generation.

    Includes: unknown region summary, trajectory safety, user profile,
    optional scenario context, and Dirichlet top-k terrain candidates
    when the scene graph has them.
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

    # Inject Dirichlet top-k candidates so the LLM names them in the question.
    if top_k_classes:
        candidates = "\n".join(
            f"  {i+1}. {name} ({int(prob * 100)}%)"
            for i, (name, prob) in enumerate(top_k_classes)
        )
        top_k_section = (
            f"\nTerrain class candidates (Dirichlet posterior):\n{candidates}\n"
            "Name these candidates in your question so the user knows what the robot thinks it sees.\n"
        )
    else:
        top_k_section = ""

    profile_section = describe_profile_for_prompt(profile)

    prompt = f"""You are an outdoor navigation robot. Your terrain perception system has detected an area it cannot classify.

Unknown region summary:
- Number of unknown regions: {n_unknown}
- Unknown area: approximately {pct_unknown}% of the scene
- {traj_note}
{context_line}{top_k_section}
{profile_section}

Generate a single clarification question to ask the human user. Strictly follow the user profile above for verbosity, expertise level, and format.

Respond in JSON:
{{"question": "<your question here>"}}"""
    return prompt
