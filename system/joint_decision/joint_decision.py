"""
Joint uncertainty decision — combines instruction branch (κ_I) and environmental
branch (κ_E) into a single PROCEED / ASK / STOP decision with a formal coverage
guarantee from conformal prediction.

Pipeline position: after both branches have computed their nonconformity scores.

    κ_joint = max(κ_I, κ_E)

If either branch signals uncertainty, the joint score is high and the robot asks.
This is deliberately conservative: it is better to ask once too many than to
drive into unknown terrain.

Novel contribution vs. KnowNo / WhenToAsk
------------------------------------------
KnowNo:     κ_I only (LLM logit-based, instruction branch only)
WhenToAsk:  κ_I + policy-incapability score (still no terrain map)
This module: κ_joint = max(κ_I, κ_E) where κ_E comes from the GP traversability
             map — the first principled terrain uncertainty score in this pipeline type.

κ_E normalization
------------------
κ_E = min(unknown_coverage / stop_threshold, 1.0)

This maps:
  unknown_coverage = 0.00 → κ_E = 0.00  (fully known scene)
  unknown_coverage = 0.40 → κ_E = 0.50  (moderate uncertainty)
  unknown_coverage = 0.80 → κ_E = 1.00  (stop threshold, maximum uncertainty)

The normalization is linear and interpretable. The stop threshold (0.80) is the
same one used in runner._decide_action(), keeping both branches aligned.

Thresholds
-----------
  ask_threshold  (default 0.15): κ_joint above this → ASK
  stop_threshold (default 1.00): κ_joint at or above this → STOP

Why ask_threshold=0.15?
  ambiguous_target (w=0.75) with p_ambiguous=0.20 → κ_I = 0.15. This is the
  lowest plausible confidence level that still carries meaningful ambiguity.
  Setting the threshold here means the robot asks only when there is genuine
  uncertainty, not noise.

Why stop_threshold=1.00?
  κ_joint=1.0 requires full environmental uncertainty (100% coverage OR explicit
  STOP from the env runner). The joint STOP is the most conservative possible
  outcome and should only fire in extreme cases.

Joint decision vs. independent decisions
-----------------------------------------
  env_decision.robot_action is the environmental decision (may be ASK or STOP
  from LCB or coverage checks). The joint decision can ONLY escalate from
  env_decision — it cannot downgrade a STOP to PROCEED. If env_decision is STOP,
  the joint always returns STOP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from system.env_uncertainty.runner import EnvironmentalUncertaintyRunner, EnvUncertaintyDecision
from system.instruction_uncertainty.ambiguity_detector import AmbiguityDetection, AmbiguityDetector

# The stop_threshold is the same value used by EnvironmentalUncertaintyRunner
# to cap κ_E at 1.0 when unknown_coverage >= this value.
_COVERAGE_STOP_THRESHOLD = 0.80


@dataclass
class JointDecision:
    """
    Output from JointDecisionMaker.decide().

    kappa_I:                 Instruction nonconformity score (from AmbiguityDetector).
    kappa_E:                 Environmental nonconformity score (normalized coverage).
    kappa_joint:             max(kappa_I, kappa_E).
    instruction_ambiguity:   Full AmbiguityDetection dataclass from instruction branch.
    env_decision:            Full EnvUncertaintyDecision from environmental branch.
    final_action:            "PROCEED", "ASK", or "STOP".
    question:                Clarification question if final_action == "ASK" or "STOP".
                             Comes from the dominant branch:
                               κ_I > κ_E → instruction clarification question
                               κ_E >= κ_I → terrain/environment question
    dominant_branch:         "instruction", "environment", or "none" — identifies
                             which branch triggered the ASK/STOP.  "none" on PROCEED.
    """

    kappa_I: float
    kappa_E: float
    kappa_joint: float
    instruction_ambiguity: AmbiguityDetection
    env_decision: EnvUncertaintyDecision
    final_action: str
    question: Optional[str]
    dominant_branch: str = "none"


_INSTRUCTION_QUESTIONS = {
    "missing_action": (
        "Could you clarify what action you'd like me to perform? "
        "I see a navigation environment but no action verb in your command."
    ),
    "ambiguous_target": (
        "Which location or object did you mean? "
        "Could you describe it more specifically — for example, name a landmark or give a direction?"
    ),
    "missing_object": (
        "What object or destination should I navigate to? "
        "Your instruction specifies an action but not a clear target."
    ),
    "ambiguous_action": (
        "Could you clarify what you'd like me to do? "
        "I'm unsure whether you want me to avoid, stop near, or pass through the area ahead."
    ),
    "missing_direction": (
        "Which direction should I go? "
        "Please specify left, right, straight ahead, or name a visible landmark."
    ),
    "missing_distance": (
        "How far should I travel? "
        "Could you give a specific distance (e.g., '5 meters') or name a stopping landmark?"
    ),
}


def _build_terrain_scene_context(env_decision: "EnvUncertaintyDecision") -> str:
    """
    Build a concise terrain description from env_decision for the instruction branch.

    This feeds real terrain observations into scene_context so the ambiguity
    detector's rule-based and LLM paths both see what the robot actually sees —
    making κ_I terrain-aware rather than purely instruction-text-aware.

    Examples:
      "Terrain ahead: grass (traversable). Unknown coverage: 0%."
      "Terrain ahead: unknown terrain (40% of view unidentified). Unknown coverage: 40%."
    """
    try:
        parts = []
        cov = getattr(env_decision, "unknown_coverage", 0.0)
        action = getattr(env_decision, "robot_action", "PROCEED")

        if cov > 0.05:
            parts.append(f"Unknown terrain covers {cov:.0%} of the robot's view.")
        if action == "STOP":
            parts.append("Environmental branch determined terrain is unsafe to traverse.")
        elif action == "ASK" and cov > 0.05:
            parts.append("Robot is uncertain about terrain ahead.")

        return " ".join(parts) if parts else "Terrain ahead appears clear and traversable."
    except Exception:
        return ""


def _terrain_context_suffix(env_decision: "EnvUncertaintyDecision") -> str:
    """
    Build a one-line terrain summary from the environmental branch decision.

    Per june1meeting lines 88–94: the scene graph should inform instruction
    disambiguation.  This suffix appends what the robot actually sees so the
    user can ground their answer in the current visual context.

    Only generates a suffix when there is MEANINGFUL uncertainty to report
    (unknown_coverage > 5%).  When the scene is fully clear, the instruction
    ambiguity is purely linguistic — no terrain context needed.

    Returns "" when there is nothing informative to add.  Never raises.
    """
    try:
        # Only append terrain context when there's actual environmental uncertainty.
        # A clear scene (unknown_coverage ≤ 0.05) needs no terrain grounding.
        if env_decision.unknown_coverage <= 0.05:
            return ""
        parts: List[str] = []
        if env_decision.unknown_coverage > 0.05:
            parts.append(
                f"{env_decision.unknown_coverage:.0%} of my view is unidentified terrain"
            )
        if not parts:
            return ""
        return "(In my current view: " + "; ".join(parts) + ".)"
    except Exception:
        return ""


def generate_instruction_question(
    amb: AmbiguityDetection,
    env_decision: Optional["EnvUncertaintyDecision"] = None,
) -> str:
    """
    Generate a targeted natural-language clarification question for an ambiguous instruction.

    The question is tailored to the detected ambiguity type so the user knows
    exactly what slot is missing.  When env_decision is supplied (and contains
    terrain observations), a brief terrain-context suffix is appended so the
    instruction question is grounded in what the robot actually sees — directly
    implementing the june1meeting requirement that scene graph and instruction
    branch must be connected (lines 88–94, notes lines 311–313).

    Args:
        amb:          AmbiguityDetection from the instruction branch.
        env_decision: Optional environmental branch output.  When provided, its
                      terrain summary is appended to the question to connect
                      scene-graph context to instruction disambiguation.

    Returns:
        Instruction-focused clarification question string.
    """
    base = _INSTRUCTION_QUESTIONS.get(amb.ambiguity_type)
    if not base:
        slots_str = ", ".join(amb.missing_slots) if amb.missing_slots else "the instruction"
        base = f"Could you clarify your instruction? I'm uncertain about: {slots_str}."

    if env_decision is not None:
        suffix = _terrain_context_suffix(env_decision)
        if suffix:
            return f"{base} {suffix}"
    return base


def compute_kappa_E(env_decision: EnvUncertaintyDecision) -> float:
    """
    Normalize environmental uncertainty to [0, 1].

    κ_E = min(unknown_coverage / stop_threshold, 1.0)

    This keeps κ_E on the same scale as κ_I so the max() comparison is
    meaningful. unknown_coverage=0.80 (full stop threshold) gives κ_E=1.0.
    """
    return min(env_decision.unknown_coverage / _COVERAGE_STOP_THRESHOLD, 1.0)


def compute_kappa_joint(kappa_I: float, kappa_E: float) -> float:
    """
    κ_joint = max(κ_I, κ_E).

    If either branch signals uncertainty, the robot asks.
    """
    return max(kappa_I, kappa_E)


class JointDecisionMaker:
    """
    Combine instruction and environmental uncertainty into one decision.

    Args:
        env_runner:          Configured EnvironmentalUncertaintyRunner instance.
        ambiguity_detector:  Configured AmbiguityDetector instance.
        ask_threshold:       κ_joint >= this value → ASK (default 0.15).
        stop_threshold:      κ_joint >= this value → STOP (default 1.0).

    The env_runner already handles its own LCB-STOP and coverage-STOP internally.
    The joint layer can only escalate: it checks κ_joint for ASK, and upgrades
    the action to STOP if κ_joint >= stop_threshold OR env_decision is STOP.
    """

    def __init__(
        self,
        env_runner: EnvironmentalUncertaintyRunner,
        ambiguity_detector: AmbiguityDetector,
        ask_threshold: float = 0.15,
        stop_threshold: float = 1.0,
    ) -> None:
        self._env_runner = env_runner
        self._ambiguity_detector = ambiguity_detector
        self._ask_threshold = ask_threshold
        self._stop_threshold = stop_threshold

    def decide(
        self,
        instruction: str,
        image,
        scene_context: str = "",
        scene_graph=None,
    ) -> JointDecision:
        """
        Run both branches and return a joint decision.

        Execution order (june1meeting lines 88-94, june4 integration fix):
          1. Environmental branch runs FIRST — produces terrain labels and coverage.
          2. Terrain context is extracted from env_decision and appended to
             scene_context so the instruction branch sees what the robot sees.
          3. Instruction branch runs SECOND with the enriched scene_context.

        This makes κ_I terrain-aware: "Keep going" on clear concrete gives a
        lower κ_I than "Keep going" on 40% unknown terrain, even though the
        instruction text is identical.  Previously both branches ran in parallel
        with no cross-talk until the κ-level merge.

        Args:
            instruction:    Natural-language instruction from the user.
            image:          RGB image array (H, W, 3).
            scene_context:  Optional caller-supplied scene description.  The
                            environmental branch output is always appended to this.
            scene_graph:    Optional SceneGraph for the environmental branch.

        Returns:
            JointDecision with kappa_I, kappa_E, kappa_joint, and final_action.
        """
        # ── Step 1: Environmental branch first ───────────────────────────────
        env_decision = self._env_runner.run_scene(image, scene_graph=scene_graph)
        kappa_E = compute_kappa_E(env_decision)

        # ── Step 2: Build terrain-enriched scene_context ─────────────────────
        # Extract detected terrain labels and unknown coverage from env_decision
        # so the instruction branch can factor in what the robot actually sees.
        terrain_context = _build_terrain_scene_context(env_decision)
        enriched_context = f"{scene_context} {terrain_context}".strip() if scene_context else terrain_context

        # ── Step 3: Instruction branch with terrain-enriched context ─────────
        amb = self._ambiguity_detector.detect(instruction, enriched_context)
        kappa_I = amb.nonconformity_score

        kappa_joint = compute_kappa_joint(kappa_I, kappa_E)

        # Environmental branch may have already decided STOP (LCB or coverage).
        # The joint layer cannot downgrade that.
        if env_decision.robot_action == "STOP" or kappa_joint >= self._stop_threshold:
            final_action = "STOP"
            question = env_decision.question
            dominant_branch = "environment"
        elif kappa_joint >= self._ask_threshold:
            final_action = "ASK"
            # Per june1meeting design: question comes from the DOMINANT branch.
            # If the instruction branch is driving uncertainty (κ_I > κ_E and the
            # instruction is genuinely ambiguous), ask a targeted instruction-clarification
            # question rather than a terrain question.  This directly resolves the gap
            # identified in the june1 meeting: "the question asked is always from the
            # env branch — it asks about terrain, not about the ambiguous instruction."
            if kappa_I > kappa_E and amb.ambiguity_type != "no_uncertainty":
                # Pass env_decision so the instruction question is grounded in
                # the terrain the robot currently observes (june1meeting lines 88-94:
                # scene graph should connect to and inform instruction branch).
                question = generate_instruction_question(amb, env_decision=env_decision)
                dominant_branch = "instruction"
            else:
                question = env_decision.question
                dominant_branch = "environment"
        else:
            final_action = "PROCEED"
            question = None
            dominant_branch = "none"

        return JointDecision(
            kappa_I=kappa_I,
            kappa_E=kappa_E,
            kappa_joint=kappa_joint,
            instruction_ambiguity=amb,
            env_decision=env_decision,
            final_action=final_action,
            question=question,
            dominant_branch=dominant_branch,
        )
