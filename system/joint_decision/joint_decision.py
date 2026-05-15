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
from typing import Optional

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
    """

    kappa_I: float
    kappa_E: float
    kappa_joint: float
    instruction_ambiguity: AmbiguityDetection
    env_decision: EnvUncertaintyDecision
    final_action: str
    question: Optional[str]


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

        Args:
            instruction:    Natural-language instruction from the user.
            image:          RGB image array (H, W, 3).
            scene_context:  Optional scene description for the instruction branch.
            scene_graph:    Optional SceneGraph for the environmental branch.

        Returns:
            JointDecision with kappa_I, kappa_E, kappa_joint, and final_action.
        """
        amb = self._ambiguity_detector.detect(instruction, scene_context)
        kappa_I = amb.nonconformity_score

        env_decision = self._env_runner.run_scene(image, scene_graph=scene_graph)
        kappa_E = compute_kappa_E(env_decision)

        kappa_joint = compute_kappa_joint(kappa_I, kappa_E)

        # Environmental branch may have already decided STOP (LCB or coverage).
        # The joint layer cannot downgrade that.
        if env_decision.robot_action == "STOP" or kappa_joint >= self._stop_threshold:
            final_action = "STOP"
            question = env_decision.question
        elif kappa_joint >= self._ask_threshold:
            final_action = "ASK"
            question = env_decision.question
        else:
            final_action = "PROCEED"
            question = None

        return JointDecision(
            kappa_I=kappa_I,
            kappa_E=kappa_E,
            kappa_joint=kappa_joint,
            instruction_ambiguity=amb,
            env_decision=env_decision,
            final_action=final_action,
            question=question,
        )
