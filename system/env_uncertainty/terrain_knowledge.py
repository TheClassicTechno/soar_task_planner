"""
Persistent semantic terrain knowledge across multiple frames.

Addresses the mentor's May 19 concern: the current SceneGraph is a per-image
local grid — it should not be called a "scene graph" (which implies cross-pose,
cross-frame spatial connections).  This module adds the cross-frame layer.

Two classes:

  LocalTerrainMap     — alias for SceneGraph. Clarifies that the 10×10 grid is
                        local to a SINGLE image. Use this name in new code.

  PersistentTerrainKnowledge — lightweight semantic memory that survives across
                        frames by tracking label-level (not pixel-level) beliefs.
                        When the user confirms "wet grass is safe," that fact
                        propagates to the next frame where grass is detected.

Design decision (why not pixel-level cross-frame?)
---------------------------------------------------
Pixel coordinates shift between frames as the robot moves — (row=75, col=200)
in frame 1 is not the same real-world location in frame 2.  Without GPS /
visual odometry / SLAM, we cannot reliably map pixel→world coordinates.

What IS stable across frames: terrain class labels.  Grass is grass in frame 1
and frame 2.  If the user confirmed "this grass patch is safe," we have high
confidence that other grass patches in subsequent frames are also safe —
unless the user gives contradictory feedback.

This is exactly the Bayesian update the GP does within one frame, extended to
the label dimension across frames.

Integration with runner.py
--------------------------
  knowledge = PersistentTerrainKnowledge()
  ...
  # After user feedback in run_with_feedback():
  knowledge.update_from_feedback(parsed.terrain_label, parsed.is_traversable,
                                 parsed.traversability_confidence)
  ...
  # When seeding GP for a new frame, use knowledge-adjusted traversability:
  trav = knowledge.adjusted_traversability(region.label, region.traversability)
  gp_map.add_observation(cy, cx, trav, h, w)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from system.env_uncertainty.traversability import TRAVERSABILITY_SCORES


# ── LocalTerrainMap (terminology alias) ──────────────────────────────────────

def LocalTerrainMap(*args, **kwargs):
    """
    Factory alias for SceneGraph.

    Use 'LocalTerrainMap' in new code to make clear this is a per-image,
    single-frame local representation — NOT a multi-pose scene graph.
    The underlying implementation is SceneGraph.
    """
    from system.env_uncertainty.scene_graph import SceneGraph
    return SceneGraph(*args, **kwargs)


# ── PersistentTerrainKnowledge ────────────────────────────────────────────────

@dataclass
class _LabelBelief:
    """
    Running Bayesian belief about a terrain label's traversability across frames.

    traversability_mean:  Current posterior mean traversability ∈ [0, 1].
    n_observations:       Number of user-feedback observations incorporated.
    is_confirmed:         True when at least one high-confidence update was applied.
    """
    traversability_mean: float
    n_observations: int = 0
    is_confirmed: bool = False


class PersistentTerrainKnowledge:
    """
    Cross-frame semantic terrain knowledge: label → traversability belief.

    Maintains one Bayesian belief per terrain label that accumulates user
    feedback across frames.  Unlike the GP (which is pixel-specific and resets
    per frame), this class survives between run_scene() calls.

    Bayesian update rule
    --------------------
    When user confirms label L is traversable (is_traversable=True):
      τ₁ = p_tp · τ₀ / (p_tp · τ₀ + p_fp · (1 − τ₀))

    When user says L is NOT traversable:
      τ₁ = (1 − p_tp) · τ₀ / ((1 − p_tp) · τ₀ + (1 − p_fp) · (1 − τ₀))

    p_tp = 0.95 (probability user says safe | terrain is safe)
    p_fp = 0.10 (probability user says safe | terrain is unsafe)

    Matches the same update used in traversability.py _bayesian_update().

    Usage
    -----
      knowledge = PersistentTerrainKnowledge()

      # After user responds in run_with_feedback():
      knowledge.update_from_feedback("grass", is_traversable=True, confidence=0.85)

      # When seeding GP for new frame:
      adjusted = knowledge.adjusted_traversability("grass", default_score=0.90)
      gp_map.add_observation(cy, cx, adjusted, h, w)

      # Check if we already know this label well enough to skip asking:
      if knowledge.should_skip_asking("grass"):
          ...
    """

    SKIP_CONFIRMATION_THRESHOLD: int = 2   # confirmations before skip
    SKIP_TRAVERSABILITY_MIN: float = 0.60  # must be reliably traversable to skip
    SKIP_TRAVERSABILITY_MAX: float = 0.40  # must be reliably NOT traversable to skip

    def __init__(self) -> None:
        # label.lower() → _LabelBelief
        self._beliefs: Dict[str, _LabelBelief] = {}

    def update_from_feedback(
        self,
        label: Optional[str],
        is_traversable: bool,
        confidence: float = 0.70,
        p_tp: float = 0.95,
        p_fp: float = 0.10,
    ) -> None:
        """
        Incorporate user feedback for a terrain label.

        If label is None (user response didn't name a terrain class), no update
        is recorded — we don't know which label the user was commenting on.

        Args:
            label:          Canonical terrain label (e.g. "grass", "mud").
            is_traversable: True if user said terrain is safe.
            confidence:     label_confidence from ParsedUserResponse ∈ [0, 1].
            p_tp:           Likelihood P(safe response | terrain safe).
            p_fp:           Likelihood P(safe response | terrain unsafe).
        """
        if not label:
            return
        key = label.lower().strip()
        prior = self._beliefs.get(key)
        if prior is None:
            default = TRAVERSABILITY_SCORES.get(key, 0.5)
            prior = _LabelBelief(traversability_mean=default)

        τ0 = prior.traversability_mean
        if is_traversable:
            τ1 = p_tp * τ0 / (p_tp * τ0 + p_fp * (1.0 - τ0) + 1e-9)
        else:
            τ1 = (1.0 - p_tp) * τ0 / ((1.0 - p_tp) * τ0 + (1.0 - p_fp) * (1.0 - τ0) + 1e-9)

        # Weight update by label confidence (low-confidence responses shift less)
        τ_new = confidence * τ1 + (1.0 - confidence) * τ0
        τ_new = float(max(0.0, min(1.0, τ_new)))

        self._beliefs[key] = _LabelBelief(
            traversability_mean=τ_new,
            n_observations=prior.n_observations + 1,
            is_confirmed=(confidence >= 0.60 or prior.is_confirmed),
        )

    def adjusted_traversability(
        self,
        label: str,
        default_score: Optional[float] = None,
    ) -> float:
        """
        Return traversability for a label, incorporating cross-frame knowledge.

        Args:
            label:         Canonical terrain label.
            default_score: Fallback if no cross-frame knowledge exists.
                           Defaults to TRAVERSABILITY_SCORES[label] or 0.5.

        Returns:
            Posterior traversability ∈ [0, 1].
        """
        key = label.lower().strip()
        if key in self._beliefs:
            return self._beliefs[key].traversability_mean
        if default_score is not None:
            return default_score
        return TRAVERSABILITY_SCORES.get(key, 0.5)

    def should_skip_asking(self, label: str) -> bool:
        """
        Return True when we have enough cross-frame evidence to skip asking about this label.

        Conditions:
          1. At least SKIP_CONFIRMATION_THRESHOLD confirmations.
          2. Either clearly safe (mean ≥ SKIP_TRAVERSABILITY_MIN) or clearly unsafe
             (mean ≤ SKIP_TRAVERSABILITY_MAX) — not in the ambiguous middle.

        This prevents repeated questions about the same terrain class across frames.
        """
        key = label.lower().strip()
        if key not in self._beliefs:
            return False
        b = self._beliefs[key]
        if b.n_observations < self.SKIP_CONFIRMATION_THRESHOLD:
            return False
        return (
            b.traversability_mean >= self.SKIP_TRAVERSABILITY_MIN
            or b.traversability_mean <= self.SKIP_TRAVERSABILITY_MAX
        )

    def has_knowledge(self, label: str) -> bool:
        """Return True if any feedback has been received for this label."""
        return label.lower().strip() in self._beliefs

    def get_belief(self, label: str) -> Optional[_LabelBelief]:
        """Return the belief for a label, or None if unseen."""
        return self._beliefs.get(label.lower().strip())

    def reset(self) -> None:
        """Clear all cross-frame knowledge (e.g., between separate navigation tasks)."""
        self._beliefs.clear()

    @property
    def n_labels_known(self) -> int:
        """Number of distinct terrain labels with at least one observation."""
        return len(self._beliefs)

    def summary(self) -> str:
        """Human-readable summary of current cross-frame knowledge."""
        if not self._beliefs:
            return "PersistentTerrainKnowledge: no observations yet"
        lines = ["PersistentTerrainKnowledge:"]
        for label, b in sorted(self._beliefs.items()):
            skip = " [SKIP-ASK]" if self.should_skip_asking(label) else ""
            lines.append(
                f"  {label:15s}  τ={b.traversability_mean:.2f}  "
                f"n={b.n_observations}  confirmed={b.is_confirmed}{skip}"
            )
        return "\n".join(lines)
