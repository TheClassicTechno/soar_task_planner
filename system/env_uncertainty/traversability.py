"""
Traversability scoring for outdoor terrain regions.

Each terrain class is assigned a traversability score in [0, 1]:
  1.0 = fully safe, confirmed navigable surface
  0.0 = impassable OR unknown (treat unknown = impassable until clarified)

Scores reflect expected risk for a wheeled outdoor robot on the RUGD
terrain vocabulary. Unknown regions (from SAM2 subtraction) always receive
score 0.0 — safety-first until user feedback is incorporated.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# Per-class traversability scores matching SAM3's 13-class vocabulary.
# Unknown is not a SAM3 class but is used by the detector for SAM2 residuals.
TRAVERSABILITY_SCORES: Dict[str, float] = {
    "sidewalk":         0.95,
    "crosswalk":        0.90,
    "road":             0.95,
    "concrete":         0.95,
    "dirt":             0.80,
    "grass":            0.90,
    "gravel":           0.70,
    "mulch":            0.65,
    "sand":             0.60,
    "vegetation":       0.60,
    "wet surface":      0.40,
    "cracked pavement": 0.35,
    "slope":            0.30,
    "rock-bed":         0.20,
    "mud":              0.10,
    "puddle":           0.05,
    "water":            0.05,
    "log":              0.05,
    "tree":             0.05,
    "person":           0.00,
    # unknown = SAM2 residual region not explained by SAM3
    "unknown":          0.00,
}

# Any score at or below this value triggers the STOP decision.
STOP_THRESHOLD = 0.20


def get_traversability(label: str) -> float:
    """
    Return the traversability score for a terrain label.

    Falls back to 0.0 (unknown) for labels not in the vocabulary.
    """
    return TRAVERSABILITY_SCORES.get(label.lower(), 0.0)


@dataclass
class TraversabilityMap:
    """
    Per-pixel traversability map for one camera frame.

    Stores a float32 array shaped (H, W) where each value is a traversability
    score in [0, 1]. Initially all zeros (fully unknown).

    After calling update_from_regions(), known areas are filled in.
    After apply_user_feedback(), user-confirmed regions are updated.
    The map is treated as immutable per-frame — updates return new instances.
    """

    _scores: np.ndarray              # (H, W) float32
    _label_map: Dict[Tuple[int, int], str] = field(default_factory=dict)

    @classmethod
    def create(cls, height: int, width: int) -> "TraversabilityMap":
        """
        Initialize a blank (all-unknown) traversability map.

        Args:
            height: Image height in pixels.
            width:  Image width in pixels.
        """
        scores = np.zeros((height, width), dtype=np.float32)
        return cls(_scores=scores)

    def update_region(self, mask: np.ndarray, label: str) -> "TraversabilityMap":
        """
        Set the traversability score for all pixels covered by mask.

        Returns a new TraversabilityMap rather than mutating in place.

        Args:
            mask:  (H, W) bool array identifying the region.
            label: Terrain class label (e.g., "grass", "unknown").
        """
        score = get_traversability(label)
        new_scores = self._scores.copy()
        new_scores[mask] = score
        return TraversabilityMap(_scores=new_scores, _label_map=dict(self._label_map))

    def apply_user_feedback(
        self,
        mask: np.ndarray,
        is_traversable: bool,
        prior_score: Optional[float] = None,
        p_tp: float = 0.95,
        p_fp: float = 0.10,
    ) -> "TraversabilityMap":
        """
        Update a region's traversability using a Bayesian posterior update.

        Instead of hard-coding 0.9 (safe) or 0.0 (unsafe), this computes:

            If is_traversable (user says "safe"):
              τ_1 = p_tp * τ_0 / (p_tp * τ_0 + p_fp * (1 - τ_0))

            If not is_traversable (user says "unsafe"):
              τ_1 = (1-p_tp) * τ_0 / ((1-p_tp) * τ_0 + (1-p_fp) * (1-τ_0))

        where τ_0 is the prior traversability score (read from the map if not
        explicitly provided) and p_tp/p_fp are the likelihood parameters.

        Default likelihood values:
          p_tp = P(response="safe" | truly traversable) = 0.95
          p_fp = P(response="safe" | truly impassable)  = 0.10

        These reflect that users rarely call safe terrain unsafe but may
        occasionally misjudge genuinely hazardous surfaces.

        The posterior is applied uniformly to all pixels in mask, replacing
        their current scores. Calling this method multiple times on the same
        region performs sequential Bayesian refinement: the posterior from
        turn k becomes the prior for turn k+1.

        Args:
            mask:           (H, W) bool array identifying the region to update.
            is_traversable: True if user confirms the region is safe.
            prior_score:    Override prior τ_0. If None, uses the mean score
                            of mask pixels in the current map.
            p_tp:           P(response="safe" | traversable=True).
            p_fp:           P(response="safe" | traversable=False).

        Returns:
            New TraversabilityMap with updated scores for mask pixels.
        """
        if prior_score is None:
            prior_score = self.mean_score_over_mask(mask)

        # Unknown regions (score 0.0) get a uniform prior instead of 0.0
        # so that the first "safe" response can actually raise the score.
        # Applies whether prior_score was read from the map or passed explicitly.
        if prior_score == 0.0:
            prior_score = 0.5

        prior_score = max(0.0, min(1.0, prior_score))
        posterior = _bayesian_update(prior_score, is_traversable, p_tp, p_fp)

        new_scores = self._scores.copy()
        new_scores[mask] = float(posterior)
        return TraversabilityMap(_scores=new_scores, _label_map=dict(self._label_map))

    def score_at(self, y: int, x: int) -> float:
        """
        Return the traversability score at pixel (y, x).

        Out-of-bounds coordinates return 0.0 (unknown).
        """
        h, w = self._scores.shape
        if not (0 <= y < h and 0 <= x < w):
            return 0.0
        return float(self._scores[y, x])

    def mean_score_over_mask(self, mask: np.ndarray) -> float:
        """
        Return the average traversability score over the pixels in mask.

        Returns 0.0 for an empty mask.

        Args:
            mask: (H, W) bool array.
        """
        if not np.any(mask):
            return 0.0
        return float(np.mean(self._scores[mask]))

    def min_score_over_mask(self, mask: np.ndarray) -> float:
        """
        Return the minimum traversability score over the pixels in mask.

        Minimum is used for safety-critical trajectory scoring: one bad pixel
        in the path makes the whole path unsafe.
        """
        if not np.any(mask):
            return 0.0
        return float(np.min(self._scores[mask]))

    def has_unknown_in_mask(self, mask: np.ndarray) -> bool:
        """
        Return True if any pixel in mask has a traversability score of 0.0.

        Used to detect whether a trajectory passes through any unknown region.
        """
        if not np.any(mask):
            return False
        return bool(np.any(self._scores[mask] == 0.0))

    @property
    def shape(self) -> Tuple[int, int]:
        """(height, width) of the map."""
        return self._scores.shape

    @property
    def scores(self) -> np.ndarray:
        """Read-only view of the (H, W) float32 score array."""
        return self._scores.view()


# ── Module-level helper ───────────────────────────────────────────────────────

def _bayesian_update(
    prior: float,
    is_safe: bool,
    p_tp: float = 0.95,
    p_fp: float = 0.10,
) -> float:
    """
    Compute the Bayesian posterior traversability given a user response.

    Model:
      P(response="safe" | traversable=True)  = p_tp  (true-positive rate)
      P(response="safe" | traversable=False) = p_fp  (false-positive rate)

    Posterior after response="safe":
      τ_1 = p_tp * τ_0 / (p_tp * τ_0 + p_fp * (1 - τ_0))

    Posterior after response="unsafe":
      τ_1 = (1-p_tp) * τ_0 / ((1-p_tp) * τ_0 + (1-p_fp) * (1-τ_0))

    Args:
        prior:    Prior traversability τ_0 ∈ [0, 1].
        is_safe:  True if user said the region is safe to traverse.
        p_tp:     P(response="safe" | traversable=True).
        p_fp:     P(response="safe" | traversable=False).

    Returns:
        Posterior traversability τ_1 ∈ [0, 1].
    """
    if is_safe:
        num = p_tp * prior
        den = p_tp * prior + p_fp * (1.0 - prior)
    else:
        num = (1.0 - p_tp) * prior
        den = (1.0 - p_tp) * prior + (1.0 - p_fp) * (1.0 - prior)
    return float(num / den) if den > 0.0 else 0.0
