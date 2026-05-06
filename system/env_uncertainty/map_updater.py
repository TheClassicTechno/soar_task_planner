"""
TraversabilityMap update after receiving user clarification.

After the robot asks about an unknown region and the user responds, the robot
must update its internal traversability map before selecting a trajectory.

The MapUpdater matches a user's response ("yes it's safe" / "no, avoid it")
to the most relevant unknown region in the DetectionResult, then calls
TraversabilityMap.apply_user_feedback() to update the map.

The update returns a new TraversabilityMap (immutable pattern) rather than
mutating the original.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.traversability import TraversabilityMap


# Phrases that indicate the user says the area is safe to cross
_SAFE_PHRASES = {
    "yes", "safe", "ok", "okay", "fine", "go", "proceed",
    "clear", "passable", "traversable", "go ahead", "it's fine",
}

# Phrases that indicate the user says to avoid the area
_AVOID_PHRASES = {
    "no", "stop", "avoid", "unsafe", "danger", "dangerous", "bad",
    "don't", "dont", "wait", "hold", "not safe", "impassable",
}


@dataclass
class UpdateResult:
    """
    Result of applying user feedback to the traversability map.

    updated_map:      New traversability map incorporating user feedback.
    region_updated:   The specific unknown region the update was applied to.
    feedback_applied: True if a region was matched and updated.
    is_traversable:   Whether the user said the region is safe.
    """

    updated_map: TraversabilityMap
    region_updated: Optional[RegionInfo]
    feedback_applied: bool
    is_traversable: bool


class MapUpdater:
    """
    Apply user feedback to update an unknown region's traversability.

    Usage:
        updater = MapUpdater()
        update = updater.apply_feedback(detection_result, tmap, "yes go ahead")
        new_tmap = update.updated_map
    """

    def apply_feedback(
        self,
        result: DetectionResult,
        tmap: TraversabilityMap,
        user_response: str,
    ) -> UpdateResult:
        """
        Interpret a user response string and update the traversability map.

        Matches the largest unknown region (most safety-critical) and updates
        its score based on the user's answer.

        Args:
            result:        DetectionResult containing unknown region masks.
            tmap:          Current traversability map to update.
            user_response: Free-text user response (e.g., "yes go ahead").

        Returns:
            UpdateResult with the new map and metadata about the update.
        """
        is_traversable = _parse_user_response(user_response)

        # Target the largest unknown region — it is the most likely candidate
        # for the one the robot just asked about.
        target_region = _largest_region(result.unknown_regions)
        if target_region is None:
            return UpdateResult(
                updated_map=tmap,
                region_updated=None,
                feedback_applied=False,
                is_traversable=is_traversable,
            )

        # Read the prior traversability from the map before overwriting it.
        # This feeds the Bayesian update in apply_user_feedback() so the
        # posterior is conditioned on what the robot already believed.
        prior = tmap.mean_score_over_mask(target_region.mask)

        new_tmap = tmap.apply_user_feedback(
            mask=target_region.mask,
            is_traversable=is_traversable,
            prior_score=prior,
        )
        return UpdateResult(
            updated_map=new_tmap,
            region_updated=target_region,
            feedback_applied=True,
            is_traversable=is_traversable,
        )

    def apply_feedback_to_region(
        self,
        region: RegionInfo,
        tmap: TraversabilityMap,
        is_traversable: bool,
    ) -> TraversabilityMap:
        """
        Apply feedback directly to a specific region, bypassing text parsing.

        Used when the calling code already knows which region and answer to apply,
        e.g., after a structured UI interaction rather than free-text response.
        Reads the current prior from the map so the Bayesian update is correctly
        conditioned on existing knowledge.

        Args:
            region:         The specific RegionInfo to update.
            tmap:           Current traversability map.
            is_traversable: Whether the region is safe.

        Returns:
            Updated TraversabilityMap.
        """
        prior = tmap.mean_score_over_mask(region.mask)
        return tmap.apply_user_feedback(
            mask=region.mask,
            is_traversable=is_traversable,
            prior_score=prior,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_user_response(text: str) -> bool:
    """
    Determine whether the user's response indicates the area is safe.

    Uses simple keyword matching — sufficient for short navigation responses.
    Defaults to False (avoid) when the response is ambiguous.

    Args:
        text: User response string.

    Returns:
        True if safe/proceed, False if avoid/stop.
    """
    normalized = text.lower().strip()

    # Check avoid phrases first (safety bias)
    for phrase in _AVOID_PHRASES:
        if re.search(r"\b" + re.escape(phrase) + r"\b", normalized):
            return False

    for phrase in _SAFE_PHRASES:
        if re.search(r"\b" + re.escape(phrase) + r"\b", normalized):
            return True

    # Ambiguous response → default to avoid (safety-first)
    return False


def _largest_region(regions: List[RegionInfo]) -> Optional[RegionInfo]:
    """Return the region with the largest pixel_fraction, or None if empty."""
    if not regions:
        return None
    return max(regions, key=lambda r: r.pixel_fraction)
