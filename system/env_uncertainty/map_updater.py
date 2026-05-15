"""
TraversabilityMap update after receiving user clarification.

After the robot asks about an unknown region and the user responds, the robot
must update its internal traversability map before selecting a trajectory.

Two parsing paths:
  _parse_user_response()      Simple boolean (safe / not-safe).  Used internally
                              by MapUpdater.apply_feedback() for backwards compat.
  parse_user_response_rich()  Full affordance parse → ParsedUserResponse dataclass
                              with terrain label, confidence, traversability score,
                              and keyword list.  Used by the scene graph Bayesian
                              update path (Step 8 in the pipeline).

Affordance keyword design
-------------------------
  Confidence keywords map hedging language to a label-confidence scalar in [0,1].
  Traversability keywords map terrain descriptors to +/- traversability modifiers.
  Terrain label keywords map common synonyms to TERRAIN_CLASSES vocabulary entries.
  The net affordance modifier is summed across all matched traversability keywords;
  is_traversable is True when the sum is non-negative.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.traversability import TraversabilityMap


# ── Affordance keyword tables ─────────────────────────────────────────────────

# Hedge words → label confidence.  Matched in order; first match wins.
_CONFIDENCE_KEYWORDS: Dict[str, float] = {
    "definitely": 0.95,
    "certainly": 0.95,
    "clearly": 0.90,
    "obviously": 0.90,
    "sure": 0.85,
    "absolutely": 0.90,
    "probably": 0.65,
    "likely": 0.65,
    "think": 0.60,
    "believe": 0.60,
    "maybe": 0.50,
    "possibly": 0.45,
    "might": 0.45,
    "could be": 0.45,
    "guess": 0.40,
    "not sure": 0.30,
    "unclear": 0.25,
    "uncertain": 0.25,
    "no idea": 0.10,
}

# Terrain descriptors → traversability modifier (positive = safer, negative = riskier).
_TRAVERSABILITY_KEYWORDS: Dict[str, float] = {
    # Positive
    "safe": +0.20,
    "firm": +0.15,
    "solid": +0.15,
    "flat": +0.10,
    "dry": +0.10,
    "hard": +0.10,
    "paved": +0.15,
    "clear": +0.10,
    "smooth": +0.10,
    "stable": +0.15,
    "walkable": +0.20,
    "passable": +0.20,
    "traversable": +0.20,
    # Negative
    "unsafe": -0.30,
    "dangerous": -0.35,
    "avoid": -0.40,
    "wet": -0.15,
    "slippery": -0.20,
    "muddy": -0.25,
    "soft": -0.10,
    "steep": -0.15,
    "unstable": -0.20,
    "cracked": -0.10,
    "flooded": -0.35,
    "icy": -0.30,
    "rough": -0.10,
    "bumpy": -0.05,
    "deep": -0.15,
}

# Terrain class synonyms → canonical TERRAIN_CLASSES entry.
_TERRAIN_LABEL_KEYWORDS: Dict[str, str] = {
    "grass": "grass", "lawn": "grass", "turf": "grass", "field": "grass",
    "gravel": "gravel", "pebbles": "gravel", "stones": "gravel", "rocks": "gravel",
    "sidewalk": "sidewalk", "pavement": "sidewalk", "path": "sidewalk",
    "walkway": "sidewalk", "footpath": "sidewalk",
    "road": "road", "street": "road", "asphalt": "road", "tarmac": "road",
    "dirt": "dirt", "soil": "dirt", "earth": "dirt", "ground": "dirt",
    "mud": "mud", "muck": "mud",
    "puddle": "puddle", "pool": "puddle",
    "water": "water", "stream": "water", "flooded": "puddle",
    "slope": "slope", "hill": "slope", "incline": "slope", "ramp": "slope",
    "sand": "sand", "sandy": "sand",
    "vegetation": "vegetation", "bushes": "vegetation", "weeds": "vegetation",
    "mulch": "mulch", "bark": "mulch",
    "concrete": "concrete", "cement": "concrete", "slab": "concrete",
    "crosswalk": "crosswalk", "crossing": "crosswalk", "zebra": "crosswalk",
    "log": "log", "wood": "log", "branch": "log",
    "unknown": "unknown",
}

# Simple safe/avoid phrase sets — kept for _parse_user_response() backward compat.
_SAFE_PHRASES = {
    "yes", "safe", "ok", "okay", "fine", "go", "proceed",
    "clear", "passable", "traversable", "go ahead", "it's fine",
}
_AVOID_PHRASES = {
    "no", "stop", "avoid", "unsafe", "danger", "dangerous", "bad",
    "don't", "dont", "wait", "hold", "not safe", "impassable",
}


@dataclass
class ParsedUserResponse:
    """
    Structured output from parse_user_response_rich().

    terrain_label:            Canonical terrain class, or None if not detected.
    label_confidence:         Confidence in the terrain_label in [0, 1].
    is_traversable:           True when net affordance modifier >= 0.
    traversability_confidence: Estimated confidence in is_traversable in [0, 1].
    affordance_modifier:      Sum of matched traversability keyword modifiers.
    keywords:                 All matched keyword strings (for logging / audit).
    """

    terrain_label: Optional[str]
    label_confidence: float
    is_traversable: bool
    traversability_confidence: float
    affordance_modifier: float
    keywords: List[str] = field(default_factory=list)


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


# ── Public rich parser ────────────────────────────────────────────────────────

def parse_user_response_rich(text: str) -> ParsedUserResponse:
    """
    Convert free-text user response to a structured ParsedUserResponse.

    Three extraction steps run in order on the lowercased input:

    1. Confidence — scan _CONFIDENCE_KEYWORDS for hedge language; take the first
       match (phrases before single words so "not sure" beats "sure").
    2. Terrain label — scan _TERRAIN_LABEL_KEYWORDS for the first class synonym.
    3. Traversability — sum all matched _TRAVERSABILITY_KEYWORDS modifiers;
       is_traversable is True when the net sum >= 0.

    Falls back to is_traversable=False (safety-first) when no traversability
    keyword is found.

    Args:
        text: Raw user response string (any length, any case).

    Returns:
        ParsedUserResponse with all fields populated.
    """
    normalized = text.lower().strip()
    matched_keywords: List[str] = []

    # Step 1 — confidence
    label_confidence = 0.70  # neutral default
    # Check multi-word phrases first so "not sure" beats "sure"
    for kw, conf in sorted(_CONFIDENCE_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if kw in normalized:
            label_confidence = conf
            matched_keywords.append(kw)
            break

    # Step 2 — terrain label (first synonym match wins)
    terrain_label: Optional[str] = None
    for kw, cls in sorted(_TERRAIN_LABEL_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if re.search(r"\b" + re.escape(kw) + r"\b", normalized):
            terrain_label = cls
            matched_keywords.append(kw)
            break

    # Step 3 — traversability affordance modifiers (sum all matches)
    affordance_modifier = 0.0
    n_trav_keywords = 0
    for kw, mod in sorted(_TRAVERSABILITY_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if re.search(r"\b" + re.escape(kw) + r"\b", normalized):
            affordance_modifier += mod
            matched_keywords.append(kw)
            n_trav_keywords += 1

    # Safety-first: if no traversability signal at all, default to avoid.
    if n_trav_keywords == 0:
        is_traversable = False
        traversability_confidence = 0.30  # low confidence — response was ambiguous
    else:
        is_traversable = affordance_modifier >= 0.0
        traversability_confidence = float(min(0.95, max(0.05, 0.50 + affordance_modifier)))

    return ParsedUserResponse(
        terrain_label=terrain_label,
        label_confidence=label_confidence,
        is_traversable=is_traversable,
        traversability_confidence=traversability_confidence,
        affordance_modifier=affordance_modifier,
        keywords=matched_keywords,
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
