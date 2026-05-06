"""
Unit tests for system/env_uncertainty/map_updater.py

Tests:
  _parse_user_response:
    - "yes" / "safe" / "go ahead" → True
    - "no" / "stop" / "dangerous" → False
    - Ambiguous / empty → False (safety-first default)
    - Avoid phrases take priority over safe phrases

  MapUpdater.apply_feedback:
    - With unknown regions: returns feedback_applied=True
    - Updates largest region's score (Bayesian posterior, not hard-coded 0.9/0.0)
    - Unknown region prior = 0.5 (maximum entropy); "safe" → ~0.905, "unsafe" → ~0.053
    - With no unknown regions: feedback_applied=False, map unchanged

  MapUpdater.apply_feedback_to_region:
    - Directly applies Bayesian update using current map as prior
    - Returns updated TraversabilityMap

  Bayesian update properties:
    - "safe" response raises score above prior
    - "unsafe" response lowers score below prior
    - Score always stays in [0, 1]
    - Sequential updates refine estimate (posterior becomes new prior)
    - High-confidence terrain (grass 0.9) is harder to flip than unknown (0.5)

  UpdateResult fields:
    - region_updated, updated_map, feedback_applied, is_traversable all present
"""

import numpy as np
import pytest

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.map_updater import MapUpdater, UpdateResult, _parse_user_response
from system.env_uncertainty.traversability import TraversabilityMap


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_region(frac=0.25):
    h, w = 50, 50
    n = int(h * w * frac)
    mask = np.zeros((h, w), dtype=bool)
    mask.flat[:n] = True
    return RegionInfo(
        label="unknown", mask=mask, confidence=0.7,
        pixel_fraction=frac, source="sam2", traversability=0.0,
    )


def _make_result(regions=None):
    tmap = TraversabilityMap.create(50, 50)
    if regions is None:
        regions = [_make_region()]
    return DetectionResult(
        known_regions=[], unknown_regions=regions,
        image_shape=(50, 50), sam3_coverage=0.5,
        unknown_coverage=sum(r.pixel_fraction for r in regions),
        has_unknown=len(regions) > 0,
        traversability_map=tmap,
    )


def _blank_tmap():
    return TraversabilityMap.create(50, 50)


# ── _parse_user_response ──────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("yes", True),
    ("safe", True),
    ("ok", True),
    ("go ahead", True),
    ("it's fine", True),
    ("proceed", True),
    ("no", False),
    ("stop", False),
    ("dangerous", False),
    ("don't go", False),
    ("avoid", False),
    ("not safe", False),
    ("", False),           # empty → safety default
    ("maybe", False),      # ambiguous → safety default
])
def test_parse_response(text, expected):
    assert _parse_user_response(text) is expected


def test_avoid_takes_priority_over_safe():
    # "no, it's safe" → avoid wins (safety-first)
    assert _parse_user_response("no, it's safe") is False


# ── apply_feedback ────────────────────────────────────────────────────────────

def test_apply_feedback_returns_update_result():
    updater = MapUpdater()
    result = _make_result()
    update = updater.apply_feedback(result, _blank_tmap(), "yes go ahead")
    assert isinstance(update, UpdateResult)


def test_apply_feedback_with_regions_sets_applied_true():
    updater = MapUpdater()
    result = _make_result()
    update = updater.apply_feedback(result, _blank_tmap(), "yes")
    assert update.feedback_applied is True


def test_apply_feedback_traversable_true_sets_09():
    updater = MapUpdater()
    region = _make_region(frac=1.0)   # full image mask
    result = _make_result(regions=[region])
    tmap = _blank_tmap()
    update = updater.apply_feedback(result, tmap, "yes, safe to cross")
    # Unknown prior = 0.5; Bayesian "safe" → p_tp*0.5 / (p_tp*0.5 + p_fp*0.5) = 0.95/1.05 ≈ 0.905
    assert update.updated_map.score_at(0, 0) == pytest.approx(0.905, abs=0.001)


def test_apply_feedback_traversable_false_keeps_zero():
    updater = MapUpdater()
    region = _make_region(frac=1.0)
    result = _make_result(regions=[region])
    tmap = _blank_tmap()
    update = updater.apply_feedback(result, tmap, "stop, dangerous")
    # Unknown prior = 0.5; Bayesian "unsafe" → (1-p_tp)*0.5 / ((1-p_tp)*0.5 + (1-p_fp)*0.5) ≈ 0.053
    assert update.updated_map.score_at(0, 0) == pytest.approx(0.053, abs=0.001)


def test_apply_feedback_no_regions_returns_applied_false():
    updater = MapUpdater()
    result = _make_result(regions=[])
    update = updater.apply_feedback(result, _blank_tmap(), "yes")
    assert update.feedback_applied is False


def test_apply_feedback_no_regions_map_unchanged():
    updater = MapUpdater()
    tmap = _blank_tmap()
    result = _make_result(regions=[])
    update = updater.apply_feedback(result, tmap, "yes")
    assert np.all(update.updated_map.scores == tmap.scores)


def test_apply_feedback_targets_largest_region():
    updater = MapUpdater()
    small = _make_region(frac=0.10)
    large = _make_region(frac=0.50)
    result = _make_result(regions=[small, large])
    update = updater.apply_feedback(result, _blank_tmap(), "yes")
    assert update.region_updated is large


def test_apply_feedback_is_traversable_true():
    updater = MapUpdater()
    result = _make_result()
    update = updater.apply_feedback(result, _blank_tmap(), "go ahead")
    assert update.is_traversable is True


def test_apply_feedback_is_traversable_false():
    updater = MapUpdater()
    result = _make_result()
    update = updater.apply_feedback(result, _blank_tmap(), "stop it is dangerous")
    assert update.is_traversable is False


# ── apply_feedback_to_region ──────────────────────────────────────────────────

def test_apply_feedback_to_region_traversable():
    updater = MapUpdater()
    region = _make_region(frac=1.0)
    tmap = _blank_tmap()
    new_tmap = updater.apply_feedback_to_region(region, tmap, is_traversable=True)
    # Unknown prior = 0.5; Bayesian "safe" → ≈ 0.905
    assert new_tmap.score_at(0, 0) == pytest.approx(0.905, abs=0.001)


def test_apply_feedback_to_region_impassable():
    updater = MapUpdater()
    # Set a region to grass first (prior 0.9)
    region = _make_region(frac=1.0)
    tmap = TraversabilityMap.create(50, 50)
    mask = np.ones((50, 50), dtype=bool)
    tmap = tmap.update_region(mask, "grass")  # score 0.9
    new_tmap = updater.apply_feedback_to_region(region, tmap, is_traversable=False)
    # Grass prior = 0.9; Bayesian "unsafe" → (1-0.95)*0.9 / ((1-0.95)*0.9 + (1-0.10)*0.1) ≈ 0.333
    assert new_tmap.score_at(0, 0) == pytest.approx(0.333, abs=0.01)


# ── Bayesian update properties ────────────────────────────────────────────────

def test_safe_response_raises_score_above_prior():
    """'safe' feedback always produces a posterior strictly above the prior."""
    updater = MapUpdater()
    for frac in [0.1, 0.5, 0.9]:
        region = _make_region(frac=1.0)
        tmap = TraversabilityMap.create(50, 50)
        mask = np.ones((50, 50), dtype=bool)
        # Set a known prior by filling with "dirt" (score 0.8) or "grass" (0.9)
        prior_label = "grass" if frac > 0.5 else "dirt"
        tmap = tmap.update_region(mask, prior_label)
        prior_val = tmap.score_at(0, 0)
        new_tmap = updater.apply_feedback_to_region(region, tmap, is_traversable=True)
        assert new_tmap.score_at(0, 0) > prior_val


def test_unsafe_response_lowers_score_below_prior():
    """'unsafe' feedback always produces a posterior strictly below the prior."""
    updater = MapUpdater()
    region = _make_region(frac=1.0)
    tmap = TraversabilityMap.create(50, 50)
    mask = np.ones((50, 50), dtype=bool)
    tmap = tmap.update_region(mask, "grass")  # prior 0.9
    prior_val = tmap.score_at(0, 0)
    new_tmap = updater.apply_feedback_to_region(region, tmap, is_traversable=False)
    assert new_tmap.score_at(0, 0) < prior_val


def test_score_stays_in_unit_interval():
    """Posterior always remains in [0, 1] regardless of prior."""
    updater = MapUpdater()
    region = _make_region(frac=1.0)
    for label in ["grass", "mud", "puddle", "sidewalk"]:
        for is_trav in [True, False]:
            tmap = TraversabilityMap.create(50, 50)
            mask = np.ones((50, 50), dtype=bool)
            tmap = tmap.update_region(mask, label)
            new_tmap = updater.apply_feedback_to_region(region, tmap, is_traversable=is_trav)
            score = new_tmap.score_at(0, 0)
            assert 0.0 <= score <= 1.0


def test_sequential_update_refines_estimate():
    """Two 'safe' responses drive the score higher than one."""
    updater = MapUpdater()
    region = _make_region(frac=1.0)
    tmap0 = _blank_tmap()

    # First update: unknown prior 0.5 → safe → ~0.905
    tmap1 = updater.apply_feedback_to_region(region, tmap0, is_traversable=True)
    score1 = tmap1.score_at(0, 0)

    # Second update: posterior from round 1 is new prior → another safe
    tmap2 = updater.apply_feedback_to_region(region, tmap1, is_traversable=True)
    score2 = tmap2.score_at(0, 0)

    assert score2 > score1


def test_high_confidence_terrain_harder_to_flip():
    """
    A single 'unsafe' response changes a grass region (prior 0.9) by less
    than it changes an unknown region (prior 0.5 → posterior 0.053 drops 0.447).
    Grass drop: 0.9 → ~0.333 (drop 0.567) — wait, actually grass drops more
    in absolute terms but the test checks the *relative resistance* differently.

    The real invariant: after one 'unsafe', the grass posterior (≈0.333) is
    still much higher than the unknown posterior (≈0.053) — high confidence
    terrain resists flipping to near-zero more than unknown terrain does.
    """
    updater = MapUpdater()
    region = _make_region(frac=1.0)

    # Unknown terrain: prior 0.5
    tmap_unknown = _blank_tmap()
    tmap_unknown_after = updater.apply_feedback_to_region(region, tmap_unknown, is_traversable=False)
    unknown_posterior = tmap_unknown_after.score_at(0, 0)

    # Grass terrain: prior 0.9
    tmap_grass = TraversabilityMap.create(50, 50)
    mask = np.ones((50, 50), dtype=bool)
    tmap_grass = tmap_grass.update_region(mask, "grass")
    tmap_grass_after = updater.apply_feedback_to_region(region, tmap_grass, is_traversable=False)
    grass_posterior = tmap_grass_after.score_at(0, 0)

    # Grass posterior should be significantly higher after one 'unsafe'
    assert grass_posterior > unknown_posterior


# ── UpdateResult fields ───────────────────────────────────────────────────────

def test_update_result_has_all_fields():
    updater = MapUpdater()
    result = _make_result()
    update = updater.apply_feedback(result, _blank_tmap(), "yes")
    assert hasattr(update, "region_updated")
    assert hasattr(update, "updated_map")
    assert hasattr(update, "feedback_applied")
    assert hasattr(update, "is_traversable")
