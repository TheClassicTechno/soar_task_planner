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
    - Updates largest region's score correctly
    - is_traversable=True sets region to 0.9
    - is_traversable=False sets region to 0.0
    - With no unknown regions: feedback_applied=False, map unchanged

  MapUpdater.apply_feedback_to_region:
    - Directly applies score without text parsing
    - Returns updated TraversabilityMap

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
    assert update.updated_map.score_at(0, 0) == pytest.approx(0.9)


def test_apply_feedback_traversable_false_keeps_zero():
    updater = MapUpdater()
    region = _make_region(frac=1.0)
    result = _make_result(regions=[region])
    tmap = _blank_tmap()
    update = updater.apply_feedback(result, tmap, "stop, dangerous")
    assert update.updated_map.score_at(0, 0) == pytest.approx(0.0)


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
    assert new_tmap.score_at(0, 0) == pytest.approx(0.9)


def test_apply_feedback_to_region_impassable():
    updater = MapUpdater()
    # Set a region to grass first
    region = _make_region(frac=1.0)
    tmap = TraversabilityMap.create(50, 50)
    mask = np.ones((50, 50), dtype=bool)
    tmap = tmap.update_region(mask, "grass")  # score 0.9
    new_tmap = updater.apply_feedback_to_region(region, tmap, is_traversable=False)
    assert new_tmap.score_at(0, 0) == pytest.approx(0.0)
