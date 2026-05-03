"""
Unit tests for system/env_uncertainty/traversability.py

Tests:
  get_traversability:
    - Known labels return expected scores
    - Unknown labels default to 0.0
    - Lookup is case-insensitive

  TraversabilityMap.create:
    - Correct shape
    - All zeros initially

  TraversabilityMap.update_region:
    - Masked pixels set to terrain score
    - Non-masked pixels unchanged
    - Returns new instance (immutable)

  TraversabilityMap.apply_user_feedback:
    - is_traversable=True → scores 0.9 in region
    - is_traversable=False → scores 0.0 in region

  TraversabilityMap.score_at:
    - Returns correct value
    - Out-of-bounds returns 0.0

  TraversabilityMap.mean_score_over_mask / min_score_over_mask:
    - Correct aggregations
    - Empty mask returns 0.0

  TraversabilityMap.has_unknown_in_mask:
    - Detects zero-score pixels
    - Returns False for fully-known mask
"""

import numpy as np
import pytest

from system.env_uncertainty.traversability import (
    STOP_THRESHOLD,
    TRAVERSABILITY_SCORES,
    TraversabilityMap,
    get_traversability,
)


# ── get_traversability ────────────────────────────────────────────────────────

def test_known_labels_return_correct_scores():
    assert get_traversability("grass") == pytest.approx(0.90)
    assert get_traversability("sidewalk") == pytest.approx(0.95)
    assert get_traversability("mud") == pytest.approx(0.10)
    assert get_traversability("unknown") == pytest.approx(0.00)


def test_unknown_label_defaults_to_zero():
    assert get_traversability("flying_saucer") == pytest.approx(0.0)
    assert get_traversability("") == pytest.approx(0.0)


def test_lookup_is_case_insensitive():
    assert get_traversability("GRASS") == get_traversability("grass")
    assert get_traversability("Sidewalk") == get_traversability("sidewalk")


def test_all_vocabulary_labels_have_scores():
    # Every label in the TRAVERSABILITY_SCORES dict must return itself on lookup
    for label, expected in TRAVERSABILITY_SCORES.items():
        assert get_traversability(label) == pytest.approx(expected)


# ── TraversabilityMap.create ──────────────────────────────────────────────────

def test_create_returns_correct_shape():
    tmap = TraversabilityMap.create(50, 80)
    assert tmap.shape == (50, 80)


def test_create_all_zeros():
    tmap = TraversabilityMap.create(10, 10)
    assert np.all(tmap.scores == 0.0)


# ── TraversabilityMap.update_region ──────────────────────────────────────────

def test_update_region_sets_correct_score():
    tmap = TraversabilityMap.create(10, 10)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    updated = tmap.update_region(mask, "grass")
    assert updated.score_at(3, 3) == pytest.approx(0.90)


def test_update_region_leaves_other_pixels_unchanged():
    tmap = TraversabilityMap.create(10, 10)
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:3, 0:3] = True
    updated = tmap.update_region(mask, "grass")
    # Pixel outside the mask should still be 0.0
    assert updated.score_at(9, 9) == pytest.approx(0.0)


def test_update_region_is_immutable():
    tmap = TraversabilityMap.create(10, 10)
    mask = np.ones((10, 10), dtype=bool)
    updated = tmap.update_region(mask, "grass")
    # Original map must be unchanged
    assert np.all(tmap.scores == 0.0)
    assert np.all(updated.scores == pytest.approx(0.90))


def test_update_region_unknown_sets_zero():
    tmap = TraversabilityMap.create(10, 10)
    mask = np.ones((10, 10), dtype=bool)
    updated = tmap.update_region(mask, "unknown")
    assert np.all(updated.scores == 0.0)


# ── TraversabilityMap.apply_user_feedback ─────────────────────────────────────

def test_feedback_traversable_sets_09():
    tmap = TraversabilityMap.create(10, 10)
    mask = np.zeros((10, 10), dtype=bool)
    mask[5:8, 5:8] = True
    updated = tmap.apply_user_feedback(mask, is_traversable=True)
    assert updated.score_at(6, 6) == pytest.approx(0.9)


def test_feedback_not_traversable_sets_00():
    tmap = TraversabilityMap.create(10, 10)
    mask = np.ones((10, 10), dtype=bool)
    updated = tmap.apply_user_feedback(mask, is_traversable=False)
    assert np.all(updated.scores == 0.0)


# ── TraversabilityMap.score_at ────────────────────────────────────────────────

def test_score_at_returns_correct_value():
    tmap = TraversabilityMap.create(10, 10)
    mask = np.zeros((10, 10), dtype=bool)
    mask[4, 4] = True
    tmap = tmap.update_region(mask, "dirt")
    assert tmap.score_at(4, 4) == pytest.approx(0.80)


def test_score_at_out_of_bounds_returns_zero():
    tmap = TraversabilityMap.create(10, 10)
    assert tmap.score_at(-1, 5) == pytest.approx(0.0)
    assert tmap.score_at(5, 100) == pytest.approx(0.0)


# ── mean/min score over mask ──────────────────────────────────────────────────

def test_mean_score_over_mask():
    tmap = TraversabilityMap.create(10, 10)
    mask_a = np.zeros((10, 10), dtype=bool)
    mask_b = np.zeros((10, 10), dtype=bool)
    mask_a[0:5, :] = True
    mask_b[5:, :] = True
    tmap = tmap.update_region(mask_a, "grass")   # score 0.9
    tmap = tmap.update_region(mask_b, "mud")     # score 0.1

    query_mask = np.ones((10, 10), dtype=bool)
    mean = tmap.mean_score_over_mask(query_mask)
    assert mean == pytest.approx(0.5, abs=0.05)


def test_min_score_over_mask():
    tmap = TraversabilityMap.create(10, 10)
    mask_a = np.zeros((10, 10), dtype=bool)
    mask_b = np.zeros((10, 10), dtype=bool)
    mask_a[0:5, :] = True
    mask_b[5:, :] = True
    tmap = tmap.update_region(mask_a, "grass")   # 0.9
    tmap = tmap.update_region(mask_b, "mud")     # 0.1

    query_mask = np.ones((10, 10), dtype=bool)
    assert tmap.min_score_over_mask(query_mask) == pytest.approx(0.1)


def test_mean_score_empty_mask_returns_zero():
    tmap = TraversabilityMap.create(10, 10)
    empty = np.zeros((10, 10), dtype=bool)
    assert tmap.mean_score_over_mask(empty) == pytest.approx(0.0)


def test_min_score_empty_mask_returns_zero():
    tmap = TraversabilityMap.create(10, 10)
    empty = np.zeros((10, 10), dtype=bool)
    assert tmap.min_score_over_mask(empty) == pytest.approx(0.0)


# ── has_unknown_in_mask ───────────────────────────────────────────────────────

def test_has_unknown_detects_zero_score_pixel():
    tmap = TraversabilityMap.create(10, 10)
    # Map is all zeros by default — every pixel is "unknown"
    mask = np.zeros((10, 10), dtype=bool)
    mask[3, 3] = True
    assert tmap.has_unknown_in_mask(mask) is True


def test_has_unknown_false_for_fully_known_mask():
    tmap = TraversabilityMap.create(10, 10)
    full_mask = np.ones((10, 10), dtype=bool)
    tmap = tmap.update_region(full_mask, "grass")  # score 0.9, no zeros
    assert tmap.has_unknown_in_mask(full_mask) is False


def test_stop_threshold_constant_in_expected_range():
    assert 0.0 < STOP_THRESHOLD < 1.0
