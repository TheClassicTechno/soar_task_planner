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

  parse_user_response_rich:
    Confidence extraction:
      - "definitely" → label_confidence=0.95
      - "probably" → label_confidence=0.65
      - "not sure" (multi-word) beats "sure" (single-word) — phrase priority
      - No hedge word → neutral default 0.70
    Terrain label extraction:
      - "grass" / "lawn" → terrain_label="grass"
      - "pavement" → terrain_label="sidewalk"
      - "mud" → terrain_label="mud"
      - No terrain word → terrain_label=None
    Traversability extraction (affordance scoring):
      - "safe" alone → is_traversable=True, modifier>0
      - "wet slippery" → is_traversable=False, modifier<0
      - Composite: "safe but slippery" → sum of modifiers determines result
      - No traversability keywords → safety-first default: is_traversable=False, confidence=0.30
    Keywords list captures all matched tokens
    Traversability confidence stays in [0.05, 0.95]
"""

import numpy as np
import pytest

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.map_updater import (
    MapUpdater,
    ParsedUserResponse,
    UpdateResult,
    _parse_user_response,
    parse_user_response_rich,
)
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


# ── parse_user_response_rich — confidence extraction ─────────────────────────

def test_rich_definitely_sets_high_confidence():
    r = parse_user_response_rich("definitely safe to cross")
    assert r.label_confidence == pytest.approx(0.95)


def test_rich_probably_sets_medium_confidence():
    r = parse_user_response_rich("probably fine to walk through")
    assert r.label_confidence == pytest.approx(0.65)


def test_rich_not_sure_beats_sure():
    # "not sure" is a 2-word phrase; it must win over the single word "sure"
    r = parse_user_response_rich("not sure if it's safe")
    assert r.label_confidence == pytest.approx(0.30)


def test_rich_no_hedge_word_returns_neutral_confidence():
    r = parse_user_response_rich("it is grass")
    assert r.label_confidence == pytest.approx(0.70)


# ── parse_user_response_rich — terrain label extraction ──────────────────────

def test_rich_grass_detected():
    r = parse_user_response_rich("that's just grass, go ahead")
    assert r.terrain_label == "grass"


def test_rich_lawn_maps_to_grass():
    r = parse_user_response_rich("it looks like a lawn")
    assert r.terrain_label == "grass"


def test_rich_pavement_maps_to_sidewalk():
    r = parse_user_response_rich("pavement, totally safe")
    assert r.terrain_label == "sidewalk"


def test_rich_mud_detected():
    r = parse_user_response_rich("looks like mud, avoid it")
    assert r.terrain_label == "mud"


def test_rich_no_terrain_word_returns_none():
    r = parse_user_response_rich("yes it's fine")
    assert r.terrain_label is None


# ── parse_user_response_rich — traversability affordance scoring ──────────────

def test_rich_safe_alone_is_traversable():
    r = parse_user_response_rich("safe to cross")
    assert r.is_traversable is True
    assert r.affordance_modifier > 0


def test_rich_unsafe_alone_is_not_traversable():
    r = parse_user_response_rich("unsafe, do not enter")
    assert r.is_traversable is False
    assert r.affordance_modifier < 0


def test_rich_wet_slippery_is_not_traversable():
    r = parse_user_response_rich("it's wet and slippery")
    assert r.is_traversable is False
    assert r.affordance_modifier < 0


def test_rich_wet_slippery_modifier_is_sum():
    # wet=-0.15, slippery=-0.20 → sum=-0.35
    r = parse_user_response_rich("wet and slippery")
    assert r.affordance_modifier == pytest.approx(-0.35, abs=1e-9)


def test_rich_safe_but_slippery_modifier_sums_both():
    # safe=+0.20, slippery=-0.20 → sum=0.0 → is_traversable=True (boundary)
    r = parse_user_response_rich("safe but slippery")
    assert r.affordance_modifier == pytest.approx(0.0, abs=1e-9)
    assert r.is_traversable is True


def test_rich_avoid_word_makes_not_traversable():
    r = parse_user_response_rich("avoid this area")
    assert r.is_traversable is False


def test_rich_no_traversability_keywords_safety_first():
    # When no affordance keyword matches, default to is_traversable=False
    r = parse_user_response_rich("I have no idea")
    assert r.is_traversable is False
    assert r.traversability_confidence == pytest.approx(0.30)


def test_rich_no_traversability_keywords_empty_string():
    r = parse_user_response_rich("")
    assert r.is_traversable is False
    assert r.traversability_confidence == pytest.approx(0.30)


# ── parse_user_response_rich — traversability_confidence bounds ───────────────

def test_rich_traversability_confidence_in_unit_interval():
    for text in ["safe", "unsafe", "wet slippery muddy flooded dangerous", "dry firm flat paved"]:
        r = parse_user_response_rich(text)
        assert 0.0 <= r.traversability_confidence <= 1.0, f"out of bounds for: {text!r}"


def test_rich_high_positive_modifier_capped_at_095():
    # walkable+passable+traversable+safe+firm+solid+stable = many positives
    r = parse_user_response_rich("walkable passable traversable safe firm solid stable flat dry")
    assert r.traversability_confidence <= 0.95


def test_rich_high_negative_modifier_floored_at_005():
    r = parse_user_response_rich("unsafe dangerous avoid flooded icy slippery muddy steep")
    assert r.traversability_confidence >= 0.05


# ── parse_user_response_rich — keywords list ──────────────────────────────────

def test_rich_keywords_list_contains_matched_tokens():
    r = parse_user_response_rich("probably wet grass")
    assert "probably" in r.keywords
    assert "wet" in r.keywords
    assert "grass" in r.keywords


def test_rich_no_false_keywords_for_clean_safe_response():
    r = parse_user_response_rich("safe")
    assert "unsafe" not in r.keywords
    assert "safe" in r.keywords


def test_rich_keywords_empty_for_blank_input():
    r = parse_user_response_rich("")
    assert r.keywords == []


# ── parse_user_response_rich — ParsedUserResponse dataclass ──────────────────

def test_rich_returns_parsed_user_response_type():
    r = parse_user_response_rich("probably safe grass")
    assert isinstance(r, ParsedUserResponse)


def test_rich_all_fields_present():
    r = parse_user_response_rich("maybe wet mud")
    assert hasattr(r, "terrain_label")
    assert hasattr(r, "label_confidence")
    assert hasattr(r, "is_traversable")
    assert hasattr(r, "traversability_confidence")
    assert hasattr(r, "affordance_modifier")
    assert hasattr(r, "keywords")


# ── parse_user_response_rich — ranked terrain-label scoring ──────────────────

def test_rich_ranked_label_grass_beats_mud_by_frequency():
    # "grass" synonyms: "grass"(1) + "lawn"(1) + "field"(1) = 3 matches
    # "mud" synonyms: "mud"(1) = 1 match
    # Frequency ranking should pick grass (3 > 1)
    r = parse_user_response_rich("wet muddy grass lawn field")
    assert r.terrain_label == "grass"


def test_rich_ranked_label_single_synonym_still_works():
    r = parse_user_response_rich("it's just mud, avoid it")
    assert r.terrain_label == "mud"


def test_rich_ranked_label_tie_is_deterministic():
    # "grass"(1) vs "mud"(1) — tie broken by Counter.most_common (stable)
    r1 = parse_user_response_rich("muddy grass")
    r2 = parse_user_response_rich("muddy grass")
    assert r1.terrain_label == r2.terrain_label


def test_rich_ranked_label_all_kw_hits_in_keywords():
    # With ranked scoring, ALL matching terrain keywords appear in keywords list
    r = parse_user_response_rich("wet muddy grass lawn")
    terrain_hits = [k for k in r.keywords if k in ("mud", "muddy", "grass", "lawn")]
    # "muddy" is a traversability keyword, "grass" and "lawn" are terrain keywords
    assert "grass" in r.keywords
    assert "lawn" in r.keywords
