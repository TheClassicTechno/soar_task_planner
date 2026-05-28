"""
Tests for SceneGraph and TerrainNode Dirichlet semantic uncertainty.

Covers:
  - pixel_to_cell: origin, max corner, midpoints, boundary clamping
  - upsert_region: creates new node with correct defaults
  - upsert_region: second call updates gp_mean/variance; does not downgrade
  - upsert_region: certainty_level UNKNOWN when no gp_mean; INFERRED when gp_mean provided
  - recall: returns None for unknown key; returns node for known key
  - should_skip_asking: False for unconfirmed; False for confirmed+high variance
  - should_skip_asking: True after mark_confirmed + low gp_variance
  - mark_confirmed: sets user_confirmed=True, certainty=CONFIRMED
  - mark_confirmed: returns None for unknown key
  - update_from_gp: upserts all regions with GP values
  - update_from_gp: adjacent_trajectory_ids populated when trajectory passes through cell
  - node_count: increments with new (label, cell) pairs; same cell different label = 2 nodes
  - CertaintyLevel ordering: UNKNOWN < INFERRED < CONFIRMED (by enum value)
  Dirichlet coverage:
  - upsert_region sets label-informed prior (alpha[label]=2.0, others=1.0)
  - upsert_region for unknown label uses uniform prior (all 1.0)
  - semantic_distribution sums to 1
  - update_from_user increments the correct class by confidence
  - update_from_user ignores unknown labels
  - semantic_entropy decreases after confident single-class update
  - expected_traversability shifts toward updated class's score
  - top_k_classes returns sorted (label, probability) pairs
  - TerrainNodeV2 is the same class as TerrainNode (alias)
"""

import math
import numpy as np
import pytest
from dataclasses import dataclass
from typing import List, Tuple
from unittest.mock import MagicMock

from system.env_uncertainty.scene_graph import (
    CertaintyLevel,
    SceneGraph,
    TerrainNode,
    TerrainNodeV2,
    TERRAIN_CLASSES,
    K_CLASSES,
)

H, W = 100, 100


# ── pixel_to_cell ─────────────────────────────────────────────────────────────

def test_pixel_to_cell_origin():
    sg = SceneGraph()
    assert sg.pixel_to_cell(0, 0, H, W) == (0, 0)


def test_pixel_to_cell_max_corner():
    sg = SceneGraph()
    cy, cx = sg.pixel_to_cell(H - 1, W - 1, H, W)
    assert cy == SceneGraph.GRID_SIZE - 1
    assert cx == SceneGraph.GRID_SIZE - 1


def test_pixel_to_cell_midpoint():
    sg = SceneGraph()
    cy, cx = sg.pixel_to_cell(50, 50, H, W)
    assert 0 <= cy < SceneGraph.GRID_SIZE
    assert 0 <= cx < SceneGraph.GRID_SIZE


def test_pixel_to_cell_clamped_within_grid():
    sg = SceneGraph()
    for y in [0, 25, 50, 75, H - 1]:
        for x in [0, 25, 50, 75, W - 1]:
            cy, cx = sg.pixel_to_cell(y, x, H, W)
            assert 0 <= cy < SceneGraph.GRID_SIZE
            assert 0 <= cx < SceneGraph.GRID_SIZE


# ── upsert_region ─────────────────────────────────────────────────────────────

def test_upsert_creates_new_node():
    sg = SceneGraph()
    node = sg.upsert_region("grass", 50, 50, H, W)
    assert isinstance(node, TerrainNode)
    assert node.label == "grass"


def test_upsert_uses_get_traversability_as_default_mean():
    from system.env_uncertainty.traversability import get_traversability
    sg = SceneGraph()
    node = sg.upsert_region("grass", 50, 50, H, W)
    assert abs(node.gp_mean - get_traversability("grass")) < 1e-9


def test_upsert_certainty_unknown_without_gp_mean():
    sg = SceneGraph()
    node = sg.upsert_region("grass", 50, 50, H, W)
    assert node.certainty_level == CertaintyLevel.UNKNOWN


def test_upsert_certainty_inferred_with_gp_mean():
    sg = SceneGraph()
    node = sg.upsert_region("grass", 50, 50, H, W, gp_mean=0.85)
    assert node.certainty_level == CertaintyLevel.INFERRED


def test_upsert_second_call_updates_gp_mean():
    sg = SceneGraph()
    sg.upsert_region("grass", 50, 50, H, W)
    node = sg.upsert_region("grass", 50, 50, H, W, gp_mean=0.75)
    assert abs(node.gp_mean - 0.75) < 1e-9


def test_upsert_second_call_updates_gp_variance():
    sg = SceneGraph()
    sg.upsert_region("grass", 50, 50, H, W)
    node = sg.upsert_region("grass", 50, 50, H, W, gp_variance=0.02)
    assert abs(node.gp_variance - 0.02) < 1e-9


def test_upsert_does_not_downgrade_confirmed():
    sg = SceneGraph()
    node = sg.upsert_region("grass", 50, 50, H, W, gp_mean=0.85)
    cell = node.position_cell_id
    node.certainty_level = CertaintyLevel.CONFIRMED
    node.user_confirmed = True
    # Update again — should not reset to UNKNOWN/INFERRED
    updated = sg.upsert_region("grass", 50, 50, H, W, gp_mean=0.80)
    assert updated.user_confirmed is True


def test_upsert_increments_node_count():
    sg = SceneGraph()
    assert sg.node_count == 0
    sg.upsert_region("grass", 50, 50, H, W)
    assert sg.node_count == 1


def test_upsert_same_cell_different_label_two_nodes():
    sg = SceneGraph()
    sg.upsert_region("grass", 50, 50, H, W)
    sg.upsert_region("unknown", 50, 50, H, W)
    assert sg.node_count == 2


def test_upsert_label_lowercased():
    sg = SceneGraph()
    node = sg.upsert_region("Grass", 50, 50, H, W)
    assert node.label == "grass"


# ── recall ────────────────────────────────────────────────────────────────────

def test_recall_returns_none_for_unknown():
    sg = SceneGraph()
    assert sg.recall("grass", 5, 5) is None


def test_recall_returns_node_after_upsert():
    sg = SceneGraph()
    node = sg.upsert_region("grass", 50, 50, H, W)
    cy, cx = node.position_cell_id
    recalled = sg.recall("grass", cy, cx)
    assert recalled is node


def test_recall_case_insensitive():
    sg = SceneGraph()
    node = sg.upsert_region("Grass", 50, 50, H, W)
    cy, cx = node.position_cell_id
    recalled = sg.recall("GRASS", cy, cx)
    assert recalled is not None


# ── should_skip_asking ────────────────────────────────────────────────────────

def test_should_skip_returns_false_for_unknown_label():
    sg = SceneGraph()
    skip, node = sg.should_skip_asking("grass", 5, 5)
    assert skip is False
    assert node is None


def test_should_skip_false_for_unconfirmed_node():
    sg = SceneGraph()
    n = sg.upsert_region("grass", 50, 50, H, W, gp_mean=0.85, gp_variance=0.01)
    cy, cx = n.position_cell_id
    skip, _ = sg.should_skip_asking("grass", cy, cx)
    assert skip is False


def test_should_skip_false_for_confirmed_high_variance():
    sg = SceneGraph()
    n = sg.upsert_region("grass", 50, 50, H, W, gp_mean=0.85, gp_variance=0.10)
    cy, cx = n.position_cell_id
    n.user_confirmed = True
    n.certainty_level = CertaintyLevel.CONFIRMED
    skip, _ = sg.should_skip_asking("grass", cy, cx)
    assert skip is False


def test_should_skip_true_after_confirmed_and_low_variance():
    sg = SceneGraph()
    n = sg.upsert_region("grass", 50, 50, H, W, gp_mean=0.85, gp_variance=0.01)
    cy, cx = n.position_cell_id
    sg.mark_confirmed("grass", cy, cx, is_traversable=True)
    skip, node = sg.should_skip_asking("grass", cy, cx)
    assert skip is True
    assert node is not None


def test_should_skip_threshold_boundary():
    sg = SceneGraph()
    n = sg.upsert_region("grass", 50, 50, H, W, gp_mean=0.85, gp_variance=SceneGraph.SKIP_VARIANCE_THRESHOLD)
    cy, cx = n.position_cell_id
    n.user_confirmed = True
    # Exactly at threshold — should NOT skip (< not <=)
    skip, _ = sg.should_skip_asking("grass", cy, cx)
    assert skip is False


# ── mark_confirmed ────────────────────────────────────────────────────────────

def test_mark_confirmed_sets_user_confirmed():
    sg = SceneGraph()
    n = sg.upsert_region("grass", 50, 50, H, W)
    cy, cx = n.position_cell_id
    sg.mark_confirmed("grass", cy, cx, is_traversable=True)
    assert n.user_confirmed is True


def test_mark_confirmed_sets_certainty_confirmed():
    sg = SceneGraph()
    n = sg.upsert_region("grass", 50, 50, H, W)
    cy, cx = n.position_cell_id
    sg.mark_confirmed("grass", cy, cx, is_traversable=True)
    assert n.certainty_level == CertaintyLevel.CONFIRMED


def test_mark_confirmed_returns_none_for_missing():
    sg = SceneGraph()
    result = sg.mark_confirmed("grass", 5, 5, is_traversable=True)
    assert result is None


# ── update_from_gp ────────────────────────────────────────────────────────────

@dataclass
class _FakeRegion:
    label: str
    mask: np.ndarray


class _FakeGPMap:
    """Mock GPTraversabilityMap that returns fixed predictions."""
    def predict(self, pixel_y, pixel_x, height, width, beta=None):
        from system.env_uncertainty.gp_traversability import GPPrediction
        return GPPrediction(mu=0.75, sigma=0.10, lcb=0.60, beta=1.5, source="posterior")


def test_update_from_gp_creates_nodes():
    sg = SceneGraph()
    mask = np.zeros((H, W), dtype=bool)
    mask[40:60, 40:60] = True
    regions = [_FakeRegion("grass", mask)]
    sg.update_from_gp(_FakeGPMap(), regions, H, W)
    assert sg.node_count == 1


def test_update_from_gp_sets_gp_mean():
    sg = SceneGraph()
    mask = np.zeros((H, W), dtype=bool)
    mask[40:60, 40:60] = True
    regions = [_FakeRegion("grass", mask)]
    sg.update_from_gp(_FakeGPMap(), regions, H, W)
    nodes = list(sg._nodes.values())
    assert abs(nodes[0].gp_mean - 0.75) < 1e-9


def test_update_from_gp_sets_gp_variance_from_sigma_squared():
    sg = SceneGraph()
    mask = np.zeros((H, W), dtype=bool)
    mask[40:60, 40:60] = True
    regions = [_FakeRegion("grass", mask)]
    sg.update_from_gp(_FakeGPMap(), regions, H, W)
    nodes = list(sg._nodes.values())
    assert abs(nodes[0].gp_variance - 0.10 ** 2) < 1e-9


def test_update_from_gp_trajectory_ids_recorded():
    sg = SceneGraph()
    mask = np.zeros((H, W), dtype=bool)
    mask[40:60, 40:60] = True
    regions = [_FakeRegion("grass", mask)]

    @dataclass
    class _Traj:
        name: str
        waypoints: List[Tuple[int, int]]

    # mask[40:60, 40:60] centroid is pixel (49, 49) → cell (4, 4)
    # Waypoint (45, 45) also maps to cell (4, 4) → triggers adjacency record
    traj = _Traj(name="forward", waypoints=[(45, 45)])
    sg.update_from_gp(_FakeGPMap(), regions, H, W, trajectories=[traj])
    nodes = list(sg._nodes.values())
    assert "forward" in nodes[0].adjacent_trajectory_ids


# ── node_count ────────────────────────────────────────────────────────────────

def test_node_count_zero_initially():
    sg = SceneGraph()
    assert sg.node_count == 0


def test_node_count_multiple_distinct_regions():
    sg = SceneGraph()
    sg.upsert_region("grass", 10, 10, H, W)
    sg.upsert_region("mud", 90, 90, H, W)
    sg.upsert_region("grass", 10, 10, H, W)  # update, not new
    assert sg.node_count == 2


# ── TerrainNode / TerrainNodeV2 Dirichlet fields ──────────────────────────────

def test_terrain_node_v2_is_same_class_as_terrain_node():
    # TerrainNodeV2 is a forward-compatibility alias — must be the same object.
    assert TerrainNodeV2 is TerrainNode


def test_upsert_label_informed_prior_scales_with_confidence():
    # For a known SAM3 label with default confidence=0.9, alpha[label] should
    # equal 1 + 0.9 * 30 = 28.0 and all other classes should remain 1.0.
    from system.env_uncertainty.scene_graph import _CONFIDENCE_PSEUDOCOUNT_SCALE
    sg = SceneGraph()
    node = sg.upsert_region("grass", 50, 50, H, W)  # default region_confidence=0.9
    grass_idx = TERRAIN_CLASSES.index("grass")
    expected = 1.0 + 0.9 * _CONFIDENCE_PSEUDOCOUNT_SCALE
    assert abs(node.dirichlet_alpha[grass_idx] - expected) < 1e-6
    for i, cls in enumerate(TERRAIN_CLASSES):
        if cls != "grass":
            assert abs(node.dirichlet_alpha[i] - 1.0) < 1e-6, f"alpha[{cls}] should be 1.0"

    # Low-confidence detection → smaller pseudocount → higher entropy
    node_low = sg.upsert_region("mud", 10, 10, H, W, region_confidence=0.30)
    mud_idx = TERRAIN_CLASSES.index("mud")
    expected_low = 1.0 + 0.30 * _CONFIDENCE_PSEUDOCOUNT_SCALE
    assert abs(node_low.dirichlet_alpha[mud_idx] - expected_low) < 1e-6


def test_upsert_unknown_label_uses_uniform_prior():
    # "unknown" is always uniform regardless of confidence — it is a sentinel
    # meaning "SAM3 found no class", so maximum entropy is the correct prior.
    sg = SceneGraph()
    node = sg.upsert_region("unknown", 50, 50, H, W)
    for alpha_val in node.dirichlet_alpha:
        assert abs(alpha_val - 1.0) < 1e-9, "unknown label must use flat uniform prior"


def test_dirichlet_alpha_length_equals_k_classes():
    sg = SceneGraph()
    node = sg.upsert_region("grass", 50, 50, H, W)
    assert len(node.dirichlet_alpha) == K_CLASSES


def test_semantic_distribution_sums_to_one():
    sg = SceneGraph()
    node = sg.upsert_region("grass", 50, 50, H, W)
    p = node.semantic_distribution()
    assert abs(p.sum() - 1.0) < 1e-6


def test_semantic_distribution_all_nonnegative():
    sg = SceneGraph()
    node = sg.upsert_region("unknown", 50, 50, H, W)
    p = node.semantic_distribution()
    assert (p >= 0).all()


def test_map_class_returns_initialized_label():
    # After upsert, the informed prior makes the SAM3-detected label most probable.
    sg = SceneGraph()
    node = sg.upsert_region("grass", 50, 50, H, W)
    assert node.map_class() == "grass"


def test_update_from_user_increments_correct_class():
    sg = SceneGraph()
    node = sg.upsert_region("unknown", 50, 50, H, W)
    before = node.dirichlet_alpha[TERRAIN_CLASSES.index("grass")]
    node.update_from_user("grass", 0.8)
    after = node.dirichlet_alpha[TERRAIN_CLASSES.index("grass")]
    assert abs(after - before - 0.8) < 1e-9


def test_update_from_user_does_not_change_other_classes():
    sg = SceneGraph()
    node = sg.upsert_region("unknown", 50, 50, H, W)
    alpha_before = list(node.dirichlet_alpha)
    node.update_from_user("grass", 1.0)
    grass_idx = TERRAIN_CLASSES.index("grass")
    for i, cls in enumerate(TERRAIN_CLASSES):
        if i != grass_idx:
            assert abs(node.dirichlet_alpha[i] - alpha_before[i]) < 1e-9


def test_update_from_user_ignores_unknown_label_string():
    sg = SceneGraph()
    node = sg.upsert_region("grass", 50, 50, H, W)
    alpha_before = list(node.dirichlet_alpha)
    node.update_from_user("completely_made_up_terrain", 1.0)
    assert node.dirichlet_alpha == alpha_before


def test_semantic_entropy_decreases_after_confident_update():
    # Concentrating mass on one class reduces uncertainty.
    sg = SceneGraph()
    node = sg.upsert_region("unknown", 50, 50, H, W)
    entropy_before = node.semantic_entropy()
    node.update_from_user("grass", 20.0)   # very confident: +20 pseudocounts
    entropy_after = node.semantic_entropy()
    assert entropy_after < entropy_before


def test_semantic_entropy_is_finite():
    sg = SceneGraph()
    node = sg.upsert_region("grass", 50, 50, H, W)
    h = node.semantic_entropy()
    assert math.isfinite(h)


def test_expected_traversability_shifts_toward_updated_class():
    # Updating strongly toward grass (trav=0.90) should push expected_trav up.
    sg = SceneGraph()
    node = sg.upsert_region("unknown", 50, 50, H, W)
    trav_before = node.expected_traversability()
    node.update_from_user("grass", 50.0)   # dominate distribution with grass
    trav_after = node.expected_traversability()
    assert trav_after > trav_before


def test_expected_traversability_shifts_toward_low_trav_class():
    # Updating toward mud (trav=0.10) should push expected_trav down.
    sg = SceneGraph()
    node = sg.upsert_region("grass", 50, 50, H, W)
    trav_before = node.expected_traversability()
    node.update_from_user("mud", 50.0)
    trav_after = node.expected_traversability()
    assert trav_after < trav_before


def test_top_k_classes_returns_k_items():
    sg = SceneGraph()
    node = sg.upsert_region("grass", 50, 50, H, W)
    top = node.top_k_classes(k=3)
    assert len(top) == 3


def test_top_k_classes_sorted_descending():
    sg = SceneGraph()
    node = sg.upsert_region("grass", 50, 50, H, W)
    top = node.top_k_classes(k=3)
    probs = [p for _, p in top]
    assert probs == sorted(probs, reverse=True)


def test_top_k_classes_first_is_map_class():
    sg = SceneGraph()
    node = sg.upsert_region("grass", 50, 50, H, W)
    node.update_from_user("grass", 10.0)   # make grass strongly dominant
    top = node.top_k_classes(k=3)
    assert top[0][0] == "grass"


def test_top_k_classes_probabilities_sum_close_to_one_for_full_k():
    # With k=K_CLASSES we get all classes; their probabilities must sum to 1.
    sg = SceneGraph()
    node = sg.upsert_region("grass", 50, 50, H, W)
    top = node.top_k_classes(k=K_CLASSES)
    total = sum(p for _, p in top)
    assert abs(total - 1.0) < 1e-6


def test_sequential_updates_accumulate_correctly():
    # Three updates to the same class should add up.
    sg = SceneGraph()
    node = sg.upsert_region("unknown", 50, 50, H, W)
    grass_idx = TERRAIN_CLASSES.index("grass")
    start = node.dirichlet_alpha[grass_idx]
    node.update_from_user("grass", 0.5)
    node.update_from_user("grass", 0.5)
    node.update_from_user("grass", 1.0)
    assert abs(node.dirichlet_alpha[grass_idx] - (start + 2.0)) < 1e-9
