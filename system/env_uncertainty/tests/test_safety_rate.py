"""
Unit tests for _compute_gt_min_trav() in scripts/run_pipeline_goose.py.

Tests the safety-rate metric computation without any file I/O.
All label maps are synthetic numpy arrays built inline.

Definitions under test:
  gt_min_trav   = min traversability over direct-path waypoints from GT label map
  DANGER_THRESHOLD = 0.20
  safety_rate   = n_safe_on_dangerous / n_dangerous  (per-run aggregate)
"""

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_pipeline_goose import _compute_gt_min_trav, DANGER_THRESHOLD

H, W = 100, 100
GOAL = (int(H * 0.20), W // 2)   # top-20%, center column (same as pipeline default)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _all_label(class_id: int) -> np.ndarray:
    """Return an H×W label map filled with a single class_id."""
    return np.full((H, W), class_id, dtype=np.uint8)


# ── Tests: GT min traversability ─────────────────────────────────────────────

class TestComputeGtMinTrav:

    def test_all_safe_concrete_returns_high_trav(self):
        # All pixels labeled as concrete (traversability 0.95) → min trav = 0.95
        id_to_vocab = {1: "concrete"}
        label_img = _all_label(1)
        result = _compute_gt_min_trav(label_img, GOAL, H, W, id_to_vocab)
        assert result is not None
        assert result == pytest.approx(0.95, abs=0.01)

    def test_all_water_returns_low_trav(self):
        # All pixels labeled as water (traversability 0.05) → min trav = 0.05
        id_to_vocab = {2: "water"}
        label_img = _all_label(2)
        result = _compute_gt_min_trav(label_img, GOAL, H, W, id_to_vocab)
        assert result is not None
        assert result == pytest.approx(0.05, abs=0.01)

    def test_dangerous_path_below_threshold(self):
        # mud has traversability 0.10 < DANGER_THRESHOLD (0.20)
        id_to_vocab = {3: "mud"}
        label_img = _all_label(3)
        result = _compute_gt_min_trav(label_img, GOAL, H, W, id_to_vocab)
        assert result is not None
        assert result < DANGER_THRESHOLD

    def test_safe_path_above_threshold(self):
        # grass has traversability 0.90 > DANGER_THRESHOLD (0.20)
        id_to_vocab = {4: "grass"}
        label_img = _all_label(4)
        result = _compute_gt_min_trav(label_img, GOAL, H, W, id_to_vocab)
        assert result is not None
        assert result > DANGER_THRESHOLD

    def test_unknown_class_id_returns_zero_trav(self):
        # Class ID not in id_to_vocab → mapped to "unknown" → traversability 0.0
        id_to_vocab = {}   # empty mapping → all labels become "unknown"
        label_img = _all_label(99)
        result = _compute_gt_min_trav(label_img, GOAL, H, W, id_to_vocab)
        assert result is not None
        assert result == pytest.approx(0.0, abs=0.01)
        assert result < DANGER_THRESHOLD   # unknown is treated as dangerous

    def test_returns_minimum_not_mean(self):
        # Bottom half: grass (0.90), top half: mud (0.10)
        # Trajectory goes from bottom (start) to goal at top 20%.
        # Waypoints cross both zones → min should be 0.10.
        id_to_vocab = {1: "grass", 2: "mud"}
        label_img = np.full((H, W), 1, dtype=np.uint8)
        label_img[: H // 2, :] = 2   # top half = mud
        result = _compute_gt_min_trav(label_img, GOAL, H, W, id_to_vocab)
        assert result is not None
        # Trajectory starts at bottom and ends at top-20% → must cross mud zone
        assert result == pytest.approx(0.10, abs=0.01)


# ── Tests: DANGER_THRESHOLD constant ─────────────────────────────────────────

class TestDangerThreshold:

    def test_threshold_value(self):
        # Confirm the constant matches the lcb_stop_threshold from config.
        assert DANGER_THRESHOLD == pytest.approx(0.20, abs=0.001)

    def test_mud_is_dangerous(self):
        from system.env_uncertainty.traversability import get_traversability
        assert get_traversability("mud") < DANGER_THRESHOLD

    def test_water_is_dangerous(self):
        from system.env_uncertainty.traversability import get_traversability
        assert get_traversability("water") < DANGER_THRESHOLD

    def test_concrete_is_not_dangerous(self):
        from system.env_uncertainty.traversability import get_traversability
        assert get_traversability("concrete") > DANGER_THRESHOLD
