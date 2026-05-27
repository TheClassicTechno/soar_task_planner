"""
Unit tests for system/env_uncertainty/detector.py — models fully mocked.

Tests:
  _compute_overlap:
    - Full overlap returns 1.0
    - No overlap returns 0.0
    - Partial overlap returns correct fraction
    - Empty mask returns 0.0

  EnvironmentalUncertaintyDetector.detect (no models):
    - Returns DetectionResult
    - With no models: known_regions=[], unknown_regions=[]
    - has_unknown=False, sam3_coverage=0.0

  EnvironmentalUncertaintyDetector.detect (mocked SAM3):
    - Known regions populated from SAM3 output
    - Traversability map updated with SAM3 scores
    - sam3_coverage > 0.0

  EnvironmentalUncertaintyDetector.detect (mocked SAM3 + SAM2):
    - SAM2 regions with high overlap → not in unknown_regions
    - SAM2 regions with low overlap → added to unknown_regions
    - has_unknown True when unknown regions exist

  RegionInfo fields:
    - label, confidence, pixel_fraction, source, traversability all present

  DetectionResult fields:
    - traversability_map shape matches image
    - known + unknown regions are disjoint in source field
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from system.env_uncertainty.detector import (
    DetectionResult,
    EnvironmentalUncertaintyDetector,
    RegionInfo,
    _compute_overlap,
)

H, W = 64, 64
IMAGE = np.zeros((H, W, 3), dtype=np.uint8)


# ── _compute_overlap ──────────────────────────────────────────────────────────

def test_compute_overlap_full():
    mask = np.ones((10, 10), dtype=bool)
    coverage = np.ones((10, 10), dtype=bool)
    assert _compute_overlap(mask, coverage) == pytest.approx(1.0)


def test_compute_overlap_none():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:5, :] = True
    coverage = np.zeros((10, 10), dtype=bool)
    coverage[5:, :] = True
    assert _compute_overlap(mask, coverage) == pytest.approx(0.0)


def test_compute_overlap_partial():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:, :] = True           # 100 pixels in mask
    coverage = np.zeros((10, 10), dtype=bool)
    coverage[:, :5] = True      # left half only
    # 50 out of 100 pixels overlap → 0.5
    result = _compute_overlap(mask, coverage)
    assert result == pytest.approx(0.5, abs=0.01)


def test_compute_overlap_empty_mask():
    mask = np.zeros((10, 10), dtype=bool)
    coverage = np.ones((10, 10), dtype=bool)
    assert _compute_overlap(mask, coverage) == pytest.approx(0.0)


# ── Detector with no models ───────────────────────────────────────────────────

def test_detect_no_models_returns_detection_result():
    det = EnvironmentalUncertaintyDetector()
    result = det.detect(IMAGE)
    assert isinstance(result, DetectionResult)


def test_detect_no_models_empty_known_regions():
    det = EnvironmentalUncertaintyDetector()
    result = det.detect(IMAGE)
    assert result.known_regions == []


def test_detect_no_models_empty_unknown_regions():
    det = EnvironmentalUncertaintyDetector()
    result = det.detect(IMAGE)
    assert result.unknown_regions == []


def test_detect_no_models_has_unknown_false():
    det = EnvironmentalUncertaintyDetector()
    result = det.detect(IMAGE)
    assert result.has_unknown is False


def test_detect_no_models_sam3_coverage_zero():
    det = EnvironmentalUncertaintyDetector()
    result = det.detect(IMAGE)
    assert result.sam3_coverage == pytest.approx(0.0)


def test_detect_traversability_map_matches_image_shape():
    det = EnvironmentalUncertaintyDetector()
    result = det.detect(IMAGE)
    assert result.traversability_map.shape == (H, W)


# ── Detector with mocked SAM3 ─────────────────────────────────────────────────

def _make_sam3_mock(label_idx=5, area_frac=0.25):
    """Build a mock SAM3 that returns one region (label_idx=5 = grass)."""
    h, w = H, W
    n_pix = int(h * w * area_frac)
    mask = np.zeros((h, w), dtype=bool)
    mask.flat[:n_pix] = True

    mock = MagicMock()
    mock.segment.return_value = {
        "masks": mask[np.newaxis, :, :],   # shape (1, H, W)
        "labels": np.array([label_idx], dtype=np.int32),
        "scores": np.array([0.85], dtype=np.float32),
    }
    return mock


def test_detect_with_sam3_populates_known_regions():
    sam3 = _make_sam3_mock(label_idx=5)   # index 5 = "grass"
    det = EnvironmentalUncertaintyDetector(sam3_model=sam3)
    result = det.detect(IMAGE)
    assert len(result.known_regions) == 1
    assert result.known_regions[0].label == "grass"
    assert result.known_regions[0].source == "sam3"


def test_detect_with_sam3_coverage_positive():
    sam3 = _make_sam3_mock(area_frac=0.25)
    det = EnvironmentalUncertaintyDetector(sam3_model=sam3)
    result = det.detect(IMAGE)
    assert result.sam3_coverage > 0.0


def test_detect_with_sam3_traversability_map_updated():
    sam3 = _make_sam3_mock(label_idx=5)  # grass → 0.9
    det = EnvironmentalUncertaintyDetector(sam3_model=sam3)
    result = det.detect(IMAGE)
    # First pixel in mask should have grass score
    assert result.traversability_map.score_at(0, 0) == pytest.approx(0.9)


def test_detect_region_confidence_matches_model_output():
    sam3 = _make_sam3_mock(label_idx=5)
    det = EnvironmentalUncertaintyDetector(sam3_model=sam3)
    result = det.detect(IMAGE)
    assert result.known_regions[0].confidence == pytest.approx(0.85)


# ── Detector with SAM3 + SAM2 ─────────────────────────────────────────────────

def _make_sam2_mock_with_regions(regions):
    """
    Build a mock SAM2 returning the given list of {mask, score} dicts.
    regions: list of (mask_array, score_float)
    """
    mock = MagicMock()
    mock.segment_everything.return_value = [
        {"mask": m, "score": s} for m, s in regions
    ]
    return mock


def test_detect_sam2_region_with_high_overlap_is_not_unknown():
    # SAM3 covers top-left quadrant; SAM2 region is also top-left → high overlap
    h, w = H, W
    coverage_mask = np.zeros((h, w), dtype=bool)
    coverage_mask[:h // 2, :w // 2] = True

    sam3_mock = MagicMock()
    sam3_mock.segment.return_value = {
        "masks": coverage_mask[np.newaxis, :, :],
        "labels": np.array([5], dtype=np.int32),   # grass
        "scores": np.array([0.9], dtype=np.float32),
    }

    # SAM2 returns the same region — fully inside SAM3 coverage
    sam2_mock = _make_sam2_mock_with_regions([(coverage_mask.copy(), 0.8)])
    det = EnvironmentalUncertaintyDetector(
        sam3_model=sam3_mock,
        sam2_model=sam2_mock,
        overlap_threshold=0.3,
    )
    result = det.detect(IMAGE)
    assert result.unknown_regions == []
    assert result.has_unknown is False


def test_detect_sam2_region_with_low_overlap_is_unknown():
    h, w = H, W
    sam3_coverage = np.zeros((h, w), dtype=bool)
    sam3_coverage[:h // 2, :w // 2] = True   # top-left only

    # SAM2 region is bottom-right — completely outside SAM3 coverage
    sam2_region = np.zeros((h, w), dtype=bool)
    sam2_region[h // 2:, w // 2:] = True

    sam3_mock = MagicMock()
    sam3_mock.segment.return_value = {
        "masks": sam3_coverage[np.newaxis, :, :],
        "labels": np.array([5], dtype=np.int32),
        "scores": np.array([0.9], dtype=np.float32),
    }
    sam2_mock = _make_sam2_mock_with_regions([(sam2_region, 0.7)])
    det = EnvironmentalUncertaintyDetector(
        sam3_model=sam3_mock,
        sam2_model=sam2_mock,
        overlap_threshold=0.3,
    )
    result = det.detect(IMAGE)
    assert len(result.unknown_regions) == 1
    assert result.has_unknown is True
    assert result.unknown_regions[0].source == "sam2"
    assert result.unknown_regions[0].label == "unknown"
    assert result.unknown_regions[0].traversability == pytest.approx(0.0)


def test_detect_unknown_coverage_reflects_unknown_region_size():
    h, w = H, W
    sam3_coverage = np.zeros((h, w), dtype=bool)
    sam3_coverage[:h // 2, :] = True   # top half known

    sam2_region = np.zeros((h, w), dtype=bool)
    sam2_region[h // 2:, :] = True     # bottom half unknown

    sam3_mock = MagicMock()
    sam3_mock.segment.return_value = {
        "masks": sam3_coverage[np.newaxis, :, :],
        "labels": np.array([5], dtype=np.int32),
        "scores": np.array([0.9], dtype=np.float32),
    }
    sam2_mock = _make_sam2_mock_with_regions([(sam2_region, 0.8)])
    det = EnvironmentalUncertaintyDetector(
        sam3_model=sam3_mock,
        sam2_model=sam2_mock,
        overlap_threshold=0.3,
        min_unknown_pixel_fraction=0.01,
    )
    result = det.detect(IMAGE)
    # Unknown region covers bottom half → ~50% of image
    assert result.unknown_coverage == pytest.approx(0.5, abs=0.02)


def test_small_sam2_region_filtered_by_min_frac():
    h, w = H, W
    sam3_mock = MagicMock()
    sam3_mock.segment.return_value = {
        "masks": np.zeros((0, h, w), dtype=bool),
        "labels": np.zeros(0, dtype=np.int32),
        "scores": np.zeros(0, dtype=np.float32),
    }
    # Tiny region: 1 pixel → well below 2% threshold
    tiny = np.zeros((h, w), dtype=bool)
    tiny[0, 0] = True
    sam2_mock = _make_sam2_mock_with_regions([(tiny, 0.9)])
    det = EnvironmentalUncertaintyDetector(
        sam3_model=sam3_mock,
        sam2_model=sam2_mock,
        overlap_threshold=0.3,
        min_unknown_pixel_fraction=0.02,
    )
    result = det.detect(IMAGE)
    assert result.unknown_regions == []


def test_detect_sam2_dictionary_format_supported():
    h, w = H, W
    sam3_mock = MagicMock()
    sam3_mock.segment.return_value = {
        "masks": np.zeros((0, h, w), dtype=bool),
        "labels": np.zeros(0, dtype=np.int32),
        "scores": np.zeros(0, dtype=np.float32),
    }

    # Create SAM2 mock returning Dict format instead of List[Dict]
    sam2_region = np.zeros((h, w), dtype=bool)
    sam2_region[h // 2:, :] = True     # bottom half unknown
    sam2_mock = MagicMock()
    sam2_mock.segment_everything.return_value = {
        "masks": sam2_region[np.newaxis, :, :],
        "scores": np.array([0.85], dtype=np.float32),
        "inference_time_s": 0.05,
    }

    det = EnvironmentalUncertaintyDetector(
        sam3_model=sam3_mock,
        sam2_model=sam2_mock,
        overlap_threshold=0.3,
        min_unknown_pixel_fraction=0.01,
    )
    result = det.detect(IMAGE)
    assert len(result.unknown_regions) == 1
    assert result.unknown_regions[0].label == "unknown"
    assert result.unknown_regions[0].confidence == pytest.approx(0.85)
    assert np.array_equal(result.unknown_regions[0].mask, sam2_region)
