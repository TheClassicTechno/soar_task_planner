"""
Unit tests for metrics.py — pure numpy, no GPU, no dataset download.
"""

import numpy as np
import pytest

from baselines.sam3.metrics import (
    compute_iou,
    compute_pixel_accuracy,
    compute_per_class_iou,
    compute_miou,
    compute_mask_coverage,
    MetricsAccumulator,
)
from baselines.sam3.data_loader import SAM3_TO_RUGD


# ── compute_iou ──────────────────────────────────────────────────────────────

def test_iou_perfect_overlap():
    mask = np.ones((4, 4), dtype=bool)
    assert compute_iou(mask, mask) == pytest.approx(1.0)


def test_iou_no_overlap():
    pred = np.zeros((4, 4), dtype=bool)
    pred[:2, :] = True
    gt = np.zeros((4, 4), dtype=bool)
    gt[2:, :] = True
    assert compute_iou(pred, gt) == pytest.approx(0.0)


def test_iou_partial_overlap():
    pred = np.zeros((4, 4), dtype=bool)
    pred[:, :2] = True   # left half
    gt = np.zeros((4, 4), dtype=bool)
    gt[:, 1:3] = True    # middle columns — overlap is col 1 only
    # intersection = 4 pixels, union = 4+4+4-4 = 12? let me compute:
    # pred true: cols 0,1 → 8 pixels; gt true: cols 1,2 → 8 pixels
    # intersection: col 1 → 4 pixels; union: 12 pixels
    assert compute_iou(pred, gt) == pytest.approx(4 / 12)


def test_iou_both_empty_returns_zero():
    pred = np.zeros((4, 4), dtype=bool)
    gt = np.zeros((4, 4), dtype=bool)
    # Edge case: both empty → IoU should be 0 (no union means safe_divide returns 0)
    assert compute_iou(pred, gt) == pytest.approx(0.0)


# ── compute_pixel_accuracy ───────────────────────────────────────────────────

def test_pixel_accuracy_all_correct():
    gt = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    pred = np.array([[1, 2], [3, 4]], dtype=np.int32)
    assert compute_pixel_accuracy(pred, gt) == pytest.approx(1.0)


def test_pixel_accuracy_all_wrong():
    gt = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    pred = np.array([[5, 6], [7, 8]], dtype=np.int32)
    assert compute_pixel_accuracy(pred, gt) == pytest.approx(0.0)


def test_pixel_accuracy_excludes_void():
    # Pixel 0 = void (class 0) — should be excluded from denominator
    gt = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    pred = np.array([[99, 1], [2, 99]], dtype=np.int32)  # void pixel wrong, 2 of 3 valid correct
    assert compute_pixel_accuracy(pred, gt) == pytest.approx(2 / 3)


def test_pixel_accuracy_all_void_returns_zero():
    gt = np.zeros((3, 3), dtype=np.uint8)
    pred = np.zeros((3, 3), dtype=np.int32)
    assert compute_pixel_accuracy(pred, gt) == pytest.approx(0.0)


# ── compute_per_class_iou ────────────────────────────────────────────────────

def _make_sam3_result(masks: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> dict:
    return {"masks": masks, "labels": labels, "scores": scores}


def test_per_class_iou_perfect_match():
    H, W = 8, 8
    queries = ["grass", "road"]
    sam3_to_rugd = {"grass": [3], "road": [10]}

    # GT: grass (id=3) in top half, road (id=10) in bottom half
    gt = np.zeros((H, W), dtype=np.uint8)
    gt[:4, :] = 3
    gt[4:, :] = 10

    # SAM3 predicts grass for label=0, road for label=1
    grass_mask = np.zeros((H, W), dtype=bool)
    grass_mask[:4, :] = True
    road_mask = np.zeros((H, W), dtype=bool)
    road_mask[4:, :] = True

    result = _make_sam3_result(
        masks=np.stack([grass_mask, road_mask]),
        labels=np.array([0, 1]),
        scores=np.array([0.9, 0.8]),
    )

    iou_map = compute_per_class_iou(result, gt, queries, sam3_to_rugd)
    assert iou_map["grass"] == pytest.approx(1.0)
    assert iou_map["road"] == pytest.approx(1.0)


def test_per_class_iou_no_predictions_returns_zero():
    H, W = 8, 8
    queries = ["grass"]
    sam3_to_rugd = {"grass": [3]}
    gt = np.full((H, W), 3, dtype=np.uint8)

    result = _make_sam3_result(
        masks=np.array([]),
        labels=np.array([]),
        scores=np.array([]),
    )
    iou_map = compute_per_class_iou(result, gt, queries, sam3_to_rugd)
    assert iou_map["grass"] == pytest.approx(0.0)


# ── compute_miou ─────────────────────────────────────────────────────────────

def test_miou_average():
    iou_map = {"road": 0.8, "grass": 0.6, "gravel": 0.4}
    assert compute_miou(iou_map) == pytest.approx(0.6)


def test_miou_empty_returns_zero():
    assert compute_miou({}) == pytest.approx(0.0)


# ── compute_mask_coverage ────────────────────────────────────────────────────

def test_mask_coverage_full():
    H, W = 4, 4
    mask = np.ones((1, H, W), dtype=bool)
    result = {"masks": mask}
    assert compute_mask_coverage(result, (H, W)) == pytest.approx(1.0)


def test_mask_coverage_half():
    H, W = 4, 4
    mask = np.zeros((1, H, W), dtype=bool)
    mask[0, :, :2] = True  # left half only
    result = {"masks": mask}
    assert compute_mask_coverage(result, (H, W)) == pytest.approx(0.5)


def test_mask_coverage_no_masks():
    result = {"masks": np.array([])}
    assert compute_mask_coverage(result, (4, 4)) == pytest.approx(0.0)


def test_mask_coverage_union_of_two_masks():
    H, W = 4, 4
    m1 = np.zeros((H, W), dtype=bool)
    m1[:, :2] = True   # left half
    m2 = np.zeros((H, W), dtype=bool)
    m2[:, 2:] = True   # right half
    result = {"masks": np.stack([m1, m2])}
    assert compute_mask_coverage(result, (H, W)) == pytest.approx(1.0)


# ── MetricsAccumulator ───────────────────────────────────────────────────────

def test_accumulator_summary_structure():
    queries = ["grass", "road"]
    acc = MetricsAccumulator(queries)
    H, W = 8, 8

    mask = np.zeros((H, W), dtype=bool)
    mask[:4, :] = True
    result = {"masks": np.stack([mask]), "labels": np.array([0]), "scores": np.array([0.9])}

    gt = np.full((H, W), 3, dtype=np.uint8)  # all grass
    acc.update(result, gt, SAM3_TO_RUGD, (H, W))

    summary = acc.summary()
    assert "n_images" in summary
    assert "mean_iou" in summary
    assert "per_class_iou" in summary
    assert "mean_coverage" in summary
    assert summary["n_images"] == 1


def test_accumulator_empty_returns_empty():
    acc = MetricsAccumulator(["grass"])
    assert acc.summary() == {}


def test_accumulator_no_gt_still_computes_coverage():
    queries = ["grass"]
    acc = MetricsAccumulator(queries)
    H, W = 8, 8
    mask = np.ones((1, H, W), dtype=bool)
    result = {"masks": mask, "labels": np.array([0]), "scores": np.array([0.9])}

    # No GT annotation
    acc.update(result, None, SAM3_TO_RUGD, (H, W))

    summary = acc.summary()
    assert summary["mean_coverage"] == pytest.approx(1.0)
    # Without GT, per_class_iou should be all 0.0
    assert summary["per_class_iou"]["grass"] == pytest.approx(0.0)
