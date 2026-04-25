"""
Segmentation metrics for the SAM3 baseline.

All functions operate on numpy arrays; no GPU or model dependency.
Two modes:
  - With RUGD ground truth: pixel accuracy + per-class IoU + mIoU
  - Without ground truth: mask coverage (what fraction of image is segmented)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or 0.0 if denominator is zero."""
    return numerator / denominator if denominator > 0 else 0.0


# ── Per-image metrics (with GT) ──────────────────────────────────────────────

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Intersection-over-Union for a single binary mask pair.

    Args:
        pred_mask: (H, W) bool or 0/1 array — predicted mask for one class.
        gt_mask:   (H, W) bool or 0/1 array — ground-truth pixels for that class.

    Returns:
        IoU in [0, 1].
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return _safe_divide(float(intersection), float(union))


def compute_pixel_accuracy(pred_class_map: np.ndarray, gt_class_map: np.ndarray) -> float:
    """
    Overall pixel accuracy: fraction of pixels where predicted class == GT class.

    Args:
        pred_class_map: (H, W) int array where each pixel = predicted class index.
                        Use -1 (background) for pixels not assigned to any query.
        gt_class_map:   (H, W) uint8 array of RUGD class IDs.

    Returns:
        Pixel accuracy in [0, 1].
    """
    valid = gt_class_map > 0  # exclude void (class 0)
    if not valid.any():
        return 0.0
    correct = (pred_class_map[valid] == gt_class_map[valid]).sum()
    return _safe_divide(float(correct), float(valid.sum()))


def compute_per_class_iou(
    sam3_results: Dict,
    gt_ann: np.ndarray,
    queries: List[str],
    sam3_to_rugd: Dict[str, List[int]],
) -> Dict[str, float]:
    """
    Compute IoU for each SAM3 concept query against the RUGD ground truth.

    For query q with RUGD class set R:
      GT mask = union of all pixels where gt_ann ∈ R
      Predicted mask = SAM3's segmentation mask for query q

    Args:
        sam3_results: Output of SAM3Repo.segment() —
                      {"masks": (N, H, W) bool, "labels": (N,) int, "scores": (N,) float}
        gt_ann:       (H, W) uint8 RUGD annotation array.
        queries:      Ordered list of concept query strings (matches config.yaml).
        sam3_to_rugd: Mapping from query string → list of RUGD class IDs.

    Returns:
        Dict mapping query string → IoU float.
    """
    masks = sam3_results.get("masks", np.array([]))   # (N, H, W)
    labels = sam3_results.get("labels", np.array([])) # (N,)

    per_class_iou: Dict[str, float] = {}

    for q_idx, query in enumerate(queries):
        rugd_ids = sam3_to_rugd.get(query, [])

        # GT: all pixels belonging to any RUGD class mapped to this query
        gt_pixels = np.zeros(gt_ann.shape, dtype=bool)
        for rid in rugd_ids:
            gt_pixels |= (gt_ann == rid)

        # Predicted: union of all SAM3 masks whose label == q_idx
        pred_pixels = np.zeros(gt_ann.shape, dtype=bool)
        if len(masks) > 0 and len(labels) > 0:
            for m_idx, lbl in enumerate(labels):
                if lbl == q_idx:
                    pred_pixels |= masks[m_idx].astype(bool)

        per_class_iou[query] = compute_iou(pred_pixels, gt_pixels)

    return per_class_iou


def compute_miou(per_class_iou: Dict[str, float]) -> float:
    """Mean IoU across all classes that appear in the mapping."""
    values = list(per_class_iou.values())
    return float(np.mean(values)) if values else 0.0


# ── Coverage metric (no GT required) ─────────────────────────────────────────

def compute_mask_coverage(sam3_results: Dict, image_hw: Tuple[int, int]) -> float:
    """
    Fraction of image pixels covered by at least one predicted mask.
    Useful when no ground-truth annotations are available.

    Args:
        sam3_results: SAM3 output dict with "masks" key.
        image_hw: (height, width) of the image.

    Returns:
        Coverage ratio in [0, 1].
    """
    masks = sam3_results.get("masks", np.array([]))
    if len(masks) == 0:
        return 0.0

    H, W = image_hw
    combined = np.zeros((H, W), dtype=bool)
    for m in masks:
        combined |= m.astype(bool)
    return _safe_divide(float(combined.sum()), float(H * W))


# ── Aggregation across a dataset split ───────────────────────────────────────

class MetricsAccumulator:
    """
    Accumulates per-image metrics across an entire dataset split,
    then computes final aggregates.

    Usage:
        acc = MetricsAccumulator(queries)
        for sample, result in zip(samples, results):
            acc.update(result, sample.load_annotation())
        summary = acc.summary()
    """

    def __init__(self, queries: List[str]):
        self.queries = queries
        self._per_class_iou_sum: Dict[str, float] = {q: 0.0 for q in queries}
        self._pixel_acc_sum = 0.0
        self._coverage_sum = 0.0
        self._count = 0

    def update(
        self,
        sam3_results: Dict,
        gt_ann: Optional[np.ndarray],
        sam3_to_rugd: Dict[str, List[int]],
        image_hw: Tuple[int, int],
    ) -> None:
        self._count += 1
        self._coverage_sum += compute_mask_coverage(sam3_results, image_hw)

        if gt_ann is not None:
            iou_map = compute_per_class_iou(sam3_results, gt_ann, self.queries, sam3_to_rugd)
            for q, iou in iou_map.items():
                self._per_class_iou_sum[q] += iou

    def summary(self) -> Dict:
        if self._count == 0:
            return {}

        per_class = {q: v / self._count for q, v in self._per_class_iou_sum.items()}
        return {
            "n_images": self._count,
            "mean_iou": compute_miou(per_class),
            "per_class_iou": per_class,
            "mean_coverage": self._coverage_sum / self._count,
        }
