"""
Environmental Uncertainty Detector — SAM3 + SAM2 spatial subtraction.

Implements the core innovation described in docs/methodology.md §4.

Algorithm:
  1. Run SAM3 on the image → known_coverage: set of terrain-labeled regions
  2. Run SAM2 in 'segment everything' mode → all_regions: all detected regions
  3. For each SAM2 region:
       overlap = |region ∩ known_coverage| / |region|
       if overlap < overlap_threshold → label this region "unknown"
  4. Return DetectionResult with known and unknown region lists

The detector is designed to accept pre-built model instances, making it
testable with mocks. Neither SAM3 nor SAM2 are loaded at import time.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from system.env_uncertainty.traversability import (
    TraversabilityMap,
    get_traversability,
)


@dataclass
class RegionInfo:
    """
    Describes one detected region in the scene.

    label:          terrain class name ("grass", "vegetation", ...) or "unknown"
    mask:           (H, W) bool array identifying the region's pixels
    confidence:     model confidence score [0, 1]
    pixel_fraction: fraction of total image pixels covered by this mask
    source:         "sam3" for known regions, "sam2" for unknown residuals
    traversability: pre-computed traversability score for this region
    """

    label: str
    mask: np.ndarray
    confidence: float
    pixel_fraction: float
    source: str                     # "sam3" or "sam2"
    traversability: float


@dataclass
class DetectionResult:
    """
    Output from one EnvironmentalUncertaintyDetector.detect() call.

    known_regions:      regions SAM3 identified with a terrain class
    unknown_regions:    SAM2 regions with <overlap_threshold overlap with SAM3
    image_shape:        (H, W) of the source image
    sam3_coverage:      fraction of image pixels covered by any SAM3 region
    unknown_coverage:   fraction of image pixels in unknown regions
    has_unknown:        True if at least one unknown region was detected
    traversability_map: per-pixel traversability map built from all regions
    """

    known_regions: List[RegionInfo]
    unknown_regions: List[RegionInfo]
    image_shape: Tuple[int, int]
    sam3_coverage: float
    unknown_coverage: float
    has_unknown: bool
    traversability_map: TraversabilityMap


class EnvironmentalUncertaintyDetector:
    """
    Detect unknown terrain regions using SAM3 + SAM2 spatial subtraction.

    Accepts pre-built model instances so it can be tested without GPU access.
    SAM3 is expected to expose a segment(image) method returning:
        {"masks": (N,H,W) bool, "labels": (N,) int, "scores": (N,) float}
    SAM2 is expected to expose a segment_everything(image) method returning:
        [{"mask": (H,W) bool, "score": float}, ...]

    Both models are optional. If sam3_model is None, known_coverage is empty.
    If sam2_model is None, unknown detection is skipped (no SAM2 regions).
    """

    def __init__(
        self,
        sam3_model: Optional[Any] = None,
        sam2_model: Optional[Any] = None,
        overlap_threshold: float = 0.3,
        min_unknown_pixel_fraction: float = 0.02,
        sam3_queries: Optional[List[str]] = None,
    ):
        """
        Args:
            sam3_model:                Model exposing segment(pil_image) → dict.
            sam2_model:                Model exposing segment_everything(image) → list.
            overlap_threshold:         SAM2 region is "unknown" if its overlap with
                                       SAM3 coverage is below this value (default 0.3).
            min_unknown_pixel_fraction: Ignore unknown regions smaller than this
                                       fraction of total image pixels.
            sam3_queries:              Ordered list of terrain class names for SAM3's
                                       label indices. Defaults to SAM3 config vocabulary.
        """
        self._sam3 = sam3_model
        self._sam2 = sam2_model
        self._overlap_threshold = overlap_threshold
        self._min_unknown_frac = min_unknown_pixel_fraction
        self._sam3_queries = sam3_queries or _DEFAULT_SAM3_QUERIES

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Run the full SAM3+SAM2 detection pipeline on one image.

        Args:
            image: (H, W, 3) uint8 numpy array (RGB).

        Returns:
            DetectionResult with known and unknown regions and traversability map.
        """
        h, w = image.shape[:2]
        tmap = TraversabilityMap.create(h, w)
        known_regions: List[RegionInfo] = []
        unknown_regions: List[RegionInfo] = []

        # ── Step 1: SAM3 — identify known terrain regions ─────────────────────
        known_coverage = np.zeros((h, w), dtype=bool)

        if self._sam3 is not None:
            sam3_output = self._sam3.segment(_to_pil(image))
            known_regions, known_coverage = self._build_known_regions(
                sam3_output, h, w
            )
            for region in known_regions:
                tmap = tmap.update_region(region.mask, region.label)

        sam3_coverage = float(np.sum(known_coverage)) / (h * w)

        # ── Step 2: SAM2 — identify all regions ───────────────────────────────
        if self._sam2 is not None:
            sam2_output = self._sam2.segment_everything(image)
            unknown_regions = self._find_unknown_regions(
                sam2_output, known_coverage, h, w
            )
            for region in unknown_regions:
                tmap = tmap.update_region(region.mask, "unknown")

        # Compute total unknown coverage
        unknown_mask_union = np.zeros((h, w), dtype=bool)
        for region in unknown_regions:
            unknown_mask_union |= region.mask
        unknown_coverage = float(np.sum(unknown_mask_union)) / (h * w)

        return DetectionResult(
            known_regions=known_regions,
            unknown_regions=unknown_regions,
            image_shape=(h, w),
            sam3_coverage=sam3_coverage,
            unknown_coverage=unknown_coverage,
            has_unknown=len(unknown_regions) > 0,
            traversability_map=tmap,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_known_regions(
        self, sam3_output: Dict, h: int, w: int
    ) -> Tuple[List[RegionInfo], np.ndarray]:
        """
        Convert SAM3 output dict into RegionInfo objects and a coverage mask.

        SAM3 returns:
            masks:  (N, H, W) bool
            labels: (N,) int  — index into self._sam3_queries
            scores: (N,) float

        Returns:
            (regions, coverage_mask) where coverage_mask is the union of all masks.
        """
        masks = sam3_output.get("masks", np.zeros((0, h, w), dtype=bool))
        labels = sam3_output.get("labels", np.zeros(0, dtype=int))
        scores = sam3_output.get("scores", np.zeros(0, dtype=float))
        total_pixels = h * w

        regions: List[RegionInfo] = []
        coverage = np.zeros((h, w), dtype=bool)

        for mask, label_idx, score in zip(masks, labels, scores):
            label_idx = int(label_idx)
            label = (
                self._sam3_queries[label_idx]
                if 0 <= label_idx < len(self._sam3_queries)
                else "unknown"
            )
            pixel_frac = float(np.sum(mask)) / total_pixels
            regions.append(
                RegionInfo(
                    label=label,
                    mask=mask.astype(bool),
                    confidence=float(score),
                    pixel_fraction=pixel_frac,
                    source="sam3",
                    traversability=get_traversability(label),
                )
            )
            coverage |= mask.astype(bool)

        return regions, coverage

    def _find_unknown_regions(
        self,
        sam2_output: List[Dict],
        known_coverage: np.ndarray,
        h: int,
        w: int,
    ) -> List[RegionInfo]:
        """
        Find SAM2 regions that are not adequately explained by SAM3.

        A SAM2 region is "unknown" when:
            overlap(region, known_coverage) < self._overlap_threshold

        where overlap is the fraction of the region's pixels that fall inside
        the SAM3-labeled coverage area.

        Small regions (< min_unknown_pixel_fraction of total image) are ignored
        to avoid noise from tiny detection artifacts.
        """
        total_pixels = h * w
        unknown: List[RegionInfo] = []

        for item in sam2_output:
            mask = np.asarray(item.get("mask", np.zeros((h, w), dtype=bool)), dtype=bool)
            score = float(item.get("score", 0.0))
            region_pixels = int(np.sum(mask))

            if region_pixels == 0:
                continue

            pixel_frac = region_pixels / total_pixels
            if pixel_frac < self._min_unknown_frac:
                continue

            overlap = _compute_overlap(mask, known_coverage)
            if overlap < self._overlap_threshold:
                unknown.append(
                    RegionInfo(
                        label="unknown",
                        mask=mask,
                        confidence=score,
                        pixel_fraction=pixel_frac,
                        source="sam2",
                        traversability=0.0,
                    )
                )

        return unknown


# ── Module-level helpers ──────────────────────────────────────────────────────

def _compute_overlap(mask: np.ndarray, coverage: np.ndarray) -> float:
    """
    Compute what fraction of mask's pixels are covered by the coverage area.

    Returns a value in [0, 1]. Returns 0.0 for an empty mask.
    """
    region_pixels = int(np.sum(mask))
    if region_pixels == 0:
        return 0.0
    intersection = int(np.sum(mask & coverage))
    return intersection / region_pixels


def _to_pil(image: np.ndarray):
    """Convert a (H, W, 3) uint8 numpy array to a PIL Image."""
    from PIL import Image
    return Image.fromarray(image.astype(np.uint8))


# SAM3's default 13-class vocabulary (matches baselines/sam3/config.yaml)
_DEFAULT_SAM3_QUERIES: List[str] = [
    "sidewalk",
    "crosswalk",
    "road",
    "dirt",
    "vegetation",
    "grass",
    "gravel",
    "puddle",
    "wet surface",
    "cracked pavement",
    "curb",
    "mud",
    "slope",
]
