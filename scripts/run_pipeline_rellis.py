"""
Run the environmental uncertainty pipeline on real RELLIS-3D images.

RELLIS-3D (Jiang et al., ICRA 2021, arXiv:2011.12954) is a real-world outdoor
dataset with 6,235 RGB frames and 20-class pixel-wise semantic annotations.
Its terrain classes overlap with our vocabulary (grass, mud, dirt, concrete, etc.)
making it a strong eval complement to RUGD.

Dataset structure expected at --rellis_dir:
  RELLIS-3D/
    RELLIS-3D_00/
      pylon_camera_node/
        frame000001-....jpg   (RGB images)
      pylon_camera_node_label_id/
        frame000001-....png   (semantic label maps, 8-bit class IDs)
    RELLIS-3D_01/ ... RELLIS-3D_04/

Download:
  1. Register at https://github.com/unmannedlab/RELLIS-3D (follow README)
  2. Download RELLIS-3D_00_img.zip + RELLIS-3D_00_label.zip from the
     release page or via the official Google Drive / OneDrive links.
  3. Extract to your --rellis_dir.
  Or use: ./scripts/download_datasets.sh rellis

Label class mapping (RELLIS-3D ID → our vocabulary):
  0  = void        → skip
  1  = dirt        → dirt (0.70)
  3  = grass       → grass (0.90)
  4  = tree        → vegetation (0.10)
  5  = pole        → unknown (0.00)
  6  = water       → water (0.05)
  7  = sky         → skip
  8  = vehicle     → unknown (0.00)
  9  = object      → unknown (0.00)
  10 = asphalt     → concrete (0.95)
  12 = building    → unknown (0.00)
  15 = log         → log (0.20)
  17 = bush        → vegetation (0.10)
  18 = concrete    → concrete (0.95)
  19 = barrier     → unknown (0.00)
  23 = puddle      → puddle (0.10)
  27 = mud         → mud (0.05)
  31 = rubble      → gravel (0.50)

Usage:
    python scripts/run_pipeline_rellis.py
    python scripts/run_pipeline_rellis.py --sequence 00 --n_images 20
    python scripts/run_pipeline_rellis.py --sequence 00 --verbosity terse --save_json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.traversability import TraversabilityMap, get_traversability, TRAVERSABILITY_SCORES
from system.env_uncertainty.user_profile import UserProfile


# ── RELLIS-3D label ID → our vocabulary ──────────────────────────────────────

_RELLIS_CLASS_MAP: Dict[int, str] = {
    # Raw label ID → our vocabulary (from RELLIS-3D GSCNN benchmark label_mapping)
    1:  "dirt",        # dirt/bare earth
    3:  "grass",       # grass
    4:  "vegetation",  # tree
    5:  "unknown",     # pole → not a terrain class
    6:  "water",       # water (deep)
    8:  "unknown",     # vehicle
    9:  "unknown",     # object (man-made)
    10: "concrete",    # asphalt
    12: "unknown",     # building
    15: "log",         # log
    17: "vegetation",  # bush
    18: "concrete",    # concrete
    19: "unknown",     # barrier
    23: "puddle",      # puddle
    27: "mud",         # mud
    29: "grass",       # also grass (alternate encoding)
    30: "grass",       # also grass
    31: "gravel",      # rubble
    32: "water",       # also water
    33: "unknown",     # fence — not a traversable terrain class
    34: "unknown",     # catch-all
}
# IDs not in map (void=0, sky=7) — not terrain
_SKIP_IDS = {0, 7}  # void and sky


# ── Label-based detector ──────────────────────────────────────────────────────

def _label_detect(image: np.ndarray, label_map: np.ndarray) -> DetectionResult:
    """
    Build a DetectionResult from a RELLIS-3D semantic label map.

    Uses ground-truth labels instead of the color detector, giving a clean
    upper-bound on what the system can achieve with perfect segmentation.

    Args:
        image:     H×W×3 RGB image (numpy).
        label_map: H×W uint8 semantic label IDs.

    Returns:
        DetectionResult with known_regions and unknown_regions.
    """
    h, w = image.shape[:2]
    total_pixels = h * w

    known_regions: List[RegionInfo] = []
    unknown_pixel_mask = np.ones((h, w), dtype=bool)

    for class_id, label in _RELLIS_CLASS_MAP.items():
        mask = (label_map == class_id)
        if not np.any(mask):
            continue
        unknown_pixel_mask &= ~mask
        pf = float(np.sum(mask)) / total_pixels
        known_regions.append(RegionInfo(
            label=label,
            mask=mask,
            confidence=0.90,
            pixel_fraction=pf,
            source="rellis_gt",
            traversability=get_traversability(label),
        ))

    # Remove skip IDs from unknown
    for skip_id in _SKIP_IDS:
        unknown_pixel_mask &= ~(label_map == skip_id)

    unknown_regions: List[RegionInfo] = []
    if np.any(unknown_pixel_mask):
        pf = float(np.sum(unknown_pixel_mask)) / total_pixels
        unknown_regions.append(RegionInfo(
            label="unknown",
            mask=unknown_pixel_mask,
            confidence=0.0,
            pixel_fraction=pf,
            source="rellis_gt",
            traversability=0.0,
        ))

    tmap = TraversabilityMap.create(h, w)
    for r in known_regions:
        tmap = tmap.update_region(r.mask, r.label)

    covered_mask = np.zeros((h, w), dtype=bool)
    for r in known_regions:
        covered_mask |= r.mask
    sam3_coverage = float(np.sum(covered_mask)) / total_pixels
    unknown_coverage = float(np.sum(unknown_pixel_mask)) / total_pixels

    return DetectionResult(
        known_regions=known_regions,
        unknown_regions=unknown_regions,
        image_shape=(h, w),
        sam3_coverage=sam3_coverage,
        unknown_coverage=unknown_coverage,
        has_unknown=len(unknown_regions) > 0,
        traversability_map=tmap,
    )


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _make_label_runner(config_path: str):
    """Build a runner using ground-truth label-based detector."""
    from system.env_uncertainty.runner import EnvironmentalUncertaintyRunner

    class _LabelDetector:
        def __init__(self):
            self._label_map: Optional[np.ndarray] = None

        def set_label(self, label_map: np.ndarray):
            self._label_map = label_map

        def detect(self, image: np.ndarray) -> DetectionResult:
            assert self._label_map is not None, "call set_label() before detect()"
            return _label_detect(image, self._label_map)

    detector = _LabelDetector()
    runner = EnvironmentalUncertaintyRunner(
        config_path=config_path,
        detector=detector,
    )
    return runner, detector


def _load_image(path: Path) -> Optional[np.ndarray]:
    try:
        import cv2
        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def _load_label(path: Path) -> Optional[np.ndarray]:
    try:
        import cv2
        lbl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return lbl
    except Exception:
        return None


def _find_label_path(image_path: Path, seq_dir: Path) -> Optional[Path]:
    """Find the corresponding label PNG for a RELLIS-3D RGB JPEG."""
    stem = image_path.stem  # e.g. frame000001-1581624831_228
    label_dir = seq_dir / "pylon_camera_node_label_id"
    label_path = label_dir / f"{stem}.png"
    return label_path if label_path.exists() else None


def _simulate_gt_feedback(runner, label_map: np.ndarray) -> List[str]:
    """
    Simulate user feedback from RELLIS-3D ground-truth labels.

    After each frame, we know exactly what terrain was present (GT labels).
    This function uses that ground truth to update PersistentTerrainKnowledge
    as if a human had confirmed each terrain class's traversability.

    In a real deployment, feedback would come from user text responses.
    Here, GT labels let us evaluate whether the cross-frame knowledge system
    converges correctly across sequential frames.

    Returns list of (label, is_traversable) strings for display.
    """
    updates: List[str] = []
    for class_id, label in _RELLIS_CLASS_MAP.items():
        if label in ("unknown",):
            continue
        mask = (label_map == class_id)
        if not np.any(mask):
            continue
        default_trav = TRAVERSABILITY_SCORES.get(label, 0.5)
        is_traversable = default_trav >= 0.50
        runner.terrain_knowledge.update_from_feedback(
            label=label,
            is_traversable=is_traversable,
            confidence=0.85,
        )
        updates.append(f"{label}={'safe' if is_traversable else 'unsafe'}")
    return updates


def run_on_rellis(
    rellis_dir: str,
    sequence: str = "00",
    n_images: int = 10,
    verbosity: str = "standard",
    save_json: Optional[str] = None,
    config_path: Optional[str] = None,
    goal_row_fraction: float = 0.20,
    use_color_detector: bool = False,
    simulate_gt_feedback: bool = True,
) -> List[dict]:
    """
    Run the uncertainty pipeline on RELLIS-3D images.

    Args:
        rellis_dir:         Path to the RELLIS-3D root (containing RELLIS-3D_00/, ...).
        sequence:           Sequence number as string e.g. "00" (default).
        n_images:           Max images to process.
        verbosity:          Question verbosity: terse / standard / verbose.
        save_json:          If set, write results to this JSON path.
        config_path:        Path to system/env_uncertainty/config.yaml.
        goal_row_fraction:  Goal pixel row as fraction of image height (default 0.20).
        use_color_detector: If True, fall back to HSV color detector (ignores labels).
                            Default False: use ground-truth RELLIS-3D labels.

    Returns:
        List of per-image result dicts.
    """
    rellis_path = Path(rellis_dir)
    seq_dir = rellis_path / f"RELLIS-3D_{sequence}"
    img_dir = seq_dir / "pylon_camera_node"

    if not img_dir.exists():
        print(f"ERROR: image directory not found: {img_dir}")
        print("  Make sure RELLIS-3D is downloaded and extracted correctly.")
        sys.exit(1)

    images = sorted(img_dir.glob("*.jpg"))[:n_images]
    if not images:
        # Try PNG fallback
        images = sorted(img_dir.glob("*.png"))[:n_images]
    if not images:
        print(f"ERROR: no JPEG or PNG images found in {img_dir}")
        sys.exit(1)

    if config_path is None:
        config_path = str(
            Path(__file__).parent.parent / "system" / "env_uncertainty" / "config.yaml"
        )

    profile = UserProfile(
        user_id="rellis_eval",
        verbosity=verbosity,
        expertise="intermediate",
        preferred_format="question",
        name=f"RELLIS-3D eval ({verbosity})",
    )

    if use_color_detector:
        from scripts.run_pipeline_rugd import _color_detect, _make_mock_runner
        runner = _make_mock_runner(config_path, verbosity)
        gt_detector = None
    else:
        runner, gt_detector = _make_label_runner(config_path)

    results = []
    n_proceed = n_ask = n_stop = 0
    detector_mode = "color_detector" if use_color_detector else "rellis_gt_labels"
    use_cross_frame = simulate_gt_feedback and not use_color_detector
    print(f"\nRunning pipeline on {len(images)} images from RELLIS-3D seq {sequence} "
          f"(verbosity={verbosity}, detector={detector_mode}, "
          f"cross_frame_knowledge={'on' if use_cross_frame else 'off'})\n{'='*65}")
    # Track which terrain labels the cross-frame knowledge has learned so far
    last_lbl: Optional[np.ndarray] = None

    for img_path in images:
        img = _load_image(img_path)
        if img is None:
            print(f"  SKIP {img_path.name}: could not load")
            continue

        # Load ground-truth label map (if using GT detector)
        lbl: Optional[np.ndarray] = None
        if not use_color_detector and gt_detector is not None:
            lbl_path = _find_label_path(img_path, seq_dir)
            if lbl_path is None:
                print(f"  SKIP {img_path.name}: no label file found")
                continue
            lbl = _load_label(lbl_path)
            if lbl is None:
                print(f"  SKIP {img_path.name}: label load failed")
                continue
            gt_detector.set_label(lbl)

        h, w = img.shape[:2]
        goal_pixel = (int(h * goal_row_fraction), w // 2)

        runner.reset_frame_state()
        t0 = time.perf_counter()
        decision = runner.run_scene(
            img,
            scene_id=img_path.stem,
            goal_pixel=goal_pixel,
            user_profile=profile,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        action = decision.robot_action
        if action == "PROCEED":  n_proceed += 1
        elif action == "ASK":    n_ask += 1
        else:                    n_stop += 1

        # Simulate GT-based user feedback → update cross-frame terrain knowledge.
        # After each frame, we know from GT labels exactly what terrain was there
        # and whether it is traversable.  This propagates to the next frame's GP
        # seed via runner._seed_gp_from_detection → terrain_knowledge.adjusted_traversability().
        gt_feedback_labels: List[str] = []
        if use_cross_frame and lbl is not None:
            gt_feedback_labels = _simulate_gt_feedback(runner, lbl)

        q_text = f'\n    Q: "{decision.question}"' if decision.question else ""
        known_labels = runner.terrain_knowledge.n_labels_known
        print(
            f"[{img_path.name[:35]:35s}]  {action:7s}  "
            f"unknown={decision.unknown_coverage:.2f}  "
            f"known={decision.n_known_regions}  "
            f"({elapsed_ms:.0f} ms)  "
            f"[cross-frame: {known_labels} labels]{q_text}"
        )

        results.append({
            "image": img_path.name,
            "sequence": sequence,
            "robot_action": action,
            "unknown_coverage": round(decision.unknown_coverage, 4),
            "sam3_coverage": round(decision.sam3_coverage, 4),
            "n_known_regions": decision.n_known_regions,
            "n_unknown_regions": decision.n_unknown_regions,
            "question": decision.question,
            "goal_pixel": list(goal_pixel),
            "elapsed_ms": round(elapsed_ms, 1),
            "cross_frame_labels_known": runner.terrain_knowledge.n_labels_known,
        })

    total = len(results)
    print(f"\n{'='*65}")
    print(f"Summary: {total} images | PROCEED={n_proceed} ASK={n_ask} STOP={n_stop}")
    if total:
        print(f"Help rate: {(n_ask + n_stop) / total:.1%}")
    print(f"Verbosity: {verbosity} | Detector: {detector_mode}")

    # Print cross-frame knowledge accumulated across all frames
    if use_cross_frame and runner.terrain_knowledge.n_labels_known > 0:
        print(f"\n{runner.terrain_knowledge.summary()}")

    if save_json and results:
        out = {
            "dataset": "RELLIS-3D",
            "sequence": sequence,
            "n_images": total,
            "verbosity": verbosity,
            "detector": detector_mode,
            "cross_frame_knowledge": use_cross_frame,
            "n_labels_learned": runner.terrain_knowledge.n_labels_known,
            "n_proceed": n_proceed,
            "n_ask": n_ask,
            "n_stop": n_stop,
            "help_rate": round((n_ask + n_stop) / total, 4) if total else 0.0,
            "per_image": results,
        }
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved → {save_json}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run uncertainty pipeline on RELLIS-3D images")
    p.add_argument(
        "--rellis_dir",
        default="/Users/julih/Documents/datasets/RELLIS-3D",
        help="Path to RELLIS-3D root (containing RELLIS-3D_00/ etc.)",
    )
    p.add_argument("--sequence", default="00", help="Sequence number string (default: 00)")
    p.add_argument("--n_images", type=int, default=10, help="Max images (default: 10)")
    p.add_argument(
        "--verbosity", choices=["terse", "standard", "verbose"],
        default="standard", help="Question verbosity",
    )
    p.add_argument("--save_json", default=None, help="Save results to JSON file")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument(
        "--color_detector", action="store_true", default=False,
        help="Use HSV color detector instead of GT labels (for comparison)",
    )
    p.add_argument(
        "--no_cross_frame", action="store_true", default=False,
        help="Disable GT-based cross-frame terrain knowledge simulation (default: on)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_on_rellis(
        rellis_dir=args.rellis_dir,
        sequence=args.sequence,
        n_images=args.n_images,
        verbosity=args.verbosity,
        save_json=args.save_json,
        config_path=args.config,
        use_color_detector=args.color_detector,
        simulate_gt_feedback=not args.no_cross_frame,
    )
