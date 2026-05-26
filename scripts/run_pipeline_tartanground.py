"""
Run the environmental uncertainty pipeline on TartanGround images.

TartanGround (Patel et al., arXiv:2505.10696, CMU AirLab) is a large-scale
synthetic dataset with 63 environments, 878 trajectories, and 17.3M images
across 3 robot platforms. Segmentation masks use per-environment label maps
(seg_label_map.json) with unique integer class IDs per semantic class.

Dataset structure expected at --tartanground_dir:
  TartanGround/
    OldTownSummer/               (or ForestEnv, Gascola, etc.)
      seg_label_map.json         (name → class_id)
      Data_omni/
        P0000/
          image_lcam_front/      (RGB .png files)
          seg_lcam_front/        (seg ID .png files, uint8)
          metadata/              (poses, robot height)

Download minimal subset (~362 MB):
    python scripts/download_tartanground.py

Usage:
    python scripts/run_pipeline_tartanground.py
    python scripts/run_pipeline_tartanground.py --env OldTownSummer --traj P0000 --n_images 20
    python scripts/run_pipeline_tartanground.py --env ForestEnv --save_json results_tg.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.traversability import (
    TraversabilityMap,
    get_traversability,
    TRAVERSABILITY_SCORES,
)
from system.env_uncertainty.user_profile import UserProfile


# ── TartanGround class name → our traversability vocabulary ──────────────────

# Maps TartanGround semantic class names → our canonical terrain label.
# Classes not listed here are treated as "unknown" (traversability 0.0).
_TARTANGROUND_CLASS_MAP: Dict[str, str] = {
    # Traversable ground surfaces
    "ground":               "dirt",         # bare ground / dirt (0.70)
    "sidewalk":             "concrete",     # paved path (0.95)
    "floor":                "concrete",     # indoor/paved floor (0.95)
    "asphalt":              "concrete",     # asphalt road (0.95)
    "road":                 "concrete",     # road surface (0.95)
    "dirt":                 "dirt",         # dirt trail (0.70)
    "grass":                "grass",        # grass (0.90)
    "sand":                 "dirt",         # sand ~= dirt (0.70)
    "gravel":               "gravel",       # gravel (0.50)
    "mud":                  "mud",          # mud (0.10)
    "water":                "water",        # water (0.05)
    "puddle":               "puddle",       # puddle (0.10)
    "groundembankment":     "dirt",         # sloped dirt embankment (0.70)
    # Vegetation on/near ground (low traversability — dense obstacles for wheeled robot)
    "bush":                 "vegetation",   # bush (0.10)
    "tree":                 "vegetation",   # tree canopy / full tree (0.10)
    "leaves":               "vegetation",   # fallen leaves on path (0.10)
    "foliage":              "vegetation",   # generic foliage / canopy (0.10)
    "instancedfoliageactor":"vegetation",   # UE4 instanced foliage actor (0.10)
    "groundplant":          "vegetation",   # low ground plants (0.10)
    "rock":                 "gravel",       # rocks treated as rough gravel (0.50)
    # Impassable natural obstacles
    "treetrunk":            "unknown",      # tree trunk — solid obstacle
    "root":                 "unknown",      # exposed tree root — trip hazard
    "stump":                "unknown",      # tree stump — solid obstacle
    # Sky / non-terrain (not labeled as unknown — just ignored)
    "sky":                  "unknown",      # sky pixels — never on ground
    # Non-traversable structures
    "building":             "unknown",
    "wall":                 "unknown",
    "roof":                 "unknown",
    "railing":              "unknown",
    "fence":                "unknown",
    "steps":                "unknown",      # stairs — not for wheeled robots
    "planter":              "unknown",
    "post":                 "unknown",
    "sign":                 "unknown",
    "storesign":            "unknown",
    "storesignsground":     "unknown",
    "bench":                "unknown",
    "chair":                "unknown",
    "table":                "unknown",
    "ceiling":              "unknown",
    "garbage":              "unknown",
    "bikerack":             "unknown",
    "light":                "unknown",
    "banner":               "unknown",
    "pipe":                 "unknown",
    "metalbars":            "unknown",
    "powerline":            "unknown",
    "awning":               "unknown",
    "tramlines":            "unknown",
    "z":                    "unknown",      # catch-all / void class
}


def _load_label_map(env_dir: Path) -> Dict[int, str]:
    """
    Load seg_label_map.json and build ID → our vocabulary dict.

    TartanGround stores the label map as {name: class_id}.
    We invert it to {class_id: our_label} for fast lookup.
    """
    label_file = env_dir / "seg_label_map.json"
    if not label_file.exists():
        return {}
    with open(label_file) as f:
        data = json.load(f)
    name_map: Dict[str, int] = data.get("name_map", {})
    id_to_label: Dict[int, str] = {}
    for name, class_id in name_map.items():
        our_label = _TARTANGROUND_CLASS_MAP.get(name, "unknown")
        id_to_label[int(class_id)] = our_label
    return id_to_label


def _label_detect(
    image: np.ndarray, seg_map: np.ndarray, id_to_label: Dict[int, str]
) -> DetectionResult:
    """
    Build DetectionResult from a TartanGround semantic segmentation map.

    Args:
        image:       H×W×3 RGB image (numpy).
        seg_map:     H×W uint8 semantic class ID map.
        id_to_label: {class_id → our vocabulary label} from _load_label_map().

    Returns:
        DetectionResult with known_regions and unknown_regions.
    """
    h, w = image.shape[:2]
    total_pixels = h * w

    known_regions: List[RegionInfo] = []
    unknown_pixel_mask = np.ones((h, w), dtype=bool)

    # Group by our canonical label (multiple class IDs → same label)
    from collections import defaultdict
    label_masks: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros((h, w), dtype=bool))

    for class_id, label in id_to_label.items():
        mask = (seg_map == class_id)
        if not np.any(mask):
            continue
        label_masks[label] |= mask
        if label != "unknown":
            unknown_pixel_mask &= ~mask

    for label, mask in label_masks.items():
        if label == "unknown" or not np.any(mask):
            continue
        pf = float(np.sum(mask)) / total_pixels
        known_regions.append(RegionInfo(
            label=label,
            mask=mask,
            confidence=0.90,
            pixel_fraction=pf,
            source="tartanground_gt",
            traversability=get_traversability(label),
        ))

    # Unknown mask: pixels not covered by any traversable class
    unknown_regions: List[RegionInfo] = []
    if np.any(unknown_pixel_mask):
        pf = float(np.sum(unknown_pixel_mask)) / total_pixels
        unknown_regions.append(RegionInfo(
            label="unknown",
            mask=unknown_pixel_mask,
            confidence=0.0,
            pixel_fraction=pf,
            source="tartanground_gt",
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


def _simulate_gt_feedback(runner, seg_map: np.ndarray, id_to_label: Dict[int, str]) -> None:
    """Simulate user feedback from GT labels to update cross-frame terrain knowledge."""
    seen_labels = set()
    for class_id, label in id_to_label.items():
        if label in ("unknown",) or label in seen_labels:
            continue
        if not np.any(seg_map == class_id):
            continue
        seen_labels.add(label)
        default_trav = TRAVERSABILITY_SCORES.get(label, 0.5)
        runner.terrain_knowledge.update_from_feedback(
            label=label,
            is_traversable=default_trav >= 0.50,
            confidence=0.85,
        )


def _make_gt_runner(config_path: str, id_to_label: Dict[int, str]):
    """Build a runner using ground-truth TartanGround segmentation."""
    from system.env_uncertainty.runner import EnvironmentalUncertaintyRunner

    class _TGDetector:
        def __init__(self):
            self._seg_map: Optional[np.ndarray] = None
            self._id_to_label = id_to_label

        def set_seg(self, seg_map: np.ndarray):
            self._seg_map = seg_map

        def detect(self, image: np.ndarray) -> DetectionResult:
            assert self._seg_map is not None, "call set_seg() before detect()"
            return _label_detect(image, self._seg_map, self._id_to_label)

    detector = _TGDetector()
    runner = EnvironmentalUncertaintyRunner(config_path=config_path, detector=detector)
    return runner, detector


def _load_image(path: Path) -> Optional[np.ndarray]:
    try:
        import cv2
        bgr = cv2.imread(str(path))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
    except Exception:
        return None


def _load_seg(path: Path) -> Optional[np.ndarray]:
    try:
        import cv2
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def run_on_tartanground(
    tartanground_dir: str,
    env: str = "OldTownSummer",
    traj: str = "P0000",
    robot: str = "omni",
    n_images: int = 10,
    verbosity: str = "standard",
    save_json: Optional[str] = None,
    config_path: Optional[str] = None,
    goal_row_fraction: float = 0.20,
    simulate_gt_feedback: bool = True,
) -> List[dict]:
    """
    Run the uncertainty pipeline on TartanGround images.

    Args:
        tartanground_dir:     Path to TartanGround root.
        env:                  Environment name (e.g. "OldTownSummer").
        traj:                 Trajectory ID (e.g. "P0000").
        robot:                Robot type: "omni", "diff", "anymal".
        n_images:             Max frames to process.
        verbosity:            Question verbosity: terse / standard / verbose.
        save_json:            If set, write results JSON to this path.
        config_path:          Path to system/env_uncertainty/config.yaml.
        goal_row_fraction:    Goal pixel row as fraction of image height.
        simulate_gt_feedback: Update cross-frame knowledge from GT labels.

    Returns:
        List of per-frame result dicts.
    """
    root = Path(os.path.expanduser(tartanground_dir))
    env_dir = root / env
    traj_dir = env_dir / f"Data_{robot}" / traj
    img_dir = traj_dir / "image_lcam_front"
    seg_dir = traj_dir / "seg_lcam_front"

    if not img_dir.exists():
        print(f"ERROR: image dir not found: {img_dir}")
        print("  Run: python scripts/download_tartanground.py")
        sys.exit(1)

    id_to_label = _load_label_map(env_dir)
    if not id_to_label:
        print(f"WARNING: no seg_label_map.json found in {env_dir}; all regions will be unknown")

    images = sorted(img_dir.glob("*.png"))[:n_images]
    if not images:
        print(f"ERROR: no PNG images in {img_dir}")
        sys.exit(1)

    if config_path is None:
        config_path = str(PROJECT_ROOT / "system" / "env_uncertainty" / "config.yaml")

    profile = UserProfile(
        user_id="tartanground_eval",
        verbosity=verbosity,
        expertise="intermediate",
        preferred_format="question",
        name=f"TartanGround eval ({env})",
    )

    runner, gt_detector = _make_gt_runner(config_path, id_to_label)

    results = []
    n_proceed = n_ask = n_stop = 0
    print(
        f"\nRunning pipeline on {len(images)} frames from TartanGround "
        f"{env}/{traj} ({robot}, verbosity={verbosity}, "
        f"cross_frame={'on' if simulate_gt_feedback else 'off'})\n{'='*70}"
    )

    for img_path in images:
        img = _load_image(img_path)
        if img is None:
            print(f"  SKIP {img_path.name}: image load failed")
            continue

        stem = img_path.stem  # e.g. "000000_lcam_front"
        seg_path = seg_dir / f"{stem}_seg.png"
        if not seg_path.exists():
            print(f"  SKIP {img_path.name}: no seg file {seg_path.name}")
            continue
        seg = _load_seg(seg_path)
        if seg is None:
            print(f"  SKIP {img_path.name}: seg load failed")
            continue

        gt_detector.set_seg(seg)
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

        if simulate_gt_feedback:
            _simulate_gt_feedback(runner, seg, id_to_label)

        q_text = f'\n    Q: "{decision.question}"' if decision.question else ""
        print(
            f"[{img_path.name[:40]:40s}]  {action:7s}  "
            f"unknown={decision.unknown_coverage:.2f}  "
            f"known={decision.n_known_regions}  "
            f"({elapsed_ms:.0f} ms)  "
            f"[xframe: {runner.terrain_knowledge.n_labels_known}]{q_text}"
        )

        results.append({
            "image": img_path.name,
            "env": env,
            "traj": traj,
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
    print(f"\n{'='*70}")
    print(f"Summary: {total} frames | PROCEED={n_proceed} ASK={n_ask} STOP={n_stop}")
    if total:
        print(f"Help rate: {(n_ask + n_stop) / total:.1%}")
    print(f"Environment: {env} | Trajectory: {traj} | Robot: {robot}")

    if simulate_gt_feedback and runner.terrain_knowledge.n_labels_known > 0:
        print(f"\n{runner.terrain_knowledge.summary()}")

    if save_json and results:
        out = {
            "dataset": "TartanGround",
            "env": env,
            "traj": traj,
            "robot": robot,
            "n_frames": total,
            "verbosity": verbosity,
            "cross_frame_knowledge": simulate_gt_feedback,
            "n_labels_learned": runner.terrain_knowledge.n_labels_known,
            "n_proceed": n_proceed,
            "n_ask": n_ask,
            "n_stop": n_stop,
            "help_rate": round((n_ask + n_stop) / total, 4) if total else 0.0,
            "per_frame": results,
        }
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved → {save_json}")

    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run uncertainty pipeline on TartanGround")
    p.add_argument(
        "--tartanground_dir",
        default="~/Documents/datasets/TartanGround",
        help="Path to TartanGround root",
    )
    p.add_argument("--env", default="OldTownSummer", help="Environment name")
    p.add_argument("--traj", default="P0000", help="Trajectory ID (default: P0000)")
    p.add_argument(
        "--robot", default="omni", choices=["omni", "diff", "anymal"],
        help="Robot type (default: omni)",
    )
    p.add_argument("--n_images", type=int, default=10, help="Max frames (default: 10)")
    p.add_argument(
        "--verbosity", choices=["terse", "standard", "verbose"],
        default="standard", help="Question verbosity",
    )
    p.add_argument("--save_json", default=None, help="Save results to JSON file")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument(
        "--no_cross_frame", action="store_true", default=False,
        help="Disable GT-feedback cross-frame knowledge simulation",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_on_tartanground(
        tartanground_dir=args.tartanground_dir,
        env=args.env,
        traj=args.traj,
        robot=args.robot,
        n_images=args.n_images,
        verbosity=args.verbosity,
        save_json=args.save_json,
        config_path=args.config,
        simulate_gt_feedback=not args.no_cross_frame,
    )
