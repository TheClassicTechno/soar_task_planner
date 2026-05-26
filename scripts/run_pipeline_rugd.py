"""
Run the environmental uncertainty pipeline on real RUGD images.

Uses a fast color-based terrain detector (no GPU required) instead of SAM3/SAM2,
so the full Steps 3-10 pipeline can be evaluated on actual RUGD frames.

The detector approximates SAM3 using HSV color analysis:
  - Green hues     → grass (traversability 0.90)
  - Brown/tan hues → dirt or gravel (traversability 0.70-0.80)
  - Dark/murky     → mud or unknown (traversability 0.0-0.10)
  - Light grey     → path/concrete (traversability 0.95)
  - Remaining      → unknown (triggers ASK)

Pipeline output per image:
  - robot_action:  PROCEED / ASK / STOP
  - question:      text if ASK/STOP, None if PROCEED
  - unknown_coverage, best_trajectory, n_known_regions

Usage:
    python scripts/run_pipeline_rugd.py
    python scripts/run_pipeline_rugd.py --sequence trail-5 --n_images 20
    python scripts/run_pipeline_rugd.py --sequence trail-5 --verbosity terse
    python scripts/run_pipeline_rugd.py --sequence trail-5 --verbosity verbose --save_json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── Add project root to path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.traversability import TraversabilityMap, get_traversability
from system.env_uncertainty.user_profile import UserProfile
from system.env_uncertainty.scenarios import SCENARIOS


# ── Color-based terrain detector ─────────────────────────────────────────────

def _color_detect(image: np.ndarray) -> DetectionResult:
    """
    Fast deterministic terrain detector using HSV color analysis.

    Segments the image into terrain regions by color, mimicking SAM3 output.
    No GPU or model weights needed — runs in milliseconds per image.

    Color rules (HSV):
      Hue 35-85, Saturation > 40   → grass
      Hue 10-35, Saturation > 25   → dirt
      Hue 0-180, Saturation < 25, Value > 80  → concrete / path (grey)
      Hue 0-180, Saturation < 25, Value < 80  → mud / unknown (dark grey)
      Hue 85-130, Saturation > 40  → water / wet surface
      Remaining pixels             → unknown
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required: pip install opencv-python")

    h_img, w_img = image.shape[:2]
    total_pixels = h_img * w_img

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0].astype(float)   # 0-179 in OpenCV
    S = hsv[:, :, 1].astype(float)   # 0-255
    V = hsv[:, :, 2].astype(float)   # 0-255

    # Binary masks for each terrain class
    grass_mask    = (H >= 35) & (H <= 85)  & (S > 40)
    dirt_mask     = (H >= 10) & (H <= 35)  & (S > 25) & ~grass_mask
    path_mask     = (S < 25)               & (V > 100)
    dark_mask     = (S < 25)               & (V <= 100) & ~path_mask
    water_mask    = (H >= 85) & (H <= 130) & (S > 60)

    # Unknown = everything not matched by a known class
    covered = grass_mask | dirt_mask | path_mask | water_mask
    unknown_mask = ~covered & ~dark_mask

    # Build RegionInfo for each known class (only if region is non-empty)
    known_regions: List[RegionInfo] = []
    for label, mask in [
        ("grass",       grass_mask),
        ("dirt",        dirt_mask),
        ("concrete",    path_mask),
        ("water",       water_mask),
    ]:
        if not np.any(mask):
            continue
        pf = float(np.sum(mask)) / total_pixels
        known_regions.append(RegionInfo(
            label=label,
            mask=mask,
            confidence=0.75,        # fixed confidence for color detector
            pixel_fraction=pf,
            source="color_detector",
            traversability=get_traversability(label),
        ))

    # Dark pixels → mud/unknown (low traversability)
    if np.any(dark_mask):
        pf = float(np.sum(dark_mask)) / total_pixels
        known_regions.append(RegionInfo(
            label="mud",
            mask=dark_mask,
            confidence=0.50,
            pixel_fraction=pf,
            source="color_detector",
            traversability=get_traversability("mud"),
        ))

    # Unknown regions — trigger ASK if on path
    unknown_regions: List[RegionInfo] = []
    if np.any(unknown_mask):
        pf = float(np.sum(unknown_mask)) / total_pixels
        unknown_regions.append(RegionInfo(
            label="unknown",
            mask=unknown_mask,
            confidence=0.0,
            pixel_fraction=pf,
            source="color_detector",
            traversability=0.0,
        ))

    # Build traversability map
    tmap = TraversabilityMap.create(h_img, w_img)
    for r in known_regions:
        tmap = tmap.update_region(r.mask, r.label)

    sam3_coverage = float(np.sum(covered | dark_mask)) / total_pixels
    unknown_coverage = float(np.sum(unknown_mask)) / total_pixels

    return DetectionResult(
        known_regions=known_regions,
        unknown_regions=unknown_regions,
        image_shape=(h_img, w_img),
        sam3_coverage=sam3_coverage,
        unknown_coverage=unknown_coverage,
        has_unknown=len(unknown_regions) > 0,
        traversability_map=tmap,
    )


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _make_mock_runner(config_path: str, verbosity: str):
    """Build a runner using the color detector (no SAM models needed)."""
    from system.env_uncertainty.runner import EnvironmentalUncertaintyRunner

    class _ColorDetector:
        def detect(self, image):
            return _color_detect(image)

    runner = EnvironmentalUncertaintyRunner(
        config_path=config_path,
        detector=_ColorDetector(),
    )
    return runner


def _load_image(path: Path) -> Optional[np.ndarray]:
    try:
        import cv2
        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def run_on_rugd(
    rugd_dir: str,
    sequence: str = "trail-5",
    n_images: int = 10,
    verbosity: str = "standard",
    save_json: Optional[str] = None,
    config_path: Optional[str] = None,
    goal_row_fraction: float = 0.20,  # goal is at top 20% of the image
    persistent_gp: bool = False,       # if True, GP accumulates across frames
) -> List[dict]:
    """
    Run the uncertainty pipeline on RUGD images and print decisions.

    Args:
        rugd_dir:          Path to RUGD_frames-with-annotations/ directory.
        sequence:          Sequence name (e.g. "trail-5").
        n_images:          Maximum number of images to process.
        verbosity:         Question verbosity: "terse" | "standard" | "verbose".
        save_json:         If set, write results to this JSON path.
        config_path:       Path to system/env_uncertainty/config.yaml.
        goal_row_fraction: Fraction of image height where goal pixel sits (default top 20%).
        persistent_gp:     If True, keep the GP map across frames (accumulates observations).
                           Default False: reset GP between frames so each frame is independent.
                           The persistent mode is unreliable for a moving robot because image-
                           relative pixel coordinates shift between frames — the same pixel (row,
                           col) maps to different real-world terrain as the robot moves forward.
                           Use persistent=True only when frames are nearly co-located.

    Returns:
        List of result dicts (one per image).
    """
    rugd_path = Path(rugd_dir)
    seq_path = rugd_path / sequence
    if not seq_path.exists():
        print(f"ERROR: sequence directory not found: {seq_path}")
        sys.exit(1)

    images = sorted(seq_path.glob("*.png"))[:n_images]
    if not images:
        print(f"ERROR: no PNG images found in {seq_path}")
        sys.exit(1)

    if config_path is None:
        config_path = str(
            Path(__file__).parent.parent / "system" / "env_uncertainty" / "config.yaml"
        )

    profile = UserProfile(
        user_id="rugd_eval",
        verbosity=verbosity,
        expertise="intermediate",
        preferred_format="question",
        name=f"RUGD eval ({verbosity})",
    )

    runner = _make_mock_runner(config_path, verbosity)

    results = []
    n_proceed = n_ask = n_stop = 0
    gp_mode = "persistent (accumulates across frames)" if persistent_gp else "per-frame reset"
    print(f"\nRunning pipeline on {len(images)} images from '{sequence}' "
          f"(verbosity={verbosity}, gp={gp_mode})\n{'='*60}")

    for img_path in images:
        img = _load_image(img_path)
        if img is None:
            print(f"  SKIP {img_path.name}: could not load")
            continue

        h, w = img.shape[:2]
        # Goal is at the top-centre of the image (robot navigates forward)
        goal_pixel = (int(h * goal_row_fraction), w // 2)

        # Reset GP between frames unless caller explicitly wants cross-frame accumulation.
        # Without reset, image-relative pixel coords drift as the robot moves, so old
        # GP observations from prior frames incorrectly influence the current decision.
        if not persistent_gp:
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

        # Print per-image result
        q_text = f'\n    Q: "{decision.question}"' if decision.question else ""
        print(
            f"[{img_path.name}]  {action:7s}  "
            f"unknown={decision.unknown_coverage:.2f}  "
            f"known_regions={decision.n_known_regions}  "
            f"({elapsed_ms:.0f} ms){q_text}"
        )

        results.append({
            "image": img_path.name,
            "robot_action": action,
            "unknown_coverage": round(decision.unknown_coverage, 4),
            "sam3_coverage": round(decision.sam3_coverage, 4),
            "n_known_regions": decision.n_known_regions,
            "n_unknown_regions": decision.n_unknown_regions,
            "question": decision.question,
            "goal_pixel": list(goal_pixel),
            "elapsed_ms": round(elapsed_ms, 1),
        })

    # Summary
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Summary: {total} images | PROCEED={n_proceed} ASK={n_ask} STOP={n_stop}")
    print(f"Help rate: {(n_ask + n_stop) / total:.1%}")
    print(f"Verbosity: {verbosity}")

    if save_json:
        out = {
            "sequence": sequence,
            "n_images": total,
            "verbosity": verbosity,
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
    p = argparse.ArgumentParser(description="Run uncertainty pipeline on RUGD images")
    p.add_argument(
        "--rugd_dir",
        default="/Users/julih/Documents/datasets/rugd/RUGD_frames-with-annotations",
        help="Path to RUGD_frames-with-annotations/",
    )
    p.add_argument("--sequence", default="trail-5", help="Sequence name (default: trail-5)")
    p.add_argument("--n_images", type=int, default=10, help="Max images to run (default: 10)")
    p.add_argument(
        "--verbosity",
        choices=["terse", "standard", "verbose"],
        default="standard",
        help="Question verbosity level (default: standard)",
    )
    p.add_argument("--save_json", default=None, help="Save results to this JSON file")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument(
        "--persistent_gp",
        action="store_true",
        default=False,
        help="Keep GP map across frames (default: reset per frame for correctness)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_on_rugd(
        rugd_dir=args.rugd_dir,
        sequence=args.sequence,
        n_images=args.n_images,
        verbosity=args.verbosity,
        save_json=args.save_json,
        config_path=args.config,
        persistent_gp=args.persistent_gp,
    )
