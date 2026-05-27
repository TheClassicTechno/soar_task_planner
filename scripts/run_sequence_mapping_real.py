#!/usr/bin/env python3
"""
Sequential, pose-aware multi-frame mapping demo on real RUGD images.

Illustrates:
  1. Simulated robot movement using MockForwardOdometry.
  2. Frame-by-frame perception processing.
  3. Spatial projection of terrain observations to metric world coordinates.
  4. Accumulation of terrain memory inside WorldGPTraversabilityMap and WorldSceneGraph.

By default, this script runs in a fast color-based mock mode so it can be verified
on any CPU-only or memory-constrained device. Run with --use_real_models to enable
the actual SAM2 and SAM3 neural networks.

Usage:
    python scripts/run_sequence_mapping_real.py --n_images 5
    python scripts/run_sequence_mapping_real.py --n_images 5 --use_real_models
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ── Add project root to path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.env_uncertainty.runner import EnvironmentalUncertaintyRunner
from system.env_uncertainty.world_coords import MockForwardOdometry, CameraMount
from system.env_uncertainty.user_profile import UserProfile


# ── Color-based terrain detector (for fast mock mode) ──────────────────────────

def _color_detect(image: np.ndarray):
    """HSV color analysis terrain detector, mimicking SAM3/SAM2 outputs."""
    import cv2
    from system.env_uncertainty.detector import DetectionResult, RegionInfo
    from system.env_uncertainty.traversability import TraversabilityMap, get_traversability

    h_img, w_img = image.shape[:2]
    total_pixels = h_img * w_img

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0].astype(float)
    S = hsv[:, :, 1].astype(float)
    V = hsv[:, :, 2].astype(float)

    grass_mask = (H >= 35) & (H <= 85) & (S > 40)
    dirt_mask = (H >= 10) & (H <= 35) & (S > 25) & ~grass_mask
    path_mask = (S < 25) & (V > 100)
    dark_mask = (S < 25) & (V <= 100) & ~path_mask
    water_mask = (H >= 85) & (H <= 130) & (S > 60)

    covered = grass_mask | dirt_mask | path_mask | water_mask
    unknown_mask = ~covered & ~dark_mask

    known_regions = []
    for label, mask in [
        ("grass", grass_mask),
        ("dirt", dirt_mask),
        ("concrete", path_mask),
        ("water", water_mask),
    ]:
        if not np.any(mask):
            continue
        pf = float(np.sum(mask)) / total_pixels
        known_regions.append(RegionInfo(
            label=label,
            mask=mask,
            confidence=0.80,
            pixel_fraction=pf,
            source="color_detector",
            traversability=get_traversability(label),
        ))

    if np.any(dark_mask):
        pf = float(np.sum(dark_mask)) / total_pixels
        known_regions.append(RegionInfo(
            label="mud",
            mask=dark_mask,
            confidence=0.60,
            pixel_fraction=pf,
            source="color_detector",
            traversability=get_traversability("mud"),
        ))

    unknown_regions = []
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


class _MockColorDetector:
    def detect(self, image):
        return _color_detect(image)


# ── CLI & Main ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run sequence pose-aware mapping demo")
    p.add_argument(
        "--rugd_dir",
        default=None,
        help="Path to RUGD dataset root folder. Overrides RUGD_DATA_PATH env var.",
    )
    p.add_argument("--sequence", default="trail-5", help="RUGD sequence to run (default: trail-5)")
    p.add_argument("--n_images", type=int, default=5, help="Number of frames to process (default: 5)")
    p.add_argument(
        "--use_real_models",
        action="store_true",
        default=False,
        help="Use real SAM3/SAM2 models (requires GPU/high memory, downloads weights)",
    )
    p.add_argument("--config", default=None, help="Path to config.yaml")
    return p.parse_args()


def load_image(path: Path) -> Optional[np.ndarray]:
    try:
        import cv2
        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def main() -> None:
    args = parse_args()

    # Resolve RUGD dir
    rugd_dir = args.rugd_dir or os.environ.get("RUGD_DATA_PATH")
    if not rugd_dir:
        # Check standard config path for rugd path
        sam3_cfg_path = PROJECT_ROOT / "baselines" / "sam3" / "config.yaml"
        if sam3_cfg_path.exists():
            import yaml
            try:
                with open(sam3_cfg_path) as f:
                    cfg = yaml.safe_load(f)
                    rugd_dir = cfg.get("rugd", {}).get("data_path")
            except Exception:
                pass

    if not rugd_dir:
        print("ERROR: RUGD dataset directory not specified. Use --rugd_dir or set RUGD_DATA_PATH.")
        sys.exit(1)

    rugd_path = Path(os.path.expanduser(rugd_dir))
    
    # Try directory layout first: val/img/<sequence>/*.png
    seq_path = rugd_path / "val" / "img" / args.sequence
    if not seq_path.exists():
        seq_path = rugd_path / "RUGD_frames-with-annotations" / args.sequence

    if seq_path.exists() and seq_path.is_dir():
        images = sorted(seq_path.glob("*.png"))[:args.n_images]
    else:
        # Fallback to flat layout: val/img/<sequence>_*.png
        flat_dir = rugd_path / "val" / "img"
        if not flat_dir.exists():
            flat_dir = rugd_path / "RUGD_frames-with-annotations"
        
        if flat_dir.exists():
            images = sorted(flat_dir.glob(f"{args.sequence}_*.png"))[:args.n_images]
        else:
            images = []

    if not images:
        print(f"ERROR: No images found for sequence '{args.sequence}' in layout at {rugd_path}")
        print("Please verify the RUGD directory layout.")
        sys.exit(1)


    config_path = args.config or str(PROJECT_ROOT / "system" / "env_uncertainty" / "config.yaml")

    # Build runner
    if args.use_real_models:
        print("\n[Sequence Mapping] Loading runner with real SAM2/SAM3 models...")
        runner = EnvironmentalUncertaintyRunner(config_path=config_path, use_real_models=True)
    else:
        print("\n[Sequence Mapping] Loading runner with fast color mock detector (No GPU needed)...")
        runner = EnvironmentalUncertaintyRunner(config_path=config_path, detector=_MockColorDetector())

    # User Profile (Grounding questions if ASK triggers)
    profile = UserProfile(
        user_id="seq_user",
        verbosity="standard",
        expertise="intermediate",
        preferred_format="question",
        name="Operator",
    )

    # Initialize simulated Odometry
    # dt = 0.2 s (5 Hz), speed = 0.5 m/s -> advances x by 0.1 meters per frame
    odometry = MockForwardOdometry(speed_mps=0.5, fps=5.0)

    print(f"\nProcessing sequence '{args.sequence}' ({len(images)} images):")
    print(f"{'='*80}")
    print(f"{'Frame':<15} | {'Robot Pose (x, y, theta)':<28} | {'Action':<7} | {'GP Obs':<6} | {'SG Nodes':<8}")
    print(f"{'-'*80}")

    for img_path in images:
        img = load_image(img_path)
        if img is None:
            print(f"{img_path.name:<15} | ERROR loading image")
            continue

        h, w = img.shape[:2]
        goal_pixel = (int(h * 0.20), w // 2)

        # Get current simulated pose and step odometry forward
        pose = odometry.next_pose()

        # Run scene with robot pose!
        # This projects pixel coordinates to world metric tiles and updates world GP
        t0 = time.perf_counter()
        decision = runner.run_scene_with_pose(
            image=img,
            pose=pose,
            scene_id=img_path.stem,
            goal_pixel=goal_pixel,
            user_profile=profile,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        pose_str = f"({pose.x:.2f}, {pose.y:.2f}, {pose.theta:.2f})"
        print(
            f"{img_path.name:<15} | "
            f"{pose_str:<28} | "
            f"{decision.robot_action:<7} | "
            f"{runner.world_gp.n_observations:<6} | "
            f"{runner.world_scene_graph.node_count:<8} "
            f"({elapsed_ms:.0f} ms)"
        )

    print(f"{'='*80}")
    print("\nSequence mapping completed!")
    print(f"Total accumulated world GP observations : {runner.world_gp.n_observations}")
    print(f"Total accumulated world Scene Graph nodes: {runner.world_scene_graph.node_count}")

    bounds = runner.world_gp.observation_bounds
    if bounds:
        min_x, max_x, min_y, max_y = bounds
        print(f"World GP Map Spatial Bounds           : x in [{min_x:.2f}, {max_x:.2f}] m, y in [{min_y:.2f}, {max_y:.2f}] m")


if __name__ == "__main__":
    main()
