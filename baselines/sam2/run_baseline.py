"""
SAM2 Baseline Runner

Runs SAM2 segment everything on a directory of RUGD images,
computes metrics (mask coverage and inference FPS),
saves visualizations with distinct random mask colors,
and writes a JSON results summary.

Usage:
    python -m baselines.sam2.run_baseline \
        --config  baselines/sam2/config.yaml \
        --rugd_dir /path/to/rugd \
        --split   val \
        --output_dir outputs/sam2_baseline \
        [--max_samples 50]
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None

from baselines.sam3.data_loader import load_rugd_split
from baselines.sam3.metrics import compute_mask_coverage
from baselines.sam2.sam2_standalone import SAM2Baseline, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM2 segment everything baseline")
    p.add_argument(
        "--config", default="baselines/sam2/config.yaml",
        help="Path to SAM2 baseline config YAML",
    )
    p.add_argument(
        "--rugd_dir", default=None,
        help="Root directory of RUGD dataset. Overrides RUGD_DATA_PATH env var.",
    )
    p.add_argument(
        "--split", default="val", choices=["train", "val", "test"],
        help="RUGD split to evaluate on (default: val)",
    )
    p.add_argument(
        "--output_dir", default="outputs/sam2_baseline",
        help="Where to write visualizations and results.json",
    )
    p.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit number of images (useful for quick smoke tests)",
    )
    p.add_argument(
        "--scene", default=None,
        help="Restrict evaluation to a single scene (e.g. 'trail-5'). "
             "Filters after loading the full split so splits remain canonical.",
    )
    return p.parse_args()


def save_visualization(
    image_pil: Image.Image,
    sam2_results: dict,
    save_path: str,
    alpha: float = 0.5,
) -> None:
    """Overlay SAM2 segment everything masks with distinct colors and save as PNG."""
    if cv2 is None:
        return
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # PIL (RGB) -> OpenCV (BGR)
    bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    output = bgr.copy()
    
    masks = sam2_results.get("masks", [])
    num_masks = len(masks)
    
    # Generate distinct colors for each mask deterministically
    np.random.seed(42)
    colors = np.random.randint(0, 256, size=(num_masks, 3), dtype=np.uint8)
    
    for i, mask in enumerate(masks):
        color = colors[i].tolist()
        mask_bool = mask.astype(bool)
        if mask_bool.shape[:2] != bgr.shape[:2]:
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (bgr.shape[1], bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_bool = mask_resized.astype(bool)
            
        colored = np.zeros_like(output)
        colored[:] = color
        output[mask_bool] = (
            (1 - alpha) * output[mask_bool] + alpha * colored[mask_bool]
        ).astype(np.uint8)
        
    cv2.imwrite(save_path, output)


def run(args: argparse.Namespace) -> None:
    load_dotenv()

    config = load_config(args.config)

    # Resolve RUGD path from args > env > config
    # We also check config for 'rugd.data_path' (sharing config pattern with sam3)
    rugd_dir = (
        args.rugd_dir
        or os.environ.get("RUGD_DATA_PATH")
        or config.get("rugd", {}).get("data_path")
    )
    if rugd_dir is None:
        # Check if we can find RUGD configuration from sam3 config if not present in sam2
        sam3_cfg_path = Path(args.config).parent.parent / "sam3" / "config.yaml"
        if sam3_cfg_path.exists():
            try:
                with open(sam3_cfg_path) as f:
                    sam3_config = yaml.safe_load(f)
                    rugd_dir = sam3_config.get("rugd", {}).get("data_path")
            except Exception:
                pass

    if rugd_dir is None:
        raise ValueError(
            "RUGD directory not specified. Use --rugd_dir or set RUGD_DATA_PATH in .env"
        )
    rugd_dir = os.path.expanduser(rugd_dir)

    print(f"\n[SAM2 Baseline] Loading RUGD {args.split} split from: {rugd_dir}")
    samples = load_rugd_split(rugd_dir, split=args.split)
    if args.scene:
        samples = [s for s in samples if s.name.startswith(args.scene)]
        print(f"  Filtered to scene '{args.scene}': {len(samples)} images")
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"  Evaluating {len(samples)} images")

    print(f"[SAM2 Baseline] Loading model...")
    model = SAM2Baseline(config_path=args.config)
    print(f"  Model loaded")

    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    results_per_image = []
    coverage_sum = 0.0
    masks_count_sum = 0
    count = 0

    print(f"\n[SAM2 Baseline] Running segmentation on {len(samples)} images...")
    for sample in tqdm(samples, desc="SAM2 segmentation"):
        image = sample.load_image()
        result = model.segment(image)
        
        H, W = image.size[1], image.size[0]
        coverage = compute_mask_coverage(result, (H, W))
        n_masks = len(result.get("masks", []))

        coverage_sum += coverage
        masks_count_sum += n_masks
        count += 1

        # Save visualization
        vis_path = str(vis_dir / f"{sample.name}_seg.png")
        save_visualization(image, result, vis_path)

        results_per_image.append({
            "name": sample.name,
            "inference_time_s": result["inference_time_s"],
            "n_detections": int(n_masks),
            "coverage": float(coverage),
        })

    mean_coverage = coverage_sum / count if count > 0 else 0.0
    mean_masks = masks_count_sum / count if count > 0 else 0.0
    mean_fps = model.mean_fps()

    summary = {
        "n_images": count,
        "mean_fps": mean_fps,
        "mean_coverage": mean_coverage,
        "mean_masks_per_image": mean_masks,
        "split": args.split,
    }

    # Write results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(
            {"summary": summary, "per_image": results_per_image},
            f, indent=2,
        )

    print(f"\n{'='*55}")
    print(f"SAM2 Baseline Results — RUGD {args.split} split")
    print(f"{'='*55}")
    print(f"  Images evaluated      : {summary['n_images']}")
    print(f"  Mean FPS              : {summary['mean_fps']:.2f}")
    print(f"  Mean coverage         : {summary['mean_coverage']:.4f}")
    print(f"  Mean masks per image  : {summary['mean_masks_per_image']:.2f}")
    print(f"\n  Results saved to : {results_path}")
    if cv2 is not None:
        print(f"  Visuals saved to : {vis_dir}")
    else:
        print(f"  Visuals skipped (OpenCV/cv2 not installed in current environment)")


if __name__ == "__main__":
    run(parse_args())
