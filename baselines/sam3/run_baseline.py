"""
SAM3 Baseline Runner

Runs SAM3 terrain segmentation on a directory of RUGD images,
computes metrics (mIoU when GT is available, mask coverage always),
saves visualizations, and writes a JSON results summary.

Usage:
    python -m baselines.sam3.run_baseline \
        --config  baselines/sam3/config.yaml \
        --rugd_dir /path/to/rugd \
        --split   val \
        --output_dir outputs/sam3_baseline \
        [--max_samples 50]

Requirements:
    - CUDA GPU
    - SAM3 checkpoint downloaded (see scripts/download_sam3_checkpoint.sh)
    - RUGD dataset downloaded (see scripts/download_rugd.py)
    - pip install -r baselines/requirements.txt
    - .env file with ANTHROPIC_API_KEY, NAV_STACK_PATH, etc.
"""

import argparse
import json
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from baselines.sam3.data_loader import (
    SAM3_TO_RUGD,
    load_rugd_split,
    build_class_index,
)
from baselines.sam3.metrics import MetricsAccumulator
from baselines.sam3.sam3_standalone import SAM3Baseline, load_config
from baselines.sam3.visualizer import load_color_map, save_visualization


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM3 terrain segmentation baseline")
    p.add_argument(
        "--config", default="baselines/sam3/config.yaml",
        help="Path to SAM3 baseline config YAML",
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
        "--output_dir", default="outputs/sam3_baseline",
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
    p.add_argument(
        "--no_gt", action="store_true",
        help="Skip mIoU computation even if GT annotations exist",
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    load_dotenv()

    config = load_config(args.config)
    queries = config["sam3"]["queries"]
    color_map = load_color_map(config["sam3"]["color_map"])

    # Resolve RUGD path from args > env > config
    rugd_dir = (
        args.rugd_dir
        or os.environ.get("RUGD_DATA_PATH")
        or config.get("rugd", {}).get("data_path")
    )
    if rugd_dir is None:
        raise ValueError(
            "RUGD directory not specified. Use --rugd_dir or set RUGD_DATA_PATH in .env"
        )

    print(f"\n[SAM3 Baseline] Loading RUGD {args.split} split from: {rugd_dir}")
    samples = load_rugd_split(rugd_dir, split=args.split)
    if args.scene:
        samples = [s for s in samples if s.name.startswith(args.scene)]
        print(f"  Filtered to scene '{args.scene}': {len(samples)} images")
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"  Evaluating {len(samples)} images")

    print(f"[SAM3 Baseline] Loading model (requires CUDA)...")
    model = SAM3Baseline(config_path=args.config)
    print(f"  Model loaded with {len(queries)} concept queries: {queries}")

    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    acc = MetricsAccumulator(queries)
    results_per_image = []

    print(f"\n[SAM3 Baseline] Running segmentation on {len(samples)} images...")
    for sample in tqdm(samples, desc="SAM3 segmentation"):
        image = sample.load_image()
        gt_ann = None if args.no_gt else sample.load_annotation()

        result = model.segment(image)
        H, W = image.size[1], image.size[0]
        acc.update(result, gt_ann, SAM3_TO_RUGD, (H, W))

        # Save visualization
        vis_path = str(vis_dir / f"{sample.name}_seg.png")
        save_visualization(image, result, queries, color_map, vis_path)

        results_per_image.append({
            "name": sample.name,
            "inference_time_s": result["inference_time_s"],
            "n_detections": int(len(result.get("labels", []))),
            "has_gt": gt_ann is not None,
        })

    summary = acc.summary()
    summary["mean_fps"] = model.mean_fps()
    summary["split"] = args.split
    summary["n_queries"] = len(queries)
    summary["queries"] = queries

    # Write results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(
            {"summary": summary, "per_image": results_per_image},
            f, indent=2,
        )

    print(f"\n{'='*55}")
    print(f"SAM3 Baseline Results — RUGD {args.split} split")
    print(f"{'='*55}")
    print(f"  Images evaluated : {summary.get('n_images', 0)}")
    print(f"  Mean FPS         : {summary['mean_fps']:.2f}")
    print(f"  Mean IoU (mIoU)  : {summary.get('mean_iou', 0):.4f}")
    print(f"  Mean coverage    : {summary.get('mean_coverage', 0):.4f}")
    if "per_class_iou" in summary:
        print(f"\n  Per-class IoU:")
        for cls, iou in summary["per_class_iou"].items():
            print(f"    {cls:<25} {iou:.4f}")
    print(f"\n  Results saved to : {results_path}")
    print(f"  Visuals saved to : {vis_dir}")


if __name__ == "__main__":
    run(parse_args())
