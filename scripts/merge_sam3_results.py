"""
Merge two SAM3 results.json files into a single combined report.

Usage:
    python scripts/merge_sam3_results.py \
        --inputs outputs/sam3_baseline/results.json \
                  outputs/sam3_trail5/results.json \
        --output  outputs/sam3_combined/results.json

The merge:
  - Concatenates per_image lists (deduplicates by name if needed)
  - Recomputes summary statistics from the merged per_image + per_class IoU
    by weighted average (weighted by image count per scene)
  - Records which source files were merged and per-scene breakdowns
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _weighted_mean(scene_summaries: List[Dict]) -> Dict:
    """
    Compute weighted mean of per-class IoU and scalar metrics across scenes.
    Weight = number of images in each scene.
    """
    total_images = sum(s["n_images"] for s in scene_summaries)
    if total_images == 0:
        return {}

    # Scalar metrics
    combined: Dict = {
        "n_images": total_images,
        "mean_coverage": sum(
            s["mean_coverage"] * s["n_images"] for s in scene_summaries
        ) / total_images,
        "mean_fps": sum(
            s["mean_fps"] * s["n_images"] for s in scene_summaries
        ) / total_images,
    }

    # Per-class IoU — only where GT exists; classes present in some scenes may
    # be 0.0 in others (class not present in that scene's images).
    all_classes = set()
    for s in scene_summaries:
        all_classes.update(s.get("per_class_iou", {}).keys())

    per_class: Dict[str, float] = {}
    for cls in sorted(all_classes):
        per_class[cls] = sum(
            s.get("per_class_iou", {}).get(cls, 0.0) * s["n_images"]
            for s in scene_summaries
        ) / total_images
    combined["per_class_iou"] = per_class
    combined["mean_iou"] = (
        sum(per_class.values()) / len(per_class) if per_class else 0.0
    )

    # Carry over metadata from first file that has it
    for s in scene_summaries:
        if "split" in s and "split" not in combined:
            combined["split"] = s["split"]
        if "queries" in s and "queries" not in combined:
            combined["queries"] = s["queries"]
            combined["n_queries"] = s.get("n_queries", len(s["queries"]))

    return combined


def merge(input_paths: List[Path], output_path: Path) -> None:
    scene_summaries = []
    all_per_image = []
    seen_names = set()

    for path in input_paths:
        with open(path) as f:
            data = json.load(f)

        summary = data["summary"]
        # Tag summary with its source file for the per-scene breakdown
        summary["_source"] = str(path)
        scene_summaries.append(summary)

        for img in data.get("per_image", []):
            if img["name"] not in seen_names:
                seen_names.add(img["name"])
                all_per_image.append(img)

    combined_summary = _weighted_mean(scene_summaries)

    # Per-scene breakdown (strip internal _source tag before saving)
    per_scene = []
    for s in scene_summaries:
        entry = {k: v for k, v in s.items() if not k.startswith("_")}
        entry["source_file"] = s["_source"]
        per_scene.append(entry)
    combined_summary["per_scene"] = per_scene

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {"summary": combined_summary, "per_image": all_per_image},
            f, indent=2,
        )

    print(f"{'='*60}")
    print(f"SAM3 Combined Results ({combined_summary['n_images']} images)")
    print(f"{'='*60}")
    print(f"  Mean IoU (mIoU)  : {combined_summary['mean_iou']:.4f}")
    print(f"  Mean coverage    : {combined_summary['mean_coverage']:.4f}")
    print(f"  Mean FPS         : {combined_summary['mean_fps']:.4f}")
    print(f"\n  Per-class IoU (combined):")
    for cls, iou in combined_summary.get("per_class_iou", {}).items():
        print(f"    {cls:<25} {iou:.4f}")
    print(f"\n  Per-scene breakdown:")
    for s in per_scene:
        print(f"    {s.get('source_file', '?')}:")
        print(f"      n_images={s['n_images']}  mIoU={s.get('mean_iou', 0):.4f}")
    print(f"\n  Saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge SAM3 results.json files")
    p.add_argument(
        "--inputs", nargs="+", required=True,
        help="Two or more results.json files to merge",
    )
    p.add_argument(
        "--output", default="outputs/sam3_combined/results.json",
        help="Output path for the merged results.json",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge(
        input_paths=[Path(p) for p in args.inputs],
        output_path=Path(args.output),
    )
