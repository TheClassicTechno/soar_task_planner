"""
Mine GOOSE val frames where coverage_threshold or simulated_ganav fail silently.

Two failure types:
  Type A: coverage_threshold says PROCEED, our system says ASK/STOP, GT is dangerous
           (coverage_threshold misses a hazard our system would catch)
  Type B: coverage_threshold says PROCEED, GT min traversability < DANGER_THRESHOLD
           (pure silent failure — GT confirms danger, baseline misses it)

Algorithm:
  1. Load up to --n_images GOOSE val frames (default 50)
  2. For each frame: run CoverageThreshold, SimulatedGANav, OurSystem equivalent
  3. Load GT label map for each frame; compute gt_min_trav
  4. Classify each frame as Type A or Type B failure
  5. Print markdown table + save to docs/failure_examples.md

Usage:
    python scripts/extract_failure_examples.py
    python scripts/extract_failure_examples.py --goose_dir /path/to/GOOSE --n_images 100
    python scripts/extract_failure_examples.py --no_goose  (show synthetic demo)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = str(PROJECT_ROOT / "system" / "env_uncertainty" / "config.yaml")

from system.env_uncertainty.traversability import get_traversability
from scripts.run_pipeline_goose import (
    DANGER_THRESHOLD,
    _compute_gt_min_trav,
    _load_goose_label_mapping,
    _load_image,
    _load_label,
    _find_label_path,
    _collect_images,
    _label_detect,
)
from scripts.eval_env_baselines import CoverageThreshold, SimulatedGANav


# ── Decision helpers ──────────────────────────────────────────────────────────

def _run_coverage_threshold(detection, h: int, w: int, goal_pixel: tuple, threshold: float = 0.60) -> str:
    b = CoverageThreshold(threshold=threshold)
    result = b.decide(detection, h, w, goal_pixel)
    return result[0] if isinstance(result, tuple) else result


def _run_simulated_ganav(detection, h: int, w: int, goal_pixel: tuple) -> str:
    b = SimulatedGANav()
    result = b.decide(detection, h, w, goal_pixel)
    return result[0] if isinstance(result, tuple) else result


def _our_system_equivalent(detection) -> str:
    """
    Simplified rule-based version of our system's decision logic.
    Mirrors OurSystemEquivalent from eval_scenario_baselines.py but
    operates on any DetectionResult directly (not tied to named scenarios).
    """
    unk = detection.unknown_coverage

    if unk >= 0.80:
        return "STOP"
    if unk >= 0.10:
        return "ASK"
    for r in detection.known_regions:
        if r.traversability < 0.50:
            return "ASK"
    return "PROCEED"


# ── Failure classification ────────────────────────────────────────────────────

def _classify_frame(
    cov_action: str,
    our_action: str,
    gt_min_trav: Optional[float],
) -> Optional[str]:
    """
    Return failure type string or None.

    Type A: cov_threshold=PROCEED, our_system≠PROCEED, GT is dangerous
    Type B: cov_threshold=PROCEED, GT is dangerous (regardless of our_system)
    """
    if cov_action != "PROCEED":
        return None
    if gt_min_trav is None or gt_min_trav >= DANGER_THRESHOLD:
        return None
    # GT confirms dangerous path
    if our_action != "PROCEED":
        return "A"  # our system would catch it
    return "B"      # even our system misses it


# ── Per-frame processing ──────────────────────────────────────────────────────

def _process_frame(
    image_path: Path,
    label_path: Path,
    id_to_vocab: Dict[int, str],
    threshold: float,
) -> Optional[Dict]:
    image = _load_image(image_path)
    label_map = _load_label(label_path)
    if image is None or label_map is None:
        return None

    h, w = image.shape[:2]
    goal_pixel = (int(h * 0.20), w // 2)

    detection = _label_detect(image, label_map, id_to_vocab)

    cov_action = _run_coverage_threshold(detection, h, w, goal_pixel, threshold)
    ganav_action = _run_simulated_ganav(detection, h, w, goal_pixel)
    our_action = _our_system_equivalent(detection)

    gt_min_trav = _compute_gt_min_trav(label_map, goal_pixel, h, w, id_to_vocab)
    failure_type = _classify_frame(cov_action, our_action, gt_min_trav)

    return {
        "image": image_path.name,
        "session": image_path.parent.name,
        "cov_threshold": cov_action,
        "simulated_ganav": ganav_action,
        "our_system": our_action,
        "gt_min_trav": round(gt_min_trav, 3) if gt_min_trav is not None else None,
        "gt_is_dangerous": (gt_min_trav is not None and gt_min_trav < DANGER_THRESHOLD),
        "failure_type": failure_type,
        "unknown_coverage": round(detection.unknown_coverage, 3),
    }


# ── GOOSE-based extraction ────────────────────────────────────────────────────

def extract_from_goose(
    goose_dir: Path,
    n_images: int = 50,
    threshold: float = 0.60,
    session_filter: Optional[str] = None,
) -> List[Dict]:
    img_base = goose_dir / "images" / "val"
    label_base = goose_dir / "labels" / "val"

    if not img_base.exists():
        print(f"  WARNING: GOOSE images directory not found: {img_base}")
        return []

    id_to_vocab = _load_goose_label_mapping(goose_dir)

    image_paths = _collect_images(img_base, session_filter, n_images)
    print(f"  Processing {len(image_paths)} GOOSE val images…")

    frames = []
    for img_path in image_paths:
        lbl_path = _find_label_path(img_path, label_base)
        if lbl_path is None:
            continue
        frame = _process_frame(img_path, lbl_path, id_to_vocab, threshold)
        if frame is not None:
            frames.append(frame)

    return frames


# ── Synthetic demo (no GOOSE) ─────────────────────────────────────────────────

def _synthetic_demo_frames() -> List[Dict]:
    """
    Return hand-crafted example frames to illustrate the failure modes
    when GOOSE is not available.  Values match expected pipeline behavior.
    """
    return [
        {
            "image": "demo_muddy_path.png",
            "session": "synthetic",
            "cov_threshold": "PROCEED",
            "simulated_ganav": "STOP",
            "our_system": "ASK",
            "gt_min_trav": 0.10,
            "gt_is_dangerous": True,
            "failure_type": "A",
            "unknown_coverage": 0.05,
        },
        {
            "image": "demo_wet_grass.png",
            "session": "synthetic",
            "cov_threshold": "PROCEED",
            "simulated_ganav": "STOP",
            "our_system": "ASK",
            "gt_min_trav": 0.15,
            "gt_is_dangerous": True,
            "failure_type": "A",
            "unknown_coverage": 0.0,
        },
        {
            "image": "demo_puddle_center.png",
            "session": "synthetic",
            "cov_threshold": "PROCEED",
            "simulated_ganav": "PROCEED",
            "our_system": "ASK",
            "gt_min_trav": 0.05,
            "gt_is_dangerous": True,
            "failure_type": "A",
            "unknown_coverage": 0.0,
        },
        {
            "image": "demo_clear_asphalt.png",
            "session": "synthetic",
            "cov_threshold": "PROCEED",
            "simulated_ganav": "PROCEED",
            "our_system": "PROCEED",
            "gt_min_trav": 0.95,
            "gt_is_dangerous": False,
            "failure_type": None,
            "unknown_coverage": 0.0,
        },
    ]


# ── Report formatting ─────────────────────────────────────────────────────────

def _markdown_table(frames: List[Dict], n_show_per_type: int = 5) -> str:
    type_a = [f for f in frames if f["failure_type"] == "A"]
    type_b = [f for f in frames if f["failure_type"] == "B"]
    n_frames = len(frames)
    n_dangerous = sum(1 for f in frames if f["gt_is_dangerous"])
    n_type_a = len(type_a)
    n_type_b = len(type_b)

    lines: List[str] = []
    lines.append("# Failure Examples: Coverage Threshold vs. Our System\n")
    lines.append(f"Frames analyzed: **{n_frames}**  |  "
                 f"GT-dangerous: **{n_dangerous}**  |  "
                 f"Type A failures: **{n_type_a}**  |  "
                 f"Type B failures: **{n_type_b}**\n")
    lines.append(f"DANGER_THRESHOLD = {DANGER_THRESHOLD}  |  "
                 f"coverage_threshold = 60%\n")

    def _table_block(title: str, desc: str, examples: List[Dict]) -> None:
        lines.append(f"## {title}\n")
        lines.append(f"*{desc}*\n")
        if not examples:
            lines.append("*No examples found.*\n")
            return
        lines.append("| Image | Session | cov_threshold | simulated_ganav | our_system | gt_min_trav | unknown_cov |")
        lines.append("|---|---|---|---|---|---|---|")
        for f in examples[:n_show_per_type]:
            gt = f"**{f['gt_min_trav']}**" if f["gt_is_dangerous"] else str(f["gt_min_trav"])
            lines.append(
                f"| {f['image']} | {f['session'][:25]} | {f['cov_threshold']} "
                f"| {f['simulated_ganav']} | {f['our_system']} | {gt} | {f['unknown_coverage']} |"
            )
        lines.append("")

    _table_block(
        "Type A — Our system catches what coverage_threshold misses",
        "coverage_threshold=PROCEED, our_system=ASK or STOP, GT path is dangerous (gt_min_trav < 0.20)",
        type_a,
    )
    _table_block(
        "Type B — Silent failure: both baselines miss GT-dangerous terrain",
        "coverage_threshold=PROCEED, our_system=PROCEED, GT path is still dangerous",
        type_b,
    )

    lines.append("## Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Frames analyzed | {n_frames} |")
    lines.append(f"| GT-dangerous frames | {n_dangerous} ({100*n_dangerous/max(n_frames,1):.1f}%) |")
    lines.append(f"| Type A failures | {n_type_a} ({100*n_type_a/max(n_dangerous,1):.1f}% of dangerous) |")
    lines.append(f"| Type B failures | {n_type_b} ({100*n_type_b/max(n_dangerous,1):.1f}% of dangerous) |")
    lines.append(f"| Our system catches dangerous frames | "
                 f"{n_type_a} / {n_dangerous} ({100*n_type_a/max(n_dangerous,1):.1f}%) |")

    return "\n".join(lines)


def run(
    goose_dir: Optional[str] = None,
    n_images: int = 50,
    threshold: float = 0.60,
    save_md: Optional[str] = None,
    session_filter: Optional[str] = None,
) -> None:
    print("\nFailure Example Extraction")
    print(f"  coverage_threshold: {threshold:.0%}  |  DANGER_THRESHOLD: {DANGER_THRESHOLD}")

    using_synthetic = False
    if goose_dir:
        goose_path = Path(goose_dir)
        if not goose_path.exists():
            print(f"  WARNING: GOOSE directory not found: {goose_path}")
            print("  Falling back to synthetic demo frames.")
            using_synthetic = True
    else:
        goose_path = None
        using_synthetic = True

    if using_synthetic:
        print("  Using synthetic demo frames (GOOSE not available).")
        frames = _synthetic_demo_frames()
    else:
        frames = extract_from_goose(
            goose_path,
            n_images=n_images,
            threshold=threshold,
            session_filter=session_filter,
        )

    type_a = [f for f in frames if f["failure_type"] == "A"]
    type_b = [f for f in frames if f["failure_type"] == "B"]

    print(f"\n  Frames processed: {len(frames)}")
    print(f"  GT-dangerous:     {sum(1 for f in frames if f['gt_is_dangerous'])}")
    print(f"  Type A failures:  {len(type_a)}")
    print(f"  Type B failures:  {len(type_b)}")

    md = _markdown_table(frames)
    print("\n" + md)

    out_path = save_md or str(PROJECT_ROOT / "docs" / "failure_examples.md")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(md)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract failure examples from GOOSE val")
    parser.add_argument("--goose_dir", type=str, default=None, help="Path to GOOSE root directory")
    parser.add_argument("--n_images", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--save_md", type=str, default=None)
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--no_goose", action="store_true", help="Skip GOOSE, show synthetic demo only")
    args = parser.parse_args()

    goose_dir = None if args.no_goose else args.goose_dir
    run(
        goose_dir=goose_dir,
        n_images=args.n_images,
        threshold=args.threshold,
        save_md=args.save_md,
        session_filter=args.session,
    )
