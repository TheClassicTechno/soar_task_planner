"""
Step-by-step trace of run_with_feedback() on one RUGD image.

Shows every stage of the pipeline clearly:
  Step 1  — Terrain detection (color detector → known/unknown regions)
  Step 4  — GP seeding from known regions
  Step 5  — Trajectory generation + LCB scoring
  Step 6  — PROCEED / ASK / STOP decision
  Step 7  — Question generation
  Step 8  — User response parsing (ParsedUserResponse)
  Step 9  — GP Bayesian update at uncertain waypoints
  Step 10 — Replan: re-run pipeline with updated GP → new decision

Usage:
    python scripts/trace_feedback_single_image.py
    python scripts/trace_feedback_single_image.py --image trail-5_00001.png
    python scripts/trace_feedback_single_image.py --response "It looks like wet mud, avoid it"
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.env_uncertainty.map_updater import parse_user_response_rich
from system.env_uncertainty.user_profile import UserProfile

RUGD_DIR = Path(os.path.expanduser(os.environ.get("RUGD_DATA_PATH", "~/Documents/datasets/rugd"))) / "RUGD_frames-with-annotations"
CONFIG_PATH = PROJECT_ROOT / "system" / "env_uncertainty" / "config.yaml"


# ── Reuse the same color detector from run_pipeline_rugd.py ──────────────────

def _color_detect(image: np.ndarray):
    import cv2
    from system.env_uncertainty.detector import DetectionResult, RegionInfo
    from system.env_uncertainty.traversability import TraversabilityMap, get_traversability

    h_img, w_img = image.shape[:2]
    total_pixels = h_img * w_img
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0].astype(float)
    S = hsv[:, :, 1].astype(float)
    V = hsv[:, :, 2].astype(float)

    grass_mask  = (H >= 35) & (H <= 85)  & (S > 40)
    dirt_mask   = (H >= 10) & (H <= 35)  & (S > 25) & ~grass_mask
    path_mask   = (S < 25)               & (V > 100)
    dark_mask   = (S < 25)               & (V <= 100) & ~path_mask
    water_mask  = (H >= 85) & (H <= 130) & (S > 60)
    covered     = grass_mask | dirt_mask | path_mask | water_mask
    unknown_mask = ~covered & ~dark_mask

    known_regions = []
    for label, mask in [("grass", grass_mask), ("dirt", dirt_mask),
                        ("concrete", path_mask), ("water", water_mask)]:
        if not np.any(mask):
            continue
        pf = float(np.sum(mask)) / total_pixels
        known_regions.append(RegionInfo(
            label=label, mask=mask, confidence=0.75, pixel_fraction=pf,
            source="color_detector", traversability=get_traversability(label),
        ))
    if np.any(dark_mask):
        pf = float(np.sum(dark_mask)) / total_pixels
        known_regions.append(RegionInfo(
            label="mud", mask=dark_mask, confidence=0.50, pixel_fraction=pf,
            source="color_detector", traversability=get_traversability("mud"),
        ))

    unknown_regions = []
    if np.any(unknown_mask):
        pf = float(np.sum(unknown_mask)) / total_pixels
        unknown_regions.append(RegionInfo(
            label="unknown", mask=unknown_mask, confidence=0.0, pixel_fraction=pf,
            source="color_detector", traversability=0.0,
        ))

    tmap = TraversabilityMap.create(h_img, w_img)
    for r in known_regions:
        tmap = tmap.update_region(r.mask, r.label)

    sam3_coverage = float(np.sum(covered | dark_mask)) / total_pixels
    unknown_coverage = float(np.sum(unknown_mask)) / total_pixels
    return DetectionResult(
        known_regions=known_regions, unknown_regions=unknown_regions,
        image_shape=(h_img, w_img), sam3_coverage=sam3_coverage,
        unknown_coverage=unknown_coverage, has_unknown=len(unknown_regions) > 0,
        traversability_map=tmap,
    )


def _load_image(path: Path) -> np.ndarray:
    import cv2
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot load {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _sep(title: str = "") -> None:
    if title:
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")
    else:
        print(f"{'─'*60}")


def trace(image_name: str, user_response: str, verbosity: str = "verbose") -> None:
    seq_dir = RUGD_DIR / "trail-5"
    img_path = seq_dir / image_name
    if not img_path.exists():
        # pick first available
        imgs = sorted(seq_dir.glob("*.png"))
        if not imgs:
            print(f"ERROR: no images found in {seq_dir}")
            sys.exit(1)
        img_path = imgs[0]
        image_name = img_path.name

    print(f"\n{'='*60}")
    print(f"  PIPELINE TRACE — run_with_feedback()")
    print(f"  Image:    {image_name}")
    print(f"  Response: \"{user_response}\"")
    print(f"  Verbosity: {verbosity}")
    print(f"{'='*60}")

    img = _load_image(img_path)
    h, w = img.shape[:2]
    goal_pixel = (int(h * 0.20), w // 2)    # top-centre goal

    # ── STEP 1: Terrain detection ─────────────────────────────────────────────
    _sep("STEP 1 — Terrain detection (color detector, no GPU)")
    t0 = time.perf_counter()
    detection = _color_detect(img)
    det_ms = (time.perf_counter() - t0) * 1000
    print(f"  Image size:         {w}×{h} px")
    print(f"  Goal pixel:         row={goal_pixel[0]}, col={goal_pixel[1]}")
    print(f"  Detection time:     {det_ms:.1f} ms")
    print(f"  SAM3 coverage:      {detection.sam3_coverage:.1%}  (pixels with known label)")
    print(f"  Unknown coverage:   {detection.unknown_coverage:.1%}  (pixels no label found)")
    print(f"  Has unknown:        {detection.has_unknown}")
    print(f"\n  Known regions ({len(detection.known_regions)}):")
    for r in detection.known_regions:
        print(f"    [{r.label:12s}]  {r.pixel_fraction:.1%} of image  "
              f"traversability={r.traversability:.2f}  conf={r.confidence:.2f}")
    print(f"\n  Unknown regions ({len(detection.unknown_regions)}):")
    for r in detection.unknown_regions:
        print(f"    [{r.label:12s}]  {r.pixel_fraction:.1%} of image  "
              f"traversability={r.traversability:.2f}  conf={r.confidence:.2f}")

    # ── STEP 4-6 via runner (initial run) ─────────────────────────────────────
    from system.env_uncertainty.runner import EnvironmentalUncertaintyRunner
    from system.env_uncertainty.gp_traversability import GPTraversabilityMap
    from system.env_uncertainty.trajectory import GoalDirectedTrajectoryGenerator

    class _ColorDet:
        def detect(self, image):
            return _color_detect(image)

    runner = EnvironmentalUncertaintyRunner(
        config_path=str(CONFIG_PATH),
        detector=_ColorDet(),
    )
    profile = UserProfile(
        user_id="trace_user", verbosity=verbosity,
        expertise="intermediate", preferred_format="question",
        name=f"Trace ({verbosity})",
    )

    # Step 4: GP seeding — show what observations will be added
    _sep("STEP 4 — GP seeding from known regions")
    print("  For each known region, we sample 10 observation points and add them")
    print("  to the GP as (pixel_y, pixel_x) → traversability score pairs.")
    print("  The GP then builds a continuous posterior over the whole image plane.")
    for r in detection.known_regions:
        ys, xs = np.where(r.mask)
        n_sample = min(10, len(ys))
        print(f"    [{r.label:12s}]  {len(ys):6d} px  "
              f"→ {n_sample} GP obs at traversability={r.traversability:.2f}")

    # Step 5: Trajectories — show what gets generated
    _sep("STEP 5 — Goal-directed trajectory generation + LCB scoring")
    tgen = GoalDirectedTrajectoryGenerator(h, w, n_waypoints=20, detour_fraction=0.25)
    trajs = tgen.generate_toward_goal(
        start_pixel=(h - 1, w // 2), goal_pixel=goal_pixel
    )
    print(f"  Generated {len(trajs)} trajectories toward goal_pixel={goal_pixel}:")
    for t in trajs:
        print(f"    [{t.name:15s}]  {len(t.waypoints)} waypoints  "
              f"start={t.waypoints[0]}  end={t.waypoints[-1]}")

    # Step 6: Initial decision
    _sep("STEP 6/7 — Initial PROCEED/ASK/STOP decision + question")
    t0 = time.perf_counter()
    initial = runner.run_scene(img, scene_id="trace_initial",
                               goal_pixel=goal_pixel, user_profile=profile)
    init_ms = (time.perf_counter() - t0) * 1000

    print(f"  Robot action:       {initial.robot_action}")
    print(f"  Unknown coverage:   {initial.unknown_coverage:.1%}")
    print(f"  Run time:           {init_ms:.1f} ms")
    if initial.best_trajectory:
        t = initial.best_trajectory
        print(f"  Best trajectory:    {t.name}  "
              f"({len(t.waypoints)} waypoints, "
              f"passes_unknown={t.passes_through_unknown})")
    else:
        print(f"  Best trajectory:    None (no safe path found)")
    if initial.question:
        print(f"\n  QUESTION GENERATED ({verbosity}):")
        print(f"  \"{initial.question}\"")
    else:
        print(f"\n  No question generated (action = {initial.robot_action})")

    # ── STEP 8: Parse user response ───────────────────────────────────────────
    _sep("STEP 8 — Parse user response → structured output")
    print(f"  User says: \"{user_response}\"")
    parsed = parse_user_response_rich(user_response)
    print(f"\n  ParsedUserResponse:")
    print(f"    terrain_label:          {parsed.terrain_label!r}")
    print(f"    label_confidence:       {parsed.label_confidence:.2f}   (0=uncertain, 1=certain)")
    print(f"    is_traversable:         {parsed.is_traversable}   (net safety judgment)")
    print(f"    traversability_conf:    {parsed.traversability_confidence:.2f}")
    print(f"    affordance_modifier:    {parsed.affordance_modifier:+.2f}  (sum of keyword modifiers)")
    print(f"    keywords_matched:       {parsed.keywords}")

    # ── STEP 9: GP update ─────────────────────────────────────────────────────
    _sep("STEP 9 — GP Bayesian update at uncertain waypoints")
    if initial.best_trajectory is not None:
        update_traj = initial.best_trajectory
        fallback = False
    else:
        raw_trajs = tgen.generate_toward_goal(start_pixel=(h - 1, w // 2), goal_pixel=goal_pixel)
        update_traj = raw_trajs[0] if raw_trajs else None
        fallback = True

    if update_traj is not None:
        wps = update_traj.waypoints
        if fallback:
            print(f"  best_trajectory was None (all paths passed through unknown).")
            print(f"  FALLBACK: updating GP along '{update_traj.name}' trajectory — the")
            print(f"  user is answering about what's on the direct path to goal.")
        else:
            print(f"  Updating GP along '{update_traj.name}' (the selected best trajectory).")
        print(f"  Waypoints to update: {len(wps)}")
        print(f"  is_traversable = {parsed.is_traversable}  "
              f"({'safe → GP score pushed up' if parsed.is_traversable else 'unsafe → GP score pushed down'})")
        print(f"  Update is area-specific — only pixels along this path are touched.")
        print(f"  Sample waypoints: {wps[:3]} … {wps[-1]}")
    else:
        print("  No trajectory available to update.")

    # ── STEP 10: Replan ───────────────────────────────────────────────────────
    _sep("STEP 10 — Replan after GP update")
    t0 = time.perf_counter()
    _, replanned = runner.run_with_feedback(
        img,
        user_response=user_response,
        scene_id="trace_replan",
        goal_pixel=goal_pixel,
        user_profile=profile,
    )
    replan_ms = (time.perf_counter() - t0) * 1000

    print(f"  Robot action:       {replanned.robot_action}")
    print(f"  Unknown coverage:   {replanned.unknown_coverage:.1%}  (unchanged — pixel masks didn't change)")
    print(f"  Run time:           {replan_ms:.1f} ms (includes full pipeline re-run)")
    if replanned.best_trajectory:
        print(f"  Best trajectory:    {replanned.best_trajectory.name}")
    if replanned.question:
        print(f"\n  NEW QUESTION (if still uncertain):")
        print(f"  \"{replanned.question}\"")
    else:
        print(f"\n  No question — robot is satisfied and ready to act.")

    # ── Summary ───────────────────────────────────────────────────────────────
    _sep("SUMMARY")
    arrow = "→"
    changed = initial.robot_action != replanned.robot_action
    print(f"  Before feedback:  {initial.robot_action}")
    print(f"  After feedback:   {replanned.robot_action}  {'← RESOLVED ✓' if changed else '← unchanged'}")
    print(f"  User said terrain is {'SAFE (traversable)' if parsed.is_traversable else 'UNSAFE (not traversable)'}")
    n_updated = len(update_traj.waypoints) if update_traj is not None else 0
    src = update_traj.name if update_traj else "none"
    print(f"  GP posterior updated at {n_updated} waypoints along '{src}'")
    if initial.question:
        print(f"\n  Question asked ({verbosity}): \"{initial.question}\"")
    print(f"\n  Total time:  detection={det_ms:.0f}ms  initial={init_ms:.0f}ms  replan={replan_ms:.0f}ms")
    print()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step-by-step run_with_feedback trace on one RUGD image")
    p.add_argument("--image", default="trail-5_00001.png", help="Image filename in trail-5/")
    p.add_argument(
        "--response",
        default="That looks like wet mud ahead, I would avoid it.",
        help="Simulated user response text",
    )
    p.add_argument(
        "--verbosity",
        choices=["terse", "standard", "verbose"],
        default="verbose",
        help="Question verbosity level",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    trace(args.image, args.response, args.verbosity)
