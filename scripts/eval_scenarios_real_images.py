"""
Run the 7 named scenarios against their real GOOSE source images.

Uses GT label maps (same approach as run_pipeline_goose.py) so no GPU is
needed.  For each scenario, compares the pipeline's actual decision to
scenario.expected_action and reports PASS / FAIL.

Usage:
    python scripts/eval_scenarios_real_images.py
    python scripts/eval_scenarios_real_images.py --save_json outputs/scenario_real_image_eval.json
    python scripts/eval_scenarios_real_images.py --goose_dir /path/to/GOOSE
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_pipeline_goose import (
    _load_goose_label_mapping,
    _load_image,
    _load_label,
    _find_label_path,
    _make_label_runner,
    _simulate_gt_feedback,
)
from system.env_uncertainty.scenarios import SCENARIOS
from system.env_uncertainty.user_profile import DEFAULT_PROFILE


def run_scenario_real_images(
    goose_dir: str = "data/datasets/GOOSE",
    save_json: Optional[str] = None,
    goal_row_fraction: float = 0.20,
) -> List[dict]:
    goose_path = Path(goose_dir)
    label_base = goose_path / "labels" / "val"

    id_to_vocab = _load_goose_label_mapping(goose_path)
    config_path = str(PROJECT_ROOT / "system" / "env_uncertainty" / "config.yaml")
    runner, gt_detector = _make_label_runner(config_path, id_to_vocab)

    results = []
    n_pass = n_fail = n_skip = 0

    print(f"\nEvaluating 7 scenarios on real GOOSE images")
    print(f"{'=' * 72}")
    header = f"{'Scenario':<28}  {'Expected':8}  {'Actual':8}  {'Result':6}  {'unk':>7}  {'lcb':>6}  {'n_obs':>5}"
    print(header)
    print("-" * 72)

    for scenario in SCENARIOS:
        if scenario.source_image is None:
            print(f"  {scenario.name:<28}  SKIP (no source_image)")
            n_skip += 1
            results.append({"scenario": scenario.name, "result": "skip", "reason": "no source_image"})
            continue

        img_path = PROJECT_ROOT / scenario.source_image
        if not img_path.exists():
            print(f"  {scenario.name:<28}  SKIP (image not found: {img_path})")
            n_skip += 1
            results.append({"scenario": scenario.name, "result": "skip", "reason": f"image not found: {img_path}"})
            continue

        img = _load_image(img_path)
        if img is None:
            print(f"  {scenario.name:<28}  SKIP (could not load image)")
            n_skip += 1
            results.append({"scenario": scenario.name, "result": "skip", "reason": "load failed"})
            continue

        lbl_path = _find_label_path(img_path, label_base)
        if lbl_path is None:
            print(f"  {scenario.name:<28}  SKIP (no label file)")
            n_skip += 1
            results.append({"scenario": scenario.name, "result": "skip", "reason": "no label file"})
            continue

        lbl = _load_label(lbl_path)
        if lbl is None:
            print(f"  {scenario.name:<28}  SKIP (label load failed)")
            n_skip += 1
            results.append({"scenario": scenario.name, "result": "skip", "reason": "label load failed"})
            continue

        gt_detector.set_label(lbl)
        h, w = img.shape[:2]
        goal_pixel = (int(h * goal_row_fraction), w // 2)

        runner.reset_all_knowledge()  # full isolation: no cross-frame bleed between scenarios
        t0 = time.perf_counter()
        decision = runner.run_scene(
            img,
            scene_id=scenario.name,
            goal_pixel=goal_pixel,
            user_profile=DEFAULT_PROFILE,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Compute best_lcb for diagnostics (re-score trajectories after the run)
        h_img, w_img = img.shape[:2]
        if decision.best_trajectory is not None:
            best_lcb_diag = runner._gp_map.score_trajectory_lcb(
                decision.best_trajectory.waypoints, h_img, w_img
            )
        else:
            best_lcb_diag = None

        _simulate_gt_feedback(runner, lbl, id_to_vocab)

        actual = decision.robot_action
        expected = scenario.expected_action
        passed = actual == expected
        result_str = "PASS" if passed else "FAIL"
        if passed:
            n_pass += 1
        else:
            n_fail += 1

        lcb_str = f"{best_lcb_diag:.3f}" if best_lcb_diag is not None else "  N/A"
        n_obs = runner._gp_map.n_observations
        print(
            f"  {scenario.name:<28}  {expected:8}  {actual:8}  {result_str:6}  "
            f"unk={decision.unknown_coverage:.3f}  lcb={lcb_str}  n_obs={n_obs}  ({elapsed_ms:.0f} ms)"
        )
        if not passed and decision.question:
            print(f"    Q: \"{decision.question}\"")

        results.append({
            "scenario": scenario.name,
            "expected_action": expected,
            "actual_action": actual,
            "result": result_str.lower(),
            "passed": passed,
            "unknown_coverage": round(decision.unknown_coverage, 4),
            "sam3_coverage": round(decision.sam3_coverage, 4),
            "n_known_regions": decision.n_known_regions,
            "n_unknown_regions": decision.n_unknown_regions,
            "question": decision.question,
            "elapsed_ms": round(elapsed_ms, 1),
            "source_image": scenario.source_image,
        })

    total_run = n_pass + n_fail
    print("-" * 72)
    print(f"\nResults: {n_pass}/{total_run} passed  |  {n_fail} failed  |  {n_skip} skipped")
    if total_run:
        print(f"Accuracy: {n_pass / total_run:.1%}")

    if save_json:
        out = {
            "n_scenarios": len(SCENARIOS),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_skip": n_skip,
            "accuracy": round(n_pass / total_run, 4) if total_run else None,
            "scenarios": results,
        }
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved → {save_json}")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate 7 scenarios on real GOOSE images")
    p.add_argument("--goose_dir", default="data/datasets/GOOSE")
    p.add_argument("--save_json", default=None)
    p.add_argument("--goal_row", type=float, default=0.20,
                   help="Goal pixel row as fraction of image height (default: 0.20)")
    args = p.parse_args()
    run_scenario_real_images(
        goose_dir=args.goose_dir,
        save_json=args.save_json,
        goal_row_fraction=args.goal_row,
    )
