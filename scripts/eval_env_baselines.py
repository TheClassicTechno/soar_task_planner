"""
Environmental uncertainty baselines comparison on RUGD.

Compares seven decision strategies on the same set of RUGD images:

  1. always_proceed           — never ask, robot always goes (help_rate = 0%)
  2. always_ask               — always ask, robot never self-decides (help_rate = 100%)
  3. coverage_threshold       — ASK if unknown_coverage > threshold (no GP, no Dirichlet)
  4. cvar_path (α=0.10)      — Conditional Value-at-Risk over path waypoints (risk-theoretic)
  5. cvar_global (α=0.10)    — CVaR over all image pixels (global, no path awareness)
  6. max_uncertainty_explorer — info-gain maximizer: seeks highest GP σ path (explores unknown)
  7. our_system               — Full pipeline: GP LCB + Dirichlet + goal-directed trajectories

  Baseline 6 is the information-gain-MAXIMIZING contrast to our conservative system.
  It represents frontier-based exploration (Yamauchi 1997) applied to traversability:
  the robot deliberately routes through uncertain terrain to learn about it, rather
  than asking the user. This is dangerous for safety-critical navigation but is a
  valid academic comparison to show why asking is safer than blind exploration.

CVaR baseline motivation:
  CVaR_α (Conditional Value-at-Risk at level α) is the expected traversability in the
  worst α-fraction of terrain on the path. It is a standard risk measure from financial
  mathematics applied to robotics traversability in:
    Asgharivaskasi & Atanasov, "Scene Informedness and Risk-aware Navigation" (2022)
    and risk-aware costmaps literature (arXiv:2107.11722).
  CVaR captures tail risk — it answers "how bad is the worst 10% of terrain I will
  cross?" rather than just "how much is unknown?". This is principled but still CPU-only
  (pure numpy).

  cvar_path vs cvar_global:
    cvar_path: CVaR computed only along the 20 waypoints of the direct trajectory to goal.
               Path-aware — comparable to our GP LCB which also scores only along the path.
    cvar_global: CVaR over ALL image pixels. No path awareness, whole-scene risk.

  Decision rule for both CVaR variants:
    CVaR_α < stop_cvar  → STOP  (tail risk is critical — worst terrain is nearly impassable)
    CVaR_α < ask_cvar   → ASK   (tail risk is elevated — worst terrain may be unsafe)
    otherwise           → PROCEED

Comparison papers:
  • BADGR (Kahn et al., RA-L 2021, arXiv:2002.05700) — robot explores, no asking.
    Not runnable without GPU/training.
  • GANav (Guan et al., RA-L 2022, arXiv:2103.04233) — classify-then-act, no asking.
    Not runnable without GPU.
  • Physical terrain probing (Frontiers 2022, arXiv:2209.00334) — physical robot probing.

Metrics:
  help_rate:        fraction of images where human was asked
  proceed_rate:     fraction where robot self-decided to go
  stop_rate:        fraction where robot decided to stop entirely
  avg_cvar:         mean CVaR_α across images (CVaR baselines only)
  avg_ms_per_image: wall-clock time per image

Usage:
    python scripts/eval_env_baselines.py
    python scripts/eval_env_baselines.py --n_images 50 --sequence village
    python scripts/eval_env_baselines.py --threshold 0.60 --alpha 0.10
"""
from __future__ import annotations

import argparse
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RUGD_DIR = Path(os.path.expanduser(os.environ.get("RUGD_DATA_PATH", "~/Documents/datasets/rugd"))) / "RUGD_frames-with-annotations"
CONFIG_PATH = str(PROJECT_ROOT / "system" / "env_uncertainty" / "config.yaml")


# ── Color detector (same as run_pipeline_rugd.py) ────────────────────────────

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


def _load_image(path: Path) -> np.ndarray | None:
    import cv2
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── Baseline runners ─────────────────────────────────────────────────────────

class AlwaysProceed:
    """Trivial lower bound: never asks. help_rate = 0%."""
    name = "always_proceed"

    def decide(self, detection, h, w, goal_pixel):
        return "PROCEED", None


class AlwaysAsk:
    """Trivial upper bound: always asks. help_rate = 100%."""
    name = "always_ask"

    _question = "I need your guidance to determine the safest path."

    def decide(self, detection, h, w, goal_pixel):
        return "ASK", self._question


class CoverageThreshold:
    """
    Heuristic baseline: ASK iff unknown_coverage > threshold.

    This captures the 'classify-then-decide' logic of GANav and similar methods
    that have no GP, no Dirichlet entropy, and no trajectory-level reasoning.
    The robot asks whenever too much terrain is unrecognized — regardless of
    whether the unrecognized region is on the planned path or not.

    Corresponds roughly to: "if I can't label enough of the scene, ask."
    This is the simplest possible environmental uncertainty strategy.
    """
    name = "coverage_threshold"

    _question = "I cannot identify enough terrain ahead. Is it safe to proceed?"

    def __init__(self, threshold: float = 0.20):
        self.threshold = threshold

    def decide(self, detection, h, w, goal_pixel):
        if detection.unknown_coverage >= 0.80:
            return "STOP", "Terrain is almost entirely unrecognized. I must stop."
        if detection.unknown_coverage >= self.threshold:
            return "ASK", self._question
        return "PROCEED", None


def _cvar(scores: np.ndarray, alpha: float) -> float:
    """
    Conditional Value-at-Risk at level alpha for a 1-D array of traversability scores.

    CVaR_α = mean traversability of the worst (lowest) α-fraction of scores.

    Formal definition:
        Sort scores ascending: s₀ ≤ s₁ ≤ ... ≤ s_{n-1}
        k = ceil(α · n)          (number of tail values to average)
        CVaR_α = (1/k) Σᵢ₌₀^{k-1} sᵢ

    Interpretation for traversability:
        Low CVaR  → worst terrain patches on the path are very unsafe → ASK or STOP
        High CVaR → even the worst patches are reasonably traversable → PROCEED

    Args:
        scores: 1-D numpy array of traversability values in [0, 1].
        alpha:  Risk level in (0, 1]. α=0.10 → worst 10% of scores.

    Returns:
        CVaR_α in [0, 1], or 0.0 if scores is empty.
    """
    if len(scores) == 0:
        return 0.0
    sorted_s = np.sort(scores)                         # ascending
    k = max(1, int(np.ceil(alpha * len(sorted_s))))    # at least 1 value
    return float(np.mean(sorted_s[:k]))


class CVaRPath:
    """
    Path-aware CVaR baseline (α configurable, default α=0.10).

    Computes CVaR_α over traversability scores at the 20 waypoints along the
    direct trajectory from the robot's current position to goal_pixel.

    This is the most directly comparable baseline to our GP LCB:
      - Both operate along the planned path (not the whole image)
      - Both use a pessimistic/tail-risk metric (CVaR vs LCB)
      - Neither uses a Gaussian Process or Dirichlet distribution

    Key difference from GP LCB:
      - CVaR reads traversability from a static lookup table (class label → score)
      - GP LCB fits a spatial posterior that interpolates and extrapolates smoothly,
        and can be updated from user feedback
      - CVaR cannot distinguish "I've seen this terrain is usually safe" from
        "I've never seen this exact patch" — the GP can

    Threshold calibration (same scale as traversability scores [0,1]):
      stop_cvar: CVaR < 0.10 → STOP  (worst 10% of path is nearly impassable)
      ask_cvar:  CVaR < 0.45 → ASK   (worst 10% of path has significant risk)
      otherwise: PROCEED

    Reference: CVaR applied to traversability cost-maps in risk-aware robot navigation
    (Asgharivaskasi & Atanasov, IROS 2022; risk-aware costmap literature arXiv:2107.11722).
    """

    def __init__(self, alpha: float = 0.10, ask_cvar: float = 0.45, stop_cvar: float = 0.10):
        self.alpha = alpha
        self.ask_cvar = ask_cvar
        self.stop_cvar = stop_cvar
        self.name = f"cvar_path_a{alpha:.2f}"

    def decide(self, detection, h, w, goal_pixel):
        from system.env_uncertainty.trajectory import GoalDirectedTrajectoryGenerator

        # Generate the direct trajectory (same start/goal as our system uses)
        start = (h - 1, w // 2)
        gen = GoalDirectedTrajectoryGenerator(h, w, n_waypoints=20)
        trajs = gen.generate_toward_goal(start, goal_pixel)

        # Use the direct path (first trajectory — straight line to goal)
        direct = trajs[0]

        # Look up traversability at each waypoint from the static label map.
        # Unknown pixels have score 0.0 (initialised by TraversabilityMap.create).
        tmap = detection.traversability_map
        waypoint_scores = np.array([
            tmap.score_at(wy, wx) for wy, wx in direct.waypoints
        ], dtype=np.float32)

        cvar = _cvar(waypoint_scores, self.alpha)

        if cvar < self.stop_cvar:
            return "STOP", (
                f"Path tail risk is critical (CVaR_{self.alpha:.0%}={cvar:.2f}). "
                f"Worst terrain on route may be impassable."
            ), cvar
        if cvar < self.ask_cvar:
            return "ASK", (
                f"Path has elevated tail risk (CVaR_{self.alpha:.0%}={cvar:.2f}). "
                f"Is it safe to cross the terrain ahead?"
            ), cvar
        return "PROCEED", None, cvar


class CVaRGlobal:
    """
    Global CVaR baseline (α configurable, default α=0.10).

    Computes CVaR_α over ALL pixels in the traversability map, regardless of
    whether they are on the planned path.

    This is the simpler, path-unaware variant. It corresponds to risk measures
    used in methods that score the whole scene without trajectory reasoning
    (e.g., global traversability maps used as a go/no-go threshold).

    Key difference from CVaRPath:
      - CVaRPath looks only at the 20 waypoints on the direct route to goal
      - CVaRGlobal considers every pixel in the image equally
      - A dangerous region far off-path will trigger CVaRGlobal but NOT CVaRPath
        (our system and CVaRPath correctly ignore off-path hazards)
    """

    def __init__(self, alpha: float = 0.10, ask_cvar: float = 0.45, stop_cvar: float = 0.10):
        self.alpha = alpha
        self.ask_cvar = ask_cvar
        self.stop_cvar = stop_cvar
        self.name = f"cvar_global_a{alpha:.2f}"

    def decide(self, detection, h, w, goal_pixel):
        # Flatten all pixel traversability scores (includes 0.0 for unknowns)
        all_scores = detection.traversability_map.scores.ravel()
        cvar = _cvar(all_scores, self.alpha)

        if cvar < self.stop_cvar:
            return "STOP", (
                f"Scene-wide tail risk is critical (CVaR_{self.alpha:.0%}={cvar:.2f})."
            ), cvar
        if cvar < self.ask_cvar:
            return "ASK", (
                f"Scene-wide tail risk is elevated (CVaR_{self.alpha:.0%}={cvar:.2f}). "
                f"Is it safe to proceed?"
            ), cvar
        return "PROCEED", None, cvar


class MaxUncertaintyExplorer:
    """
    Information-gain-maximizing baseline (high info-gain exploration).

    Contrast to our system which is information-gain MINIMIZING (avoids unknown
    areas, asks before entering them).  This baseline does the opposite: it
    deliberately seeks the trajectory with the HIGHEST GP uncertainty (max σ_GP),
    exploring unknown terrain rather than avoiding it.

    This mirrors frontier-based exploration (Yamauchi 1997) and active learning
    robots that resolve uncertainty by going there — not asking the user.

    Decision rule:
      • Score each of 3 goal-directed trajectories by MAX σ_GP (vs our min LCB).
      • Pick the highest-uncertainty trajectory.
      • If max σ > ask_sigma: robot reports it is "exploring" (treated as PROCEED).
      • If max σ ≤ stop_sigma: scene is already well-known, PROCEED normally.

    Key insight for the paper: this baseline proceeds through unknown terrain
    without asking — dangerous for safety-critical outdoor navigation where unknown
    = potentially impassable mud/water.  Illustrates why asking is better than
    blindly exploring.
    """
    name = "max_uncertainty_explorer"

    def __init__(self, config_path: str, ask_sigma: float = 0.30):
        from system.env_uncertainty.gp_traversability import GPTraversabilityMap
        from system.env_uncertainty.trajectory import GoalDirectedTrajectoryGenerator
        self._gp = GPTraversabilityMap()
        self.ask_sigma = ask_sigma
        self._config_path = config_path

    def decide(self, detection, h, w, goal_pixel):
        import numpy as np
        from system.env_uncertainty.trajectory import GoalDirectedTrajectoryGenerator

        self._gp.reset()

        # Seed GP from known regions (same as our system)
        for region in detection.known_regions:
            mask = np.asarray(region.mask)
            if not np.any(mask):
                continue
            ys, xs = np.where(mask)
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
            self._gp.add_observation(cy, cx, region.traversability, h, w)

        # Generate 3 goal-directed trajectories
        start = (h - 1, w // 2)
        gen = GoalDirectedTrajectoryGenerator(h, w, n_waypoints=20)
        trajs = gen.generate_toward_goal(start, goal_pixel)

        # Score each path by MAX GP sigma (exploration: seek high uncertainty)
        best_traj = None
        best_max_sigma = -1.0
        for traj in trajs:
            sigmas = [
                self._gp.predict(wy, wx, h, w).sigma
                for wy, wx in traj.waypoints
            ]
            max_sigma = float(np.max(sigmas)) if sigmas else 0.0
            if max_sigma > best_max_sigma:
                best_max_sigma = max_sigma
                best_traj = traj

        # Explorer always proceeds — it wants to go to the uncertain area
        # to learn. It only asks/stops if it detects a known danger (µ < 0.20).
        if self._gp.n_observations > 0 and best_traj is not None:
            mus = [self._gp.predict(wy, wx, h, w).mu for wy, wx in best_traj.waypoints]
            min_mu = float(np.min(mus))
            if min_mu < 0.20:
                return "STOP", "Known dangerous terrain ahead — stopping."
        return "PROCEED", None


class SimulatedGANav:
    """
    Simulated GANav baseline (oracle segmentation, no neural net).

    GANav (Guan et al., RA-L 2022, arXiv:2103.04233) classifies terrain into
    6 navigability groups via group-wise semantic attention, then acts without
    asking.  Here we replicate its decision logic using our label→traversability
    table as a proxy for navigability group assignment.

    Navigability groups (mapped from traversability scores):
      Group A — freely navigable   (τ ≥ 0.80): concrete, sidewalk, road, grass, dirt
      Group B — navigable w/ caution (0.50 ≤ τ < 0.80): gravel, sand, mulch, vegetation
      Group C — non-navigable       (τ < 0.50): mud, water, puddle, slope, rock-bed,
                                                 log, tree, unknown

    Decision rule:
      ANY on-path region is Group C  → STOP  (non-navigable, hard refusal)
      Otherwise (all Group A/B)      → PROCEED

    Key differentiator for the paper:
      SimulatedGANav NEVER ASKs.  It either proceeds or stops silently.
      A human cannot unblock it.  When it encounters unknown terrain or low-
      traversability labels, it refuses forever — no clarification dialog.
      This directly contrasts with our Steps 6–9 human-in-the-loop resolution.
    """
    name = "simulated_ganav"
    _NON_NAVIGABLE_THRESHOLD = 0.50   # τ below this → Group C → STOP

    def decide(self, detection, h, w, goal_pixel):
        from system.env_uncertainty.traversability import get_traversability

        # Unknown regions (SAM2 residuals with no label) are always Group C.
        if detection.unknown_regions:
            return "STOP", None

        # Check traversability of every known region on the path.
        # GANav refuses to enter any Group C region regardless of path focus.
        for region in detection.known_regions:
            trav = get_traversability(region.label)
            if trav < self._NON_NAVIGABLE_THRESHOLD:
                return "STOP", None

        return "PROCEED", None


class CachingDetector:
    """Wrapper that caches the output of detect() to avoid redundant deep network runs."""
    def __init__(self, actual_detector):
        self.actual_detector = actual_detector
        self.cache = {}

    def detect(self, image: np.ndarray):
        import hashlib
        import time
        img_hash = hashlib.md5(image.tobytes()).hexdigest()
        if img_hash not in self.cache:
            t_start = time.perf_counter()
            self.cache[img_hash] = self.actual_detector.detect(image)
            t_elapsed = time.perf_counter() - t_start
            print(f"        [Inference] SAM3+SAM2 forward pass took {t_elapsed:.2f}s")
        return self.cache[img_hash]


class OurSystem:
    """Full pipeline: GP LCB + Dirichlet + goal-directed trajectories."""
    name = "our_system"

    def __init__(self, config_path: str, use_real_models: bool = False, device: Optional[str] = None):
        from system.env_uncertainty.runner import EnvironmentalUncertaintyRunner
        from system.env_uncertainty.user_profile import UserProfile

        if use_real_models:
            self._runner = EnvironmentalUncertaintyRunner(
                config_path=config_path, use_real_models=True, device=device
            )
        else:
            class _Det:
                def detect(self, image):
                    return _color_detect(image)

            self._runner = EnvironmentalUncertaintyRunner(
                config_path=config_path, detector=_Det()
            )
        self._profile = UserProfile(
            user_id="eval", verbosity="standard",
            expertise="intermediate", preferred_format="question",
            name="Eval",
        )


    def decide(self, detection, h, w, goal_pixel):
        # We already ran detection externally — but runner re-runs it internally.
        # Use reset_frame_state so GP doesn't accumulate across frames.
        self._runner.reset_frame_state()
        return None, None  # handled by run_on_image below

    def run_on_image(self, image: np.ndarray, goal_pixel):
        self._runner.reset_frame_state()
        h, w = image.shape[:2]
        d = self._runner.run_scene(image, goal_pixel=goal_pixel, user_profile=self._profile)
        return d.robot_action, d.question


# ── Evaluation loop ───────────────────────────────────────────────────────────

def _eval_baseline(baseline, images: List[Path], use_real_models: bool = False, runner_for_detection = None) -> Dict:
    n_ask = n_proceed = n_stop = 0
    question_lens = []
    cvar_values: List[float] = []
    total_ms = 0.0

    is_cvar = isinstance(baseline, (CVaRPath, CVaRGlobal))

    for idx, img_path in enumerate(images):
        if use_real_models:
            print(f"      [Frame {idx+1}/{len(images)}] Processing {img_path.name}...", flush=True)

        img = _load_image(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        goal_pixel = (int(h * 0.20), w // 2)

        t0 = time.perf_counter()
        if isinstance(baseline, OurSystem):
            action, question = baseline.run_on_image(img, goal_pixel)
        else:
            if use_real_models and runner_for_detection is not None:
                detection = runner_for_detection._detector.detect(img)
            else:
                detection = _color_detect(img)

            if is_cvar:
                action, question, cvar_val = baseline.decide(detection, h, w, goal_pixel)
                cvar_values.append(cvar_val)
            else:
                action, question = baseline.decide(detection, h, w, goal_pixel)
        total_ms += (time.perf_counter() - t0) * 1000

        if action == "ASK":    n_ask += 1
        elif action == "STOP": n_stop += 1
        else:                  n_proceed += 1

        if question:
            question_lens.append(len(question))

    n = n_ask + n_proceed + n_stop
    return {
        "baseline": baseline.name,
        "n_images": n,
        "n_ask": n_ask,
        "n_proceed": n_proceed,
        "n_stop": n_stop,
        "help_rate": round(n_ask / n, 3) if n else 0.0,
        "proceed_rate": round(n_proceed / n, 3) if n else 0.0,
        "stop_rate": round(n_stop / n, 3) if n else 0.0,
        "avg_question_len": round(sum(question_lens) / len(question_lens), 1) if question_lens else 0.0,
        "avg_cvar": round(float(np.mean(cvar_values)), 3) if cvar_values else None,
        "avg_ms_per_image": round(total_ms / n, 1) if n else 0.0,
    }


def run_comparison(
    sequence: str = "trail-5",
    n_images: int = 20,
    threshold: float = 0.60,
    alpha: float = 0.10,
    use_real_models: bool = False,
    device: Optional[str] = None,
) -> None:
    # Resolve dataset path / layout
    rugd_path = Path(os.path.expanduser(os.environ.get("RUGD_DATA_PATH", "~/Documents/datasets/rugd")))
    
    # Try directory layout first
    seq_dir = rugd_path / "RUGD_frames-with-annotations" / sequence
    if not seq_dir.exists():
        seq_dir = rugd_path / "val" / "img" / sequence

    if seq_dir.exists() and seq_dir.is_dir():
        images = sorted(seq_dir.glob("*.png"))[:n_images]
    else:
        # Fallback to flat layout
        flat_dir = rugd_path / "val" / "img"
        if not flat_dir.exists():
            flat_dir = rugd_path / "RUGD_frames-with-annotations"
        
        if flat_dir.exists():
            images = sorted(flat_dir.glob(f"{sequence}_*.png"))[:n_images]
        else:
            images = []

    if not images:
        print(f"ERROR: no PNG images found for sequence '{sequence}' in {rugd_path}")
        sys.exit(1)

    print(f"\nEnv Uncertainty Baselines — {len(images)} images from '{sequence}'")
    print(f"  coverage_threshold: {threshold:.0%}   CVaR alpha: {alpha:.0%}")
    print(f"  CVaR thresholds: ask < 0.45, stop < 0.10\n")

    our_sys = OurSystem(config_path=CONFIG_PATH, use_real_models=use_real_models, device=device)
    if use_real_models:
        our_sys._runner._detector = CachingDetector(our_sys._runner._detector)
        runner_for_detection = our_sys._runner
    else:
        runner_for_detection = None

    baselines = [
        AlwaysProceed(),
        AlwaysAsk(),
        CoverageThreshold(threshold=threshold),
        CVaRPath(alpha=alpha),
        CVaRGlobal(alpha=alpha),
        MaxUncertaintyExplorer(config_path=CONFIG_PATH),
        SimulatedGANav(),
        our_sys,
    ]

    rows = []
    for b in baselines:
        print(f"  Running {b.name} ...", flush=True)
        result = _eval_baseline(b, images, use_real_models=use_real_models, runner_for_detection=runner_for_detection)
        rows.append(result)
        cvar_str = f"  avg_CVaR={result['avg_cvar']:.3f}" if result["avg_cvar"] is not None else ""
        print(f"  Done {b.name} ({result['avg_ms_per_image']:.0f} ms/img){cvar_str}\n", flush=True)

    # ── Print comparison table ────────────────────────────────────────────────
    W = 90
    print(f"\n{'='*W}")
    print(f"{'Baseline':<26} {'Help%':>6} {'Proceed%':>9} {'Stop%':>6} {'AvgCVaR':>8} {'AvgQ':>5} {'ms/img':>7}")
    print(f"{'-'*W}")
    for r in rows:
        cvar_str = f"{r['avg_cvar']:.3f}" if r["avg_cvar"] is not None else "  n/a"
        print(
            f"{r['baseline']:<26} "
            f"{r['help_rate']*100:>5.1f}%  "
            f"{r['proceed_rate']*100:>8.1f}%  "
            f"{r['stop_rate']*100:>5.1f}%  "
            f"{cvar_str:>8}  "
            f"{r['avg_question_len']:>4.0f}  "
            f"{r['avg_ms_per_image']:>6.0f}"
        )
    print(f"{'='*W}")

    print(f"""
Sequence: {sequence}  |  n={len(images)}  |  CVaR α={alpha:.0%}  |  coverage threshold={threshold:.0%}

Baseline descriptions:
  always_proceed        Trivial lower bound — never asks. Analog of BADGR/GANav (autonomous
                        exploration, never defers to human).
  always_ask            Trivial upper bound — always defers to human.
  coverage_threshold    Asks when unknown pixel fraction > {threshold:.0%}. No path awareness,
                        no risk modelling of *known* terrain.
  cvar_path (α={alpha:.0%})   CVaR of traversability scores along the 20 waypoints of the direct
                        trajectory. Path-aware tail risk. No GP, no Dirichlet.
                        Reference: risk-aware costmaps (arXiv:2107.11722).
  cvar_global (α={alpha:.0%}) CVaR over all image pixels. No path awareness — hazards far from
                        the route still trigger ASK/STOP.
  our_system            GP LCB + Dirichlet semantic entropy + goal-directed trajectories.
                        GP captures spatial correlations; Dirichlet catches label ambiguity;
                        trajectory filter ignores off-path hazards; updateable from user feedback.

Key distinctions vs CVaR baselines:
  cvar_path is the fairest comparison to our_system (both are path-aware and pessimistic).
  Difference: CVaR reads a fixed label→score table; GP LCB builds a continuous spatial
  posterior that smoothly interpolates between observations and updates from user feedback.
  cvar_global shows what happens without path awareness — same as coverage_threshold weakness.
""")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare env uncertainty baselines on RUGD")
    p.add_argument("--sequence", default="trail-5")
    p.add_argument("--n_images", type=int, default=20)
    p.add_argument(
        "--threshold", type=float, default=0.60,
        help="Unknown-coverage threshold for coverage_threshold baseline (default 0.60)",
    )
    p.add_argument(
        "--alpha", type=float, default=0.10,
        help="CVaR risk level α: fraction of worst-case waypoints averaged (default 0.10)",
    )
    p.add_argument(
        "--use_real_models",
        action="store_true",
        default=False,
        help="Use real SAM3/SAM2 models instead of color heuristic",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device override for PyTorch (e.g. cpu, mps, cuda)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_comparison(
        sequence=args.sequence,
        n_images=args.n_images,
        threshold=args.threshold,
        alpha=args.alpha,
        use_real_models=args.use_real_models,
        device=args.device,
    )
