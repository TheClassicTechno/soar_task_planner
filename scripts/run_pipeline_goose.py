"""
Run the environmental uncertainty pipeline on real GOOSE images.

GOOSE (Mortimer et al., ICRA 2024, arXiv:2310.16788) is a real-world outdoor
dataset with ~10,000 labeled frames, 64 fine-grained terrain classes (12 coarse
categories), captured across 4 seasons. License: CC BY-SA 4.0.

GOOSE has significantly more class diversity than RELLIS-3D or RUGD, including
snow, cobblestone, crops, moss, rail_track, hedge, etc. — making it a strong
stress-test for unknown-terrain detection.

Dataset structure expected at --goose_dir:
  GOOSE/
    goose_label_mapping.csv      (class_name, label_key, has_instance, hex)
    images/
      val/
        2022-08-30_siegertsbrunn_feldwege/
          *_windshield_vis.png   (visible-spectrum images — use these)
          *_windshield_nir.png   (near-infrared — skip)
        ...
    labels/
      val/
        2022-08-30_siegertsbrunn_feldwege/
          *_labelids.png         (uint8, pixel value = class label_key)
          *_color.png            (RGB visualization — not used here)
          *_instanceids.png      (instance IDs — not used here)
        ...

Label resolution: for image path
  images/val/SESSION/SESSION__FRAME_TIMESTAMP_windshield_vis.png
the label is at:
  labels/val/SESSION/SESSION__FRAME_TIMESTAMP_labelids.png

Download:
  1. Register at https://goose-dataset.de/ and accept the CC BY-SA 4.0 license
  2. Download goose_2d_val.zip (2.9 GB) — includes both images and labels
  3. Extract to your --goose_dir
  Or use academictorrents (search "GOOSE traversability")

Usage:
    python scripts/run_pipeline_goose.py
    python scripts/run_pipeline_goose.py --split val --n_images 20
    python scripts/run_pipeline_goose.py --split val --verbosity terse --save_json results/goose_val.json
    python scripts/run_pipeline_goose.py --split val --n_images 50 --session 2023-05-15_neubiberg_rain
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.traversability import TraversabilityMap, get_traversability, TRAVERSABILITY_SCORES
from system.env_uncertainty.user_profile import UserProfile


# ── GOOSE class_name → our vocabulary ────────────────────────────────────────
#
# Keyed by the exact strings in goose_label_mapping.csv column "class_name".
# Classes missing from this map (or mapped to "unknown") get traversability 0.0.
#
# GOOSE val sessions (8 total, 4 seasons):
#   2022-07-22_flight            summer, aerial/ground mix
#   2022-08-30_siegertsbrunn     summer, field paths
#   2022-09-21_garching          autumn, urban/campus
#   2022-12-07_aying_hills       winter, hills
#   2023-01-20_aying_mangfall_2  winter, river/trail
#   2023-03-03_garching_2        spring, campus
#   2023-05-15_neubiberg_rain    spring, rain, urban
#   2023-05-17_neubiberg_sunny   spring, sunny, urban

_GOOSE_NAME_MAP: Dict[str, str] = {
    # ── Paved flat surfaces ──────────────────────────────────────────────────
    "asphalt":              "concrete",
    "sidewalk":             "concrete",
    "bikeway":              "concrete",
    "pedestrian_crossing":  "concrete",
    "road_marking":         "concrete",
    "curb":                 "concrete",
    "cobble":               "gravel",       # cobblestone → uneven, like gravel
    "rail_track":           "concrete",     # traversable flat surface
    # ── Unpaved flat surfaces ────────────────────────────────────────────────
    "gravel":               "gravel",
    "soil":                 "dirt",
    # ── Nature ───────────────────────────────────────────────────────────────
    "low_grass":            "grass",
    "high_grass":           "grass",
    "forest":               "vegetation",
    "bush":                 "vegetation",
    "moss":                 "vegetation",
    "crops":                "vegetation",
    "scenery_vegetation":   "vegetation",
    "hedge":                "vegetation",
    "leaves":               "vegetation",
    "tree_crown":           "tree",
    "tree_trunk":           "tree",
    "tree_root":            "tree",
    "rock":                 "rock-bed",
    "snow":                 "unknown",      # no traversability model for snow
    # ── Water ────────────────────────────────────────────────────────────────
    "water":                "water",
    # ── Human ────────────────────────────────────────────────────────────────
    "person":               "person",
    "rider":                "person",
    # ── Vehicles (not terrain) ───────────────────────────────────────────────
    "car":                  "unknown",
    "truck":                "unknown",
    "bus":                  "unknown",
    "motorcycle":           "unknown",
    "bicycle":              "unknown",
    "kick_scooter":         "unknown",
    "trailer":              "unknown",
    "caravan":              "unknown",
    "on_rails":             "unknown",
    "heavy_machinery":      "unknown",
    "military_vehicle":     "unknown",
    "ego_vehicle":          "unknown",
    # ── Infrastructure / obstacles ───────────────────────────────────────────
    "building":             "unknown",
    "wall":                 "unknown",
    "fence":                "unknown",
    "guard_rail":           "unknown",
    "bridge":               "unknown",
    "tunnel":               "unknown",
    "pole":                 "unknown",
    "traffic_light":        "unknown",
    "traffic_sign":         "unknown",
    "misc_sign":            "unknown",
    "street_light":         "unknown",
    "traffic_cone":         "unknown",
    "road_block":           "unknown",
    "boom_barrier":         "unknown",
    "barrier_tape":         "unknown",
    "debris":               "unknown",
    "obstacle":             "unknown",
    "container":            "unknown",
    "barrel":               "unknown",
    "pipe":                 "unknown",
    "wire":                 "unknown",
    "animal":               "unknown",
    "outlier":              "unknown",
}

# Skip entirely — not terrain and not navigable
_GOOSE_SKIP_NAMES = {"sky", "undefined", "ego_vehicle"}


# ── Load goose_label_mapping.csv ─────────────────────────────────────────────

def _load_goose_label_mapping(goose_dir: Path) -> Dict[int, str]:
    """
    Parse goose_label_mapping.csv → {label_key: our_vocabulary_label}.

    CSV columns: class_name, label_key, has_instance, hex
    Returns empty dict if CSV is absent (all pixels treated as unknown).
    """
    csv_path = goose_dir / "goose_label_mapping.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found — all pixels will be 'unknown'.")
        return {}

    id_to_vocab: Dict[int, str] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("class_name", "").strip().lower()
            try:
                class_id = int(row.get("label_key", -1))
            except (ValueError, TypeError):
                continue
            if class_id < 0 or name in _GOOSE_SKIP_NAMES:
                continue
            vocab = _GOOSE_NAME_MAP.get(name, "unknown")
            id_to_vocab[class_id] = vocab

    print(f"  Loaded {len(id_to_vocab)} class mappings from {csv_path.name}")
    return id_to_vocab


# ── Label-based detector ──────────────────────────────────────────────────────

def _label_detect(
    image: np.ndarray,
    label_map: np.ndarray,
    id_to_vocab: Dict[int, str],
) -> DetectionResult:
    """
    Build a DetectionResult from a GOOSE uint8 semantic label map.

    Merges all class IDs that map to the same vocabulary label so the
    result has one RegionInfo per vocabulary class, not per GOOSE class.
    """
    h, w = image.shape[:2]
    total_pixels = h * w

    # Aggregate pixels per vocabulary label
    label_masks: Dict[str, np.ndarray] = {}
    covered_mask = np.zeros((h, w), dtype=bool)

    for class_id, vocab in id_to_vocab.items():
        mask = (label_map == class_id)
        if not np.any(mask):
            continue
        covered_mask |= mask
        if vocab == "unknown":
            continue
        if vocab in label_masks:
            label_masks[vocab] |= mask
        else:
            label_masks[vocab] = mask.copy()

    known_regions: List[RegionInfo] = []
    for vocab, mask in label_masks.items():
        pf = float(np.sum(mask)) / total_pixels
        known_regions.append(RegionInfo(
            label=vocab,
            mask=mask,
            confidence=0.90,
            pixel_fraction=pf,
            source="goose_gt",
            traversability=get_traversability(vocab),
        ))

    unknown_pixel_mask = ~covered_mask
    unknown_regions: List[RegionInfo] = []
    if np.any(unknown_pixel_mask):
        pf = float(np.sum(unknown_pixel_mask)) / total_pixels
        unknown_regions.append(RegionInfo(
            label="unknown",
            mask=unknown_pixel_mask,
            confidence=0.0,
            pixel_fraction=pf,
            source="goose_gt",
            traversability=0.0,
        ))

    tmap = TraversabilityMap.create(h, w)
    for r in known_regions:
        tmap = tmap.update_region(r.mask, r.label)

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


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _make_label_runner(config_path: str, id_to_vocab: Dict[int, str]):
    from system.env_uncertainty.runner import EnvironmentalUncertaintyRunner

    class _GooseLabelDetector:
        def __init__(self, id_to_vocab: Dict[int, str]):
            self._id_to_vocab = id_to_vocab
            self._label_map: Optional[np.ndarray] = None

        def set_label(self, label_map: np.ndarray) -> None:
            self._label_map = label_map

        def detect(self, image: np.ndarray) -> DetectionResult:
            assert self._label_map is not None, "call set_label() before detect()"
            return _label_detect(image, self._label_map, self._id_to_vocab)

    detector = _GooseLabelDetector(id_to_vocab)
    runner = EnvironmentalUncertaintyRunner(
        config_path=config_path,
        detector=detector,
    )
    return runner, detector


def _load_image(path: Path) -> Optional[np.ndarray]:
    try:
        import cv2
        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def _load_label(path: Path) -> Optional[np.ndarray]:
    """Load a GOOSE labelids PNG (uint8, pixel = class label_key)."""
    try:
        import cv2
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def _find_label_path(image_path: Path, label_base: Path) -> Optional[Path]:
    """
    Resolve the label file for a GOOSE visible-spectrum image.

    image_path: .../images/val/SESSION/SESSION__FRAME_TS_windshield_vis.png
    label:      .../labels/val/SESSION/SESSION__FRAME_TS_labelids.png
    """
    session = image_path.parent.name
    # Strip _windshield_vis or _windshield_nir suffix, append _labelids
    stem = image_path.stem
    for suffix in ("_windshield_vis", "_windshield_nir", "_vis", "_nir"):
        if stem.endswith(suffix):
            base = stem[: -len(suffix)]
            break
    else:
        base = stem
    candidate = label_base / session / f"{base}_labelids.png"
    return candidate if candidate.exists() else None


def _collect_images(img_base: Path, session_filter: Optional[str], n_images: int) -> List[Path]:
    """Walk session subdirs and return up to n_images visible-spectrum PNGs."""
    images: List[Path] = []
    sessions = sorted(img_base.iterdir()) if img_base.exists() else []
    for session_dir in sessions:
        if not session_dir.is_dir():
            continue
        if session_filter and session_filter not in session_dir.name:
            continue
        vis = sorted(session_dir.glob("*_windshield_vis.png"))
        images.extend(vis)
        if len(images) >= n_images:
            break
    return images[:n_images]


def _simulate_gt_feedback(runner, label_map: np.ndarray, id_to_vocab: Dict[int, str]) -> List[str]:
    """Propagate GT terrain classes into cross-frame PersistentTerrainKnowledge."""
    seen: Dict[str, bool] = {}
    for class_id, vocab in id_to_vocab.items():
        if vocab in ("unknown", "person") or vocab in seen:
            continue
        if not np.any(label_map == class_id):
            continue
        is_trav = TRAVERSABILITY_SCORES.get(vocab, 0.5) >= 0.50
        runner.terrain_knowledge.update_from_feedback(label=vocab, is_traversable=is_trav, confidence=0.85)
        seen[vocab] = is_trav
    return [f"{v}={'safe' if t else 'unsafe'}" for v, t in seen.items()]


# ── Safety-rate helper ───────────────────────────────────────────────────────

DANGER_THRESHOLD = 0.20  # GT min traversability below this → "dangerous frame"


def _compute_gt_min_trav(
    label_img: np.ndarray,
    goal_pixel: tuple,
    h: int,
    w: int,
    id_to_vocab: Dict[int, str],
) -> Optional[float]:
    """
    Compute the minimum GT traversability along the direct Bézier path to goal.

    Uses the GOOSE ground-truth label map (uint8 pixel values = class IDs) to
    determine actual terrain traversability at each waypoint on the planned path.
    This gives a per-frame "ground truth safety" signal independent of the robot's
    own uncertainty estimate — used to compute safety_rate.

    Args:
        label_img:   GT semantic label map, uint8, same H×W as the RGB image.
        goal_pixel:  (row, col) goal location in image coordinates.
        h, w:        Image dimensions.
        id_to_vocab: Mapping from GOOSE class ID (int) → our vocabulary label (str).

    Returns:
        Minimum traversability across all 20 direct-path waypoints, or None if
        the trajectory cannot be generated.
    """
    from system.env_uncertainty.trajectory import GoalDirectedTrajectoryGenerator

    start = (h - 1, w // 2)
    gen = GoalDirectedTrajectoryGenerator(h, w, n_waypoints=20)
    trajs = gen.generate_toward_goal(start, goal_pixel)
    if not trajs:
        return None

    # Use the "direct" trajectory — straight line to goal (index 0)
    direct = trajs[0]
    travs = []
    for py, px in direct.waypoints:
        py_c = int(np.clip(py, 0, h - 1))
        px_c = int(np.clip(px, 0, w - 1))
        class_id = int(label_img[py_c, px_c])
        vocab_label = id_to_vocab.get(class_id, "unknown")
        travs.append(get_traversability(vocab_label))

    return float(min(travs)) if travs else None


# ── Main pipeline function ────────────────────────────────────────────────────

def run_on_goose(
    goose_dir: str,
    split: str = "val",
    n_images: int = 10,
    verbosity: str = "standard",
    save_json: Optional[str] = None,
    config_path: Optional[str] = None,
    goal_row_fraction: float = 0.20,
    simulate_gt_feedback: bool = True,
    session_filter: Optional[str] = None,
) -> List[dict]:
    """
    Run the environmental uncertainty pipeline on GOOSE images.

    Args:
        goose_dir:          Path to GOOSE root (images/, labels/, CSV).
        split:              "val" or "train" (default: "val").
        n_images:           Max images to process.
        verbosity:          terse / standard / verbose.
        save_json:          If set, write results to this JSON path.
        config_path:        Path to system/env_uncertainty/config.yaml.
        goal_row_fraction:  Goal pixel row as fraction of image height (top 20%).
        simulate_gt_feedback: Propagate GT labels to cross-frame terrain knowledge.
        session_filter:     If set, only process sessions whose name contains
                            this string (e.g. "neubiberg", "aying").

    Returns:
        List of per-image result dicts.
    """
    goose_path = Path(goose_dir)
    img_base = goose_path / "images" / split
    label_base = goose_path / "labels" / split

    if not img_base.exists():
        print(f"ERROR: image directory not found: {img_base}")
        print("  Check that GOOSE is extracted to the correct path.")
        sys.exit(1)
    if not label_base.exists():
        print(f"ERROR: label directory not found: {label_base}")
        sys.exit(1)

    images = _collect_images(img_base, session_filter, n_images)
    if not images:
        print(f"ERROR: no *_windshield_vis.png images found under {img_base}")
        sys.exit(1)

    id_to_vocab = _load_goose_label_mapping(goose_path)
    if not id_to_vocab:
        print("  WARNING: no label mapping loaded — all pixels will be 'unknown'.")

    if config_path is None:
        config_path = str(PROJECT_ROOT / "system" / "env_uncertainty" / "config.yaml")

    profile = UserProfile(
        user_id="goose_eval",
        verbosity=verbosity,
        expertise="intermediate",
        preferred_format="question",
        name=f"GOOSE eval ({split}, {verbosity})",
    )

    runner, gt_detector = _make_label_runner(config_path, id_to_vocab)
    use_cross_frame = simulate_gt_feedback and bool(id_to_vocab)

    # Print session breakdown
    sessions_seen = sorted({img.parent.name for img in images})
    print(
        f"\nRunning pipeline on {len(images)} images from GOOSE/{split}"
        f"\nSessions: {', '.join(sessions_seen)}"
        f"\nVerbosity: {verbosity} | Cross-frame knowledge: {'on' if use_cross_frame else 'off'}"
        f"\n{'=' * 70}"
    )

    results = []
    n_proceed = n_ask = n_stop = 0
    n_dangerous = n_safe_on_dangerous = 0

    for img_path in images:
        img = _load_image(img_path)
        if img is None:
            print(f"  SKIP {img_path.name}: could not load image")
            continue

        lbl_path = _find_label_path(img_path, label_base)
        if lbl_path is None:
            print(f"  SKIP {img_path.name}: no label file found")
            continue
        lbl = _load_label(lbl_path)
        if lbl is None:
            print(f"  SKIP {img_path.name}: label load failed")
            continue

        gt_detector.set_label(lbl)

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
        if action == "PROCEED":    n_proceed += 1
        elif action == "ASK":      n_ask += 1
        else:                      n_stop += 1

        gt_min_trav = _compute_gt_min_trav(lbl, goal_pixel, h, w, id_to_vocab)
        if gt_min_trav is not None and gt_min_trav < DANGER_THRESHOLD:
            n_dangerous += 1
            if action in ("ASK", "STOP"):
                n_safe_on_dangerous += 1

        if use_cross_frame:
            _simulate_gt_feedback(runner, lbl, id_to_vocab)

        q_text = f'\n    Q: "{decision.question}"' if decision.question else ""
        known_labels = runner.terrain_knowledge.n_labels_known
        session_short = img_path.parent.name[-20:]
        print(
            f"[{session_short:20s}/{img_path.name[:20]:20s}]  {action:7s}  "
            f"unk={decision.unknown_coverage:.2f}  "
            f"known={decision.n_known_regions}  "
            f"({elapsed_ms:.0f} ms)  "
            f"[xframe: {known_labels}]{q_text}"
        )

        results.append({
            "image": img_path.name,
            "session": img_path.parent.name,
            "split": split,
            "robot_action": action,
            "unknown_coverage": round(decision.unknown_coverage, 4),
            "sam3_coverage": round(decision.sam3_coverage, 4),
            "n_known_regions": decision.n_known_regions,
            "n_unknown_regions": decision.n_unknown_regions,
            "question": decision.question,
            "goal_pixel": list(goal_pixel),
            "elapsed_ms": round(elapsed_ms, 1),
            "cross_frame_labels_known": runner.terrain_knowledge.n_labels_known,
            "gt_min_trav": round(gt_min_trav, 4) if gt_min_trav is not None else None,
            "gt_is_dangerous": (gt_min_trav is not None and gt_min_trav < DANGER_THRESHOLD),
        })

    total = len(results)
    safety_rate = round(n_safe_on_dangerous / n_dangerous, 4) if n_dangerous else None
    print(f"\n{'=' * 70}")
    print(f"Summary: {total} images | PROCEED={n_proceed}  ASK={n_ask}  STOP={n_stop}")
    if total:
        print(f"Help rate: {(n_ask + n_stop) / total:.1%}")
    if n_dangerous:
        print(
            f"Safety rate: {safety_rate:.1%}  "
            f"({n_safe_on_dangerous}/{n_dangerous} dangerous frames correctly caught)"
        )
    else:
        print("Safety rate: n/a (no GT-dangerous frames found in this run)")

    if use_cross_frame and runner.terrain_knowledge.n_labels_known > 0:
        print(f"\n{runner.terrain_knowledge.summary()}")

    if save_json and results:
        out = {
            "dataset": "GOOSE",
            "split": split,
            "sessions": sessions_seen,
            "n_images": total,
            "verbosity": verbosity,
            "detector": "goose_gt",
            "cross_frame_knowledge": use_cross_frame,
            "n_labels_learned": runner.terrain_knowledge.n_labels_known,
            "n_proceed": n_proceed,
            "n_ask": n_ask,
            "n_stop": n_stop,
            "help_rate": round((n_ask + n_stop) / total, 4) if total else 0.0,
            "n_dangerous_frames": n_dangerous,
            "n_safe_on_dangerous": n_safe_on_dangerous,
            "safety_rate": safety_rate,
            "per_image": results,
        }
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved → {save_json}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run uncertainty pipeline on GOOSE images")
    p.add_argument(
        "--goose_dir",
        default="data/datasets/GOOSE",
        help="Path to GOOSE root (default: data/datasets/GOOSE)",
    )
    p.add_argument(
        "--split", choices=["train", "val"], default="val",
        help="Dataset split (default: val)",
    )
    p.add_argument("--n_images", type=int, default=10, help="Max images (default: 10)")
    p.add_argument(
        "--verbosity", choices=["terse", "standard", "verbose"], default="standard",
    )
    p.add_argument("--save_json", default=None, help="Save results to this JSON file")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument(
        "--no_cross_frame", action="store_true", default=False,
        help="Disable cross-frame terrain knowledge (default: on)",
    )
    p.add_argument(
        "--goal_row", type=float, default=0.20,
        help="Goal pixel row as fraction of image height (default: 0.20)",
    )
    p.add_argument(
        "--session", default=None,
        help="Only process sessions whose name contains this string (e.g. 'neubiberg')",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_on_goose(
        goose_dir=args.goose_dir,
        split=args.split,
        n_images=args.n_images,
        verbosity=args.verbosity,
        save_json=args.save_json,
        config_path=args.config,
        goal_row_fraction=args.goal_row,
        simulate_gt_feedback=not args.no_cross_frame,
        session_filter=args.session,
    )
