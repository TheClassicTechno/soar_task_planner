"""
Mine GOOSE val label maps to find the best real frame for each scenario.

For each of the 7 scenarios in scenarios.py, this script:
  1. Scans every *_labelids.png in data/datasets/GOOSE/labels/val/
  2. Computes terrain-class coverage fractions using the GOOSE→vocab mapping
  3. Scores each frame against each scenario's matching criteria
  4. Picks the highest-scoring frame per scenario
  5. Updates scenarios.py with `source_image` field on each Scenario

Matching criteria per scenario:
  unrecognized_patch     — unknown_coverage 0.15–0.45 + concrete/sidewalk present
  mud_or_gravel_ambiguity— gravel or soil (dirt) present + low unknown
  wet_grass_low_lcb      — high grass coverage (>30%) + low unknown (<15%)
  clear_sidewalk         — high concrete coverage (>40%) + minimal unknown (<5%)
  dry_gravel_path        — high gravel+dirt coverage (>30%) + minimal unknown (<5%)
  campus_crosswalk       — high concrete coverage (>40%) + minimal unknown (<5%)
  flooded_trail          — very high unknown (>0.70) OR water present + high unknown

Usage:
    python scripts/link_scenarios_to_images.py
    python scripts/link_scenarios_to_images.py --dry_run   (print matches, don't edit)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GOOSE_DIR = PROJECT_ROOT / "data" / "datasets" / "GOOSE"
SCENARIOS_PY = PROJECT_ROOT / "system" / "env_uncertainty" / "scenarios.py"


# ── GOOSE vocab mapping (inline so this script is standalone) ─────────────────

_GOOSE_NAME_MAP: Dict[str, str] = {
    "asphalt": "concrete", "sidewalk": "concrete", "bikeway": "concrete",
    "pedestrian_crossing": "concrete", "road_marking": "concrete",
    "curb": "concrete", "cobble": "gravel", "rail_track": "concrete",
    "gravel": "gravel", "soil": "dirt",
    "low_grass": "grass", "high_grass": "grass",
    "forest": "vegetation", "bush": "vegetation", "moss": "vegetation",
    "crops": "vegetation", "scenery_vegetation": "vegetation",
    "hedge": "vegetation", "leaves": "vegetation",
    "tree_crown": "tree", "tree_trunk": "tree", "tree_root": "tree",
    "rock": "rock-bed", "snow": "unknown",
    "water": "water",
    "person": "person", "rider": "person",
}
_SKIP = {"sky", "undefined", "ego_vehicle"}


def _load_id_to_vocab(goose_dir: Path) -> Dict[int, str]:
    csv_path = goose_dir / "goose_label_mapping.csv"
    mapping: Dict[int, str] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("class_name", "").strip().lower()
            try:
                class_id = int(row.get("label_key", -1))
            except (ValueError, TypeError):
                continue
            if class_id < 0 or name in _SKIP:
                continue
            mapping[class_id] = _GOOSE_NAME_MAP.get(name, "unknown")
    return mapping


def _coverage(label_img: np.ndarray, id_to_vocab: Dict[int, str]) -> Dict[str, float]:
    """Return fractional coverage per vocab label (including 'unknown' for unmapped)."""
    total = label_img.size
    counts: Dict[str, int] = {}
    for class_id, vocab in id_to_vocab.items():
        n = int(np.sum(label_img == class_id))
        if n:
            counts[vocab] = counts.get(vocab, 0) + n
    # pixels not covered by any mapped class → unknown
    covered = sum(counts.values())
    counts["unknown"] = counts.get("unknown", 0) + (total - covered)
    return {k: v / total for k, v in counts.items()}


# ── Per-scenario scoring functions ────────────────────────────────────────────

def _score(cov: Dict[str, float], criteria_fn) -> float:
    return criteria_fn(cov)


def _score_unrecognized_patch(cov: Dict[str, float]) -> float:
    unk = cov.get("unknown", 0)
    concrete = cov.get("concrete", 0)
    # Want: unknown 0.15–0.45, some concrete/sidewalk present
    unk_ok = 1.0 - abs(unk - 0.30) / 0.15 if 0.10 <= unk <= 0.50 else 0.0
    return unk_ok * (1.0 + concrete)


def _score_mud_or_gravel_ambiguity(cov: Dict[str, float]) -> float:
    unk = cov.get("unknown", 0)
    gravel = cov.get("gravel", 0)
    dirt = cov.get("dirt", 0)
    if unk > 0.20:
        return 0.0
    return (gravel + dirt) * (1.0 - unk * 3)


def _score_wet_grass_low_lcb(cov: Dict[str, float]) -> float:
    unk = cov.get("unknown", 0)
    grass = cov.get("grass", 0)
    if unk > 0.20:
        return 0.0
    return grass * max(0, 1.0 - unk * 5)


def _score_clear_sidewalk(cov: Dict[str, float]) -> float:
    unk = cov.get("unknown", 0)
    concrete = cov.get("concrete", 0)
    if unk > 0.08:
        return 0.0
    return concrete * max(0, 1.0 - unk * 10)


def _score_dry_gravel_path(cov: Dict[str, float]) -> float:
    unk = cov.get("unknown", 0)
    gravel = cov.get("gravel", 0)
    dirt = cov.get("dirt", 0)
    if unk > 0.08:
        return 0.0
    return (gravel + dirt) * max(0, 1.0 - unk * 10)


def _score_campus_crosswalk(cov: Dict[str, float]) -> float:
    # Same concrete requirement as clear_sidewalk but with higher concrete threshold
    # to prefer dense urban/campus scenes over rural roads
    unk = cov.get("unknown", 0)
    concrete = cov.get("concrete", 0)
    if unk > 0.08 or concrete < 0.30:
        return 0.0
    # Bonus for very high concrete density (campus/urban feel)
    return concrete * concrete * max(0, 1.0 - unk * 10)


def _score_flooded_trail(cov: Dict[str, float]) -> float:
    unk = cov.get("unknown", 0)
    water = cov.get("water", 0)
    # Prefer high unknown + any water
    return unk * 2 + water * 3


_SCORERS = {
    "unrecognized_patch":      _score_unrecognized_patch,
    "mud_or_gravel_ambiguity": _score_mud_or_gravel_ambiguity,
    "wet_grass_low_lcb":       _score_wet_grass_low_lcb,
    "clear_sidewalk":          _score_clear_sidewalk,
    "dry_gravel_path":         _score_dry_gravel_path,
    "campus_crosswalk":        _score_campus_crosswalk,
    "flooded_trail":           _score_flooded_trail,
}

# Preferred sessions per scenario (for tie-breaking — not exclusive)
_PREFERRED_SESSIONS = {
    "unrecognized_patch":      {"2022-09-21_garching", "2023-05-15_neubiberg_rain"},
    "mud_or_gravel_ambiguity": {"2022-08-30_siegertsbrunn_feldwege", "2022-12-07_aying_hills"},
    "wet_grass_low_lcb":       {"2023-01-20_aying_mangfall_2", "2022-08-30_siegertsbrunn_feldwege"},
    "clear_sidewalk":          {"2022-12-07_aying_hills", "2023-01-20_aying_mangfall_2"},
    "dry_gravel_path":         {"2022-08-30_siegertsbrunn_feldwege", "2022-07-22_flight"},
    "campus_crosswalk":        {"2023-03-03_garching_2", "2022-09-21_garching", "2023-05-17_neubiberg_sunny"},
    "flooded_trail":           {"2023-01-20_aying_mangfall_2", "2022-08-30_siegertsbrunn_feldwege"},
}


# ── Scan all label maps ────────────────────────────────────────────────────────

def scan_goose(goose_dir: Path) -> List[Tuple[Path, Dict[str, float], str]]:
    """
    Return list of (label_path, coverage_dict, session_name) for every labelids.png.
    """
    try:
        import cv2
    except ImportError:
        print("ERROR: cv2 not available. Install opencv-python.")
        sys.exit(1)

    id_to_vocab = _load_id_to_vocab(goose_dir)
    label_base = goose_dir / "labels" / "val"
    results = []
    sessions = sorted(label_base.iterdir())
    for session_dir in sessions:
        if not session_dir.is_dir():
            continue
        label_files = sorted(session_dir.glob("*_labelids.png"))
        for lp in label_files:
            img = cv2.imread(str(lp), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            cov = _coverage(img, id_to_vocab)
            results.append((lp, cov, session_dir.name))
    print(f"  Scanned {len(results)} label maps across {len(sessions)} sessions.")
    return results


# ── Find best match per scenario ──────────────────────────────────────────────

def find_best_matches(
    frames: List[Tuple[Path, Dict[str, float], str]],
) -> Dict[str, Dict]:
    """
    Return {scenario_name: {label_path, image_path, session, coverage, score}}.

    Each scenario gets a unique label file — once a frame is assigned to the
    highest-scoring scenario it is excluded from the remaining ones.
    """
    # First pass: score every (scenario, frame) pair
    all_scores: Dict[str, List[Tuple[float, Path, str]]] = {name: [] for name in _SCORERS}

    for label_path, cov, session in frames:
        for scenario_name, scorer in _SCORERS.items():
            score = scorer(cov)
            preferred = _PREFERRED_SESSIONS.get(scenario_name, set())
            if any(p in session for p in preferred):
                score *= 1.20
            if score > 0:
                all_scores[scenario_name].append((score, label_path, session))

    for name in all_scores:
        all_scores[name].sort(key=lambda x: -x[0])

    # Second pass: greedy assignment — highest-priority scenario gets its top frame;
    # that frame is then unavailable to lower-priority scenarios.
    used: set = set()
    best: Dict[str, Dict] = {}

    for scenario_name in _SCORERS:  # preserve declaration order
        for score, label_path, session in all_scores[scenario_name]:
            if label_path in used:
                continue
            used.add(label_path)
            stem = label_path.stem.replace("_labelids", "")
            # label_path: .../GOOSE/labels/val/SESSION/FILE_labelids.png
            # parents[3] = .../GOOSE/
            img_path = (
                label_path.parents[3] / "images" / "val"
                / session / f"{stem}_windshield_vis.png"
            )
            # Retrieve coverage dict for this label path
            cov = next(c for lp, c, s in frames if lp == label_path)
            best[scenario_name] = {
                "score": score,
                "label_path": label_path,
                "image_path": img_path,
                "session": session,
                "coverage": cov,
            }
            break
        else:
            best[scenario_name] = {"score": -1.0}

    return best


# ── Patch scenarios.py with source_image ──────────────────────────────────────

def _relative_image_path(img_path: Path) -> str:
    """Return path relative to PROJECT_ROOT for storage in scenarios.py."""
    try:
        return str(img_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(img_path)


def patch_scenarios_py(matches: Dict[str, Dict]) -> None:
    """
    Add/update `source_image=` field in each Scenario() block in scenarios.py.
    - If `source_image=` already exists in the block, replaces it.
    - Otherwise inserts after `goal_description=`.
    """
    text = SCENARIOS_PY.read_text()

    for scenario_name, info in matches.items():
        if info["score"] <= 0:
            print(f"  WARNING: no positive-score match for '{scenario_name}' — skipping")
            continue

        rel_path = _relative_image_path(info["image_path"])

        # Find block by name= field; end of block is the first ),  after it
        block_pattern = rf'(name="{re.escape(scenario_name)}".*?)(    \),)'
        match = re.search(block_pattern, text, re.DOTALL)
        if not match:
            print(f"  WARNING: could not find scenario block for '{scenario_name}'")
            continue

        block_body = match.group(1)
        closer = match.group(2)

        if "source_image=" in block_body:
            # Replace existing source_image line within this block
            new_body = re.sub(
                r'        source_image=[^\n]+\n',
                f'        source_image="{rel_path}",\n',
                block_body,
                count=1,
            )
            text = text[:match.start()] + new_body + closer + text[match.end():]
            print(f"  Updated source_image for '{scenario_name}'")
        else:
            # Insert after goal_description= line
            goal_match = re.search(r'(        goal_description=[^\n]+\n)', block_body)
            if not goal_match:
                print(f"  WARNING: no goal_description line for '{scenario_name}'")
                continue
            insert_pos = match.start() + goal_match.end()
            text = text[:insert_pos] + f'        source_image="{rel_path}",\n' + text[insert_pos:]
            print(f"  Added source_image for '{scenario_name}'")

    SCENARIOS_PY.write_text(text)


def add_source_image_field_to_dataclass() -> None:
    """Add optional source_image field to Scenario dataclass if not already present."""
    text = SCENARIOS_PY.read_text()
    if "source_image:" in text:
        return  # already there

    # Add after goal_description field
    old = "    goal_description: str\n"
    new = "    goal_description: str\n    source_image: Optional[str] = None\n"
    if old in text:
        text = text.replace(old, new, 1)
        SCENARIOS_PY.write_text(text)
        print("  Added `source_image: Optional[str] = None` field to Scenario dataclass.")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(dry_run: bool = False) -> None:
    print("\nScenario → Real Image Linker")
    print(f"  GOOSE dir: {GOOSE_DIR}")

    if not GOOSE_DIR.exists():
        print("  ERROR: GOOSE directory not found. Cannot link scenarios to images.")
        sys.exit(1)

    print("\nStep 1: Scanning GOOSE label maps…")
    frames = scan_goose(GOOSE_DIR)

    print("\nStep 2: Scoring frames against scenario criteria…")
    matches = find_best_matches(frames)

    print("\nStep 3: Best matches per scenario:")
    print(f"  {'Scenario':<30} {'Score':>6}  {'Session':<35}  Coverage summary")
    print("  " + "-" * 110)
    for name, info in matches.items():
        if info["score"] <= 0:
            print(f"  {name:<30} {'N/A':>6}  (no match found)")
            continue
        cov = info["coverage"]
        top = sorted(cov.items(), key=lambda x: -x[1])[:4]
        cov_str = "  ".join(f"{k}={v:.2f}" for k, v in top if v > 0.01)
        print(f"  {name:<30} {info['score']:>6.3f}  {info['session']:<35}  {cov_str}")
        if not dry_run:
            print(f"    → {_relative_image_path(info['image_path'])}")

    if dry_run:
        print("\n[dry_run] No files modified.")
        return

    print("\nStep 4: Patching scenarios.py with source_image fields…")
    add_source_image_field_to_dataclass()
    patch_scenarios_py(matches)

    print("\nDone. Run `pytest system/env_uncertainty/tests/` to verify no regressions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link scenarios to real GOOSE images")
    parser.add_argument("--dry_run", action="store_true", help="Print matches but don't modify files")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
