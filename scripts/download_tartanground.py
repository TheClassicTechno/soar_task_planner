"""
Download a minimal TartanGround subset for pipeline testing.

Downloads OldTownSummer / P0000 (omni robot, front camera only):
  - image_lcam_front.zip   ~353 MB  (RGB frames)
  - seg_lcam_front.zip     ~8 MB    (semantic segmentation IDs)
  - metadata.zip           ~1 MB    (poses, heights)
  - seg_labels.zip         ~0.002 MB (class ID → name mapping)

Total: ~362 MB

Uses huggingface_hub directly — no CUDA / cupy required.

Usage:
    python scripts/download_tartanground.py
    python scripts/download_tartanground.py --out_dir /path/to/tartanground
    python scripts/download_tartanground.py --env ForestEnv --traj P0000
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_tartanground(
    out_dir: str = "~/Documents/datasets/TartanGround",
    env: str = "OldTownSummer",
    traj: str = "P0000",
    robot: str = "omni",
    unzip: bool = True,
) -> Path:
    """
    Download RGB + seg + metadata for one TartanGround trajectory.

    Args:
        out_dir:  Local root directory (created if missing).
        env:      Environment name (e.g. "OldTownSummer", "ForestEnv", "Gascola").
        traj:     Trajectory ID (omni: "P0000"–"P0003", diff: "P1000"+, anymal: "P2000"+).
        robot:    Robot type: "omni", "diff", or "anymal".
        unzip:    If True, extract zips after download.

    Returns:
        Path to the downloaded trajectory directory.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    repo_id = "theairlabcmu/TartanGround"
    robot_dir = f"Data_{robot}"
    traj_prefix = f"{env}/{robot_dir}/{traj}"
    seg_labels_path = f"{env}/seg_labels.zip"

    patterns = [
        f"{traj_prefix}/image_lcam_front.zip",
        f"{traj_prefix}/seg_lcam_front.zip",
        f"{traj_prefix}/metadata.zip",
        seg_labels_path,
    ]

    out_path = Path(os.path.expanduser(out_dir))
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading TartanGround subset from HuggingFace ({repo_id})")
    print(f"  Environment : {env}")
    print(f"  Trajectory  : {traj} ({robot} robot)")
    print(f"  Camera      : lcam_front")
    print(f"  Modalities  : image + seg + metadata + seg_labels")
    print(f"  Output      : {out_path}")
    print(f"  Est. size   : ~362 MB\n")

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(out_path),
            allow_patterns=patterns,
        )
        print(f"\nDownload complete → {local_dir}")
    except Exception as e:
        print(f"ERROR: Download failed: {e}")
        sys.exit(1)

    traj_dir = out_path / env / robot_dir / traj

    if unzip:
        print("\nExtracting zip files...")
        for pattern in patterns:
            zip_path = out_path / pattern
            if not zip_path.exists():
                print(f"  SKIP (not found): {pattern}")
                continue
            dest = zip_path.parent
            print(f"  Extracting {zip_path.name} → {dest}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest)

    return traj_dir


def _summarize(traj_dir: Path) -> None:
    """Print a summary of what was downloaded."""
    img_dir = traj_dir / "image_lcam_front"
    seg_dir = traj_dir / "seg_lcam_front"

    if img_dir.exists():
        imgs = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
        print(f"\n  RGB images  : {len(imgs)} frames in {img_dir}")
        if imgs:
            print(f"    First: {imgs[0].name}  Last: {imgs[-1].name}")
    else:
        print(f"\n  RGB dir not found: {img_dir}")

    if seg_dir.exists():
        segs = sorted(seg_dir.glob("*.png")) + sorted(seg_dir.glob("*.npy"))
        print(f"  Seg masks   : {len(segs)} frames in {seg_dir}")
    else:
        print(f"  Seg dir not found: {seg_dir}")

    label_file = traj_dir.parent.parent / "seg_label_map.json"
    if not label_file.exists():
        # Try zip-extracted location variants
        for candidate in [
            traj_dir.parent.parent / "seg_labels" / "seg_label_map.json",
            traj_dir.parent.parent.parent / "seg_label_map.json",
        ]:
            if candidate.exists():
                label_file = candidate
                break
    if label_file.exists():
        print(f"  Label map   : {label_file}")
    else:
        print(f"  Label map   : not found (check seg_labels.zip extraction)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download minimal TartanGround subset")
    p.add_argument(
        "--out_dir",
        default="~/Documents/datasets/TartanGround",
        help="Root output directory",
    )
    p.add_argument(
        "--env", default="OldTownSummer",
        help="Environment name (default: OldTownSummer ~362 MB). "
             "Outdoor alternatives: ForestEnv (~2.4 GB), Gascola (~914 MB), "
             "SeasonalForestAutumn (~1.23 GB)",
    )
    p.add_argument("--traj", default="P0000", help="Trajectory ID (default: P0000)")
    p.add_argument(
        "--robot", default="omni", choices=["omni", "diff", "anymal"],
        help="Robot type (default: omni)",
    )
    p.add_argument(
        "--no_unzip", action="store_true", default=False,
        help="Skip extraction of downloaded zip files",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    traj_dir = download_tartanground(
        out_dir=args.out_dir,
        env=args.env,
        traj=args.traj,
        robot=args.robot,
        unzip=not args.no_unzip,
    )
    _summarize(traj_dir)
