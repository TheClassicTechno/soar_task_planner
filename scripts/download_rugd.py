"""
Download RUGD dataset via the dataset-tools package.

Usage:
    pip install dataset-tools
    python scripts/download_rugd.py [--output_dir /path/to/rugd]

The dataset-tools package downloads RUGD into the structure:
    {output_dir}/train/img/   + {output_dir}/train/ann/
    {output_dir}/val/img/     + {output_dir}/val/ann/
    {output_dir}/test/img/    + {output_dir}/test/ann/

This matches the layout expected by baselines/sam3/data_loader.py.
"""

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download RUGD via dataset-tools")
    p.add_argument(
        "--output_dir",
        default=os.environ.get("RUGD_DATA_PATH", "/Users/julih/Documents/datasets/rugd"),
        help="Where to store the dataset (default: RUGD_DATA_PATH env or ~/Documents/datasets/rugd)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading RUGD to: {out}")
    print("This may take several minutes (~3.5 GB total).")

    try:
        import dataset_tools as dtools
    except ImportError:
        raise ImportError(
            "dataset-tools is not installed.\n"
            "Run: pip install dataset-tools"
        )

    dtools.download(dataset="RUGD", dst_dir=str(out))

    # Handle nested folder structure - move contents up
    inner_rugd = out / "rugd"
    if inner_rugd.exists():
        import shutil
        for item in inner_rugd.iterdir():
            dest = out / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        inner_rugd.rmdir()  # Remove empty rugd folder

    # Verify
    splits = ["train", "val", "test"]
    print("\nDownload complete. Verifying structure:")
    for split in splits:
        img_dir = out / split / "img"
        ann_dir = out / split / "ann"
        n_img = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
        n_ann = len(list(ann_dir.glob("*.json"))) if ann_dir.exists() else 0
        print(f"  {split}: {n_img} images, {n_ann} annotations")

    print(f"\nSet RUGD_DATA_PATH={out} in your .env file.")


if __name__ == "__main__":
    main()
