"""
RUGD dataset loader for the SAM3 baseline.

Handles two sources:
  1. dataset-tools download  (train/img + train/ann structure, grayscale labels)
  2. Official RUGD download  (RUGD_frames-with-annotations + RUGD_annotations,
                              scene-based folders, RGB color-coded labels)

RUGD has 24 semantic classes. We map our 13 SAM3 concept queries to
the closest RUGD class IDs so we can compute mIoU when GT labels exist.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ── RUGD 24-class definitions ──────────────────────────────────────────────
# Source: RUGD dataset paper (Wigness et al. 2019), Table 1.
RUGD_CLASSES: Dict[int, str] = {
    0:  "void",
    1:  "dirt",
    2:  "sand",
    3:  "grass",
    4:  "tree",
    5:  "pole",
    6:  "water",
    7:  "sky",
    8:  "vehicle",
    9:  "container/generic-object",
    10: "asphalt",
    11: "gravel",
    12: "building",
    13: "mulch",
    14: "rock/boulder",
    15: "log",
    16: "bicycle",
    17: "person",
    18: "fence",
    19: "bush",
    20: "sign",
    21: "rock",
    22: "bridge",
    23: "concrete",
    24: "picnic-table",
}

# ── RGB colormap from RUGD_annotation-colormap.txt ───────────────────────────
# Maps (R, G, B) tuple → class ID. Used to decode official RUGD annotations,
# which are stored as RGB color images (not grayscale class-ID images).
RUGD_COLORMAP: Dict[Tuple[int, int, int], int] = {
    (0,   0,   0):   0,   # void
    (108, 64,  20):  1,   # dirt
    (255, 229, 204): 2,   # sand
    (0,   102, 0):   3,   # grass
    (0,   255, 0):   4,   # tree
    (0,   153, 153): 5,   # pole
    (0,   128, 255): 6,   # water
    (0,   0,   255): 7,   # sky
    (255, 255, 0):   8,   # vehicle
    (255, 0,   127): 9,   # container/generic-object
    (64,  64,  64):  10,  # asphalt
    (255, 128, 0):   11,  # gravel
    (255, 0,   0):   12,  # building
    (153, 76,  0):   13,  # mulch
    (102, 102, 0):   14,  # rock-bed
    (102, 0,   0):   15,  # log
    (0,   255, 128): 16,  # bicycle
    (204, 153, 255): 17,  # person
    (102, 0,   204): 18,  # fence
    (255, 153, 204): 19,  # bush
    (0,   102, 102): 20,  # sign
    (153, 204, 255): 21,  # rock
    (102, 255, 255): 22,  # bridge
    (101, 101, 11):  23,  # concrete
    (114, 85,  47):  24,  # picnic-table
}

# ── Official RUGD train/val/test split (Wigness et al. 2019) ─────────────────
RUGD_SPLIT_SCENES: Dict[str, List[str]] = {
    "train": [
        "creek", "park-2", "trail", "trail-3", "trail-4",
        "trail-6", "trail-9", "trail-10", "trail-11", "trail-12", "village",
    ],
    "val":  ["park-1", "trail-5"],
    "test": ["park-8", "trail-7", "trail-13", "trail-14", "trail-15"],
}

# ── SAM3 query → RUGD class ID mapping ──────────────────────────────────────
SAM3_TO_RUGD: Dict[str, List[int]] = {
    "sidewalk":         [23],
    "crosswalk":        [10, 23],
    "road":             [10],
    "dirt":             [1],
    "vegetation":       [3, 19, 4],
    "grass":            [3],
    "gravel":           [11],
    "puddle":           [6],
    "wet surface":      [6, 10],
    "cracked pavement": [10, 23],
    "curb":             [23, 10],
    "mud":              [1, 2],
    "slope":            [1, 3, 11],
}

# Pre-built lookup array for fast RGB → class ID conversion (official format).
# Shape: (256, 256, 256) uint8. Built once at import time.
_RGB_TO_CLASS: Optional[np.ndarray] = None


def _get_rgb_lookup() -> np.ndarray:
    global _RGB_TO_CLASS
    if _RGB_TO_CLASS is None:
        lut = np.zeros((256, 256, 256), dtype=np.uint8)
        for (r, g, b), cls_id in RUGD_COLORMAP.items():
            lut[r, g, b] = cls_id
        _RGB_TO_CLASS = lut
    return _RGB_TO_CLASS


def _rgb_annotation_to_class_ids(ann_rgb: np.ndarray) -> np.ndarray:
    """Convert an (H, W, 3) uint8 RGB annotation to an (H, W) uint8 class-ID array."""
    lut = _get_rgb_lookup()
    return lut[ann_rgb[:, :, 0], ann_rgb[:, :, 1], ann_rgb[:, :, 2]]


class RUGDSample:
    """One image + optional GT annotation pair."""

    def __init__(
        self,
        image_path: Path,
        annotation_path: Optional[Path] = None,
        ann_is_rgb_color: bool = False,
    ):
        self.image_path = image_path
        self.annotation_path = annotation_path
        self._ann_is_rgb_color = ann_is_rgb_color

    def load_image(self) -> Image.Image:
        return Image.open(self.image_path).convert("RGB")

    def load_annotation(self) -> Optional[np.ndarray]:
        """Return (H, W) uint8 array of RUGD class IDs, or None if no GT."""
        if self.annotation_path is None or not self.annotation_path.exists():
            return None

        if self.annotation_path.suffix == ".json":
            # Load Supervisely JSON format (DatasetNinja layout)
            import json
            import base64
            import zlib
            import io

            with open(self.annotation_path) as f:
                data = json.load(f)

            height = data["size"]["height"]
            width = data["size"]["width"]
            ann = np.zeros((height, width), dtype=np.uint8)

            CLASS_NAME_TO_ID = {
                "void": 0, "dirt": 1, "sand": 2, "grass": 3, "tree": 4, "pole": 5, "water": 6, "sky": 7,
                "vehicle": 8, "container/generic-object": 9, "generic-object": 9, "container": 9,
                "asphalt": 10, "gravel": 11, "building": 12, "mulch": 13, "rock/boulder": 14, "rock-bed": 14,
                "log": 15, "bicycle": 16, "person": 17, "fence": 18, "bush": 19, "sign": 20, "rock": 21,
                "bridge": 22, "concrete": 23, "picnic-table": 24, "table": 24
            }

            for obj in data.get("objects", []):
                title = obj.get("classTitle", "")
                geom = obj.get("geometryType", "")
                class_id = CLASS_NAME_TO_ID.get(title.lower(), 0)

                if geom == "bitmap" and "bitmap" in obj:
                    origin = obj["bitmap"].get("origin", [0, 0])
                    origin_x, origin_y = origin[0], origin[1]
                    base64_data = obj["bitmap"].get("data", "")
                    if not base64_data:
                        continue

                    try:
                        png_bytes = zlib.decompress(base64.b64decode(base64_data))
                        img = Image.open(io.BytesIO(png_bytes))
                        img_np = np.array(img)
                        if img_np.ndim == 3 and img_np.shape[2] == 4:
                            mask = img_np[:, :, 3] > 0
                        else:
                            mask = img_np > 0

                        h_obj, w_obj = mask.shape
                        y1 = max(0, origin_y)
                        y2 = min(height, origin_y + h_obj)
                        x1 = max(0, origin_x)
                        x2 = min(width, origin_x + w_obj)

                        mask_y1 = y1 - origin_y
                        mask_y2 = y2 - origin_y
                        mask_x1 = x1 - origin_x
                        mask_x2 = x2 - origin_x

                        ann[y1:y2, x1:x2][mask[mask_y1:mask_y2, mask_x1:mask_x2]] = class_id
                    except Exception:
                        pass
            return ann

        ann = np.array(Image.open(self.annotation_path))
        if self._ann_is_rgb_color:
            # Official RUGD: RGB color-coded annotations
            if ann.ndim == 3:
                return _rgb_annotation_to_class_ids(ann)
            # Grayscale fallback (shouldn't happen with official data)
            return ann.astype(np.uint8)
        else:
            # dataset-tools: grayscale where pixel value = class ID
            if ann.ndim == 3:
                ann = ann[:, :, 0]
            return ann.astype(np.uint8)

    @property
    def name(self) -> str:
        return self.image_path.stem


def _collect_dataset_tools_samples(data_path: Path, split: str) -> List[RUGDSample]:
    """
    Load samples from the dataset-tools directory layout:
      {data_path}/{split}/img/*.png
      {data_path}/{split}/ann/*.png  (grayscale, pixel = class ID)
    """
    img_dir = data_path / split / "img"
    ann_dir = data_path / split / "ann"

    if not img_dir.exists():
        raise FileNotFoundError(f"RUGD img directory not found: {img_dir}")

    samples = []
    for img_path in sorted(img_dir.glob("*.png")):
        ann_path = None
        if ann_dir.exists():
            # Try png first
            p_png = ann_dir / img_path.name
            if p_png.exists():
                ann_path = p_png
            else:
                # Try .png.json
                p_json = ann_dir / (img_path.name + ".json")
                if p_json.exists():
                    ann_path = p_json
                else:
                    # Try .json
                    p_json_short = ann_dir / (img_path.stem + ".json")
                    if p_json_short.exists():
                        ann_path = p_json_short

        samples.append(RUGDSample(img_path, ann_path, ann_is_rgb_color=False))
    return samples


def _collect_official_samples(root: Path, split: str) -> List[RUGDSample]:
    """
    Load samples from the official RUGD layout:
      {root}/RUGD_frames-with-annotations/{scene}/*.png  (RGB images)
      {root}/RUGD_annotations/{scene}/*.png              (RGB color labels)

    Uses the official train/val/test scene assignment from Wigness et al. 2019.
    """
    frames_root = root / "RUGD_frames-with-annotations"
    ann_root = root / "RUGD_annotations"

    if not frames_root.exists():
        raise FileNotFoundError(f"RUGD frames directory not found: {frames_root}")

    scenes = RUGD_SPLIT_SCENES.get(split, [])
    if not scenes:
        raise ValueError(f"Unknown split '{split}'. Choose from: train, val, test")

    samples = []
    for scene in scenes:
        scene_img_dir = frames_root / scene
        scene_ann_dir = ann_root / scene
        if not scene_img_dir.exists():
            continue
        for img_path in sorted(scene_img_dir.glob("*.png")):
            ann_path = (
                scene_ann_dir / img_path.name
                if scene_ann_dir.exists()
                else None
            )
            samples.append(
                RUGDSample(img_path, ann_path, ann_is_rgb_color=True)
            )
    return samples


def load_rugd_split(
    data_path: str,
    split: str = "val",
    max_samples: Optional[int] = None,
) -> List[RUGDSample]:
    """
    Load a RUGD split from either layout (dataset-tools or official).

    Args:
        data_path: Root directory of the RUGD download.
                   Official layout: contains RUGD_frames-with-annotations/ and RUGD_annotations/
                   dataset-tools layout: contains {split}/img/ subdirectories
        split: One of "train", "val", "test".
        max_samples: Cap number of images (useful for quick smoke tests).

    Returns:
        List of RUGDSample objects.
    """
    root = Path(os.path.expanduser(data_path))
    if not root.exists():
        raise FileNotFoundError(
            f"RUGD data path not found: {root}\n"
            f"Download from http://rugd.vision/data/RUGD_frames-with-annotations.zip"
        )

    # Official layout: has RUGD_frames-with-annotations/ subdirectory
    if (root / "RUGD_frames-with-annotations").exists():
        samples = _collect_official_samples(root, split)
    elif (root / split / "img").exists():
        samples = _collect_dataset_tools_samples(root, split)
    else:
        raise FileNotFoundError(
            f"Unrecognized RUGD layout at: {root}\n"
            f"Expected either RUGD_frames-with-annotations/ or {split}/img/ subdirectory."
        )

    return samples[:max_samples]


def build_class_index(queries: List[str]) -> Dict[str, int]:
    """Map each SAM3 query string to its index in the queries list."""
    return {q: i for i, q in enumerate(queries)}
