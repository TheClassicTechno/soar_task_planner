"""
Unit tests for data_loader.py — no GPU, no RUGD download required.
Uses temporary directories with synthetic PNG files.
"""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image


from baselines.sam3.data_loader import (
    RUGD_CLASSES,
    RUGD_COLORMAP,
    RUGD_SPLIT_SCENES,
    SAM3_TO_RUGD,
    RUGDSample,
    load_rugd_split,
    build_class_index,
    _rgb_annotation_to_class_ids,
)


# ── helpers ────────────────────────────────────────────────────────────────

def _make_rgb_png(path: Path, w: int = 64, h: int = 48) -> None:
    """Write a tiny random RGB PNG to path."""
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _make_gray_png(path: Path, class_id: int = 3, w: int = 64, h: int = 48) -> None:
    """Write a grayscale annotation PNG filled with a single class_id."""
    arr = np.full((h, w), class_id, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _make_color_ann_png(path: Path, class_id: int = 3, w: int = 64, h: int = 48) -> None:
    """Write an RGB color-coded annotation PNG (official RUGD format) for one class."""
    from baselines.sam3.data_loader import RUGD_COLORMAP
    color = next(rgb for rgb, cid in RUGD_COLORMAP.items() if cid == class_id)
    arr = np.full((h, w, 3), color, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _build_official_layout(root: Path, split: str, n_per_scene: int = 3):
    """Create a minimal official RUGD directory tree."""
    from baselines.sam3.data_loader import RUGD_SPLIT_SCENES, RUGD_COLORMAP
    scenes = RUGD_SPLIT_SCENES[split][:2]  # use first 2 scenes only
    color = next(rgb for rgb, cid in RUGD_COLORMAP.items() if cid == 3)  # grass
    for scene in scenes:
        img_dir = root / "RUGD_frames-with-annotations" / scene
        ann_dir = root / "RUGD_annotations" / scene
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)
        for i in range(n_per_scene):
            name = f"{scene}_{i:05d}.png"
            _make_rgb_png(img_dir / name)
            arr = np.full((48, 64, 3), color, dtype=np.uint8)
            Image.fromarray(arr).save(ann_dir / name)


# ── RUGD_COLORMAP ───────────────────────────────────────────────────────────

def test_colormap_has_25_entries():
    assert len(RUGD_COLORMAP) == 25


def test_colormap_void_is_black():
    assert RUGD_COLORMAP[(0, 0, 0)] == 0


def test_colormap_all_ids_valid():
    valid = set(RUGD_CLASSES.keys())
    assert all(v in valid for v in RUGD_COLORMAP.values())


def test_rgb_annotation_to_class_ids_grass():
    # grass color is (0, 102, 0) → class ID 3
    ann_rgb = np.full((4, 4, 3), [0, 102, 0], dtype=np.uint8)
    result = _rgb_annotation_to_class_ids(ann_rgb)
    assert result.shape == (4, 4)
    assert (result == 3).all()


def test_rgb_annotation_to_class_ids_mixed():
    ann_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    ann_rgb[0, 0] = [108, 64, 20]   # dirt → 1
    ann_rgb[0, 1] = [255, 128, 0]   # gravel → 11
    ann_rgb[1, 0] = [0, 0, 0]       # void → 0
    ann_rgb[1, 1] = [64, 64, 64]    # asphalt → 10
    result = _rgb_annotation_to_class_ids(ann_rgb)
    assert result[0, 0] == 1
    assert result[0, 1] == 11
    assert result[1, 0] == 0
    assert result[1, 1] == 10


# ── RUGD_SPLIT_SCENES ────────────────────────────────────────────────────────

def test_split_scenes_has_three_splits():
    assert set(RUGD_SPLIT_SCENES.keys()) == {"train", "val", "test"}


def test_split_scenes_no_overlap():
    train = set(RUGD_SPLIT_SCENES["train"])
    val   = set(RUGD_SPLIT_SCENES["val"])
    test  = set(RUGD_SPLIT_SCENES["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_split_scenes_total_count():
    total = sum(len(v) for v in RUGD_SPLIT_SCENES.values())
    assert total == 18  # 11 train + 2 val + 5 test


# ── RUGD_CLASSES ────────────────────────────────────────────────────────────

def test_rugd_classes_has_24_entries():
    # void (0) + 24 named classes = 25 total entries
    assert len(RUGD_CLASSES) == 25


def test_rugd_classes_zero_is_void():
    assert RUGD_CLASSES[0] == "void"


def test_rugd_classes_all_strings():
    assert all(isinstance(v, str) for v in RUGD_CLASSES.values())


# ── SAM3_TO_RUGD ────────────────────────────────────────────────────────────

def test_sam3_to_rugd_all_queries_present():
    expected = [
        "sidewalk", "crosswalk", "road", "dirt", "vegetation", "grass",
        "gravel", "puddle", "wet surface", "cracked pavement", "curb",
        "mud", "slope",
    ]
    for q in expected:
        assert q in SAM3_TO_RUGD, f"Missing query in SAM3_TO_RUGD: {q}"


def test_sam3_to_rugd_all_ids_valid():
    valid_ids = set(RUGD_CLASSES.keys())
    for query, ids in SAM3_TO_RUGD.items():
        for cid in ids:
            assert cid in valid_ids, f"Invalid RUGD class ID {cid} for query '{query}'"


def test_sam3_to_rugd_non_empty_lists():
    for query, ids in SAM3_TO_RUGD.items():
        assert len(ids) > 0, f"Empty class list for query '{query}'"


# ── RUGDSample ──────────────────────────────────────────────────────────────

def test_rugd_sample_load_image(tmp_path):
    img_path = tmp_path / "frame_001.png"
    _make_rgb_png(img_path)
    sample = RUGDSample(img_path)
    img = sample.load_image()
    assert img.mode == "RGB"
    assert img.size == (64, 48)


def test_rugd_sample_load_annotation_grayscale(tmp_path):
    img_path = tmp_path / "frame_001.png"
    ann_path = tmp_path / "frame_001_ann.png"
    _make_rgb_png(img_path)
    _make_gray_png(ann_path, class_id=3)

    sample = RUGDSample(img_path, ann_path)
    ann = sample.load_annotation()

    assert ann is not None
    assert ann.shape == (48, 64)
    assert ann.dtype == np.uint8
    assert ann[0, 0] == 3


def test_rugd_sample_annotation_none_when_path_is_none(tmp_path):
    img_path = tmp_path / "frame_001.png"
    _make_rgb_png(img_path)
    sample = RUGDSample(img_path, annotation_path=None)
    assert sample.load_annotation() is None


def test_rugd_sample_annotation_none_when_file_missing(tmp_path):
    img_path = tmp_path / "frame_001.png"
    _make_rgb_png(img_path)
    missing = tmp_path / "nonexistent.png"
    sample = RUGDSample(img_path, missing)
    assert sample.load_annotation() is None


def test_rugd_sample_name(tmp_path):
    img_path = tmp_path / "creek_001.png"
    _make_rgb_png(img_path)
    sample = RUGDSample(img_path)
    assert sample.name == "creek_001"


def test_rugd_sample_load_annotation_rgb_color(tmp_path):
    img_path = tmp_path / "frame_001.png"
    ann_path = tmp_path / "frame_001_ann.png"
    _make_rgb_png(img_path)
    _make_color_ann_png(ann_path, class_id=3)  # grass

    sample = RUGDSample(img_path, ann_path, ann_is_rgb_color=True)
    ann = sample.load_annotation()

    assert ann is not None
    assert ann.shape == (48, 64)
    assert ann.dtype == np.uint8
    assert (ann == 3).all()


def test_rugd_sample_rgb_annotation_gravel(tmp_path):
    img_path = tmp_path / "frame_001.png"
    ann_path = tmp_path / "frame_001_ann.png"
    _make_rgb_png(img_path)
    _make_color_ann_png(ann_path, class_id=11)  # gravel

    sample = RUGDSample(img_path, ann_path, ann_is_rgb_color=True)
    ann = sample.load_annotation()
    assert (ann == 11).all()


# ── load_rugd_split ─────────────────────────────────────────────────────────

def _build_dataset_tools_layout(root: Path, split: str, n: int = 5):
    """Create a minimal dataset-tools directory tree with n images."""
    img_dir = root / split / "img"
    ann_dir = root / split / "ann"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    for i in range(n):
        _make_rgb_png(img_dir / f"frame_{i:03d}.png")
        _make_gray_png(ann_dir / f"frame_{i:03d}.png", class_id=i % 25)


def test_load_rugd_split_dataset_tools_layout(tmp_path):
    _build_dataset_tools_layout(tmp_path, "val", n=5)
    samples = load_rugd_split(str(tmp_path), split="val")
    assert len(samples) == 5
    # Each sample should have both image and annotation paths
    for s in samples:
        assert s.image_path.exists()
        assert s.annotation_path is not None


def test_load_rugd_split_max_samples(tmp_path):
    _build_dataset_tools_layout(tmp_path, "val", n=10)
    samples = load_rugd_split(str(tmp_path), split="val", max_samples=3)
    assert len(samples) == 3


def test_load_rugd_split_missing_path_raises():
    with pytest.raises(FileNotFoundError, match="RUGD data path not found"):
        load_rugd_split("/nonexistent/path/to/rugd", split="val")


def test_load_rugd_split_missing_split_raises(tmp_path):
    # Root exists but the split subdirectory does not
    (tmp_path / "val" / "img").mkdir(parents=True)  # only 'val' exists
    with pytest.raises(FileNotFoundError):
        load_rugd_split(str(tmp_path), split="train")


def test_load_rugd_split_official_layout(tmp_path):
    _build_official_layout(tmp_path, "val", n_per_scene=3)
    samples = load_rugd_split(str(tmp_path), split="val")
    # val has 2 scenes; we created 2 (first 2), 3 images each
    assert len(samples) == 6
    for s in samples:
        assert s.image_path.exists()
        assert s.annotation_path is not None
        assert s._ann_is_rgb_color is True


def test_load_rugd_split_official_annotations_decode(tmp_path):
    _build_official_layout(tmp_path, "val", n_per_scene=2)
    samples = load_rugd_split(str(tmp_path), split="val")
    ann = samples[0].load_annotation()
    assert ann is not None
    assert ann.dtype == np.uint8
    assert (ann == 3).all()  # all pixels are grass (class 3)


def test_load_rugd_split_official_max_samples(tmp_path):
    _build_official_layout(tmp_path, "val", n_per_scene=4)
    samples = load_rugd_split(str(tmp_path), split="val", max_samples=3)
    assert len(samples) == 3


def test_load_rugd_split_unrecognized_layout_raises(tmp_path):
    (tmp_path / "some_other_dir").mkdir()
    with pytest.raises(FileNotFoundError, match="Unrecognized RUGD layout"):
        load_rugd_split(str(tmp_path), split="val")


# ── build_class_index ────────────────────────────────────────────────────────

def test_build_class_index_basic():
    queries = ["road", "grass", "gravel"]
    idx = build_class_index(queries)
    assert idx == {"road": 0, "grass": 1, "gravel": 2}


def test_build_class_index_empty():
    assert build_class_index([]) == {}
