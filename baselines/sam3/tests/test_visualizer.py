"""
Unit tests for visualizer.py — no GPU, uses synthetic numpy images.
"""

import os
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from baselines.sam3.visualizer import (
    overlay_masks,
    draw_legend,
    save_visualization,
    load_color_map,
)


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def bgr_image():
    """64×48 solid gray BGR image."""
    return np.full((48, 64, 3), 128, dtype=np.uint8)


@pytest.fixture
def color_map():
    """3-class BGR color map."""
    return np.array([[255, 255, 0], [0, 0, 255], [0, 255, 0]], dtype=np.uint8)


@pytest.fixture
def queries():
    return ["road", "grass", "gravel"]


def _make_mask(H, W, fill=True):
    m = np.zeros((H, W), dtype=bool)
    if fill:
        m[:H//2, :] = True
    return m


def _make_result(mask_array, labels, scores=None):
    if scores is None:
        scores = np.ones(len(labels), dtype=float)
    return {"masks": mask_array, "labels": np.array(labels), "scores": np.array(scores)}


# ── load_color_map ────────────────────────────────────────────────────────────

def test_load_color_map_shape():
    raw = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    cm = load_color_map(raw)
    assert cm.shape == (3, 3)
    assert cm.dtype == np.uint8


def test_load_color_map_values():
    raw = [[100, 200, 50]]
    cm = load_color_map(raw)
    assert list(cm[0]) == [100, 200, 50]


# ── overlay_masks ─────────────────────────────────────────────────────────────

def test_overlay_masks_returns_same_shape(bgr_image, color_map, queries):
    H, W = bgr_image.shape[:2]
    mask = _make_mask(H, W, fill=True)
    result = _make_result(np.stack([mask]), labels=[0])

    out = overlay_masks(bgr_image, result, queries, color_map, alpha=0.5)
    assert out.shape == bgr_image.shape
    assert out.dtype == np.uint8


def test_overlay_masks_no_masks_unchanged(bgr_image, color_map, queries):
    result = _make_result(np.array([]), labels=[], scores=[])
    out = overlay_masks(bgr_image, result, queries, color_map, alpha=0.5)
    np.testing.assert_array_equal(out, bgr_image)


def test_overlay_masks_changes_masked_pixels(color_map, queries):
    # Use a black background so blended result = class color exactly at alpha=1.0
    H, W = 48, 64
    black = np.zeros((H, W, 3), dtype=np.uint8)
    mask = _make_mask(H, W, fill=True)  # top half True
    result = _make_result(np.stack([mask]), labels=[0])  # class 0 = [255,255,0]

    out = overlay_masks(black, result, queries, color_map, alpha=1.0)
    # Every pixel in the top half (masked area) must match the class color
    expected = color_map[0]  # [255, 255, 0]
    masked_pixels = out[:H // 2, :, :]
    assert np.all(masked_pixels == expected), (
        f"Masked pixels should be {expected}, got unique rows: "
        f"{np.unique(masked_pixels.reshape(-1, 3), axis=0)}"
    )


def test_overlay_masks_leaves_unmasked_pixels_unchanged(bgr_image, color_map, queries):
    H, W = bgr_image.shape[:2]
    mask = _make_mask(H, W, fill=True)  # top half masked
    result = _make_result(np.stack([mask]), labels=[0])

    out = overlay_masks(bgr_image, result, queries, color_map, alpha=1.0)
    # Bottom half (unmasked) should be unchanged
    np.testing.assert_array_equal(out[H//2:], bgr_image[H//2:])


# ── draw_legend ───────────────────────────────────────────────────────────────

def test_draw_legend_returns_same_shape(bgr_image, color_map, queries):
    out = draw_legend(bgr_image.copy(), queries, color_map, active_labels=[0, 1])
    assert out.shape == bgr_image.shape


def test_draw_legend_modifies_image(color_map, queries):
    # Must be wide enough for the legend (x_start = W-220, min-clamped to 2)
    W, H = 640, 480
    large_image = np.full((H, W, 3), 128, dtype=np.uint8)
    original = large_image.copy()
    out = draw_legend(large_image, queries, color_map, active_labels=[0])
    # At least one pixel should have changed after drawing the legend
    assert not np.array_equal(out, original)


# ── save_visualization ────────────────────────────────────────────────────────

def test_save_visualization_creates_file(tmp_path, color_map, queries):
    H, W = 48, 64
    pil_image = Image.fromarray(np.full((H, W, 3), 100, dtype=np.uint8))
    mask = _make_mask(H, W, fill=True)
    result = _make_result(np.stack([mask]), labels=[0])

    out_path = str(tmp_path / "output" / "vis.png")
    save_visualization(pil_image, result, queries, color_map, out_path)

    assert Path(out_path).exists()


def test_save_visualization_creates_parent_dirs(tmp_path, color_map, queries):
    H, W = 48, 64
    pil_image = Image.fromarray(np.full((H, W, 3), 50, dtype=np.uint8))
    result = _make_result(np.array([]), labels=[])

    deep_path = str(tmp_path / "a" / "b" / "c" / "vis.png")
    save_visualization(pil_image, result, queries, color_map, deep_path)
    assert Path(deep_path).exists()


def test_save_visualization_output_is_valid_image(tmp_path, color_map, queries):
    H, W = 48, 64
    pil_image = Image.fromarray(np.full((H, W, 3), 200, dtype=np.uint8))
    result = _make_result(np.array([]), labels=[])

    out_path = str(tmp_path / "vis.png")
    save_visualization(pil_image, result, queries, color_map, out_path)

    # Read back and verify dimensions
    loaded = Image.open(out_path)
    assert loaded.size == (W, H)
