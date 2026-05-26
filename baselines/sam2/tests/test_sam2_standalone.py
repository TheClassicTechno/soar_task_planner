"""
Unit tests for baselines/sam2/sam2_standalone.py.

We mock the third-party 'sam2' library to avoid needing GPU or local weight files during testing.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image
import yaml

# Mock the sam2 module dependencies to allow running tests in any environment
mock_build_sam_module = MagicMock()
mock_generator_module = MagicMock()

mock_build_sam2 = MagicMock()
mock_generator_cls = MagicMock()

mock_build_sam_module.build_sam2 = mock_build_sam2
mock_generator_module.SAM2AutomaticMaskGenerator = mock_generator_cls

sys.modules["sam2"] = MagicMock()
sys.modules["sam2.build_sam"] = mock_build_sam_module
sys.modules["sam2.automatic_mask_generator"] = mock_generator_module

from baselines.sam2.sam2_standalone import (
    SAM2Baseline,
    _select_device,
    _filter_overlapping_masks,
    load_config,
)


# ── Test helper functions ─────────────────────────────────────────────────────

def test_select_device_explicit():
    assert _select_device("cpu") == "cpu"
    assert _select_device("cuda") == "cuda"
    assert _select_device("mps") == "mps"


@patch("baselines.sam2.sam2_standalone._TORCH_AVAILABLE", False)
def test_select_device_torch_missing():
    with pytest.raises(ImportError, match="pip install torch"):
        _select_device("auto")


@patch("baselines.sam2.sam2_standalone._TORCH_AVAILABLE", True)
def test_select_device_auto():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    
    with patch("baselines.sam2.sam2_standalone.torch", mock_torch):
        assert _select_device("auto") == "cuda"

    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    with patch("baselines.sam2.sam2_standalone.torch", mock_torch):
        assert _select_device("auto") == "mps"

    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    with patch("baselines.sam2.sam2_standalone.torch", mock_torch):
        assert _select_device("auto") == "cpu"


def test_filter_overlapping_masks_empty():
    masks = np.zeros((0, 32, 32), dtype=bool)
    scores = np.zeros(0, dtype=np.float32)
    m_out, s_out = _filter_overlapping_masks(masks, scores)
    assert len(m_out) == 0
    assert len(s_out) == 0


def test_filter_overlapping_masks_disjoint():
    # Two masks in different parts of the image
    masks = np.zeros((2, 10, 10), dtype=bool)
    masks[0, 0:3, 0:3] = True
    masks[1, 7:10, 7:10] = True
    scores = np.array([0.9, 0.8], dtype=np.float32)

    m_out, s_out = _filter_overlapping_masks(masks, scores, iou_threshold=0.5)
    assert len(m_out) == 2
    assert np.array_equal(s_out, scores)


def test_filter_overlapping_masks_overlap():
    # Two masks overlapping heavily. One has score 0.9, other 0.8.
    masks = np.zeros((2, 10, 10), dtype=bool)
    masks[0, 2:8, 2:8] = True  # 36 pixels
    masks[1, 3:8, 2:8] = True  # 30 pixels (heavily overlapping)
    scores = np.array([0.8, 0.9], dtype=np.float32)

    # Intersection is 30 pixels. Union is 36. IoU = 30/36 = 0.83 > 0.5 threshold.
    # The mask with score 0.9 should be kept, and score 0.8 removed.
    m_out, s_out = _filter_overlapping_masks(masks, scores, iou_threshold=0.5)
    assert len(m_out) == 1
    assert s_out[0] == pytest.approx(0.9)
    assert np.array_equal(m_out[0], masks[1])


# ── Test SAM2Baseline Class ───────────────────────────────────────────────────

@pytest.fixture
def mock_config_path(tmp_path):
    config = {
        "sam2": {
            "model_cfg": "sam2_hiera_t.yaml",
            "checkpoint": "checkpoints/sam2_hiera_tiny.pt",
            "device": "cpu",
            "points_per_side": 16,
            "detection_threshold": 0.6,
            "iou_threshold": 0.8,
        }
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(config))
    return str(cfg_file)


@patch("baselines.sam2.sam2_standalone._TORCH_AVAILABLE", True)
def test_sam2_baseline_init(mock_config_path):
    mock_build_sam2.reset_mock()
    mock_generator_cls.reset_mock()

    mock_model_instance = MagicMock()
    mock_build_sam2.return_value = mock_model_instance

    baseline = SAM2Baseline(config_path=mock_config_path, device="cpu")

    # Verify build_sam2 called with config parameters
    mock_build_sam2.assert_called_once_with(
        "sam2_hiera_t.yaml",
        "checkpoints/sam2_hiera_tiny.pt",
        device="cpu"
    )

    # Verify generator initialized with config parameters
    mock_generator_cls.assert_called_once_with(
        model=mock_model_instance,
        points_per_side=16,
        pred_iou_thresh=0.6,
        stability_score_thresh=0.5,
        box_nms_thresh=0.8
    )


@patch("baselines.sam2.sam2_standalone._TORCH_AVAILABLE", True)
def test_segment_everything(mock_config_path):
    mock_generator_instance = MagicMock()
    mock_generator_cls.return_value = mock_generator_instance

    # Mock generator returning two masks
    # (segmentation is (H, W) bool array, predicted_iou is float)
    dummy_mask_1 = np.zeros((10, 10), dtype=bool)
    dummy_mask_1[0:5, 0:5] = True  # sum = 25 pixels, > 100 threshold?
    # Wait, the baseline filters masks < 100 pixels in code line 162. Let's make it larger!
    mask1 = np.zeros((15, 15), dtype=bool)
    mask1[0:12, 0:12] = True  # 144 pixels > 100
    
    mask2 = np.zeros((15, 15), dtype=bool)
    mask2[10:15, 10:15] = True  # 25 pixels (this should be filtered out because <= 100)

    mock_generator_instance.generate.return_value = [
        {"segmentation": mask1, "predicted_iou": 0.85},
        {"segmentation": mask2, "predicted_iou": 0.90},
    ]

    baseline = SAM2Baseline(config_path=mock_config_path, device="cpu")
    
    # Run segment
    img = Image.new("RGB", (15, 15))
    result = baseline.segment_everything(img)

    assert "masks" in result
    assert "scores" in result
    assert "inference_time_s" in result
    
    # mask2 is filtered out (size <= 100 pixels), only mask1 is kept
    assert len(result["masks"]) == 1
    assert result["scores"][0] == pytest.approx(0.85)
    assert np.array_equal(result["masks"][0], mask1)


@patch("baselines.sam2.sam2_standalone._TORCH_AVAILABLE", True)
def test_segment_everything_failed(mock_config_path):
    mock_generator_instance = MagicMock()
    mock_generator_cls.return_value = mock_generator_instance
    mock_generator_instance.generate.side_effect = RuntimeError("Inference failed")

    baseline = SAM2Baseline(config_path=mock_config_path, device="cpu")
    img = Image.new("RGB", (15, 15))
    result = baseline.segment_everything(img)

    # Should handle gracefully and return empty outputs
    assert len(result["masks"]) == 0
    assert len(result["scores"]) == 0
    assert result["inference_time_s"] >= 0.0


@patch("baselines.sam2.sam2_standalone._TORCH_AVAILABLE", True)
def test_segment_alias(mock_config_path):
    baseline = SAM2Baseline(config_path=mock_config_path, device="cpu")
    
    with patch.object(baseline, "segment_everything") as mock_seg:
        mock_seg.return_value = {"masks": [], "scores": []}
        img = Image.new("RGB", (15, 15))
        baseline.segment(img)
        mock_seg.assert_called_once_with(img)


@patch("baselines.sam2.sam2_standalone._TORCH_AVAILABLE", True)
def test_timing_and_fps(mock_config_path):
    mock_generator_instance = MagicMock()
    mock_generator_cls.return_value = mock_generator_instance
    mock_generator_instance.generate.return_value = []

    baseline = SAM2Baseline(config_path=mock_config_path, device="cpu")
    
    # Before segmenting, mean_fps should be 0.0
    assert baseline.mean_fps() == 0.0
    
    # Run segment
    img = Image.new("RGB", (15, 15))
    baseline.segment(img)
    
    # After segmenting, mean_fps should be > 0.0
    assert len(baseline._timing) == 1
    assert baseline.mean_fps() > 0.0
    
    # Run segment again to accumulate timing
    baseline.segment(img)
    assert len(baseline._timing) == 2
    
    # Reset timing
    baseline.reset_timing()
    assert len(baseline._timing) == 0
    assert baseline.mean_fps() == 0.0
