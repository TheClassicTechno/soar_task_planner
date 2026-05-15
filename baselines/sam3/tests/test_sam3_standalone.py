"""
Unit tests for sam3_standalone.py — HuggingFace model fully mocked.

SAM3 is 0.9B params and requires approved HF access, so we patch
Sam3Model.from_pretrained and Sam3Processor.from_pretrained to avoid
any network or GPU dependency in tests.
"""

import numpy as np
import pytest
import torch
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from PIL import Image

from baselines.sam3.sam3_standalone import (
    SAM3Baseline,
    load_config,
    _to_numpy_bool,
    _to_float,
    _select_device,
)

CONFIG_PATH = str(Path(__file__).parents[1] / "config.yaml")


# ── helpers ────────────────────────────────────────────────────────────────────

def _rgb_image(h=32, w=48):
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


def _make_post_result(n_masks=2, h=32, w=48):
    """Fake post_process_instance_segmentation output for one concept."""
    return [{
        "masks": torch.zeros(n_masks, h, w, dtype=torch.bool),
        "scores": torch.full((n_masks,), 0.85),
        "boxes": torch.zeros(n_masks, 4),
    }]


def _build_mock_classes(n_masks=2, h=32, w=48, queries=None):
    """
    Build mock Sam3Processor and Sam3Model classes.

    Returns (MockProcCls, MockModelCls, proc_instance, model_instance).
    """
    queries = queries or ["gravel", "mud"]

    MockProcCls = MagicMock()
    MockModelCls = MagicMock()

    proc = MagicMock()
    model = MagicMock()

    MockProcCls.from_pretrained.return_value = proc
    MockModelCls.from_pretrained.return_value = model
    model.to.return_value = model
    model.eval.return_value = None

    # processor(images=..., text=..., return_tensors=...) → mock_inputs
    mock_sizes = MagicMock()
    mock_sizes.tolist.return_value = [[h, w]]
    mock_inputs = MagicMock()
    mock_inputs.get.return_value = mock_sizes
    mock_inputs.to.return_value = mock_inputs
    proc.return_value = mock_inputs

    # post_process returns n_masks masks per concept
    proc.post_process_instance_segmentation.return_value = _make_post_result(n_masks, h, w)

    return MockProcCls, MockModelCls, proc, model


@pytest.fixture
def baseline_2q(tmp_path):
    """SAM3Baseline with 2 terrain concepts and 2 detections per concept."""
    config = {
        "sam3": {
            "queries": ["gravel", "mud"],
            "detection_threshold": 0.5,
            "hf_model_id": "facebook/sam3",
            "device": "cpu",
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))

    MockProcCls, MockModelCls, proc, model = _build_mock_classes(
        n_masks=2, h=32, w=48, queries=["gravel", "mud"]
    )

    with patch("baselines.sam3.sam3_standalone.Sam3Model", MockModelCls), \
         patch("baselines.sam3.sam3_standalone.Sam3Processor", MockProcCls):
        b = SAM3Baseline(str(config_path))

    return b, proc, model


@pytest.fixture
def baseline_no_detections(tmp_path):
    """SAM3Baseline where post_process returns 0 masks."""
    config = {
        "sam3": {
            "queries": ["gravel"],
            "detection_threshold": 0.5,
            "hf_model_id": "facebook/sam3",
            "device": "cpu",
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))

    MockProcCls, MockModelCls, proc, model = _build_mock_classes(
        n_masks=0, h=32, w=48, queries=["gravel"]
    )

    with patch("baselines.sam3.sam3_standalone.Sam3Model", MockModelCls), \
         patch("baselines.sam3.sam3_standalone.Sam3Processor", MockProcCls):
        b = SAM3Baseline(str(config_path))

    return b, proc, model


# ── init: model loading ────────────────────────────────────────────────────────

def test_init_calls_processor_from_pretrained(tmp_path):
    config = {"sam3": {"queries": ["gravel"], "detection_threshold": 0.5,
                        "hf_model_id": "test/sam3", "device": "cpu"}}
    cp = tmp_path / "c.yaml"
    cp.write_text(yaml.dump(config))
    MockProcCls, MockModelCls, _, _ = _build_mock_classes(queries=["gravel"])
    with patch("baselines.sam3.sam3_standalone.Sam3Model", MockModelCls), \
         patch("baselines.sam3.sam3_standalone.Sam3Processor", MockProcCls):
        SAM3Baseline(str(cp))
    MockProcCls.from_pretrained.assert_called_once_with("test/sam3")


def test_init_calls_model_from_pretrained(tmp_path):
    config = {"sam3": {"queries": ["gravel"], "detection_threshold": 0.5,
                        "hf_model_id": "test/sam3", "device": "cpu"}}
    cp = tmp_path / "c.yaml"
    cp.write_text(yaml.dump(config))
    MockProcCls, MockModelCls, _, _ = _build_mock_classes(queries=["gravel"])
    with patch("baselines.sam3.sam3_standalone.Sam3Model", MockModelCls), \
         patch("baselines.sam3.sam3_standalone.Sam3Processor", MockProcCls):
        SAM3Baseline(str(cp))
    MockModelCls.from_pretrained.assert_called_once_with("test/sam3")


# ── queries property ───────────────────────────────────────────────────────────

def test_queries_property_returns_list(baseline_2q):
    b, _, _ = baseline_2q
    assert isinstance(b.queries, list)


def test_queries_property_matches_config(baseline_2q):
    b, _, _ = baseline_2q
    assert b.queries == ["gravel", "mud"]


def test_queries_with_real_config():
    """queries matches the 13-entry list in the actual config.yaml."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    MockProcCls, MockModelCls, _, _ = _build_mock_classes(
        queries=config["sam3"]["queries"]
    )
    with patch("baselines.sam3.sam3_standalone.Sam3Model", MockModelCls), \
         patch("baselines.sam3.sam3_standalone.Sam3Processor", MockProcCls):
        b = SAM3Baseline(CONFIG_PATH)
    assert b.queries == config["sam3"]["queries"]


# ── segment: output keys and types ────────────────────────────────────────────

def test_segment_returns_expected_keys(baseline_2q):
    b, _, _ = baseline_2q
    result = b.segment(_rgb_image())
    for key in ("masks", "labels", "scores", "inference_time_s"):
        assert key in result, f"Missing key: {key}"


def test_segment_masks_dtype_is_bool(baseline_2q):
    b, _, _ = baseline_2q
    result = b.segment(_rgb_image())
    assert result["masks"].dtype == bool


def test_segment_labels_dtype_is_int32(baseline_2q):
    b, _, _ = baseline_2q
    result = b.segment(_rgb_image())
    assert result["labels"].dtype == np.int32


def test_segment_scores_dtype_is_float32(baseline_2q):
    b, _, _ = baseline_2q
    result = b.segment(_rgb_image())
    assert result["scores"].dtype == np.float32


# ── segment: shape correctness ─────────────────────────────────────────────────

def test_segment_masks_shape(baseline_2q):
    b, _, _ = baseline_2q
    # 2 concepts × 2 detections = 4 total masks, each (32, 48)
    result = b.segment(_rgb_image(h=32, w=48))
    assert result["masks"].shape == (4, 32, 48)


def test_segment_labels_length_matches_masks(baseline_2q):
    b, _, _ = baseline_2q
    result = b.segment(_rgb_image())
    assert result["labels"].shape[0] == result["masks"].shape[0]


def test_segment_scores_length_matches_masks(baseline_2q):
    b, _, _ = baseline_2q
    result = b.segment(_rgb_image())
    assert result["scores"].shape[0] == result["masks"].shape[0]


def test_segment_labels_values_are_valid_indices(baseline_2q):
    b, _, _ = baseline_2q
    result = b.segment(_rgb_image())
    assert result["labels"].min() >= 0
    assert result["labels"].max() < len(b.queries)


def test_segment_label_grouping(baseline_2q):
    """Label 0 = gravel (2 masks), label 1 = mud (2 masks)."""
    b, _, _ = baseline_2q
    result = b.segment(_rgb_image())
    labels = result["labels"]
    assert list(labels) == [0, 0, 1, 1]


# ── segment: empty detections ──────────────────────────────────────────────────

def test_segment_empty_masks_when_no_detections(baseline_no_detections):
    b, _, _ = baseline_no_detections
    result = b.segment(_rgb_image(h=32, w=48))
    assert result["masks"].shape == (0, 32, 48)
    assert result["labels"].shape == (0,)
    assert result["scores"].shape == (0,)


def test_segment_inference_time_non_negative(baseline_2q):
    b, _, _ = baseline_2q
    result = b.segment(_rgb_image())
    assert result["inference_time_s"] >= 0.0


def test_segment_converts_rgba_to_rgb(baseline_2q):
    b, _, _ = baseline_2q
    rgba = Image.new("RGBA", (48, 32), (100, 150, 200, 128))
    result = b.segment(rgba)
    assert "masks" in result


# ── segment: model call count ──────────────────────────────────────────────────

def test_model_called_once_per_query(baseline_2q):
    """Each segment() call invokes the model once per terrain concept."""
    b, _, model = baseline_2q
    b.segment(_rgb_image())
    assert model.call_count == len(b.queries)


def test_processor_called_once_per_query(baseline_2q):
    b, proc, _ = baseline_2q
    b.segment(_rgb_image())
    assert proc.call_count == len(b.queries)


# ── timing and fps ─────────────────────────────────────────────────────────────

def test_mean_fps_zero_before_any_calls(baseline_2q):
    b, _, _ = baseline_2q
    assert b.mean_fps() == 0.0


def test_mean_fps_positive_after_segment(baseline_2q):
    b, _, _ = baseline_2q
    b.segment(_rgb_image())
    assert b.mean_fps() > 0.0


def test_reset_timing_clears_history(baseline_2q):
    b, _, _ = baseline_2q
    b.segment(_rgb_image())
    b.reset_timing()
    assert b.mean_fps() == 0.0


def test_timing_accumulates_across_calls(baseline_2q):
    b, _, _ = baseline_2q
    b.segment(_rgb_image())
    b.segment(_rgb_image())
    fps = b.mean_fps()
    assert fps > 0.0


# ── helpers ────────────────────────────────────────────────────────────────────

def test_to_numpy_bool_from_tensor():
    t = torch.ones(3, 4, dtype=torch.bool)
    result = _to_numpy_bool(t)
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool


def test_to_numpy_bool_from_ndarray():
    arr = np.ones((3, 4), dtype=np.uint8)
    result = _to_numpy_bool(arr)
    assert result.dtype == bool


def test_to_float_from_tensor():
    t = torch.tensor(0.75)
    assert _to_float(t) == pytest.approx(0.75)


def test_to_float_from_scalar():
    assert _to_float(0.5) == pytest.approx(0.5)


def test_select_device_explicit_cpu():
    d = _select_device("cpu")
    assert str(d) == "cpu"


def test_load_config_returns_dict():
    result = load_config(CONFIG_PATH)
    assert isinstance(result, dict)
    assert "sam3" in result
