"""
Unit tests for SAM2 run_baseline.py.

Mocks dataset reading, model execution, and visualization to allow running offline.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.sam2.run_baseline import parse_args, run


class MockRUGDSample:
    def __init__(self, name):
        self.name = name

    def load_image(self):
        return Image.new("RGB", (32, 32))

    def load_annotation(self):
        return None


def test_parse_args():
    with patch("sys.argv", ["run_baseline.py", "--split", "test", "--max_samples", "10"]):
        args = parse_args()
        assert args.split == "test"
        assert args.max_samples == 10
        assert args.scene is None


@patch("baselines.sam2.run_baseline.load_dotenv", MagicMock())
@patch("baselines.sam2.run_baseline.SAM2Baseline")
@patch("baselines.sam2.run_baseline.load_rugd_split")
@patch("baselines.sam2.run_baseline.save_visualization")
def test_run_baseline_pipeline(
    mock_save_vis,
    mock_load_rugd,
    mock_sam2_baseline_cls,
    tmp_path,
):
    # Setup mock config file
    config_content = """
sam2:
  model_cfg: "sam2_hiera_t.yaml"
  checkpoint: "checkpoints/sam2_hiera_tiny.pt"
  device: "cpu"
rugd:
  data_path: "~/Documents/datasets/rugd"
"""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(config_content)

    # Setup mocks
    mock_samples = [MockRUGDSample("trail-5_0001"), MockRUGDSample("trail-5_0002")]
    mock_load_rugd.return_value = mock_samples

    mock_model = MagicMock()
    mock_sam2_baseline_cls.return_value = mock_model
    mock_model.segment.return_value = {
        "masks": np.zeros((2, 32, 32), dtype=bool),  # 2 masks
        "scores": np.array([0.9, 0.8], dtype=np.float32),
        "inference_time_s": 0.05,
    }
    mock_model.mean_fps.return_value = 20.0

    output_dir = tmp_path / "output"

    # Run runner pipeline
    with patch("sys.argv", [
        "run_baseline.py",
        "--config", str(cfg_file),
        "--rugd_dir", str(tmp_path / "dummy_rugd"),
        "--split", "val",
        "--output_dir", str(output_dir),
    ]):
        args = parse_args()
        run(args)

    # Verify model calls
    mock_sam2_baseline_cls.assert_called_once_with(config_path=str(cfg_file))
    assert mock_model.segment.call_count == 2
    assert mock_save_vis.call_count == 2

    # Verify outputs
    results_json_path = output_dir / "results.json"
    assert results_json_path.exists()

    with open(results_json_path) as f:
        data = json.load(f)
        assert "summary" in data
        assert "per_image" in data
        assert len(data["per_image"]) == 2
        assert data["summary"]["n_images"] == 2
        assert data["summary"]["mean_fps"] == 20.0
        assert data["summary"]["mean_masks_per_image"] == 2.0
