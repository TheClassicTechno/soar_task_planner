"""
SAM3 terrain segmentation baseline using HuggingFace Transformers.

Uses facebook/sam3 via the transformers library — no ROS2 or nav_stack dependency.
Loops over each terrain concept query and collects all instance masks.

Prerequisites:
  pip install transformers accelerate
  huggingface-cli login   (or set HF_TOKEN env var)
  HF access approved at huggingface.co/facebook/sam3
"""

import os
import time
from typing import Dict, List

import numpy as np
import yaml
from PIL import Image
from transformers import Sam3Model, Sam3Processor

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _select_device(device_str: str = "auto") -> "torch.device":
    """Return the best available torch device."""
    if not _TORCH_AVAILABLE:
        raise ImportError("pip install torch to run SAM3Baseline")
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SAM3Baseline:
    """
    Standalone SAM3 terrain segmentation.

    Loads facebook/sam3 from HuggingFace on first instantiation (~3.6 GB).
    Subsequent calls use the local HF cache. For each call to segment(), the
    model runs once per terrain concept query (13 queries → 13 forward passes).
    """

    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to baselines/sam3/config.yaml
        """
        with open(config_path) as f:
            self._config = yaml.safe_load(f)

        sam3_cfg = self._config["sam3"]
        self._queries: List[str] = sam3_cfg["queries"]
        self._threshold: float = sam3_cfg.get("detection_threshold", 0.5)
        self._timing: List[float] = []

        local_model_path = sam3_cfg.get("local_model_path", None)
        if local_model_path:
            local_model_path = os.path.expanduser(local_model_path)

        local_files_only = sam3_cfg.get("local_files_only", False)
        self._device = _select_device(sam3_cfg.get("device", "auto"))

        if local_model_path and os.path.exists(local_model_path) and os.path.isdir(local_model_path):
            model_id = local_model_path
            print(f"[SAM3] Loading from local model path: {model_id} → {self._device}")
        else:
            model_id = sam3_cfg.get("hf_model_id", "facebook/sam3")
            if local_model_path:
                print(f"[SAM3] Warning: local_model_path '{local_model_path}' not found or not a directory. Falling back to '{model_id}'.")
            else:
                print(f"[SAM3] Loading {model_id} → {self._device}")
                print("       (first run downloads ~3.6 GB to HF cache)")

        self._processor = Sam3Processor.from_pretrained(model_id, local_files_only=local_files_only)
        self._model = Sam3Model.from_pretrained(model_id, local_files_only=local_files_only).to(self._device)
        self._model.eval()
        print(f"[SAM3] Ready — {len(self._queries)} terrain concepts")

    @property
    def queries(self) -> List[str]:
        """Terrain concept labels this baseline segments."""
        return list(self._queries)

    def segment(self, image: Image.Image) -> Dict:
        """
        Run SAM3 for every configured terrain concept on a single image.

        Args:
            image: PIL Image (RGB or any mode — converted internally).

        Returns:
            Dict with:
              "masks"            — (N, H, W) bool np.ndarray
              "labels"           — (N,) int32 np.ndarray  (index into self.queries)
              "scores"           — (N,) float32 np.ndarray
              "inference_time_s" — total wall-clock seconds for all concepts
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        t0 = time.perf_counter()
        all_masks: List[np.ndarray] = []
        all_labels: List[int] = []
        all_scores: List[float] = []

        for idx, concept in enumerate(self._queries):
            inputs = self._processor(
                images=image, text=concept, return_tensors="pt"
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self._threshold,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

            masks = results.get("masks", torch.zeros(0, dtype=torch.bool))
            scores = results.get("scores", torch.zeros(0))
            for mask, score in zip(masks, scores):
                all_masks.append(_to_numpy_bool(mask))
                all_labels.append(idx)
                all_scores.append(_to_float(score))

        elapsed = time.perf_counter() - t0
        self._timing.append(elapsed)

        if all_masks:
            return {
                "masks": np.stack(all_masks),
                "labels": np.array(all_labels, dtype=np.int32),
                "scores": np.array(all_scores, dtype=np.float32),
                "inference_time_s": elapsed,
            }

        h, w = image.size[1], image.size[0]
        return {
            "masks": np.zeros((0, h, w), dtype=bool),
            "labels": np.zeros(0, dtype=np.int32),
            "scores": np.zeros(0, dtype=np.float32),
            "inference_time_s": elapsed,
        }

    def mean_fps(self) -> float:
        """Mean frames-per-second across all segment() calls. 0.0 before any calls."""
        if not self._timing:
            return 0.0
        return 1.0 / (sum(self._timing) / len(self._timing))

    def reset_timing(self) -> None:
        """Clear accumulated timing history."""
        self._timing.clear()


def load_config(config_path: str) -> Dict:
    """Load and return the YAML config dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_numpy_bool(x) -> np.ndarray:
    """Convert a torch tensor or array-like to a bool numpy array."""
    if hasattr(x, "cpu"):
        return x.cpu().numpy().astype(bool)
    return np.asarray(x, dtype=bool)


def _to_float(x) -> float:
    """Convert a torch scalar tensor or numeric to a Python float."""
    if hasattr(x, "item"):
        return x.item()
    return float(x)
