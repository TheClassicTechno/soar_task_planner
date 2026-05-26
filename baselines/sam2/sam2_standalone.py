"""
SAM2 segment everything baseline using Meta's official SAM2 repository.

Uses facebook/sam2-hiera-large via the official sam2 package.
Implements automatic mask generation (segment everything) via SAM2AutomaticMaskGenerator.
"""

import time
from typing import Dict, List, Optional

import numpy as np
import yaml
from PIL import Image

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _select_device(device_str: str = "auto") -> str:
    """Return the best available torch device as string."""
    if not _TORCH_AVAILABLE:
        raise ImportError("pip install torch to run SAM2Baseline")
    if device_str != "auto":
        return device_str
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _filter_overlapping_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.7,
) -> tuple:
    """
    Filter overlapping masks using IoU.
    
    Keeps masks with high scores and removes highly overlapping ones.
    """
    if len(masks) == 0:
        return masks, scores
    
    keep_mask = np.ones(len(masks), dtype=bool)
    
    for i in range(len(masks)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(masks)):
            if not keep_mask[j]:
                continue
            
            # Compute IoU
            inter = np.logical_and(masks[i], masks[j]).sum()
            union = np.logical_or(masks[i], masks[j]).sum()
            if union > 0:
                iou = inter / union
                if iou > iou_threshold:
                    if scores[i] < scores[j]:
                        keep_mask[i] = False
                    else:
                        keep_mask[j] = False
    
    return masks[keep_mask], scores[keep_mask]


class SAM2Baseline:
    """
    Standalone SAM2 segment everything.
    
    Loads official SAM2 checkpoint using build_sam2.
    Uses SAM2AutomaticMaskGenerator to generate masks for all objects in the image.
    """

    def __init__(
        self,
        config_path: str,
        device: Optional[str] = "auto",
    ):
        """
        Args:
            config_path: Path to baselines/sam2/config.yaml
            device: Device to run on ("auto", "cuda", "mps", "cpu")
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required to run SAM2Baseline")
        
        self._device = _select_device(device)
        
        # Default config if file doesn't exist
        self._config = {"sam2": {}}
        if config_path:
            try:
                with open(config_path) as f:
                    self._config = yaml.safe_load(f)
            except FileNotFoundError:
                pass
        
        sam2_cfg = self._config.get("sam2", {})
        model_cfg = sam2_cfg.get("model_cfg", "sam2_hiera_l.yaml")
        checkpoint_path = sam2_cfg.get("checkpoint", "checkpoints/sam2_hiera_large.pt")
        self._points_per_side = sam2_cfg.get("points_per_side", 32)
        self._threshold = sam2_cfg.get("detection_threshold", 0.5)
        self._iou_threshold = sam2_cfg.get("iou_threshold", 0.7)
        
        print(f"[SAM2] Loading {model_cfg} with checkpoint {checkpoint_path} → {self._device}")
        
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        
        # Build model and generator
        self._model = build_sam2(model_cfg, checkpoint_path, device=self._device)
        self._generator = SAM2AutomaticMaskGenerator(
            model=self._model,
            points_per_side=self._points_per_side,
            pred_iou_thresh=self._threshold,
            stability_score_thresh=0.5,
            box_nms_thresh=self._iou_threshold,
        )
        print("[SAM2] Ready")

    def segment_everything(self, image: Image.Image) -> Dict:
        """
        Run SAM2 segment everything on a single image.
        
        Args:
            image: PIL Image (RGB)
        
        Returns:
            Dict with:
              "masks"            — (N, H, W) bool np.ndarray
              "scores"           — (N,) float32 np.ndarray
              "inference_time_s" — total wall-clock seconds
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        h, w = image.size[1], image.size[0]
        
        t0 = time.perf_counter()
        
        try:
            # Convert PIL image to numpy array
            image_np = np.array(image)
            
            # Generate masks
            masks_data = self._generator.generate(image_np)
            
            # Format outputs
            valid_masks = []
            valid_scores = []
            
            for item in masks_data:
                mask = item["segmentation"]
                score = item["predicted_iou"]
                
                # Filter tiny masks (e.g. less than 100 pixels)
                if mask.sum() > 100:
                    valid_masks.append(mask)
                    valid_scores.append(score)
            
            all_masks = valid_masks
            all_scores = valid_scores
            
        except Exception as e:
            print(f"[SAM2] Warning: segment_everything failed: {e}")
            all_masks = []
            all_scores = []
        
        elapsed = time.perf_counter() - t0
        
        if all_masks:
            masks_array = np.stack(all_masks)
            scores_array = np.array(all_scores, dtype=np.float32)
            masks_filtered, scores_filtered = _filter_overlapping_masks(
                masks_array, scores_array, iou_threshold=self._iou_threshold
            )
        else:
            masks_filtered = np.zeros((0, h, w), dtype=bool)
            scores_filtered = np.zeros(0, dtype=np.float32)
        
        return {
            "masks": masks_filtered,
            "scores": scores_filtered,
            "inference_time_s": elapsed,
        }

    def segment(self, image: Image.Image) -> Dict:
        """
        Alias for segment_everything - matches SAM3 interface.
        """
        return self.segment_everything(image)


def load_config(config_path: str) -> Dict:
    """Load and return the YAML config dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)