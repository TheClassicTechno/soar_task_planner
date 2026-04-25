"""
Visualization utilities for the SAM3 baseline.

Overlays segmentation masks on the original image with per-class colors
and saves the result as a PNG. No GPU dependency.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def overlay_masks(
    image: np.ndarray,
    sam3_results: Dict,
    queries: List[str],
    color_map: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Blend SAM3 segmentation masks onto the original image.

    Args:
        image:       (H, W, 3) BGR uint8 array (OpenCV format).
        sam3_results: Output of SAM3Repo.segment() with "masks" and "labels".
        queries:     Ordered list of concept query strings.
        color_map:   (N_classes, 3) BGR uint8 array — one color per query.
        alpha:       Mask opacity in [0, 1].

    Returns:
        (H, W, 3) BGR uint8 blended image.
    """
    output = image.copy()
    masks = sam3_results.get("masks", [])
    labels = sam3_results.get("labels", [])

    for i, mask in enumerate(masks):
        if i >= len(labels):
            break
        label_idx = int(labels[i])
        if label_idx >= len(color_map):
            continue

        color = color_map[label_idx].astype(np.uint8)
        mask_bool = mask.astype(bool)
        if mask_bool.shape[:2] != image.shape[:2]:
            # Resize mask to image dimensions if needed
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_bool = mask_resized.astype(bool)

        # Blend color over masked region
        colored = np.zeros_like(output)
        colored[:] = color
        output[mask_bool] = (
            (1 - alpha) * output[mask_bool] + alpha * colored[mask_bool]
        ).astype(np.uint8)

    return output


def draw_legend(
    image: np.ndarray,
    queries: List[str],
    color_map: np.ndarray,
    active_labels: Optional[List[int]] = None,
    font_scale: float = 0.45,
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw a color legend in the top-right corner of the image.
    Only shows classes that were actually detected (active_labels).

    Args:
        image:         (H, W, 3) BGR uint8 image.
        queries:       Ordered list of query strings.
        color_map:     (N_classes, 3) BGR uint8 color array.
        active_labels: Indices of classes actually present in this result.
                       If None, show all classes.
        font_scale:    OpenCV font scale.
        thickness:     Text stroke thickness.

    Returns:
        Image with legend overlaid (modifies in-place and returns).
    """
    if active_labels is None:
        active_labels = list(range(len(queries)))

    # Ensure legend stays within image bounds
    x_start = max(image.shape[1] - 220, 2)
    y_start = 15
    box_size = 14

    for rank, idx in enumerate(active_labels):
        if idx >= len(queries):
            continue
        y = y_start + rank * (box_size + 4)
        color = tuple(int(c) for c in color_map[idx])

        # Colored square
        cv2.rectangle(image, (x_start, y), (x_start + box_size, y + box_size), color, -1)
        # Label text in white with black shadow for readability
        text = queries[idx]
        cv2.putText(
            image, text,
            (x_start + box_size + 4, y + box_size - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1,
        )
        cv2.putText(
            image, text,
            (x_start + box_size + 4, y + box_size - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness,
        )

    return image


def save_visualization(
    image_pil: Image.Image,
    sam3_results: Dict,
    queries: List[str],
    color_map: np.ndarray,
    save_path: str,
    alpha: float = 0.5,
    draw_legend_flag: bool = True,
) -> None:
    """
    Full pipeline: PIL image → overlay masks → draw legend → save PNG.

    Args:
        image_pil:       PIL RGB image.
        sam3_results:    Output of SAM3Repo.segment().
        queries:         Ordered concept query strings.
        color_map:       (N_classes, 3) BGR uint8 color array.
        save_path:       Destination file path (will create parent dirs).
        alpha:           Mask opacity.
        draw_legend_flag: Whether to draw a class legend on the image.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # PIL (RGB) → OpenCV (BGR)
    bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    blended = overlay_masks(bgr, sam3_results, queries, color_map, alpha)

    if draw_legend_flag:
        active = list(set(int(l) for l in sam3_results.get("labels", [])))
        draw_legend(blended, queries, color_map, active_labels=sorted(active))

    cv2.imwrite(save_path, blended)


def load_color_map(color_list: List[List[int]]) -> np.ndarray:
    """
    Convert a list of [B, G, R] lists (from YAML) to a (N, 3) uint8 array.

    Args:
        color_list: List of [B, G, R] int triplets from config.yaml.

    Returns:
        (N, 3) numpy uint8 array.
    """
    return np.array(color_list, dtype=np.uint8)
