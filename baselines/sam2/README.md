# SAM2 Baseline (Meta Official Implementation)

This directory contains the standalone Segment Anything 2 (SAM2) baseline using Meta's official implementation to support automatic mask generation ("segment everything" mode) via dense grid point prompts.

## Quick Setup

To configure the environment and download the required weights, activate your conda environment and run the automated python setup script:

```bash
conda activate soar-task-planner
python baselines/sam2/setup_sam2.py
```

### What this script does:
1. **Installs Meta's official `sam2` package** directly from GitHub (`facebookresearch/sam2`).
2. **Handles hardware compilation flags automatically**:
   * On macOS or systems without CUDA, it disables CUDA compilation via `SAM2_BUILD_CUDA=0` (preventing setup compilation errors).
   * On CUDA-capable systems, it automatically compiles the optimized custom CUDA kernels.
3. **Downloads the model checkpoint** `sam2_hiera_large.pt` (approx. 856 MB) to the `checkpoints/` directory at the project root.

---

## Configuration

Settings are controlled via [config.yaml](file:///Users/max.liu/Desktop/soar_task_planner/baselines/sam2/config.yaml):

```yaml
sam2:
  model_cfg: "sam2_hiera_l.yaml"               # Meta config filename (internal package path)
  checkpoint: "checkpoints/sam2_hiera_large.pt" # Local checkpoint path
  device: "auto"                              # auto / cuda / mps / cpu
  points_per_side: 32                         # Density grid of prompt points (32x32 = 1024 points)
  detection_threshold: 0.5                     # Predicted IoU cutoff
  iou_threshold: 0.7                           # Non-maximum suppression overlap threshold
```

---

## Verification

To verify that the model compiles, loads, and executes correctly, you can run a demo inference script:

```bash
# Verify using a synthetic image containing geometric shapes
python /Users/max.liu/.gemini/antigravity/brain/1ab89cda-4480-4ff5-81fc-ce7c2500c26d/scratch/run_sam2_demo.py
```

Alternatively, you can run the pytest suite to ensure no regressions were introduced to the environmental uncertainty system:

```bash
pytest system/env_uncertainty/tests/
```
