#!/usr/bin/env bash
# Download terrain datasets for environmental uncertainty evaluation.
#
# Datasets:
#   RELLIS-3D — real-world outdoor RGB + semantic labels (recommended first)
#   GOOSE     — real-world outdoor RGB + LiDAR, 4 seasons
#   TartanGround — synthetic RGB (pre-training only, NOT real-world)
#
# Usage:
#   chmod +x scripts/download_datasets.sh
#   ./scripts/download_datasets.sh rellis        # download RELLIS-3D only
#   ./scripts/download_datasets.sh goose         # download GOOSE only
#   ./scripts/download_datasets.sh tartanground   # download TartanGround sample
#   ./scripts/download_datasets.sh all           # download all three

set -euo pipefail

DATASET_DIR="${DATASET_DIR:-data/datasets}"
mkdir -p "$DATASET_DIR"

# ── RELLIS-3D ──────────────────────────────────────────────────────────────────
# Paper: Jiang et al., ICRA 2021. arXiv: 2011.12954
# 6,235 RGB images + pixel-wise semantic labels. 20 terrain classes.
# Classes compatible with our 21-class TRAVERSABILITY_SCORES vocabulary.
# License: CC BY-NC-SA 4.0
download_rellis() {
    echo "=== Downloading RELLIS-3D RGB subset ==="
    local dest="$DATASET_DIR/RELLIS-3D"
    mkdir -p "$dest"

    # RELLIS-3D is distributed as individual sequence ZIPs.
    # Official repo: https://github.com/unmannedlab/RELLIS-3D
    # Sequences 00–04 cover different outdoor environments (forest, trail, etc.)
    # NOTE: The S3 bucket (rellis.s3.amazonaws.com) returned 403 as of May 2026.
    # Official distribution is now via Google Drive / OneDrive linked from the
    # RELLIS-3D GitHub releases page.  See: https://github.com/unmannedlab/RELLIS-3D/releases
    echo ""
    echo "  RELLIS-3D requires manual download (S3 bucket is no longer public)."
    echo ""
    echo "  Steps:"
    echo "    1. Visit: https://github.com/unmannedlab/RELLIS-3D/releases"
    echo "    2. Download RELLIS-3D_00_img.zip and RELLIS-3D_00_label.zip"
    echo "       (or sequences 00–04 for full eval)"
    echo "    3. Extract to $dest/"
    echo "       Expected layout:"
    echo "         $dest/RELLIS-3D_00/pylon_camera_node/*.jpg"
    echo "         $dest/RELLIS-3D_00/pylon_camera_node_label_id/*.png"
    echo ""
    echo "  After extraction, run:"
    echo "    python scripts/run_pipeline_rellis.py --rellis_dir $dest"
}

# ── GOOSE ──────────────────────────────────────────────────────────────────────
# Paper: Mortimer et al., ICRA 2024. arXiv: 2310.16788
# 10,000 labeled frames, 64 terrain classes, 4 seasons. License: CC BY-SA 4.0
# Official site: https://goose-dataset.de
download_goose() {
    echo "=== Downloading GOOSE dataset ==="
    local dest="$DATASET_DIR/GOOSE"
    mkdir -p "$dest"

    echo "  GOOSE requires manual registration at https://goose-dataset.de/"
    echo "  Steps:"
    echo "    1. Go to https://goose-dataset.de/"
    echo "    2. Register and accept the CC BY-SA license"
    echo "    3. Download goose_2d_train.zip and goose_2d_val.zip"
    echo "    4. Extract to $dest/"
    echo ""
    echo "  Alternatively, use the GitHub downloader:"
    echo "    git clone https://github.com/FraunhoferIOSB/goose_dataset.git /tmp/goose_tools"
    echo "    python /tmp/goose_tools/download.py --output $dest"
}

# ── TartanGround ───────────────────────────────────────────────────────────────
# Paper: CMU AirLab, arXiv: 2505.10696 (May 2025)
# 910 trajectories, ~15 TB total. SYNTHETIC (simulation), not real-world.
# Use for pre-training only; not for real-terrain evaluation.
download_tartanground() {
    echo "=== Downloading TartanGround sample ==="
    local dest="$DATASET_DIR/TartanGround"
    mkdir -p "$dest"

    # tartanairpy is the official downloader
    if ! python -c "import tartanairpy" 2>/dev/null; then
        echo "  Installing tartanairpy..."
        pip install tartanairpy --quiet
    fi

    echo "  Downloading a small sample (Nature environment, 5 trajectories)..."
    python - <<'PYEOF'
import tartanairpy as ta
import os

dest = os.environ.get("DATASET_DIR", "data/datasets") + "/TartanGround"
ta.download(
    output_dir=dest,
    env=["Nature"],           # one of 6 environment types
    difficulty=["easy"],
    trajectory_id=["P000", "P001", "P002"],
    modality=["image", "seg"],  # RGB frames + semantic segmentation
)
print(f"  TartanGround sample saved to {dest}/")
PYEOF
}

# ── Dispatch ───────────────────────────────────────────────────────────────────
case "${1:-all}" in
    rellis)       download_rellis ;;
    goose)        download_goose ;;
    tartanground) download_tartanground ;;
    all)
        download_rellis
        echo ""
        download_goose
        echo ""
        download_tartanground
        ;;
    *)
        echo "Usage: $0 {rellis|goose|tartanground|all}"
        exit 1
        ;;
esac

echo ""
echo "Dataset directory: $DATASET_DIR"
ls -lh "$DATASET_DIR" 2>/dev/null || true
