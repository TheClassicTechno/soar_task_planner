#!/usr/bin/env bash
# Download the SAM3 checkpoint from HuggingFace.
#
# Prerequisites:
#   1. Request access at: https://huggingface.co/facebook/sam3
#      (Meta approves these — usually same day)
#   2. Generate a HuggingFace token at: https://huggingface.co/settings/tokens
#   3. Run: pip install huggingface_hub && hf auth login
#
# Then run this script:
#   bash scripts/download_sam3_checkpoint.sh
#
# The checkpoint will be saved to the path in baselines/sam3/config.yaml:
#   ~/Documents/navigation_stack/ws/src/perception/ckpt/sam3/

set -e

CKPT_DIR="${SAM3_CKPT_PATH:-$HOME/Documents/navigation_stack/ws/src/perception/ckpt/sam3}"

echo "Downloading SAM3 checkpoint to: $CKPT_DIR"
mkdir -p "$CKPT_DIR"

python3 - <<'PYEOF'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

ckpt_dir = Path(os.path.expanduser(os.environ.get(
    "SAM3_CKPT_PATH",
    "~/Documents/navigation_stack/ws/src/perception/ckpt/sam3"
)))
ckpt_dir.mkdir(parents=True, exist_ok=True)

# Main model checkpoint
print("Downloading sam3.pt ...")
hf_hub_download(
    repo_id="facebook/sam3",
    filename="sam3.pt",
    local_dir=str(ckpt_dir),
)

# BPE vocabulary file (required by SAM3Repo)
print("Downloading bpe_simple_vocab_16e6.txt.gz ...")
hf_hub_download(
    repo_id="facebook/sam3",
    filename="bpe_simple_vocab_16e6.txt.gz",
    local_dir=str(ckpt_dir),
)

print(f"\nCheckpoint files saved to: {ckpt_dir}")
print("Contents:")
for f in sorted(ckpt_dir.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
PYEOF
