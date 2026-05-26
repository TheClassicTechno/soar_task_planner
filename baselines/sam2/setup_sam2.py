"""
Setup script for Meta SAM2.
Installs the official sam2 library and downloads the sam2_hiera_large.pt checkpoint.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def check_torch():
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        return torch.cuda.is_available()
    except ImportError:
        print("Error: PyTorch is not installed. Please install PyTorch first.")
        sys.exit(1)

def install_sam2(cuda_available=False):
    print("\n--- Step 1: Installing Meta SAM2 Package ---")
    
    # Set environment variables for macOS or CPU-only builds
    env = os.environ.copy()
    is_mac = sys.platform == "darwin"
    
    if is_mac or not cuda_available:
        print("Disabling CUDA compilation for SAM2 (building CPU/MPS only)...")
        env["SAM2_BUILD_CUDA"] = "0"
    else:
        print("CUDA detected, enabling CUDA compilation for SAM2...")
        env["SAM2_BUILD_CUDA"] = "1"
        
    cmd = [sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/sam2.git"]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, env=env, check=True)
        print("Successfully installed SAM2 package.")
    except subprocess.CalledProcessError as e:
        print(f"Error during SAM2 package installation: {e}")
        sys.exit(1)

def download_checkpoint():
    print("\n--- Step 2: Downloading Model Checkpoint ---")
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / "sam2_hiera_large.pt"
    url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    
    if checkpoint_path.exists():
        print(f"Checkpoint already exists at {checkpoint_path} (size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB). Skipping download.")
        return
        
    print(f"Downloading sam2_hiera_large.pt (approx. 856 MB)...")
    print(f"Source: {url}")
    print(f"Destination: {checkpoint_path}")
    
    try:
        # Simple progress reporter
        def report_progress(block_num, block_size, total_size):
            read_so_far = block_num * block_size
            if total_size > 0:
                percent = read_so_far * 1e2 / total_size
                s = f"\rProgress: {percent:.1f}% ({read_so_far / 1024 / 1024:.1f} MB of {total_size / 1024 / 1024:.1f} MB)"
                sys.stdout.write(s)
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\rProgress: {read_so_far / 1024 / 1024:.1f} MB")
                sys.stdout.flush()
                
        urllib.request.urlretrieve(url, str(checkpoint_path), reporthook=report_progress)
        print("\nSuccessfully downloaded checkpoint.")
    except Exception as e:
        print(f"\nError downloading checkpoint: {e}")
        # Clean up partial download
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        sys.exit(1)

def main():
    print("====================================================")
    print("          SAM2 Environment Setup Script             ")
    print("====================================================")
    
    cuda_available = check_torch()
    install_sam2(cuda_available)
    download_checkpoint()
    
    print("\n====================================================")
    print("Setup completed successfully!")
    print("You can run verification with:")
    print(f"  python -m baselines.sam2.sam2_standalone")
    print("====================================================")

if __name__ == "__main__":
    main()
