"""
Setup script for Meta SAM2.
Installs the official sam2 library and downloads the sam2_hiera_large.pt checkpoint.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False

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
    print("\n--- Step 2: Checking/Downloading Model Checkpoint ---")
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / "sam2_hiera_large.pt"
    
    # 1. Check if checkpoints/sam2_hiera_large.pt already exists
    if checkpoint_path.exists():
        print(f"Checkpoint already exists at {checkpoint_path} (size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB). Skipping download.")
        return
        
    # 2. Check if SAM2_CKPT_PATH environment variable is specified and exists
    env_ckpt_path = os.environ.get("SAM2_CKPT_PATH", None)
    if env_ckpt_path:
        resolved_env_path = Path(os.path.expanduser(env_ckpt_path))
        if resolved_env_path.exists() and resolved_env_path.is_file():
            print(f"Found pre-downloaded SAM2 checkpoint via SAM2_CKPT_PATH: {resolved_env_path}")
            print(f"Creating symbolic link to {checkpoint_path}...")
            try:
                # Remove any existing broken symlink or file at target path
                if checkpoint_path.exists() or checkpoint_path.is_symlink():
                    checkpoint_path.unlink()
                checkpoint_path.symlink_to(resolved_env_path)
                print("Symlink created successfully!")
                return
            except Exception as link_err:
                print(f"Failed to create symlink: {link_err}. Attempting to copy file instead...")
                try:
                    import shutil
                    shutil.copy2(resolved_env_path, checkpoint_path)
                    print("Successfully copied checkpoint file!")
                    return
                except Exception as copy_err:
                    print(f"Failed to copy checkpoint: {copy_err}.")
        else:
            print(f"Warning: SAM2_CKPT_PATH is set to '{env_ckpt_path}' but file was not found.")

    # 3. Fallback: Download from Facebook URL
    url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
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
        if checkpoint_path.exists() or checkpoint_path.is_symlink():
            try:
                checkpoint_path.unlink()
            except Exception:
                pass
        print("\n====================================================")
        print("OFFLINE DETECTED OR DOWNLOAD FAILED.")
        print("If you are offline, please manually download the model weight file:")
        print(f"  {url}")
        print("And do one of the following:")
        print(f"  1. Place the file directly at: {checkpoint_path}")
        print(f"  2. Set SAM2_CKPT_PATH in your .env file to the download path (e.g. SAM2_CKPT_PATH=~/Downloads/sam2_hiera_large.pt)")
        print("====================================================")
        sys.exit(1)

def main():
    print("====================================================")
    print("          SAM2 Environment Setup Script             ")
    print("====================================================")
    
    if _DOTENV_AVAILABLE:
        load_dotenv()
        
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
