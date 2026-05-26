#!/usr/bin/env python3
"""
Setup and verification script for SAM3.

Checks dependencies, verifies HuggingFace authentication and access to facebook/sam3,
and triggers model warmup download to the default HuggingFace cache.

Usage:
    python baselines/sam3/setup_sam3.py [--check-only]
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies() -> bool:
    print("Checking dependencies...")
    libs = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("huggingface_hub", "HuggingFace Hub Tools"),
        ("yaml", "PyYAML"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
    ]
    all_ok = True
    for lib, name in libs:
        try:
            __import__(lib)
            print(f"  [✓] {name} is installed.")
        except ImportError:
            print(f"  [✗] {name} ({lib}) is MISSING.")
            all_ok = False
    return all_ok


def check_hf_access() -> bool:
    print("\nChecking Hugging Face authentication and permissions...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
    except ImportError:
        print("  [✗] huggingface_hub is not installed.")
        return False

    # 1. Check if logged in
    try:
        user_info = api.whoami()
        username = user_info.get("username", "Unknown")
        print(f"  [✓] Authenticated with Hugging Face as user: {username}")
    except Exception:
        print("  [✗] Not logged in to Hugging Face or HF_TOKEN is invalid.")
        print("      facebook/sam3 is a gated model. You MUST be logged in to download it.")
        print("      Please run: huggingface-cli login  (or set the HF_TOKEN environment variable).")
        return False

    # 2. Check access to facebook/sam3
    try:
        api.repo_info("facebook/sam3")
        print("  [✓] Access to facebook/sam3 repository is APPROVED!")
        return True
    except Exception as e:
        print("  [✗] Cannot access facebook/sam3 repository.")
        print("      Meta requires approval before downloading this model.")
        print("      Please visit: https://huggingface.co/facebook/sam3 and request access.")
        print("      Ensure you use the same HF account credentials.")
        return False


def run_warmup() -> bool:
    print("\nStarting model warmup (downloading and caching weights to default HF cache)...")
    print("This requires downloading ~3.6 GB. Please wait...")
    try:
        from transformers import Sam3Processor, Sam3Model
        
        print("Loading Sam3Processor...")
        processor = Sam3Processor.from_pretrained("facebook/sam3")
        print("Loading Sam3Model (this may take a few minutes for the first download)...")
        model = Sam3Model.from_pretrained("facebook/sam3")
        
        print("  [✓] SAM3 Model and Processor successfully cached!")
        return True
    except Exception as e:
        print(f"  [✗] Model warmup failed: {e}")
        return False


def run_sanity_check() -> bool:
    print("\nRunning basic sanity check on a dummy image...")
    try:
        import numpy as np
        from PIL import Image
        from baselines.sam3.sam3_standalone import SAM3Baseline

        config_path = str(PROJECT_ROOT / "baselines" / "sam3" / "config.yaml")
        
        # Instantiate baseline (force CPU for verification safety)
        print("Initializing SAM3Baseline on CPU...")
        baseline = SAM3Baseline(config_path=config_path)
        
        # Override queries list temporarily to a single fast query
        baseline._queries = ["sidewalk"]
        
        # Create a dummy grey image
        dummy = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
        
        # Run segmentation
        print("Running segment() on a dummy image...")
        result = baseline.segment(dummy)
        
        print("  [✓] Sanity check completed successfully!")
        print(f"      Result keys: {list(result.keys())}")
        print(f"      Masks shape: {result['masks'].shape}")
        return True
    except Exception as e:
        print(f"  [✗] Sanity check failed: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup and verify SAM3 environment")
    parser.add_argument(
        "--check-only", action="store_true", default=False,
        help="Only check dependencies and permissions, do not download weights"
    )
    args = parser.parse_args()

    print("==================================================")
    print("            SAM3 Environment Setup Checker        ")
    print("==================================================")

    if not check_dependencies():
        print("\nERROR: Missing dependencies. Please install required libraries first.")
        sys.exit(1)

    if not check_hf_access():
        print("\nERROR: Hugging Face authentication check failed.")
        sys.exit(1)

    if args.check_only:
        print("\nVerification check succeeded (dependencies & access verify only).")
        print("Run without --check-only to download weights and verify model execution.")
        sys.exit(0)

    if not run_warmup():
        print("\nERROR: Model warmup download failed.")
        sys.exit(1)

    if not run_sanity_check():
        print("\nERROR: Model execution sanity check failed.")
        sys.exit(1)

    print("\n==================================================")
    print("      SAM3 is fully set up and ready to use!      ")
    print("==================================================")


if __name__ == "__main__":
    main()
