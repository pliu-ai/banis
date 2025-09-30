#!/usr/bin/env python3
"""
Install dependencies for evaluation module.
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def main():
    """Install all required packages for evaluation."""
    print("Installing evaluation dependencies...")
    print("=" * 40)
    
    packages = [
        "numpy",
        "scipy", 
        "scikit-image",
        "tifffile",
        "zarr",
        "SimpleITK"
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print("=" * 40)
    print(f"Installation completed: {success_count}/{len(packages)} packages installed successfully")
    
    if success_count == len(packages):
        print("✓ All dependencies installed successfully!")
        print("\nYou can now use the evaluation module:")
        print("  python evaluate_segmentation.py --pred pred.tif --gt gt.nii.gz --output results.txt")
    else:
        print("✗ Some packages failed to install. Please install them manually:")
        for package in packages:
            print(f"  pip install {package}")

if __name__ == "__main__":
    main()



