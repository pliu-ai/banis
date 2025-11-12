#!/usr/bin/env python3
"""
Batch convert all zarr prediction files to TIFF format.
"""

import os
import glob
from pathlib import Path
from extract_segmentation import extract_segmentation_from_zarr


def batch_convert_predictions(base_dir, output_dir, threshold=0.5):
    """
    Convert all zarr prediction files in a directory to TIFF.
    
    Args:
        base_dir: Base directory containing zarr files
        output_dir: Output directory for TIFF files
        threshold: Threshold for binarizing affinities
    """
    
    # Find all zarr files
    zarr_pattern = os.path.join(base_dir, "**", "*.zarr")
    zarr_files = glob.glob(zarr_pattern, recursive=True)
    
    print(f"Found {len(zarr_files)} zarr files")
    
    if not zarr_files:
        print("No zarr files found!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    success_count = 0
    for zarr_file in zarr_files:
        print(f"\n{'='*60}")
        
        # Generate output filename
        rel_path = os.path.relpath(zarr_file, base_dir)
        output_name = rel_path.replace('.zarr', '_segmentation.tif')
        output_path = os.path.join(output_dir, output_name)
        
        # Extract segmentation
        success = extract_segmentation_from_zarr(zarr_file, output_path, threshold)
        
        if success:
            success_count += 1
            print(f"✓ Successfully converted: {zarr_file}")
        else:
            print(f"✗ Failed to convert: {zarr_file}")
    
    print(f"\n{'='*60}")
    print(f"Batch conversion completed!")
    print(f"Successfully converted: {success_count}/{len(zarr_files)} files")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    # Configuration - modify these paths as needed
    base_dirs = [
        "/projects/weilab/liupeng/banis/rib_outputs",
        "/projects/weilab/liupeng/banis/mito_outputs"
    ]
    output_dir = "/projects/weilab/liupeng/banis/tiff_outputs"
    threshold = 0.5
    
    print("Batch converting zarr predictions to TIFF...")
    print(f"Output directory: {output_dir}")
    print(f"Threshold: {threshold}")
    
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            print(f"\nProcessing directory: {base_dir}")
            batch_convert_predictions(base_dir, output_dir, threshold)
        else:
            print(f"Directory not found: {base_dir}")


if __name__ == "__main__":
    main()
