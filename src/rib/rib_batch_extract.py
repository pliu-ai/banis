#!/usr/bin/env python3
"""
Rib segmentation batch extraction script for specific file naming pattern:
- Input: pred_aff_test_{image_id}.zarr
- Ground Truth: {image_id}-rib-seg.nii.gz
"""

import os
import subprocess
import sys
from pathlib import Path

def run_rib_batch_extraction(input_dir, gt_dir, output_dir=None, **kwargs):
    """
    Run batch extraction for rib segmentation with specific file naming pattern.
    
    Args:
        input_dir: Directory containing pred_aff_test_*.zarr files
        gt_dir: Directory containing *-rib-seg.nii.gz files
        output_dir: Output directory (default: same as input_dir)
        **kwargs: Additional parameters for extraction
    """
    
    # Default parameters
    params = {
        'threshold': 0.5,
        'min_size': 200,
        'max_instances': None,
        'morphological': ['opening', 'closing'],
        'save_results': True
    }
    params.update(kwargs)
    
    # Build command
    cmd = [
        'python', 'extract_segmentation.py',
        '--input', input_dir,
        '--gt', gt_dir,
        '--batch',
        '--threshold', str(params['threshold']),
        '--min-size', str(params['min_size']),
        '--max-instances', str(params['max_instances']),
    ]
    
    if params['morphological']:
        cmd.extend(['--morphological'] + params['morphological'])
    
    if output_dir:
        cmd.extend(['--output', output_dir])
    
    if params['save_results']:
        cmd.append('--save-results')
    
    print("Running rib batch extraction...")
    print("Command:", ' '.join(cmd))
    print()
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    """
    Main function with example usage.
    """
    print("Rib Segmentation Batch Extraction")
    print("=" * 40)
    print()
    
    # Example usage - modify these paths according to your setup
    input_dir = "/path/to/pred_aff_test_files"  # Directory with pred_aff_test_*.zarr
    gt_dir = "/path/to/ground_truth_files"      # Directory with *-rib-seg.nii.gz
    output_dir = "/path/to/output"              # Output directory (optional)
    
    print("Example 1: Basic batch extraction")
    print("-" * 30)
    success = run_rib_batch_extraction(
        input_dir=input_dir,
        gt_dir=gt_dir,
        output_dir=output_dir,
        threshold=0.5,
        min_size=200,
        max_instances=None
    )
    
    if success:
        print("✓ Batch extraction completed successfully!")
    else:
        print("✗ Batch extraction failed!")
    
    print()
    
    print("Example 2: Custom parameters")
    print("-" * 30)
    success = run_rib_batch_extraction(
        input_dir=input_dir,
        gt_dir=gt_dir,
        output_dir=output_dir,
        threshold=0.4,  # Lower threshold
        min_size=100,   # Smaller minimum size
        max_instances=None,  # No limit on instances
        morphological=['opening']  # Only opening
    )
    
    print()
    print("File naming pattern:")
    print("Input files:  pred_aff_test_{image_id}.zarr")
    print("GT files:     {image_id}-rib-seg.nii.gz")
    print("Output files: {image_id}.tif")
    print()
    print("Example file pairs:")
    print("  pred_aff_test_001.zarr -> 001-rib-seg.nii.gz -> 001.tif")
    print("  pred_aff_test_002.zarr -> 002-rib-seg.nii.gz -> 002.tif")
    print("  pred_aff_test_003.zarr -> 003-rib-seg.nii.gz -> 003.tif")

if __name__ == "__main__":
    main()
