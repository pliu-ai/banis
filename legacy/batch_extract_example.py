#!/usr/bin/env python3
"""
Example script showing how to use the batch extraction functionality.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_batch_extraction():
    """
    Example of how to run batch extraction with different configurations.
    """
    
    # Example 1: Basic batch processing
    print("Example 1: Basic batch processing")
    print("=" * 50)
    cmd1 = [
        "python", "extract_segmentation.py",
        "--input", "/path/to/zarr/files",  # Directory containing .zarr files
        "--batch",  # Enable batch processing
        "--threshold", "0.5",
        "--min-size", "100",
        "--max-instances", "22"
    ]
    print("Command:", " ".join(cmd1))
    print()
    
    # Example 2: Batch processing with ground truth evaluation
    print("Example 2: Batch processing with ground truth evaluation")
    print("=" * 50)
    cmd2 = [
        "python", "extract_segmentation.py",
        "--input", "/path/to/zarr/files",
        "--output", "/path/to/output/directory",
        "--gt", "/path/to/ground/truth/files",
        "--batch",
        "--threshold", "0.4",
        "--min-size", "50",
        "--max-instances", "30",
        "--morphological", "opening", "closing"
    ]
    print("Command:", " ".join(cmd2))
    print()
    
    # Example 3: Custom file patterns
    print("Example 3: Custom file patterns")
    print("=" * 50)
    cmd3 = [
        "python", "extract_segmentation.py",
        "--input", "/path/to/predictions",
        "--pattern", "pred_aff_*.zarr",  # Custom pattern for input files
        "--gt", "/path/to/ground/truth",
        "--gt-pattern", "gt_*.tif",  # Custom pattern for ground truth files
        "--batch",
        "--threshold", "0.6"
    ]
    print("Command:", " ".join(cmd3))
    print()
    
    # Example 4: Single file processing (original functionality)
    print("Example 4: Single file processing")
    print("=" * 50)
    cmd4 = [
        "python", "extract_segmentation.py",
        "--input", "/path/to/single_file.zarr",
        "--output", "/path/to/output.tif",
        "--gt", "/path/to/ground_truth.tif",
        "--threshold", "0.5"
    ]
    print("Command:", " ".join(cmd4))
    print()

def create_test_script():
    """
    Create a test script for the batch extraction functionality.
    """
    script_content = '''#!/bin/bash
# Test script for batch extraction

# Set paths (modify these according to your setup)
ZARR_DIR="/path/to/your/zarr/files"
OUTPUT_DIR="/path/to/output"
GT_DIR="/path/to/ground/truth"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run batch extraction
python extract_segmentation.py \\
    --input "$ZARR_DIR" \\
    --output "$OUTPUT_DIR" \\
    --gt "$GT_DIR" \\
    --batch \\
    --threshold 0.5 \\
    --min-size 100 \\
    --max-instances 22 \\
    --morphological opening closing \\
    --save-results

echo "Batch extraction completed!"
echo "Check the results in: $OUTPUT_DIR"
echo "Summary saved as: $OUTPUT_DIR/batch_results.json"
'''
    
    with open("test_batch_extraction.sh", "w") as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod("test_batch_extraction.sh", 0o755)
    print("Created test script: test_batch_extraction.sh")

if __name__ == "__main__":
    print("Batch Extraction Examples")
    print("=" * 30)
    print()
    
    run_batch_extraction()
    create_test_script()
    
    print("Usage Summary:")
    print("- Use --batch flag or provide a directory as input for batch processing")
    print("- Use --gt to specify ground truth directory for evaluation")
    print("- Use --pattern and --gt-pattern to customize file matching")
    print("- Results are saved as .tif files with optional evaluation")
    print("- Batch results summary is saved as batch_results.json")

