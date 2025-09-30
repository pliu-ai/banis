#!/bin/bash
# Rib segmentation batch extraction script
# Usage: ./run_rib_extraction.sh <input_dir> <gt_dir> [output_dir]

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_dir> <gt_dir> [output_dir]"
    echo "  input_dir: Directory containing pred_aff_test_*.zarr files"
    echo "  gt_dir: Directory containing *-rib-seg.nii.gz files"
    echo "  output_dir: Output directory (optional, defaults to input_dir)"
    exit 1
fi

INPUT_DIR="$1"
GT_DIR="$2"
OUTPUT_DIR="${3:-$INPUT_DIR}"

# Check if directories exist
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

if [ ! -d "$GT_DIR" ]; then
    echo "Error: Ground truth directory does not exist: $GT_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Rib Segmentation Batch Extraction"
echo "================================="
echo "Input directory: $INPUT_DIR"
echo "Ground truth directory: $GT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo

# Count input files
INPUT_COUNT=$(find "$INPUT_DIR" -name "pred_aff_test_*.zarr" | wc -l)
echo "Found $INPUT_COUNT input files"

# Count ground truth files
GT_COUNT=$(find "$GT_DIR" -name "*-rib-seg.nii.gz" | wc -l)
echo "Found $GT_COUNT ground truth files"
echo

if [ $INPUT_COUNT -eq 0 ]; then
    echo "Error: No input files found matching pattern 'pred_aff_test_*.zarr'"
    exit 1
fi

# Run batch extraction
echo "Starting batch extraction..."
python extract_segmentation.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --gt "$GT_DIR" \
    --batch \
    --threshold 0.5 \
    --min-size 200 \
    --morphological opening closing \
    --save-results

# Check if successful
if [ $? -eq 0 ]; then
    echo
    echo "✓ Batch extraction completed successfully!"
    echo "Results saved in: $OUTPUT_DIR"
    echo "Summary saved as: $OUTPUT_DIR/batch_results.json"
    
    # Show some statistics if results file exists
    if [ -f "$OUTPUT_DIR/batch_results.json" ]; then
        echo
        echo "Quick summary:"
        python -c "
import json
try:
    with open('$OUTPUT_DIR/batch_results.json', 'r') as f:
        results = json.load(f)
    print(f'  Total files: {results[\"total_files\"]}')
    print(f'  Successful: {results[\"successful\"]}')
    print(f'  Failed: {results[\"failed\"]}')
    print(f'  Evaluated: {results[\"evaluated\"]}')
    if 'overall_stats' in results and 'average_dice' in results['overall_stats']:
        print(f'  Average Dice: {results[\"overall_stats\"][\"average_dice\"]:.4f}')
except Exception as e:
    print(f'  Could not read results: {e}')
"
    fi
else
    echo
    echo "✗ Batch extraction failed!"
    exit 1
fi
