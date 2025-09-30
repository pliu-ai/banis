# Batch Segmentation Extraction

This document describes how to use the enhanced `extract_segmentation.py` script for batch processing of zarr files.

## Features

- **Single File Processing**: Process individual zarr files (original functionality)
- **Batch Processing**: Process entire directories of zarr files
- **Automatic Ground Truth Matching**: Automatically find corresponding ground truth files
- **Evaluation**: Compute Dice scores and other metrics when ground truth is available
- **Progress Tracking**: Real-time progress bars and detailed logging
- **Results Summary**: JSON summary of batch processing results

## Usage

### Single File Processing

```bash
python extract_segmentation.py \
    --input /path/to/file.zarr \
    --output /path/to/output.tif \
    --gt /path/to/ground_truth.tif \
    --threshold 0.5 \
    --min-size 200
```

### Batch Processing

```bash
python extract_segmentation.py \
    --input /path/to/zarr/directory \
    --output /path/to/output/directory \
    --gt /path/to/ground_truth/directory \
    --batch \
    --threshold 0.5 \
    --min-size 200
```

### Automatic Batch Processing (when input is a directory)

```bash
python extract_segmentation.py \
    --input /path/to/zarr/directory \
    --output /path/to/output/directory \
    --gt /path/to/ground_truth/directory \
    --threshold 0.5
```

### Rib Segmentation Specific Pattern

For files with pattern `pred_aff_test_{image_id}.zarr` and ground truth `{image_id}-rib-seg.nii.gz`:

```bash
python extract_segmentation.py \
    --input /path/to/pred_aff_test_files \
    --output /path/to/output \
    --gt /path/to/ground_truth_files \
    --batch \
    --threshold 0.5 \
    --min-size 200
```

Or use the specialized script:
```bash
python rib_batch_extract.py
```

## Command Line Arguments

### Required Arguments
- `--input, -i`: Input zarr file path or directory for batch processing

### Optional Arguments

#### Input/Output
- `--output, -o`: Output TIFF file path or directory (for batch processing)
- `--batch`: Process all files in input directory (optional if input is a directory)

#### Ground Truth
- `--gt`: Path to ground truth segmentation file or directory

#### Processing Parameters
- `--threshold, -t`: Threshold for binarizing affinities (default: 0.5)
- `--min-size`: Minimum instance size in voxels (default: 200)
- `--max-instances`: Maximum number of instances to keep (default: None)
- `--morphological`: Morphological operations ['opening', 'closing']

#### Batch Processing
- `--save-results`: Save batch processing results summary (default: True)

## File Matching

The script automatically matches input zarr files with ground truth files using several strategies:

1. **Exact name matching**: `file.zarr` → `file.tif`
2. **Prefix removal**: `pred_aff_file.zarr` → `file.tif`
3. **Format conversion**: `file.zarr` → `file.nii.gz`
4. **Pattern matching**: Uses partial name matching for flexible file organization

## Output Files

### Individual Files
- `filename.tif`: Segmentation mask (TIFF format)
- `filename_eval.txt`: Evaluation results (if ground truth provided)

### Batch Results
- `batch_results.json`: Complete summary of batch processing including:
  - Total files processed
  - Success/failure counts
  - Individual file results
  - Overall statistics (average Dice, instance counts, etc.)

## Example Directory Structure

### General Case
```
input_directory/
├── pred_aff_file1.zarr
├── pred_aff_file2.zarr
└── pred_aff_file3.zarr

ground_truth_directory/
├── file1.tif
├── file2.tif
└── file3.tif

output_directory/
├── file1.tif
├── file1_eval.txt
├── file2.tif
├── file2_eval.txt
├── file3.tif
├── file3_eval.txt
└── batch_results.json
```

### Rib Segmentation Specific Pattern
```
input_directory/
├── pred_aff_test_001.zarr
├── pred_aff_test_002.zarr
└── pred_aff_test_003.zarr

ground_truth_directory/
├── 001-rib-seg.nii.gz
├── 002-rib-seg.nii.gz
└── 003-rib-seg.nii.gz

output_directory/
├── 001.tif
├── 001_eval.txt
├── 002.tif
├── 002_eval.txt
├── 003.tif
├── 003_eval.txt
└── batch_results.json
```

## Evaluation Metrics

When ground truth is provided, the script computes:
- **Dice Coefficient**: Per-instance and average Dice scores
- **Instance Matching**: Optimal matching between predicted and ground truth instances
- **Instance Counts**: Number of predicted vs ground truth instances

## Error Handling

- **File Not Found**: Skips missing files and reports in summary
- **Processing Errors**: Captures and reports individual file errors
- **Memory Issues**: Handles large files with appropriate error messages
- **Format Issues**: Attempts multiple file format readers

## Performance Tips

1. **Memory Management**: Process large datasets in smaller batches if memory is limited
2. **Parallel Processing**: Consider running multiple instances on different file subsets
3. **Storage**: Ensure sufficient disk space for output files
4. **Ground Truth**: Provide ground truth for evaluation when available

## Troubleshooting

### Common Issues

1. **No files found**: Check file patterns and directory paths
2. **Memory errors**: Reduce batch size or process files individually
3. **Ground truth not found**: Check file naming conventions and patterns
4. **Low Dice scores**: Adjust threshold and morphological parameters

### Debug Mode

Add verbose logging by modifying the script or using Python's logging module:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Example Scripts

See `batch_extract_example.py` for complete usage examples and `test_batch_extraction.sh` for a ready-to-use test script.
