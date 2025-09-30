# Zarr to TIFF Conversion Scripts

This directory contains scripts to convert zarr prediction files to TIFF format for easier visualization and analysis.

## Files Created

1. **`extract_segmentation.py`** - Main script for converting single zarr files
2. **`batch_convert.py`** - Batch processing script for multiple files
3. **`example_convert.py`** - Example usage scripts
4. **`zarr_to_tiff.py`** - Full-featured conversion script with command-line interface

## Quick Start

### Convert a Single File

```bash
python extract_segmentation.py \
    --input /path/to/pred_aff_val_RibFrac400.zarr \
    --output /path/to/output/RibFrac400_segmentation.tif \
    --threshold 0.5
```

### Batch Convert All Files

```bash
python batch_convert.py
```

This will automatically find all zarr files in your output directories and convert them.

### Using the Full-Featured Script

```bash
python zarr_to_tiff.py \
    --input /path/to/zarr/files/ \
    --output /path/to/tiff/output/ \
    --threshold 0.5 \
    --save-affinities
```

## How It Works

1. **Load Zarr Data**: Reads affinity predictions from zarr files
2. **Convert to Probabilities**: Applies sigmoid function to convert logits to probabilities
3. **Extract Short Range**: Uses only the first 3 channels (short range affinities)
4. **Binarize**: Applies threshold to create binary connections
5. **Connected Components**: Uses depth-first search to group connected pixels
6. **Save as TIFF**: Saves the resulting segmentation as a compressed TIFF file

## Parameters

- **`--threshold`**: Threshold for binarizing affinities (default: 0.5)
- **`--input`**: Input zarr file or directory
- **`--output`**: Output TIFF file or directory
- **`--save-affinities`**: Also save individual affinity channels as TIFF

## Output Files

For each input zarr file, the scripts will create:
- `{name}_segmentation.tif` - Instance segmentation mask
- `{name}_affinity_ch1.tif` - X-direction short range affinity
- `{name}_affinity_ch2.tif` - Y-direction short range affinity  
- `{name}_affinity_ch3.tif` - Z-direction short range affinity
- `{name}_affinity_ch4.tif` - X-direction long range affinity
- `{name}_affinity_ch5.tif` - Y-direction long range affinity
- `{name}_affinity_ch6.tif` - Z-direction long range affinity

## Example Usage

```python
from extract_segmentation import extract_segmentation_from_zarr

# Convert a single file
success = extract_segmentation_from_zarr(
    "pred_aff_val_RibFrac400.zarr",
    "RibFrac400_segmentation.tif",
    threshold=0.5
)

if success:
    print("Conversion successful!")
```

## Notes

- The scripts use only short range affinities (first 3 channels) for segmentation
- Long range affinities are used during training but not for final segmentation
- The threshold parameter can be adjusted to optimize segmentation quality
- Output TIFF files use LZW compression to reduce file size
