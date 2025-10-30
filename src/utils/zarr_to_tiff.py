#!/usr/bin/env python3
"""
Convert zarr prediction files to TIFF format.

This script reads affinity predictions from zarr files and converts them to 
instance segmentation masks, then saves as TIFF files.
"""

import argparse
import os
import numpy as np
import zarr
from pathlib import Path
from tqdm import tqdm
import tifffile
from inference import compute_connected_component_segmentation, scale_sigmoid
import torch


def load_zarr_predictions(zarr_path):
    """Load affinity predictions from zarr file."""
    print(f"Loading predictions from: {zarr_path}")
    zarr_array = zarr.open(zarr_path, mode='r')
    
    # Get the data shape
    print(f"Zarr shape: {zarr_array.shape}")
    print(f"Zarr chunks: {zarr_array.chunks}")
    print(f"Zarr dtype: {zarr_array.dtype}")
    
    # Load the data
    data = zarr_array[:]
    print(f"Loaded data shape: {data.shape}")
    
    return data


def affinity_to_segmentation(aff_pred, threshold=0.5, use_short_range_only=True):
    """
    Convert affinity predictions to instance segmentation.
    
    Args:
        aff_pred: Affinity predictions, shape (channels, x, y, z)
        threshold: Threshold for binarizing affinities
        use_short_range_only: If True, only use first 3 channels (short range)
    
    Returns:
        Instance segmentation mask, shape (x, y, z)
    """
    print(f"Converting affinities to segmentation with threshold {threshold}")
    
    # Convert to numpy if it's a torch tensor
    if isinstance(aff_pred, torch.Tensor):
        aff_pred = aff_pred.cpu().numpy()
    
    # Apply sigmoid to convert logits to probabilities
    aff_prob = scale_sigmoid(torch.from_numpy(aff_pred)).numpy()
    
    # Use only short range affinities (first 3 channels)
    if use_short_range_only and aff_prob.shape[0] >= 3:
        short_range_aff = aff_prob[:3]
    else:
        short_range_aff = aff_prob
    
    print(f"Using {short_range_aff.shape[0]} affinity channels")
    
    # Binarize with threshold
    hard_aff = (short_range_aff > threshold).astype(np.uint8)
    
    # Compute connected components
    print("Computing connected components...")
    seg = compute_connected_component_segmentation(hard_aff)
    
    print(f"Segmentation shape: {seg.shape}")
    print(f"Number of instances: {len(np.unique(seg)) - 1}")  # -1 for background
    
    return seg


def save_as_tiff(data, output_path, compression='lzw'):
    """Save data as TIFF file."""
    print(f"Saving to: {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as TIFF
    tifffile.imwrite(
        output_path, 
        data, 
        compression=compression,
        metadata={'axes': 'ZYX' if len(data.shape) == 3 else 'CYX' if len(data.shape) == 3 else 'ZYX'}
    )
    print(f"Saved successfully!")


def process_single_zarr(zarr_path, output_dir, threshold=0.5, save_affinities=False):
    """Process a single zarr file."""
    print(f"\n{'='*60}")
    print(f"Processing: {zarr_path}")
    print(f"{'='*60}")
    
    # Load predictions
    aff_pred = load_zarr_predictions(zarr_path)
    
    # Convert to segmentation
    seg = affinity_to_segmentation(aff_pred, threshold=threshold)
    
    # Generate output filename
    zarr_name = Path(zarr_path).stem
    output_path = os.path.join(output_dir, f"{zarr_name}_segmentation.tif")
    
    # Save segmentation
    save_as_tiff(seg, output_path)
    
    # Optionally save individual affinity channels
    if save_affinities:
        print("Saving individual affinity channels...")
        for i in range(min(6, aff_pred.shape[0])):  # Save up to 6 channels
            aff_channel = aff_pred[i]
            aff_output_path = os.path.join(output_dir, f"{zarr_name}_affinity_ch{i+1}.tif")
            save_as_tiff(aff_channel, aff_output_path)
    
    return seg


def main():
    parser = argparse.ArgumentParser(description="Convert zarr predictions to TIFF")
    parser.add_argument("--input", "-i", required=True, 
                       help="Input zarr file or directory containing zarr files")
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory for TIFF files")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Threshold for binarizing affinities (default: 0.5)")
    parser.add_argument("--save-affinities", action="store_true",
                       help="Also save individual affinity channels as TIFF")
    parser.add_argument("--pattern", default="*.zarr",
                       help="Pattern to match zarr files (default: *.zarr)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Find input files
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        zarr_files = [str(input_path)]
    elif input_path.is_dir():
        # Directory - find all zarr files
        zarr_files = list(input_path.glob(args.pattern))
        zarr_files = [str(f) for f in zarr_files if f.is_file()]
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    
    if not zarr_files:
        print(f"No zarr files found matching pattern: {args.pattern}")
        return
    
    print(f"Found {len(zarr_files)} zarr files to process")
    
    # Process each file
    for zarr_file in tqdm(zarr_files, desc="Processing zarr files"):
        try:
            process_single_zarr(
                zarr_file, 
                args.output, 
                threshold=args.threshold,
                save_affinities=args.save_affinities
            )
        except Exception as e:
            print(f"Error processing {zarr_file}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Processing completed!")
    print(f"Output directory: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
