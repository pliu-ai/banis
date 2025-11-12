#!/usr/bin/env python3
"""
Simple script to convert zarr predictions to TIFF files.
"""

import os
import numpy as np
import zarr
import tifffile
from pathlib import Path
from tqdm import tqdm
from inference import compute_connected_component_segmentation, scale_sigmoid
import torch


def convert_zarr_to_tiff(zarr_path, output_dir, threshold=0.5):
    """Convert a single zarr file to TIFF segmentation."""
    
    print(f"Processing: {zarr_path}")
    
    # Load zarr data
    zarr_array = zarr.open(zarr_path, mode='r')
    aff_pred = zarr_array[:]
    print(f"Loaded affinity predictions shape: {aff_pred.shape}")
    
    # Convert to probabilities
    aff_prob = scale_sigmoid(torch.from_numpy(aff_pred)).numpy()
    
    # Use only short range affinities (first 3 channels)
    short_range_aff = aff_prob[:3]
    print(f"Using short range affinities shape: {short_range_aff.shape}")
    
    # Binarize with threshold
    hard_aff = (short_range_aff > threshold).astype(np.uint8)
    
    # Compute connected components to get segmentation
    print("Computing connected components...")
    seg = compute_connected_component_segmentation(hard_aff)
    
    print(f"Segmentation shape: {seg.shape}")
    print(f"Number of instances: {len(np.unique(seg)) - 1}")
    
    # Save segmentation
    zarr_name = Path(zarr_path).stem
    output_path = os.path.join(output_dir, f"{zarr_name}_segmentation.tif")
    
    os.makedirs(output_dir, exist_ok=True)
    tifffile.imwrite(output_path, seg, compression='lzw')
    print(f"Saved segmentation to: {output_path}")
    
    # Also save individual affinity channels
    for i in range(min(6, aff_pred.shape[0])):
        aff_channel = aff_pred[i]
        aff_output_path = os.path.join(output_dir, f"{zarr_name}_affinity_ch{i+1}.tif")
        tifffile.imwrite(aff_output_path, aff_channel, compression='lzw')
        print(f"Saved affinity channel {i+1} to: {aff_output_path}")
    
    return seg


def main():
    # Configuration
    input_dir = "/projects/weilab/liupeng/banis/rib_outputs/25-09-09_15-32-14-552514ds_ribFrac_lrng10_s0_b4_mS_k3_lr0.001_wd0.01_schTrue_syn_1.0_drop0.05_shift0.05_intTrue_noise0.5_affine0.5_ns500000_ss128_sdt1_sdtw1.0"
    output_dir = "/projects/weilab/liupeng/banis/tiff_outputs"
    threshold = 0.5
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Threshold: {threshold}")
    
    # Find all zarr files
    zarr_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.zarr'):
                zarr_files.append(os.path.join(root, file))
    
    print(f"Found {len(zarr_files)} zarr files")
    
    # Process each file
    for zarr_file in tqdm(zarr_files, desc="Converting zarr to tiff"):
        try:
            convert_zarr_to_tiff(zarr_file, output_dir, threshold)
        except Exception as e:
            print(f"Error processing {zarr_file}: {e}")
            continue
    
    print("Conversion completed!")


if __name__ == "__main__":
    main()
