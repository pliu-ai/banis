#!/usr/bin/env python3
"""
Simple zarr to tiff conversion without compression dependencies.
"""

import os
import numpy as np
import zarr
import tifffile
from pathlib import Path
from inference import compute_connected_component_segmentation, scale_sigmoid
import torch
from scipy import ndimage
from skimage import morphology, measure


def remove_small_instances(seg, min_size=100):
    """Remove instances smaller than min_size voxels."""
    print(f"Removing instances smaller than {min_size} voxels...")
    
    unique_labels = np.unique(seg)
    unique_labels = unique_labels[unique_labels > 0]
    
    initial_count = len(unique_labels)
    
    for label in unique_labels:
        mask = (seg == label)
        size = np.sum(mask)
        if size < min_size:
            seg[mask] = 0
    
    final_labels = np.unique(seg)
    final_labels = final_labels[final_labels > 0]
    final_count = len(final_labels)
    
    print(f"Removed {initial_count - final_count} small instances")
    print(f"Remaining instances: {final_count}")
    
    return seg


def keep_largest_instances(seg, max_instances=22):
    """Keep only the largest max_instances instances."""
    print(f"Keeping only the largest {max_instances} instances...")
    
    unique_labels = np.unique(seg)
    unique_labels = unique_labels[unique_labels > 0]
    
    initial_count = len(unique_labels)
    
    if initial_count <= max_instances:
        print(f"Only {initial_count} instances found, keeping all")
        return seg
    
    # Calculate size for each instance
    instance_sizes = []
    for label in unique_labels:
        mask = (seg == label)
        size = np.sum(mask)
        instance_sizes.append((label, size))
    
    # Sort by size (descending)
    instance_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Keep only the largest instances
    labels_to_keep = [label for label, _ in instance_sizes[:max_instances]]
    
    # Remove smaller instances
    for label in unique_labels:
        if label not in labels_to_keep:
            mask = (seg == label)
            seg[mask] = 0
    
    # Relabel to ensure consecutive IDs
    seg = relabel_consecutive(seg)
    
    final_count = len(np.unique(seg)) - 1
    print(f"Kept {final_count} largest instances (removed {initial_count - final_count})")
    
    return seg


def relabel_consecutive(seg):
    """Relabel segmentation to have consecutive IDs starting from 1."""
    unique_labels = np.unique(seg)
    unique_labels = unique_labels[unique_labels > 0]
    
    if len(unique_labels) == 0:
        return seg
    
    # Create mapping from old labels to new consecutive labels
    label_mapping = {old_label: new_label + 1 for new_label, old_label in enumerate(unique_labels)}
    
    # Apply mapping
    new_seg = np.zeros_like(seg)
    for old_label, new_label in label_mapping.items():
        mask = (seg == old_label)
        new_seg[mask] = new_label
    
    return new_seg


def simple_convert(zarr_path, output_path, threshold=0.5, min_size=100, max_instances=22):
    """Simple conversion without compression."""
    
    print(f"Converting: {zarr_path}")
    
    # Load zarr data
    zarr_array = zarr.open(zarr_path, mode='r')
    aff_pred = zarr_array[:]
    print(f"Loaded data shape: {aff_pred.shape}, dtype: {aff_pred.dtype}")
    
    # Convert to probabilities
    if aff_pred.dtype == np.float16:
        aff_pred = aff_pred.astype(np.float32)
    
    aff_prob = scale_sigmoid(torch.from_numpy(aff_pred)).numpy()
    print(f"Probability range: {aff_prob.min():.4f} to {aff_prob.max():.4f}")
    
    # Use short range affinities
    short_range_aff = aff_prob[:3]
    print(f"Using {short_range_aff.shape[0]} channels")
    
    # Try multiple thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7]
    best_seg = None
    best_thresh = None
    
    for thresh in thresholds:
        print(f"Trying threshold {thresh}...")
        hard_aff = (short_range_aff > thresh).astype(np.uint8)
        connections = hard_aff.sum()
        print(f"  Connections: {connections}")
        
        if connections > 0:
            try:
                seg = compute_connected_component_segmentation(hard_aff)
                unique_ids = np.unique(seg)
                num_instances = len(unique_ids) - 1
                print(f"  Instances: {num_instances}")
                
                if num_instances > 0:
                    # Apply post-processing
                    seg = remove_small_instances(seg, min_size)
                    seg = keep_largest_instances(seg, max_instances)
                    best_seg = seg
                    best_thresh = thresh
                    print(f"  ✓ Found {num_instances} instances with threshold {thresh}")
                    break
            except Exception as e:
                print(f"  ✗ Error: {e}")
        else:
            print(f"  ✗ No connections")
    
    if best_seg is None:
        print("No valid segmentation found with any threshold!")
        return False
    
    # Save without compression
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tifffile.imwrite(output_path, best_seg)
    print(f"✓ Saved segmentation to: {output_path}")
    print(f"  Used threshold: {best_thresh}")
    print(f"  Instances: {len(np.unique(best_seg)) - 1}")
    
    return True


def main():
    """Convert all zarr files in the output directory."""
    
    # Input directory
    input_dir = "/projects/weilab/liupeng/banis/rib_outputs/25-09-09_15-32-14-552514ds_ribFrac_lrng10_s0_b4_mS_k3_lr0.001_wd0.01_schTrue_syn_1.0_drop0.05_shift0.05_intTrue_noise0.5_affine0.5_ns500000_ss128_sdt1_sdtw1.0"
    output_dir = "/projects/weilab/liupeng/banis/tiff_outputs"
    min_size = 100  # Minimum instance size in voxels
    max_instances = 22  # Maximum number of instances to keep
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Minimum instance size: {min_size} voxels")
    print(f"Maximum instances: {max_instances}")
    
    # Find zarr files
    zarr_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.zarr'):
                zarr_files.append(os.path.join(root, file))
    
    print(f"Found {len(zarr_files)} zarr files")
    
    # Convert each file
    success_count = 0
    for zarr_file in zarr_files:
        print(f"\n{'='*60}")
        
        # Generate output filename
        rel_path = os.path.relpath(zarr_file, input_dir)
        output_name = rel_path.replace('.zarr', '_segmentation.tif')
        output_path = os.path.join(output_dir, output_name)
        
        # Convert
        if simple_convert(zarr_file, output_path, min_size=min_size, max_instances=max_instances):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Conversion completed!")
    print(f"Successfully converted: {success_count}/{len(zarr_files)} files")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
