#!/usr/bin/env python3
"""
Convert BANIS segmentation outputs to neuron shape reasoning training format.
This script processes BANIS prediction files and converts them to point clouds.
"""

import argparse
import os
import glob
import numpy as np
import torch
import zarr
import tifffile
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd
from scipy import ndimage
from skimage import morphology, measure
from typing import List, Tuple, Dict


def load_mask(mask_path: str) -> np.ndarray:
    """Load mask from various formats."""
    if mask_path.endswith('.zarr'):
        mask = zarr.open(mask_path, mode='r')
        return np.array(mask)
    elif mask_path.endswith(('.tif', '.tiff')):
        return tifffile.imread(mask_path)
    elif mask_path.endswith('.npy'):
        return np.load(mask_path)
    else:
        raise ValueError(f"Unsupported file format: {mask_path}")


def remove_small_instances(seg: np.ndarray, min_size: int = 100) -> np.ndarray:
    """Remove instances smaller than min_size voxels."""
    unique_labels = np.unique(seg)
    unique_labels = unique_labels[unique_labels > 0]
    
    for label in unique_labels:
        mask = (seg == label)
        size = np.sum(mask)
        if size < min_size:
            seg[mask] = 0
    
    return seg


def mask_to_pointcloud(mask: np.ndarray, 
                      voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                      samples_per_neuron: int = 1024,
                      min_size: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert 3D segmentation mask to point cloud.
    
    Args:
        mask: 3D segmentation mask (H, W, D)
        voxel_size: Voxel size in (z, y, x) order
        samples_per_neuron: Number of points to sample
        min_size: Minimum instance size in voxels
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (points, labels)
    """
    # Remove small instances
    mask = remove_small_instances(mask, min_size)
    
    # Get unique labels (excluding background)
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]
    
    if len(unique_labels) == 0:
        # Return empty point cloud
        empty_points = torch.zeros(samples_per_neuron, 3)
        empty_labels = torch.zeros(samples_per_neuron)
        return empty_points, empty_labels
    
    # For simplicity, take the largest instance
    largest_label = get_largest_instance(mask, unique_labels)
    instance_mask = (mask == largest_label)
    
    # Get coordinates of this instance
    coords = np.where(instance_mask)
    
    if len(coords[0]) == 0:
        empty_points = torch.zeros(samples_per_neuron, 3)
        empty_labels = torch.zeros(samples_per_neuron)
        return empty_points, empty_labels
    
    # Convert to physical coordinates
    z_coords = coords[0] * voxel_size[0]
    y_coords = coords[1] * voxel_size[1]
    x_coords = coords[2] * voxel_size[2]
    
    # Stack coordinates
    points = np.column_stack((x_coords, y_coords, z_coords))
    
    # Sample points if we have too many
    if len(points) > samples_per_neuron:
        indices = np.random.choice(len(points), samples_per_neuron, replace=False)
        points = points[indices]
    
    # Pad with zeros if we have too few
    if len(points) < samples_per_neuron:
        padding = np.zeros((samples_per_neuron - len(points), 3))
        points = np.vstack([points, padding])
    
    # Create labels (all points belong to the same instance)
    labels = torch.ones(samples_per_neuron, dtype=torch.float32)
    
    return torch.tensor(points, dtype=torch.float32), labels


def get_largest_instance(mask: np.ndarray, labels: np.ndarray) -> int:
    """Get the label of the largest instance."""
    largest_size = 0
    largest_label = labels[0]
    
    for label in labels:
        size = np.sum(mask == label)
        if size > largest_size:
            largest_size = size
            largest_label = label
    
    return largest_label


def process_single_file(mask_path: str, 
                       output_dir: str,
                       voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                       samples_per_neuron: int = 1024,
                       min_size: int = 100) -> Dict:
    """Process a single mask file."""
    print(f"Processing {mask_path}...")
    
    try:
        # Load mask
        mask = load_mask(mask_path)
        
        # Convert to point cloud
        points, labels = mask_to_pointcloud(mask, voxel_size, samples_per_neuron, min_size)
        
        # Save point cloud
        base_name = Path(mask_path).stem
        output_path = Path(output_dir) / f"{base_name}_pointcloud.pt"
        
        torch.save({
            'points': points,
            'labels': labels,
            'voxel_size': voxel_size,
            'samples_per_neuron': samples_per_neuron
        }, output_path)
        
        return {
            'input_path': mask_path,
            'output_path': str(output_path),
            'n_points': len(points),
            'n_instances': len(torch.unique(labels[labels > 0])),
            'success': True
        }
        
    except Exception as e:
        print(f"Error processing {mask_path}: {e}")
        return {
            'input_path': mask_path,
            'error': str(e),
            'success': False
        }


def create_metadata_files(output_dir: str, processed_files: List[Dict]) -> None:
    """Create metadata files for neuron shape reasoning."""
    
    # Create family mapping
    family_mapping = {
        "rib_frac": 0,
        "background": 1
    }
    
    mapping_path = Path(output_dir) / "family_to_id.json"
    with open(mapping_path, 'w') as f:
        json.dump(family_mapping, f, indent=2)
    
    # Create neuron ID CSV
    data = []
    for i, file_info in enumerate(processed_files):
        if file_info['success']:
            data.append({
                'root_id': Path(file_info['input_path']).stem,
                'family': 'rib_frac'
            })
    
    csv_path = Path(output_dir) / "neuron_ids.csv"
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    print(f"Created metadata files:")
    print(f"  Family mapping: {mapping_path}")
    print(f"  Neuron IDs CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert BANIS masks to neuron shape reasoning format")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Input directory containing BANIS prediction files")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Output directory for point clouds")
    parser.add_argument("--voxel_size", type=float, nargs=3, default=[1.0, 1.0, 1.0], 
                       help="Voxel size in z, y, x order")
    parser.add_argument("--samples_per_neuron", type=int, default=1024, 
                       help="Number of points to sample per neuron")
    parser.add_argument("--min_size", type=int, default=100, 
                       help="Minimum instance size in voxels")
    parser.add_argument("--pattern", type=str, default="*.zarr", 
                       help="File pattern to match")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find mask files
    mask_files = glob.glob(os.path.join(args.input_dir, args.pattern))
    
    if not mask_files:
        print(f"No files found matching pattern {args.pattern} in {args.input_dir}")
        return
    
    print(f"Found {len(mask_files)} mask files")
    
    # Process each file
    processed_files = []
    for mask_file in tqdm(mask_files, desc="Processing masks"):
        result = process_single_file(
            mask_file, 
            args.output_dir,
            tuple(args.voxel_size),
            args.samples_per_neuron,
            args.min_size
        )
        processed_files.append(result)
    
    # Create metadata files
    create_metadata_files(args.output_dir, processed_files)
    
    # Print summary
    successful = [f for f in processed_files if f['success']]
    failed = [f for f in processed_files if not f['success']]
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(successful)} files")
    print(f"Failed: {len(failed)} files")
    
    if successful:
        total_points = sum(f['n_points'] for f in successful)
        total_instances = sum(f['n_instances'] for f in successful)
        print(f"Total points: {total_points}")
        print(f"Total instances: {total_instances}")
    
    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  {f['input_path']}: {f['error']}")


if __name__ == "__main__":
    main()

