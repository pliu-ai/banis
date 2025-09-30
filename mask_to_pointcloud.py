#!/usr/bin/env python3
"""
Convert BANIS segmentation masks to point clouds for neuron shape reasoning training.
This script converts 3D segmentation masks to SWC format and point clouds.
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
import navis
from scipy import ndimage
from skimage import morphology, measure
import json
from typing import List, Tuple, Dict
import pandas as pd


def mask_to_swc(mask: np.ndarray, 
                voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                min_size: int = 100) -> navis.TreeNeuron:
    """
    Convert 3D segmentation mask to SWC format.
    
    Args:
        mask: 3D segmentation mask (H, W, D)
        voxel_size: Voxel size in (z, y, x) order
        min_size: Minimum instance size in voxels
    
    Returns:
        navis.TreeNeuron: Neuron object in SWC format
    """
    # Remove small instances
    mask = remove_small_instances(mask, min_size)
    
    # Get unique labels (excluding background)
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]
    
    if len(unique_labels) == 0:
        # Create empty neuron if no instances found
        return create_empty_neuron()
    
    # For now, take the largest instance
    # TODO: Handle multiple instances properly
    largest_label = get_largest_instance(mask, unique_labels)
    instance_mask = (mask == largest_label)
    
    # Skeletonize the instance
    skeleton = morphology.skeletonize_3d(instance_mask)
    
    # Convert skeleton to SWC format
    swc_data = skeleton_to_swc(skeleton, voxel_size)
    
    # Create navis TreeNeuron
    neuron = navis.TreeNeuron(swc_data)
    
    return neuron


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


def skeleton_to_swc(skeleton: np.ndarray, 
                   voxel_size: Tuple[float, float, float]) -> pd.DataFrame:
    """
    Convert 3D skeleton to SWC format.
    
    Args:
        skeleton: 3D skeleton array (H, W, D)
        voxel_size: Voxel size in (z, y, x) order
    
    Returns:
        pd.DataFrame: SWC data
    """
    # Get skeleton points
    coords = np.where(skeleton > 0)
    
    if len(coords[0]) == 0:
        return create_empty_swc()
    
    # Convert to physical coordinates
    z_coords = coords[0] * voxel_size[0]
    y_coords = coords[1] * voxel_size[1]
    x_coords = coords[2] * voxel_size[2]
    
    # Create SWC data
    n_points = len(coords[0])
    swc_data = []
    
    for i in range(n_points):
        # SWC format: id, type, x, y, z, radius, parent_id
        # For simplicity, create a linear chain
        parent_id = i if i == 0 else i - 1
        swc_data.append({
            'node_id': i + 1,
            'type': 3,  # dendrite
            'x': x_coords[i],
            'y': y_coords[i], 
            'z': z_coords[i],
            'radius': 1.0,  # default radius
            'parent_id': parent_id
        })
    
    return pd.DataFrame(swc_data)


def create_empty_neuron() -> navis.TreeNeuron:
    """Create an empty neuron."""
    empty_swc = create_empty_swc()
    return navis.TreeNeuron(empty_swc)


def create_empty_swc() -> pd.DataFrame:
    """Create empty SWC data."""
    return pd.DataFrame({
        'node_id': [1],
        'type': [1],
        'x': [0.0],
        'y': [0.0],
        'z': [0.0],
        'radius': [1.0],
        'parent_id': [-1]
    })


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
    
    all_points = []
    all_labels = []
    
    for i, label in enumerate(unique_labels):
        instance_mask = (mask == label)
        
        # Get coordinates of this instance
        coords = np.where(instance_mask)
        
        if len(coords[0]) == 0:
            continue
            
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
        
        # Create labels
        labels = np.full(samples_per_neuron, i + 1, dtype=np.float32)
        
        all_points.append(points)
        all_labels.append(labels)
    
    if not all_points:
        # Return empty point cloud
        empty_points = torch.zeros(samples_per_neuron, 3)
        empty_labels = torch.zeros(samples_per_neuron)
        return empty_points, empty_labels
    
    # Concatenate all points and labels
    points = np.vstack(all_points)
    labels = np.concatenate(all_labels)
    
    return torch.tensor(points, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def process_mask_file(mask_path: str, 
                     output_dir: str,
                     voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                     samples_per_neuron: int = 1024,
                     min_size: int = 100) -> Dict:
    """
    Process a single mask file and convert to point cloud.
    
    Args:
        mask_path: Path to mask file
        output_dir: Output directory
        voxel_size: Voxel size
        samples_per_neuron: Number of points to sample
        min_size: Minimum instance size
    
    Returns:
        Dict: Processing results
    """
    print(f"Processing {mask_path}...")
    
    # Load mask
    if mask_path.endswith('.zarr'):
        mask = zarr.open(mask_path, mode='r')
        mask = np.array(mask)
    elif mask_path.endswith('.tif') or mask_path.endswith('.tiff'):
        mask = tifffile.imread(mask_path)
    else:
        raise ValueError(f"Unsupported file format: {mask_path}")
    
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
    
    # Also save as SWC for compatibility
    swc_path = Path(output_dir) / f"{base_name}.swc"
    try:
        neuron = mask_to_swc(mask, voxel_size, min_size)
        neuron.to_swc(swc_path)
    except Exception as e:
        print(f"Warning: Could not create SWC file: {e}")
    
    return {
        'input_path': mask_path,
        'output_path': str(output_path),
        'swc_path': str(swc_path),
        'n_points': len(points),
        'n_instances': len(torch.unique(labels[labels > 0]))
    }


def create_family_mapping(output_dir: str) -> str:
    """Create a family mapping file for neuron shape reasoning."""
    mapping = {
        "rib_frac": 0,
        "background": 1
    }
    
    mapping_path = Path(output_dir) / "family_to_id.json"
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    return str(mapping_path)


def create_neuron_id_csv(output_dir: str, processed_files: List[Dict]) -> str:
    """Create a neuron ID CSV file for neuron shape reasoning."""
    data = []
    
    for i, file_info in enumerate(processed_files):
        data.append({
            'root_id': Path(file_info['input_path']).stem,
            'family': 'rib_frac'
        })
    
    csv_path = Path(output_dir) / "neuron_ids.csv"
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    return str(csv_path)


def main():
    parser = argparse.ArgumentParser(description="Convert BANIS masks to point clouds for neuron shape reasoning")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing mask files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for point clouds")
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
        try:
            result = process_mask_file(
                mask_file, 
                args.output_dir,
                tuple(args.voxel_size),
                args.samples_per_neuron,
                args.min_size
            )
            processed_files.append(result)
        except Exception as e:
            print(f"Error processing {mask_file}: {e}")
            continue
    
    # Create metadata files
    family_mapping_path = create_family_mapping(args.output_dir)
    neuron_id_csv_path = create_neuron_id_csv(args.output_dir, processed_files)
    
    print(f"\nProcessing complete!")
    print(f"Processed {len(processed_files)} files")
    print(f"Point clouds saved to: {args.output_dir}")
    print(f"Family mapping: {family_mapping_path}")
    print(f"Neuron IDs CSV: {neuron_id_csv_path}")
    
    # Print summary
    total_points = sum(f['n_points'] for f in processed_files)
    total_instances = sum(f['n_instances'] for f in processed_files)
    print(f"Total points: {total_points}")
    print(f"Total instances: {total_instances}")


if __name__ == "__main__":
    main()

