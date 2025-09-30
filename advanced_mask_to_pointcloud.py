#!/usr/bin/env python3
"""
Advanced conversion of BANIS segmentation masks to neuron shape reasoning format.
This version handles multiple instances and creates more realistic neuron structures.
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
from skimage import morphology, measure, segmentation
from typing import List, Tuple, Dict
import networkx as nx
from scipy.spatial.distance import cdist


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


def skeletonize_instance(instance_mask: np.ndarray) -> np.ndarray:
    """Skeletonize a single instance."""
    # Ensure binary
    instance_mask = instance_mask.astype(bool)
    
    # Skeletonize
    skeleton = morphology.skeletonize_3d(instance_mask)
    
    return skeleton


def extract_skeleton_points(skeleton: np.ndarray, 
                           voxel_size: Tuple[float, float, float]) -> np.ndarray:
    """Extract skeleton points and convert to physical coordinates."""
    coords = np.where(skeleton > 0)
    
    if len(coords[0]) == 0:
        return np.array([]).reshape(0, 3)
    
    # Convert to physical coordinates
    z_coords = coords[0] * voxel_size[0]
    y_coords = coords[1] * voxel_size[1]
    x_coords = coords[2] * voxel_size[2]
    
    return np.column_stack((x_coords, y_coords, z_coords))


def create_swc_from_skeleton(skeleton_points: np.ndarray, 
                            voxel_size: Tuple[float, float, float]) -> pd.DataFrame:
    """Create SWC format from skeleton points."""
    if len(skeleton_points) == 0:
        return create_empty_swc()
    
    # Create a simple linear chain for now
    # TODO: Implement proper tree structure detection
    n_points = len(skeleton_points)
    swc_data = []
    
    for i in range(n_points):
        parent_id = i if i == 0 else i - 1
        swc_data.append({
            'node_id': i + 1,
            'type': 3,  # dendrite
            'x': skeleton_points[i, 0],
            'y': skeleton_points[i, 1],
            'z': skeleton_points[i, 2],
            'radius': 1.0,  # default radius
            'parent_id': parent_id
        })
    
    return pd.DataFrame(swc_data)


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


def mask_to_pointcloud_advanced(mask: np.ndarray, 
                               voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                               samples_per_neuron: int = 1024,
                               min_size: int = 100,
                               use_skeleton: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert 3D segmentation mask to point cloud using advanced methods.
    
    Args:
        mask: 3D segmentation mask (H, W, D)
        voxel_size: Voxel size in (z, y, x) order
        samples_per_neuron: Number of points to sample
        min_size: Minimum instance size in voxels
        use_skeleton: Whether to use skeletonization
    
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
        
        if use_skeleton:
            # Use skeletonization for more realistic neuron structure
            skeleton = skeletonize_instance(instance_mask)
            skeleton_points = extract_skeleton_points(skeleton, voxel_size)
            
            if len(skeleton_points) == 0:
                # Fallback to surface points
                coords = np.where(instance_mask)
                z_coords = coords[0] * voxel_size[0]
                y_coords = coords[1] * voxel_size[1]
                x_coords = coords[2] * voxel_size[2]
                points = np.column_stack((x_coords, y_coords, z_coords))
            else:
                points = skeleton_points
        else:
            # Use all voxels
            coords = np.where(instance_mask)
            z_coords = coords[0] * voxel_size[0]
            y_coords = coords[1] * voxel_size[1]
            x_coords = coords[2] * voxel_size[2]
            points = np.column_stack((x_coords, y_coords, z_coords))
        
        if len(points) == 0:
            continue
        
        # Sample points if we have too many
        if len(points) > samples_per_neuron:
            indices = np.random.choice(len(points), samples_per_neuron, replace=False)
            points = points[indices]
        
        # Pad with zeros if we have too few
        if len(points) < samples_per_neuron:
            padding = np.zeros((samples_per_neuron - len(points), 3))
            points = np.vstack([points, padding])
        
        # Create labels (all points belong to the same instance)
        labels = torch.ones(samples_per_neuron, dtype=torch.float32) * (i + 1)
        
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


def process_single_file_advanced(mask_path: str, 
                                output_dir: str,
                                voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                                samples_per_neuron: int = 1024,
                                min_size: int = 100,
                                use_skeleton: bool = True) -> Dict:
    """Process a single mask file with advanced methods."""
    print(f"Processing {mask_path}...")
    
    try:
        # Load mask
        mask = load_mask(mask_path)
        
        # Convert to point cloud
        points, labels = mask_to_pointcloud_advanced(
            mask, voxel_size, samples_per_neuron, min_size, use_skeleton
        )
        
        # Save point cloud
        base_name = Path(mask_path).stem
        output_path = Path(output_dir) / f"{base_name}_pointcloud.pt"
        
        torch.save({
            'points': points,
            'labels': labels,
            'voxel_size': voxel_size,
            'samples_per_neuron': samples_per_neuron,
            'use_skeleton': use_skeleton
        }, output_path)
        
        # Also create SWC file
        swc_path = Path(output_dir) / f"{base_name}.swc"
        try:
            # Get the largest instance for SWC
            unique_labels = torch.unique(labels[labels > 0])
            if len(unique_labels) > 0:
                largest_label = unique_labels[0].item()
                instance_mask = (labels == largest_label)
                instance_points = points[instance_mask]
                
                if len(instance_points) > 0:
                    skeleton_points = instance_points.numpy()
                    swc_data = create_swc_from_skeleton(skeleton_points, voxel_size)
                    swc_data.to_csv(swc_path, sep=' ', index=False, header=False)
        except Exception as e:
            print(f"Warning: Could not create SWC file: {e}")
        
        return {
            'input_path': mask_path,
            'output_path': str(output_path),
            'swc_path': str(swc_path),
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
    parser = argparse.ArgumentParser(description="Advanced conversion of BANIS masks to neuron shape reasoning format")
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
    parser.add_argument("--use_skeleton", action="store_true", default=True,
                       help="Use skeletonization for more realistic neuron structure")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find mask files
    mask_files = glob.glob(os.path.join(args.input_dir, args.pattern))
    
    if not mask_files:
        print(f"No files found matching pattern {args.pattern} in {args.input_dir}")
        return
    
    print(f"Found {len(mask_files)} mask files")
    print(f"Using skeletonization: {args.use_skeleton}")
    
    # Process each file
    processed_files = []
    for mask_file in tqdm(mask_files, desc="Processing masks"):
        result = process_single_file_advanced(
            mask_file, 
            args.output_dir,
            tuple(args.voxel_size),
            args.samples_per_neuron,
            args.min_size,
            args.use_skeleton
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

