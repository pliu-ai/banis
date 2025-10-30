#!/usr/bin/env python3
"""
Extract segmentation from zarr prediction files and save as TIFF.
Supports both single file and batch processing of entire folders.
"""

import argparse
import os
import glob
import json
import numpy as np
import zarr
import tifffile
from pathlib import Path
from tqdm import tqdm
from inference import compute_connected_component_segmentation, scale_sigmoid
import torch
from scipy import ndimage
from skimage import morphology, measure
from scipy.optimize import linear_sum_assignment
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import time
import pandas as pd
import csv


def remove_small_instances(seg, min_size=100):
    """
    Remove instances smaller than min_size voxels.
    
    Args:
        seg: Segmentation array
        min_size: Minimum instance size in voxels
    
    Returns:
        Filtered segmentation array
    """
    print(f"Removing instances smaller than {min_size} voxels...")
    
    # Get unique labels (excluding background)
    unique_labels = np.unique(seg)
    unique_labels = unique_labels[unique_labels > 0]
    
    initial_count = len(unique_labels)
    
    # Count voxels for each instance and remove small ones
    for label in unique_labels:
        mask = (seg == label)
        size = np.sum(mask)
        if size < min_size:
            seg[mask] = 0  # Remove small instances
    
    # Get final count
    final_labels = np.unique(seg)
    final_labels = final_labels[final_labels > 0]
    final_count = len(final_labels)
    
    print(f"Removed {initial_count - final_count} small instances")
    print(f"Remaining instances: {final_count}")
    
    return seg


def keep_largest_instances(seg, max_instances=22):
    """
    Keep only the largest max_instances instances.
    
    Args:
        seg: Segmentation array
        max_instances: Maximum number of instances to keep
    
    Returns:
        Filtered segmentation array
    """
    print(f"Keeping only the largest {max_instances} instances...")
    
    # Get unique labels (excluding background)
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
    """
    Relabel segmentation to have consecutive IDs starting from 1.
    
    Args:
        seg: Segmentation array
    
    Returns:
        Relabeled segmentation array
    """
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


def save_evaluation_csv(results, output_path, case_name=None):
    """
    Save evaluation results to CSV file.
    
    Args:
        results: Dictionary containing evaluation results
        output_path: Path to save CSV file
        case_name: Name of the case (optional)
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare data for CSV
        csv_data = []
        
        # Add case-level summary
        csv_data.append({
            "case_name": case_name or "unknown",
            "label_id": "SUMMARY",
            "dice": results.get("average_dice", 0.0),
            "matched": results.get("matched_labels", 0),
            "total_gt_labels": results.get("total_gt_labels", 0),
            "match_rate": results.get("matched_labels", 0) / max(results.get("total_gt_labels", 1), 1)
        })
        
        # Add individual label results
        if "label_details" in results:
            for label_info in results["label_details"]:
                csv_data.append({
                    "case_name": case_name or "unknown",
                    "label_id": f"Label_{label_info['gt_label']}",
                    "dice": label_info["dice"],
                    "matched": 1 if label_info["matched"] else 0,
                    "total_gt_labels": results.get("total_gt_labels", 0),
                    "match_rate": results.get("matched_labels", 0) / max(results.get("total_gt_labels", 1), 1)
                })
        
        # Write to CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["case_name", "label_id", "dice", "matched", "total_gt_labels", "match_rate"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        
        print(f"Evaluation results saved to CSV: {output_path}")
        
    except Exception as e:
        print(f"Error saving CSV file: {e}")


def save_batch_evaluation_csv(batch_results, output_path):
    """
    Save batch evaluation results to CSV file.
    
    Args:
        batch_results: Dictionary containing batch processing results
        output_path: Path to save CSV file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare data for CSV
        csv_data = []
        
        for file_result in batch_results.get("file_results", []):
            case_name = os.path.basename(file_result.get("input_file", "unknown"))
            case_name = case_name.replace(".zarr", "").replace("pred_aff_test_", "")
            
            # Add case-level summary
            csv_data.append({
                "case_name": case_name,
                "label_id": "SUMMARY",
                "dice": file_result.get("dice_score", 0.0),
                "matched": "N/A",  # Not available in batch results
                "total_gt_labels": "N/A",
                "match_rate": "N/A",
                "instances": file_result.get("instances", 0),
                "success": file_result.get("success", False),
                "processing_time": file_result.get("processing_time", 0.0),
                "error": file_result.get("error", "")
            })
        
        # Write to CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["case_name", "label_id", "dice", "matched", "total_gt_labels", 
                         "match_rate", "instances", "success", "processing_time", "error"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        
        print(f"Batch evaluation results saved to CSV: {output_path}")
        
    except Exception as e:
        print(f"Error saving batch CSV file: {e}")


def compute_iou(mask1, mask2):
    """
    Compute Intersection over Union (IoU) between two binary masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
    
    Returns:
        IoU value
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def match_pred_to_gt(pred_seg, gt_seg):
    """
    Match prediction labels to ground truth labels based on maximum IoU.
    
    Args:
        pred_seg: Prediction segmentation array
        gt_seg: Ground truth segmentation array
    
    Returns:
        matched_pred_seg: Prediction segmentation with matched labels
        iou_matrix: IoU matrix between pred and gt labels
    """
    print("Matching prediction labels to ground truth labels...")
    
    # Get unique labels (excluding background)
    pred_labels = np.unique(pred_seg)
    pred_labels = pred_labels[pred_labels > 0]
    
    gt_labels = np.unique(gt_seg)
    gt_labels = gt_labels[gt_labels > 0]
    
    print(f"Prediction labels: {len(pred_labels)}")
    print(f"Ground truth labels: {len(gt_labels)}")
    
    if len(pred_labels) == 0 or len(gt_labels) == 0:
        print("No labels found in prediction or ground truth")
        return pred_seg, np.array([])
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_labels), len(gt_labels)))
    
    for i, pred_label in enumerate(pred_labels):
        pred_mask = (pred_seg == pred_label)
        for j, gt_label in enumerate(gt_labels):
            gt_mask = (gt_seg == gt_label)
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)
    
    # Use Hungarian algorithm for optimal matching
    pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)  # Negative for maximization
    
    # Create mapping from pred labels to gt labels
    label_mapping = {}
    matched_pred_seg = np.zeros_like(pred_seg)
    
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        pred_label = pred_labels[pred_idx]
        gt_label = gt_labels[gt_idx]
        iou_value = iou_matrix[pred_idx, gt_idx]
        
        if iou_value > 0:  # Only match if IoU > 0
            label_mapping[pred_label] = gt_label
            print(f"Matched pred {pred_label} -> gt {gt_label} (IoU: {iou_value:.3f})")
    
    # Apply mapping
    for pred_label, gt_label in label_mapping.items():
        mask = (pred_seg == pred_label)
        matched_pred_seg[mask] = gt_label
    
    print(f"Successfully matched {len(label_mapping)} prediction labels")
    
    return matched_pred_seg, iou_matrix


def compute_dice(pred_seg, gt_seg, label_id):
    """
    Compute Dice coefficient for a specific label.
    
    Args:
        pred_seg: Prediction segmentation
        gt_seg: Ground truth segmentation
        label_id: Label ID to compute Dice for
    
    Returns:
        Dice coefficient
    """
    pred_mask = (pred_seg == label_id)
    gt_mask = (gt_seg == label_id)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2.0 * intersection / total


def evaluate_segmentation(pred_seg, gt_seg):
    """
    Evaluate segmentation by matching labels and computing Dice coefficients.
    
    Args:
        pred_seg: Prediction segmentation
        gt_seg: Ground truth segmentation
    
    Returns:
        dict: Evaluation results including average Dice
    """
    print("Evaluating segmentation...")
    
    # Match prediction to ground truth
    matched_pred_seg, iou_matrix = match_pred_to_gt(pred_seg, gt_seg)
    
    # Get ground truth labels
    gt_labels = np.unique(gt_seg)
    gt_labels = gt_labels[gt_labels > 0]
    
    if len(gt_labels) == 0:
        print("No ground truth labels found")
        return {"average_dice": 0.0, "dice_scores": [], "matched_labels": 0}
    
    # Compute Dice for each ground truth label
    dice_scores = []
    matched_labels = 0
    label_details = []
    
    for gt_label in gt_labels:
        dice = compute_dice(matched_pred_seg, gt_seg, gt_label)
        dice_scores.append(dice)
        
        # Store detailed information for each label
        label_info = {
            "gt_label": gt_label,
            "dice": dice,
            "matched": dice > 0
        }
        label_details.append(label_info)
        
        if dice > 0:
            matched_labels += 1
            print(f"Label {gt_label}: Dice = {dice:.3f}")
        else:
            print(f"Label {gt_label}: Dice = {dice:.3f} (no match)")
    
    average_dice = np.mean(dice_scores)
    
    print(f"Average Dice: {average_dice:.3f}")
    print(f"Matched labels: {matched_labels}/{len(gt_labels)}")
    
    return {
        "average_dice": average_dice,
        "dice_scores": dice_scores,
        "matched_labels": matched_labels,
        "total_gt_labels": len(gt_labels),
        "iou_matrix": iou_matrix,
        "label_details": label_details
    }


def morphological_cleanup(seg, operations=['opening', 'closing']):
    """
    Apply morphological operations to clean up segmentation.
    
    Args:
        seg: Segmentation array
        operations: List of morphological operations to apply
    
    Returns:
        Cleaned segmentation array
    """
    print(f"Applying morphological operations: {operations}")
    
    # Convert to binary mask
    binary_mask = (seg > 0).astype(np.uint8)
    
    # Apply operations
    if 'opening' in operations:
        binary_mask = morphology.binary_opening(binary_mask, morphology.ball(1))
        print("Applied opening operation")
    
    if 'closing' in operations:
        binary_mask = morphology.binary_closing(binary_mask, morphology.ball(1))
        print("Applied closing operation")
    
    # Recompute connected components
    from skimage.measure import label
    seg_cleaned = label(binary_mask)
    
    print(f"Instances after cleanup: {len(np.unique(seg_cleaned)) - 1}")
    return seg_cleaned


def postprocess_segmentation(seg, min_size=200, max_instances=None, morphological_ops=None, verbose=True):
    """
    Apply post-processing to segmentation.
    
    Args:
        seg: Input segmentation
        min_size: Minimum instance size in voxels (default: 200)
        max_instances: Maximum number of instances to keep (keep largest ones, default: None)
        morphological_ops: List of morphological operations ['opening', 'closing']
        verbose: Whether to print progress messages
    
    Returns:
        Post-processed segmentation
    """
    if verbose:
        print("Starting post-processing...")
    initial_instances = len(np.unique(seg)) - 1
    if verbose:
        print(f"Initial instances: {initial_instances}")
    
    # Remove small instances
    if min_size > 0:
        seg = remove_small_instances(seg, min_size)
    
    # Keep only largest instances
    if max_instances is not None:
        seg = keep_largest_instances(seg, max_instances)
    
    # Morphological cleanup
    if morphological_ops:
        seg = morphological_cleanup(seg, morphological_ops)
    
    final_instances = len(np.unique(seg)) - 1
    if verbose:
        print(f"Final instances: {final_instances}")
    
    return seg


def extract_segmentation_from_zarr(zarr_path, output_path, threshold=0.5, min_size=200, max_instances=None, morphological_ops=None, gt_path=None, verbose=True):
    """
    Extract segmentation from zarr file and save as TIFF.
    
    Args:
        zarr_path: Path to input zarr file
        output_path: Path to output TIFF file
        threshold: Threshold for binarizing affinities
        min_size: Minimum instance size in voxels (default: 200)
        max_instances: Maximum number of instances to keep (default: None)
        morphological_ops: List of morphological operations ['opening', 'closing'] (default: None)
        gt_path: Path to ground truth segmentation for evaluation (default: None)
        verbose: Whether to print progress messages (default: True)
    """
    if verbose:
        print(f"Processing: {zarr_path}")
    
    # Load zarr data with memory mapping for efficiency
    try:
        zarr_array = zarr.open(zarr_path, mode='r')
        # Use memory mapping for large files
        if zarr_array.nbytes > 1e9:  # > 1GB
            aff_pred = zarr_array[:]  # Load all data
        else:
            aff_pred = zarr_array[:]
        if verbose:
            print(f"Loaded data shape: {aff_pred.shape}, dtype: {aff_pred.dtype}")
    except Exception as e:
        if verbose:
            print(f"Error loading zarr file: {e}")
        return False
    
    # Convert to probabilities if needed
    if aff_pred.dtype == np.float16:
        aff_pred = aff_pred.astype(np.float32)
    
    # Apply sigmoid to convert logits to probabilities
    #aff_prob = scale_sigmoid(torch.from_numpy(aff_pred)).numpy()
    aff_prob = aff_pred
    
    # Use only short range affinities (first 3 channels)
    if aff_prob.shape[0] >= 3:
        short_range_aff = aff_prob[:3]
        if verbose:
            print(f"Using first 3 channels for segmentation")
    else:
        short_range_aff = aff_prob
        if verbose:
            print(f"Using all {aff_prob.shape[0]} channels for segmentation")
    
    # Binarize with threshold
    hard_aff = (short_range_aff > threshold).astype(np.uint8)
    if verbose:
        print(f"Binarized affinities with threshold {threshold}")
        print(f"Affinity statistics - Min: {short_range_aff.min():.4f}, Max: {short_range_aff.max():.4f}, Mean: {short_range_aff.mean():.4f}")
        print(f"Hard affinities - True pixels: {hard_aff.sum()}, Total pixels: {hard_aff.size}")
    
    # If no connections found, try lower threshold
    if hard_aff.sum() == 0:
        if verbose:
            print("No connections found with current threshold, trying lower threshold...")
        lower_threshold = threshold * 0.5
        hard_aff = (short_range_aff > lower_threshold).astype(np.uint8)
        if verbose:
            print(f"Tried threshold {lower_threshold}, found {hard_aff.sum()} connections")
    
    # Compute connected components
    if verbose:
        print("Computing connected components...")
    try:
        seg = compute_connected_component_segmentation(hard_aff)
        # transpose Seg
        seg = np.transpose(seg,(2,1,0))
        if verbose:
            print(f"Segmentation shape: {seg.shape}")
        unique_ids = np.unique(seg)
        num_instances = len(unique_ids) - 1  # -1 for background (ID 0)
        if verbose:
            print(f"Number of instances: {num_instances}")
            print(f"Instance IDs: {unique_ids}")
        
        if num_instances == 0:
            if verbose:
                print("Warning: No instances found! This might indicate:")
                print("1. Threshold too high")
                print("2. Affinity predictions are too low")
                print("3. Data format issue")
                print("Trying to save anyway...")
        else:
            # Apply post-processing
            seg = postprocess_segmentation(seg, min_size=min_size, max_instances=max_instances, morphological_ops=morphological_ops, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"Error computing connected components: {e}")
        return False
    
    # Save as TIFF
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Try LZW compression first, fallback to no compression if not available
        try:
            tifffile.imwrite(output_path, seg.astype(np.uint16), compression='zlib')
        except Exception as comp_error:
            print(f"LZW compression failed, using no compression: {comp_error}")
            tifffile.imwrite(output_path, seg)
        print(f"Saved segmentation to: {output_path}")
        
        # Evaluate if ground truth is provided
        if gt_path and os.path.exists(gt_path):
            print(f"\nEvaluating against ground truth: {gt_path}")
            try:
                # Load ground truth
                if gt_path.endswith('.zarr'):
                    gt_array = zarr.open(gt_path, mode='r')
                    gt_seg = gt_array[:]
                elif gt_path.endswith('.nii.gz') or gt_path.endswith('.nii'):
                    # Use SimpleITK for NIfTI files
                    gt_image = sitk.ReadImage(gt_path)
                    gt_seg = sitk.GetArrayFromImage(gt_image)
                else:
                    gt_seg = tifffile.imread(gt_path)
                
                # Ensure same shape
                if gt_seg.shape != seg.shape:
                    print(f"Warning: GT shape {gt_seg.shape} != pred shape {seg.shape}")
                    # Resize GT to match prediction
                    from scipy.ndimage import zoom
                    zoom_factors = [seg.shape[i] / gt_seg.shape[i] for i in range(len(seg.shape))]
                    gt_seg = zoom(gt_seg, zoom_factors, order=0)  # Nearest neighbor interpolation
                
                # Evaluate
                eval_results = evaluate_segmentation(seg, gt_seg)
                
                # Save evaluation results
                eval_output_path = output_path.replace('.tif', '_eval.txt')
                with open(eval_output_path, 'w') as f:
                    f.write(f"Evaluation Results\n")
                    f.write(f"==================\n")
                    f.write(f"Average Dice: {eval_results['average_dice']:.4f}\n")
                    f.write(f"Matched labels: {eval_results['matched_labels']}/{eval_results['total_gt_labels']}\n")
                    f.write(f"Individual Dice scores:\n")
                    for i, dice in enumerate(eval_results['dice_scores']):
                        f.write(f"  Label {i+1}: {dice:.4f}\n")
                
                print(f"Evaluation results saved to: {eval_output_path}")
                
                # Save evaluation results to CSV
                case_name = os.path.basename(zarr_path).replace('.zarr', '').replace('pred_aff_test_', '')
                csv_output_path = output_path.replace('.tif', '_eval.csv')
                save_evaluation_csv(eval_results, csv_output_path, case_name)
                
            except Exception as e:
                print(f"Error during evaluation: {e}")
        
        return True
    except Exception as e:
        print(f"Error saving TIFF file: {e}")
        return False


def find_matching_files(input_dir, gt_dir=None):
    """
    Find input files and their corresponding ground truth files.
    
    Args:
        input_dir: Directory containing input zarr files
        gt_dir: Directory containing ground truth files (default: None)
    
    Returns:
        List of tuples (input_path, gt_path, output_path)
    """
    # Use fixed pattern for rib segmentation: pred_aff_test_*.zarr
    input_files = glob.glob(os.path.join(input_dir, "pred_aff_test_*.zarr"))
    input_files.sort()
    
    file_pairs = []
    
    for input_file in input_files:
        # Generate output path
        input_name = os.path.basename(input_file)
        output_name = input_name.replace('.zarr', '.tif')
        output_path = os.path.join(input_dir, output_name)
        
        # Find corresponding ground truth file
        gt_path = None
        if gt_dir:
            # Extract image_id from input filename
            # Handle pattern: pred_aff_test_{image_id}.zarr -> {image_id}-rib-seg.nii.gz
            if input_name.startswith('pred_aff_test_') and input_name.endswith('.zarr'):
                # Extract image_id from pred_aff_test_{image_id}.zarr
                image_id = input_name.replace('pred_aff_test_', '').replace('.zarr', '')
                # Construct ground truth filename: {image_id}-rib-seg.nii.gz
                gt_candidate = os.path.join(gt_dir, f"{image_id}-rib-seg.nii.gz")
                if os.path.exists(gt_candidate):
                    gt_path = gt_candidate
                    print(f"Matched GT: {os.path.basename(gt_candidate)}")
            
            # If no match with the specific pattern, try other naming conventions
            if gt_path is None:
                gt_candidates = [
                    os.path.join(gt_dir, input_name.replace('.zarr', '.tif')),
                    os.path.join(gt_dir, input_name.replace('.zarr', '.nii.gz')),
                    os.path.join(gt_dir, input_name.replace('.zarr', '.nii')),
                    os.path.join(gt_dir, input_name.replace('pred_aff_', 'gt_').replace('.zarr', '.tif')),
                    os.path.join(gt_dir, input_name.replace('pred_aff_', '').replace('.zarr', '.tif')),
                ]
                
                for candidate in gt_candidates:
                    if os.path.exists(candidate):
                        gt_path = candidate
                        break
            
            # If no exact match, try pattern matching with fixed pattern
            if gt_path is None:
                gt_files = glob.glob(os.path.join(gt_dir, "*-rib-seg.nii.gz"))
                for gt_file in gt_files:
                    gt_name = os.path.basename(gt_file)
                    # Simple name matching (remove common prefixes/suffixes)
                    if (input_name.replace('pred_aff_test_', '').replace('.zarr', '') in gt_name or
                        gt_name.replace('-rib-seg.nii.gz', '') in input_name):
                        gt_path = gt_file
                        break
        
        file_pairs.append((input_file, gt_path, output_path))
    
    return file_pairs


def process_single_file(args):
    """
    Process a single file - used for parallel processing.
    
    Args:
        args: Tuple of (input_path, gt_path, output_path, threshold, min_size, 
                       max_instances, morphological_ops, verbose)
    
    Returns:
        dict: File processing result
    """
    (input_path, gt_path, output_path, threshold, min_size, 
     max_instances, morphological_ops, verbose) = args
    
    file_result = {
        "input_file": input_path,
        "output_file": output_path,
        "gt_file": gt_path,
        "success": False,
        "instances": 0,
        "dice_score": 0.0,
        "error": None,
        "processing_time": 0.0
    }
    
    start_time = time.time()
    
    try:
        # Extract segmentation
        success = extract_segmentation_from_zarr(
            input_path,
            output_path,
            threshold,
            min_size,
            max_instances,
            morphological_ops,
            gt_path,
            verbose=verbose
        )
        
        if success:
            file_result["success"] = True
            
            # Count instances in output file
            if os.path.exists(output_path):
                seg = tifffile.imread(output_path)
                instances = len(np.unique(seg)) - 1
                file_result["instances"] = instances
            
            # Load evaluation results if available
            eval_file = output_path.replace('.tif', '_eval.txt')
            if os.path.exists(eval_file):
                try:
                    with open(eval_file, 'r') as f:
                        content = f.read()
                        # Extract average dice from file
                        for line in content.split('\n'):
                            if 'Average Dice:' in line:
                                dice = float(line.split(':')[1].strip())
                                file_result["dice_score"] = dice
                                break
                except Exception as e:
                    if verbose:
                        print(f"Error reading evaluation file: {e}")
        else:
            file_result["error"] = "Extraction failed"
            
    except Exception as e:
        file_result["error"] = str(e)
    
    file_result["processing_time"] = time.time() - start_time
    return file_result


def batch_extract_segmentation(input_dir, output_dir=None, threshold=0.5, min_size=200, 
                              max_instances=None, morphological_ops=None, gt_dir=None, 
                              save_results=True, save_csv=True, n_workers=None, verbose=True):
    """
    Batch extract segmentation from multiple zarr files with parallel processing.
    
    Args:
        input_dir: Directory containing input zarr files
        output_dir: Directory to save output files (default: same as input_dir)
        threshold: Threshold for binarizing affinities
        min_size: Minimum instance size in voxels (default: 200)
        max_instances: Maximum number of instances to keep (default: None)
        morphological_ops: List of morphological operations
        gt_dir: Directory containing ground truth files
        save_results: Whether to save batch results summary
        n_workers: Number of parallel workers (default: CPU count)
        verbose: Whether to print progress messages
    
    Returns:
        dict: Summary of batch processing results
    """
    if output_dir is None:
        output_dir = input_dir
    
    if verbose:
        print(f"Batch processing files in: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Ground truth directory: {gt_dir}")
    
    # Find all file pairs
    file_pairs = find_matching_files(input_dir, gt_dir)
    
    if not file_pairs:
        if verbose:
            print(f"No files found matching pattern 'pred_aff_test_*.zarr' in {input_dir}")
        return {"success": False, "message": "No files found"}
    
    if verbose:
        print(f"Found {len(file_pairs)} files to process")
    
    # Set number of workers
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(file_pairs))
    
    if verbose:
        print(f"Using {n_workers} parallel workers")
    
    # Results tracking
    results = {
        "total_files": len(file_pairs),
        "successful": 0,
        "failed": 0,
        "evaluated": 0,
        "file_results": [],
        "overall_stats": {
            "average_dice": 0.0,
            "total_instances": 0,
            "total_gt_instances": 0,
            "total_processing_time": 0.0
        }
    }
    
    dice_scores = []
    
    # Prepare arguments for parallel processing
    process_args = []
    for input_path, gt_path, output_path in file_pairs:
        # Update output path to use output_dir
        output_filename = os.path.basename(output_path)
        output_path = os.path.join(output_dir, output_filename)
        
        process_args.append((
            input_path, gt_path, output_path, threshold, min_size,
            max_instances, morphological_ops, verbose
        ))
    
    # Process files in parallel
    start_time = time.time()
    
    if n_workers > 1 and len(file_pairs) > 1:
        if verbose:
            print(f"Starting parallel processing with {n_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            file_results = list(tqdm(
                executor.map(process_single_file, process_args),
                total=len(process_args),
                desc="Processing files"
            ))
    else:
        # Sequential processing for single file or single worker
        if verbose:
            print("Processing files sequentially...")
        
        file_results = []
        for args in tqdm(process_args, desc="Processing files"):
            file_results.append(process_single_file(args))
    
    total_processing_time = time.time() - start_time
    results["overall_stats"]["total_processing_time"] = total_processing_time
    
    # Process results
    for file_result in file_results:
        results["file_results"].append(file_result)
        
        if file_result["success"]:
            results["successful"] += 1
            results["overall_stats"]["total_instances"] += file_result["instances"]
            
            if file_result["dice_score"] > 0:
                dice_scores.append(file_result["dice_score"])
                results["evaluated"] += 1
        else:
            results["failed"] += 1
    
    # Calculate overall statistics
    if dice_scores:
        results["overall_stats"]["average_dice"] = np.mean(dice_scores)
        results["overall_stats"]["dice_std"] = np.std(dice_scores)
        results["overall_stats"]["dice_min"] = np.min(dice_scores)
        results["overall_stats"]["dice_max"] = np.max(dice_scores)
    
    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total files: {results['total_files']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"Evaluated: {results['evaluated']}")
        print(f"Total instances: {results['overall_stats']['total_instances']}")
        print(f"Total processing time: {total_processing_time:.2f} seconds")
        print(f"Average time per file: {total_processing_time/len(file_pairs):.2f} seconds")
        
        if dice_scores:
            print(f"Average Dice: {results['overall_stats']['average_dice']:.4f} ± {results['overall_stats']['dice_std']:.4f}")
            print(f"Dice range: {results['overall_stats']['dice_min']:.4f} - {results['overall_stats']['dice_max']:.4f}")
    
    # Save results summary
    if save_results:
        results_file = os.path.join(output_dir, "batch_results.json")
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            if verbose:
                print(f"Results saved to: {results_file}")
        except Exception as e:
            if verbose:
                print(f"Error saving results: {e}")
        
        # Save batch evaluation results to CSV
        if save_csv:
            csv_file = os.path.join(output_dir, "batch_evaluation_results.csv")
            try:
                save_batch_evaluation_csv(results, csv_file)
            except Exception as e:
                if verbose:
                    print(f"Error saving CSV results: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract segmentation from zarr files")
    
    # Input/Output arguments
    parser.add_argument("--input", "-i", required=True, 
                       help="Input zarr file path or directory for batch processing")
    parser.add_argument("--output", "-o", 
                       help="Output TIFF file path or directory (for batch processing)")
    parser.add_argument("--batch", action="store_true",
                       help="Process all files in input directory")
    
    # Ground truth arguments
    parser.add_argument("--gt", type=str,
                       help="Path to ground truth segmentation file or directory")
    
    # Processing parameters
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Threshold for binarizing affinities (default: 0.5)")
    parser.add_argument("--min-size", type=int, default=200,
                       help="Minimum instance size in voxels (default: 200)")
    parser.add_argument("--max-instances", type=int, default=None,
                       help="Maximum number of instances to keep (keep largest ones, default: None)")
    parser.add_argument("--morphological", nargs="+", choices=['opening', 'closing'],
                       help="Morphological operations to apply (e.g., --morphological opening closing)")
    
    # Batch processing options
    parser.add_argument("--save-results", action="store_true", default=True,
                       help="Save batch processing results summary (default: True)")
    parser.add_argument("--save-csv", action="store_true", default=True,
                       help="Save evaluation results to CSV (default: True)")
    parser.add_argument("--n-workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Print progress messages (default: True)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress messages")
    
    args = parser.parse_args()
    
    # Handle verbose/quiet flags
    if args.quiet:
        args.verbose = False
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Input path does not exist: {args.input}")
        return
    
    # Determine if batch processing
    if args.batch or os.path.isdir(args.input):
        # Batch processing
        if args.verbose:
            print("Starting batch processing...")
        results = batch_extract_segmentation(
            input_dir=args.input,
            output_dir=args.output,
            threshold=args.threshold,
            min_size=args.min_size,
            max_instances=args.max_instances,
            morphological_ops=args.morphological,
            gt_dir=args.gt,
            save_results=args.save_results,
            save_csv=args.save_csv,
            n_workers=args.n_workers,
            verbose=args.verbose
        )
        
        if results.get("successful", 0) > 0:
            if args.verbose:
                print("Batch processing completed successfully!")
        else:
            if args.verbose:
                print("Batch processing failed!")
    else:
        # Single file processing
        if not args.output:
            print("Output path required for single file processing")
            return
        
        if args.verbose:
            print("Processing single file...")
        success = extract_segmentation_from_zarr(
            args.input, 
            args.output, 
            args.threshold,
            args.min_size,
            args.max_instances,
            args.morphological,
            args.gt,
            verbose=args.verbose
        )
        
        if success:
            if args.verbose:
                print("Extraction completed successfully!")
        else:
            if args.verbose:
                print("Extraction failed!")


if __name__ == "__main__":
    main()
