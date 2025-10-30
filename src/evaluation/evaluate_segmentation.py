#!/usr/bin/env python3
"""
Evaluate segmentation results against ground truth.
"""

import argparse
import os
import numpy as np
import tifffile
import zarr
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import zoom
import json
import SimpleITK as sitk


def compute_iou(mask1, mask2):
    """Compute Intersection over Union (IoU) between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def match_pred_to_gt(pred_seg, gt_seg):
    """
    Match prediction labels to ground truth labels based on maximum IoU.
    Uses Hungarian algorithm for optimal matching.
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
        return pred_seg, np.array([]), {}
    
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
    
    return matched_pred_seg, iou_matrix, label_mapping


def compute_dice(pred_seg, gt_seg, label_id):
    """Compute Dice coefficient for a specific label."""
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
    
    Returns:
        dict: Evaluation results including average Dice
    """
    print("Evaluating segmentation...")
    
    # Match prediction to ground truth
    matched_pred_seg, iou_matrix, label_mapping = match_pred_to_gt(pred_seg, gt_seg)
    
    # Get ground truth labels
    gt_labels = np.unique(gt_seg)
    gt_labels = gt_labels[gt_labels > 0]
    
    if len(gt_labels) == 0:
        print("No ground truth labels found")
        return {"average_dice": 0.0, "dice_scores": [], "matched_labels": 0}
    
    # Compute Dice for each ground truth label
    dice_scores = []
    matched_labels = 0
    label_results = {}
    
    for gt_label in gt_labels:
        dice = compute_dice(matched_pred_seg, gt_seg, gt_label)
        dice_scores.append(dice)
        label_results[gt_label] = dice
        
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
        "label_results": label_results,
        "matched_labels": matched_labels,
        "total_gt_labels": len(gt_labels),
        "iou_matrix": iou_matrix.tolist(),
        "label_mapping": label_mapping
    }


def load_segmentation(file_path):
    """Load segmentation from file (supports TIFF, Zarr, and NIfTI)."""
    if file_path.endswith('.zarr'):
        array = zarr.open(file_path, mode='r')
        return array[:]
    elif file_path.endswith('.nii.gz') or file_path.endswith('.nii'):
        # Use SimpleITK for NIfTI files
        image = sitk.ReadImage(file_path)
        return sitk.GetArrayFromImage(image)
    else:
        return tifffile.imread(file_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation against ground truth")
    parser.add_argument("--pred", "-p", required=True,
                       help="Path to prediction segmentation file")
    parser.add_argument("--gt", "-g", required=True,
                       help="Path to ground truth segmentation file")
    parser.add_argument("--output", "-o", required=True,
                       help="Path to save evaluation results")
    parser.add_argument("--resize-gt", action="store_true",
                       help="Resize ground truth to match prediction shape")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.pred):
        print(f"Prediction file does not exist: {args.pred}")
        return
    
    if not os.path.exists(args.gt):
        print(f"Ground truth file does not exist: {args.gt}")
        return
    
    print(f"Loading prediction: {args.pred}")
    pred_seg = load_segmentation(args.pred)
    print(f"Prediction shape: {pred_seg.shape}")
    
    print(f"Loading ground truth: {args.gt}")
    gt_seg = load_segmentation(args.gt)
    print(f"Ground truth shape: {gt_seg.shape}")
    
    # Resize ground truth if needed
    if args.resize_gt and gt_seg.shape != pred_seg.shape:
        print(f"Resizing ground truth from {gt_seg.shape} to {pred_seg.shape}")
        zoom_factors = [pred_seg.shape[i] / gt_seg.shape[i] for i in range(len(pred_seg.shape))]
        gt_seg = zoom(gt_seg, zoom_factors, order=0)  # Nearest neighbor interpolation
    
    # Evaluate
    results = evaluate_segmentation(pred_seg, gt_seg)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save as JSON
    json_output = args.output.replace('.txt', '.json')
    with open(json_output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"JSON results saved to: {json_output}")
    
    # Save as text
    with open(args.output, 'w') as f:
        f.write(f"Segmentation Evaluation Results\n")
        f.write(f"================================\n")
        f.write(f"Prediction file: {args.pred}\n")
        f.write(f"Ground truth file: {args.gt}\n")
        f.write(f"Prediction shape: {pred_seg.shape}\n")
        f.write(f"Ground truth shape: {gt_seg.shape}\n")
        f.write(f"\n")
        f.write(f"Average Dice: {results['average_dice']:.4f}\n")
        f.write(f"Matched labels: {results['matched_labels']}/{results['total_gt_labels']}\n")
        f.write(f"\n")
        f.write(f"Individual Dice scores:\n")
        for label_id, dice in results['label_results'].items():
            f.write(f"  Label {label_id}: {dice:.4f}\n")
        f.write(f"\n")
        f.write(f"Label mapping (pred -> gt):\n")
        for pred_label, gt_label in results['label_mapping'].items():
            f.write(f"  {pred_label} -> {gt_label}\n")
    
    print(f"Text results saved to: {args.output}")
    print(f"Evaluation completed!")


if __name__ == "__main__":
    main()
