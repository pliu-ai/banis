#!/usr/bin/env python3
"""
Batch evaluate multiple segmentation results against ground truth.
"""

import os
import glob
import json
import numpy as np
from pathlib import Path
from evaluate_segmentation import evaluate_segmentation, load_segmentation


def find_matching_files(pred_dir, gt_dir, pred_pattern="*_segmentation.tif", gt_pattern="*.tif"):
    """
    Find matching prediction and ground truth files.
    
    Args:
        pred_dir: Directory containing prediction files
        gt_dir: Directory containing ground truth files
        pred_pattern: Pattern for prediction files
        gt_pattern: Pattern for ground truth files
    
    Returns:
        List of (pred_path, gt_path) tuples
    """
    pred_files = glob.glob(os.path.join(pred_dir, pred_pattern))
    gt_files = glob.glob(os.path.join(gt_dir, gt_pattern))
    
    # Create mapping from filename to full path
    pred_map = {}
    for pred_file in pred_files:
        filename = os.path.basename(pred_file)
        # Remove common suffixes to match with GT
        base_name = filename.replace('_segmentation.tif', '').replace('_seg.tif', '')
        pred_map[base_name] = pred_file
    
    gt_map = {}
    for gt_file in gt_files:
        filename = os.path.basename(gt_file)
        # Remove common suffixes
        base_name = filename.replace('.tif', '').replace('.tiff', '')
        gt_map[base_name] = gt_file
    
    # Find matches
    matches = []
    for base_name in pred_map:
        if base_name in gt_map:
            matches.append((pred_map[base_name], gt_map[base_name]))
        else:
            print(f"Warning: No GT found for {base_name}")
    
    return matches


def batch_evaluate(pred_dir, gt_dir, output_dir, pred_pattern="*_segmentation.tif", gt_pattern="*.tif"):
    """
    Batch evaluate multiple segmentation results.
    
    Args:
        pred_dir: Directory containing prediction files
        gt_dir: Directory containing ground truth files
        output_dir: Directory to save evaluation results
        pred_pattern: Pattern for prediction files
        gt_pattern: Pattern for ground truth files
    """
    print(f"Batch evaluation:")
    print(f"  Prediction directory: {pred_dir}")
    print(f"  Ground truth directory: {gt_dir}")
    print(f"  Output directory: {output_dir}")
    
    # Find matching files
    matches = find_matching_files(pred_dir, gt_dir, pred_pattern, gt_pattern)
    
    if not matches:
        print("No matching files found!")
        return
    
    print(f"Found {len(matches)} matching file pairs")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate each pair
    all_results = []
    dice_scores = []
    
    for i, (pred_path, gt_path) in enumerate(matches):
        print(f"\n{'='*60}")
        print(f"Evaluating {i+1}/{len(matches)}: {os.path.basename(pred_path)}")
        print(f"{'='*60}")
        
        try:
            # Load segmentations
            pred_seg = load_segmentation(pred_path)
            gt_seg = load_segmentation(gt_path)
            
            # Resize GT to match prediction if needed
            if gt_seg.shape != pred_seg.shape:
                print(f"Resizing GT from {gt_seg.shape} to {pred_seg.shape}")
                from scipy.ndimage import zoom
                zoom_factors = [pred_seg.shape[i] / gt_seg.shape[i] for i in range(len(pred_seg.shape))]
                gt_seg = zoom(gt_seg, zoom_factors, order=0)
            
            # Evaluate
            results = evaluate_segmentation(pred_seg, gt_seg)
            
            # Add file information
            results['pred_file'] = pred_path
            results['gt_file'] = gt_path
            results['pred_shape'] = pred_seg.shape
            results['gt_shape'] = gt_seg.shape
            
            all_results.append(results)
            dice_scores.append(results['average_dice'])
            
            # Save individual results
            base_name = os.path.splitext(os.path.basename(pred_path))[0]
            result_file = os.path.join(output_dir, f"{base_name}_eval.json")
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to: {result_file}")
            
        except Exception as e:
            print(f"Error evaluating {pred_path}: {e}")
            continue
    
    # Compute overall statistics
    if dice_scores:
        overall_stats = {
            "total_files": len(matches),
            "successful_evaluations": len(dice_scores),
            "mean_dice": np.mean(dice_scores),
            "std_dice": np.std(dice_scores),
            "min_dice": np.min(dice_scores),
            "max_dice": np.max(dice_scores),
            "median_dice": np.median(dice_scores)
        }
        
        print(f"\n{'='*60}")
        print(f"OVERALL STATISTICS")
        print(f"{'='*60}")
        print(f"Total files: {overall_stats['total_files']}")
        print(f"Successful evaluations: {overall_stats['successful_evaluations']}")
        print(f"Mean Dice: {overall_stats['mean_dice']:.4f}")
        print(f"Std Dice: {overall_stats['std_dice']:.4f}")
        print(f"Min Dice: {overall_stats['min_dice']:.4f}")
        print(f"Max Dice: {overall_stats['max_dice']:.4f}")
        print(f"Median Dice: {overall_stats['median_dice']:.4f}")
        
        # Save overall results
        overall_file = os.path.join(output_dir, "overall_results.json")
        with open(overall_file, 'w') as f:
            json.dump({
                "overall_stats": overall_stats,
                "individual_results": all_results
            }, f, indent=2)
        
        print(f"Overall results saved to: {overall_file}")
    
    print(f"\nBatch evaluation completed!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch evaluate segmentation results")
    parser.add_argument("--pred-dir", "-p", required=True,
                       help="Directory containing prediction files")
    parser.add_argument("--gt-dir", "-g", required=True,
                       help="Directory containing ground truth files")
    parser.add_argument("--output-dir", "-o", required=True,
                       help="Directory to save evaluation results")
    parser.add_argument("--pred-pattern", default="*_segmentation.tif",
                       help="Pattern for prediction files (default: *_segmentation.tif)")
    parser.add_argument("--gt-pattern", default="*.tif",
                       help="Pattern for ground truth files (default: *.tif)")
    
    args = parser.parse_args()
    
    batch_evaluate(
        args.pred_dir,
        args.gt_dir,
        args.output_dir,
        args.pred_pattern,
        args.gt_pattern
    )


if __name__ == "__main__":
    main()



