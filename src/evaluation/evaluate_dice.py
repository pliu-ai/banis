#!/usr/bin/env python3
"""
Evaluate Dice coefficient between prediction and ground truth segmentation files.
Supports both single file and batch processing of entire folders.
"""

import argparse
import os
import glob
import json
import numpy as np
import tifffile
import zarr
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
import csv


def load_segmentation(file_path):
    """
    Load segmentation from various file formats.
    
    Args:
        file_path: Path to segmentation file
    
    Returns:
        numpy array: Segmentation data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.endswith('.zarr'):
        zarr_array = zarr.open(file_path, mode='r')
        return zarr_array[:]
    elif file_path.endswith('.nii.gz') or file_path.endswith('.nii'):
        # Use SimpleITK for NIfTI files
        image = sitk.ReadImage(file_path)
        return sitk.GetArrayFromImage(image)
    elif file_path.endswith('.tif') or file_path.endswith('.tiff'):
        return tifffile.imread(file_path)
    else:
        # Try to load as TIFF by default
        return tifffile.imread(file_path)


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


def evaluate_dice(pred_path, gt_path, resize_gt=True, verbose=True):
    """
    Evaluate Dice coefficient between prediction and ground truth segmentation.
    
    Args:
        pred_path: Path to prediction segmentation file
        gt_path: Path to ground truth segmentation file
        resize_gt: Whether to resize GT to match prediction shape
        verbose: Whether to print progress messages
    
    Returns:
        dict: Evaluation results including average Dice
    """
    if verbose:
        print(f"Evaluating Dice coefficient...")
        print(f"Prediction: {pred_path}")
        print(f"Ground truth: {gt_path}")
    
    try:
        # Load segmentations
        pred_seg = load_segmentation(pred_path)
        gt_seg = load_segmentation(gt_path)
        
        if verbose:
            print(f"Prediction shape: {pred_seg.shape}, dtype: {pred_seg.dtype}")
            print(f"Ground truth shape: {gt_seg.shape}, dtype: {gt_seg.dtype}")
        
        # Ensure same shape
        if gt_seg.shape != pred_seg.shape:
            if verbose:
                print(f"Warning: GT shape {gt_seg.shape} != pred shape {pred_seg.shape}")
            if resize_gt:
                from scipy.ndimage import zoom
                zoom_factors = [pred_seg.shape[i] / gt_seg.shape[i] for i in range(len(pred_seg.shape))]
                gt_seg = zoom(gt_seg, zoom_factors, order=0)  # Nearest neighbor interpolation
                if verbose:
                    print(f"Resized GT to: {gt_seg.shape}")
            else:
                raise ValueError(f"Shape mismatch: pred {pred_seg.shape} vs gt {gt_seg.shape}")
        
        # Match prediction to ground truth
        matched_pred_seg, iou_matrix = match_pred_to_gt(pred_seg, gt_seg)
        
        # Get ground truth labels
        gt_labels = np.unique(gt_seg)
        gt_labels = gt_labels[gt_labels > 0]
        
        if len(gt_labels) == 0:
            if verbose:
                print("No ground truth labels found")
            return {"average_dice": 0.0, "dice_scores": [], "matched_labels": 0, "total_gt_labels": 0}
        
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
                if verbose:
                    print(f"Label {gt_label}: Dice = {dice:.3f}")
            else:
                if verbose:
                    print(f"Label {gt_label}: Dice = {dice:.3f} (no match)")
        
        average_dice = np.mean(dice_scores)
        
        if verbose:
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
        
    except Exception as e:
        if verbose:
            print(f"Error during evaluation: {e}")
        return {"error": str(e), "average_dice": 0.0, "dice_scores": [], "matched_labels": 0, "total_gt_labels": 0}


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


def find_matching_files(pred_dir, gt_dir):
    """
    Find prediction files and their corresponding ground truth files.
    Supports specific naming patterns:
    - Prediction: refined_RibFrac*.tif
    - Ground truth: RibFrac*-rib-seg.nii.gz
    
    Args:
        pred_dir: Directory containing prediction files
        gt_dir: Directory containing ground truth files
    
    Returns:
        List of tuples (pred_path, gt_path)
    """
    # Common file extensions
    pred_extensions = ['*.tif', '*.tiff', '*.nii', '*.nii.gz', '*.zarr']
    gt_extensions = ['*.tif', '*.tiff', '*.nii', '*.nii.gz', '*.zarr']
    
    file_pairs = []
    
    # Find prediction files
    pred_files = []
    for ext in pred_extensions:
        pred_files.extend(glob.glob(os.path.join(pred_dir, ext)))
    pred_files.sort()
    
    for pred_file in pred_files:
        pred_name = os.path.basename(pred_file)
        pred_base = os.path.splitext(pred_name)[0]
        
        # Try to find matching ground truth file
        gt_path = None
        
        # Special handling for refined_RibFrac*.tif -> RibFrac*-rib-seg.nii.gz pattern
        if pred_name.startswith('refined_RibFrac') and pred_name.endswith('.tif'):
            # Extract the RibFrac ID from refined_RibFrac{ID}.tif
            ribfrac_id = pred_name.replace('refined_RibFrac', '').replace('.tif', '')
            # Look for RibFrac{ID}-rib-seg.nii.gz
            gt_candidate = os.path.join(gt_dir, f"RibFrac{ribfrac_id}-rib-seg.nii.gz")
            if os.path.exists(gt_candidate):
                gt_path = gt_candidate
                print(f"Matched (refined pattern): {pred_name} -> {os.path.basename(gt_candidate)}")
        
        # Try exact name match if special pattern didn't work
        if gt_path is None:
            for ext in gt_extensions:
                gt_candidate = os.path.join(gt_dir, pred_base + ext.replace('*', ''))
                if os.path.exists(gt_candidate):
                    gt_path = gt_candidate
                    print(f"Matched (exact): {pred_name} -> {os.path.basename(gt_candidate)}")
                    break
        
        # Try pattern matching if exact match fails
        if gt_path is None:
            for ext in gt_extensions:
                gt_files = glob.glob(os.path.join(gt_dir, ext))
                for gt_file in gt_files:
                    gt_name = os.path.basename(gt_file)
                    gt_base = os.path.splitext(gt_name)[0]
                    
                    # Simple name matching (remove common prefixes/suffixes)
                    if (pred_base in gt_base or gt_base in pred_base or
                        pred_base.replace('pred_', '').replace('prediction_', '').replace('refined_', '') in gt_base or
                        gt_base.replace('gt_', '').replace('ground_truth_', '').replace('-rib-seg', '') in pred_base):
                        gt_path = gt_file
                        print(f"Matched (pattern): {pred_name} -> {gt_name}")
                        break
                
                if gt_path:
                    break
        
        if gt_path:
            file_pairs.append((pred_file, gt_path))
        else:
            print(f"No ground truth found for: {pred_file}")
    
    return file_pairs


def process_single_evaluation(args):
    """
    Process a single file pair - used for parallel processing.
    
    Args:
        args: Tuple of (pred_path, gt_path, resize_gt, verbose)
    
    Returns:
        dict: File evaluation result
    """
    pred_path, gt_path, resize_gt, verbose = args
    
    file_result = {
        "pred_file": pred_path,
        "gt_file": gt_path,
        "success": False,
        "dice_score": 0.0,
        "matched_labels": 0,
        "total_gt_labels": 0,
        "error": None,
        "processing_time": 0.0
    }
    
    start_time = time.time()
    
    try:
        # Evaluate Dice
        results = evaluate_dice(pred_path, gt_path, resize_gt, verbose)
        
        if "error" not in results:
            file_result["success"] = True
            file_result["dice_score"] = results["average_dice"]
            file_result["matched_labels"] = results["matched_labels"]
            file_result["total_gt_labels"] = results["total_gt_labels"]
        else:
            file_result["error"] = results["error"]
            
    except Exception as e:
        file_result["error"] = str(e)
    
    file_result["processing_time"] = time.time() - start_time
    return file_result


def batch_evaluate_dice(pred_dir, gt_dir, output_dir=None, resize_gt=True, 
                       save_results=True, save_csv=True, n_workers=None, verbose=True):
    """
    Batch evaluate Dice coefficient for multiple file pairs with parallel processing.
    
    Args:
        pred_dir: Directory containing prediction files
        gt_dir: Directory containing ground truth files
        output_dir: Directory to save output files (default: same as pred_dir)
        resize_gt: Whether to resize GT to match prediction shape
        save_results: Whether to save batch results summary
        save_csv: Whether to save evaluation results to CSV
        n_workers: Number of parallel workers (default: CPU count)
        verbose: Whether to print progress messages
    
    Returns:
        dict: Summary of batch evaluation results
    """
    if output_dir is None:
        output_dir = pred_dir
    
    if verbose:
        print(f"Batch evaluating files...")
        print(f"Prediction directory: {pred_dir}")
        print(f"Ground truth directory: {gt_dir}")
        print(f"Output directory: {output_dir}")
    
    # Find all file pairs
    file_pairs = find_matching_files(pred_dir, gt_dir)
    
    if not file_pairs:
        if verbose:
            print(f"No matching files found between {pred_dir} and {gt_dir}")
        return {"success": False, "message": "No matching files found"}
    
    if verbose:
        print(f"Found {len(file_pairs)} file pairs to evaluate")
    
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
        "file_results": [],
        "overall_stats": {
            "average_dice": 0.0,
            "dice_std": 0.0,
            "dice_min": 0.0,
            "dice_max": 0.0,
            "total_processing_time": 0.0
        }
    }
    
    dice_scores = []
    
    # Prepare arguments for parallel processing
    process_args = []
    for pred_path, gt_path in file_pairs:
        process_args.append((pred_path, gt_path, resize_gt, verbose))
    
    # Process files in parallel
    start_time = time.time()
    
    if n_workers > 1 and len(file_pairs) > 1:
        if verbose:
            print(f"Starting parallel processing with {n_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            file_results = list(tqdm(
                executor.map(process_single_evaluation, process_args),
                total=len(process_args),
                desc="Evaluating files"
            ))
    else:
        # Sequential processing for single file or single worker
        if verbose:
            print("Processing files sequentially...")
        
        file_results = []
        for args in tqdm(process_args, desc="Evaluating files"):
            file_results.append(process_single_evaluation(args))
    
    total_processing_time = time.time() - start_time
    results["overall_stats"]["total_processing_time"] = total_processing_time
    
    # Process results
    for file_result in file_results:
        results["file_results"].append(file_result)
        
        if file_result["success"]:
            results["successful"] += 1
            dice_scores.append(file_result["dice_score"])
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
        print("BATCH EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total files: {results['total_files']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"Total processing time: {total_processing_time:.2f} seconds")
        print(f"Average time per file: {total_processing_time/len(file_pairs):.2f} seconds")
        
        if dice_scores:
            print(f"Average Dice: {results['overall_stats']['average_dice']:.4f} ± {results['overall_stats']['dice_std']:.4f}")
            print(f"Dice range: {results['overall_stats']['dice_min']:.4f} - {results['overall_stats']['dice_max']:.4f}")
    
    # Save results summary
    if save_results:
        results_file = os.path.join(output_dir, "dice_evaluation_results.json")
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            if verbose:
                print(f"Results saved to: {results_file}")
        except Exception as e:
            if verbose:
                print(f"Error saving results: {e}")
        
        # Save evaluation results to CSV
        if save_csv:
            csv_file = os.path.join(output_dir, "dice_evaluation_results.csv")
            try:
                save_batch_evaluation_csv(results, csv_file)
            except Exception as e:
                if verbose:
                    print(f"Error saving CSV results: {e}")
    
    return results


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
            case_name = os.path.basename(file_result.get("pred_file", "unknown"))
            case_name = os.path.splitext(case_name)[0]
            
            # Add case-level summary
            csv_data.append({
                "case_name": case_name,
                "dice": file_result.get("dice_score", 0.0),
                "matched_labels": file_result.get("matched_labels", 0),
                "total_gt_labels": file_result.get("total_gt_labels", 0),
                "match_rate": file_result.get("matched_labels", 0) / max(file_result.get("total_gt_labels", 1), 1),
                "success": file_result.get("success", False),
                "processing_time": file_result.get("processing_time", 0.0),
                "error": file_result.get("error", "")
            })
        
        # Write to CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["case_name", "dice", "matched_labels", "total_gt_labels", 
                         "match_rate", "success", "processing_time", "error"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        
        print(f"Batch evaluation results saved to CSV: {output_path}")
        
    except Exception as e:
        print(f"Error saving batch CSV file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Dice coefficient between prediction and ground truth segmentation files.\n"
                   "Supports file naming patterns:\n"
                   "  - Prediction: refined_RibFrac*.tif\n"
                   "  - Ground truth: RibFrac*-rib-seg.nii.gz\n\n"
                   "Examples:\n"
                   "  # Single file evaluation\n"
                   "  python evaluate_dice.py -p refined_RibFrac501.tif -g RibFrac501-rib-seg.nii.gz\n\n"
                   "  # Batch evaluation\n"
                   "  python evaluate_dice.py -p /path/to/predictions/ -g /path/to/ground_truth/ --batch\n\n"
                   "  # With custom output directory\n"
                   "  python evaluate_dice.py -p /path/to/preds/ -g /path/to/gts/ --batch -o /path/to/results/",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument("--pred", "-p", required=True, 
                       help="Prediction segmentation file path or directory for batch processing")
    parser.add_argument("--gt", "-g", required=True,
                       help="Ground truth segmentation file path or directory for batch processing")
    parser.add_argument("--output", "-o", 
                       help="Output directory for batch processing results")
    parser.add_argument("--batch", action="store_true",
                       help="Process all files in prediction directory")
    
    # Processing parameters
    parser.add_argument("--resize-gt", action="store_true", default=True,
                       help="Resize ground truth to match prediction shape (default: True)")
    parser.add_argument("--no-resize", action="store_true",
                       help="Do not resize ground truth (will fail if shapes don't match)")
    
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
    
    # Handle resize flags
    if args.no_resize:
        args.resize_gt = False
    
    # Check if inputs exist
    if not os.path.exists(args.pred):
        print(f"Prediction path does not exist: {args.pred}")
        return
    
    if not os.path.exists(args.gt):
        print(f"Ground truth path does not exist: {args.gt}")
        return
    
    # Determine if batch processing
    if args.batch or (os.path.isdir(args.pred) and os.path.isdir(args.gt)):
        # Batch processing
        if args.verbose:
            print("Starting batch evaluation...")
        results = batch_evaluate_dice(
            pred_dir=args.pred,
            gt_dir=args.gt,
            output_dir=args.output,
            resize_gt=args.resize_gt,
            save_results=args.save_results,
            save_csv=args.save_csv,
            n_workers=args.n_workers,
            verbose=args.verbose
        )
        
        if results.get("successful", 0) > 0:
            if args.verbose:
                print("Batch evaluation completed successfully!")
        else:
            if args.verbose:
                print("Batch evaluation failed!")
    else:
        # Single file processing
        if args.verbose:
            print("Evaluating single file pair...")
        results = evaluate_dice(
            args.pred, 
            args.gt, 
            resize_gt=args.resize_gt,
            verbose=args.verbose
        )
        
        if "error" not in results:
            if args.verbose:
                print("Evaluation completed successfully!")
                print(f"Average Dice: {results['average_dice']:.4f}")
                print(f"Matched labels: {results['matched_labels']}/{results['total_gt_labels']}")
        else:
            if args.verbose:
                print(f"Evaluation failed: {results['error']}")


if __name__ == "__main__":
    main()
