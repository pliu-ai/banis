#!/usr/bin/env python3
"""
Fast batch processing script for segmentation extraction.
Optimized for large-scale processing with parallel execution.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add current directory to path to import extract_segmentation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_segmentation import batch_extract_segmentation


def main():
    parser = argparse.ArgumentParser(description="Fast batch processing for segmentation extraction")
    
    # Required arguments
    parser.add_argument("--input-dir", "-i", required=True,
                       help="Input directory containing zarr files")
    parser.add_argument("--output-dir", "-o", 
                       help="Output directory (default: same as input)")
    parser.add_argument("--gt-dir", "-g",
                       help="Ground truth directory for evaluation")
    
    # Processing parameters
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Threshold for binarizing affinities (default: 0.5)")
    parser.add_argument("--min-size", type=int, default=200,
                       help="Minimum instance size in voxels (default: 200)")
    parser.add_argument("--max-instances", type=int, default=None,
                       help="Maximum number of instances to keep (default: None)")
    
    # Performance options
    parser.add_argument("--n-workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress messages")
    parser.add_argument("--no-eval", action="store_true",
                       help="Skip evaluation (faster processing)")
    
    # Advanced options
    parser.add_argument("--morphological", nargs="+", choices=['opening', 'closing'],
                       help="Morphological operations to apply")
    parser.add_argument("--save-results", action="store_true", default=True,
                       help="Save batch results summary")
    parser.add_argument("--save-csv", action="store_true", default=True,
                       help="Save evaluation results to CSV")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Skip evaluation if requested
    if args.no_eval:
        args.gt_dir = None
    
    print("="*60)
    print("FAST BATCH PROCESSING")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Ground truth directory: {args.gt_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Min size: {args.min_size}")
    print(f"Max instances: {args.max_instances}")
    print(f"Workers: {args.n_workers or 'auto'}")
    print(f"Evaluation: {'disabled' if args.no_eval else 'enabled'}")
    print("="*60)
    
    # Start processing
    start_time = time.time()
    
    try:
        results = batch_extract_segmentation(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
            min_size=args.min_size,
            max_instances=args.max_instances,
            morphological_ops=args.morphological,
            gt_dir=args.gt_dir,
            save_results=args.save_results,
            save_csv=args.save_csv,
            n_workers=args.n_workers,
            verbose=not args.quiet
        )
        
        total_time = time.time() - start_time
        
        # Print final summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETED")
        print("="*60)
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Files processed: {results['total_files']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        
        if results['evaluated'] > 0:
            print(f"Evaluated: {results['evaluated']}")
            print(f"Average Dice: {results['overall_stats']['average_dice']:.4f}")
        
        print(f"Total instances: {results['overall_stats']['total_instances']}")
        print(f"Average time per file: {total_time/results['total_files']:.2f} seconds")
        
        if results['successful'] > 0:
            print("\n✓ Processing completed successfully!")
            return 0
        else:
            print("\n✗ Processing failed!")
            return 1
            
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
