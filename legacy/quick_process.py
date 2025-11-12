#!/usr/bin/env python3
"""
Quick processing script using predefined configurations.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_segmentation import batch_extract_segmentation
from processing_configs import get_config, list_configs, print_configs


def main():
    parser = argparse.ArgumentParser(description="Quick processing with predefined configurations")
    
    # Required arguments
    parser.add_argument("--input-dir", "-i", required=True,
                       help="Input directory containing zarr files")
    parser.add_argument("--config", "-c", default="balanced",
                       help="Processing configuration (default: balanced)")
    
    # Optional overrides
    parser.add_argument("--output-dir", "-o",
                       help="Output directory (default: same as input)")
    parser.add_argument("--gt-dir", "-g",
                       help="Ground truth directory for evaluation")
    parser.add_argument("--list-configs", action="store_true",
                       help="List available configurations and exit")
    
    # Quick overrides
    parser.add_argument("--threshold", type=float,
                       help="Override threshold")
    parser.add_argument("--min-size", type=int,
                       help="Override min_size")
    parser.add_argument("--max-instances", type=int,
                       help="Override max_instances")
    parser.add_argument("--workers", type=int,
                       help="Override number of workers")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress messages")
    parser.add_argument("--save-csv", action="store_true", default=True,
                       help="Save evaluation results to CSV")
    
    args = parser.parse_args()
    
    # List configurations if requested
    if args.list_configs:
        print_configs()
        return 0
    
    # Get configuration
    try:
        config = get_config(args.config)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nAvailable configurations:")
        list_configs()
        return 1
    
    # Apply overrides
    if args.threshold is not None:
        config["threshold"] = args.threshold
    if args.min_size is not None:
        config["min_size"] = args.min_size
    if args.max_instances is not None:
        config["max_instances"] = args.max_instances
    if args.workers is not None:
        config["n_workers"] = args.workers
    if args.quiet:
        config["verbose"] = False
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("QUICK PROCESSING")
    print("="*60)
    print(f"Configuration: {args.config}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Ground truth directory: {args.gt_dir}")
    print(f"Threshold: {config['threshold']}")
    print(f"Min size: {config['min_size']}")
    print(f"Max instances: {config['max_instances']}")
    print(f"Workers: {config['n_workers'] or 'auto'}")
    print("="*60)
    
    # Start processing
    start_time = time.time()
    
    try:
        results = batch_extract_segmentation(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            threshold=config["threshold"],
            min_size=config["min_size"],
            max_instances=config["max_instances"],
            morphological_ops=config["morphological_ops"],
            gt_dir=args.gt_dir,
            save_results=True,
            save_csv=args.save_csv,
            n_workers=config["n_workers"],
            verbose=config["verbose"]
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
