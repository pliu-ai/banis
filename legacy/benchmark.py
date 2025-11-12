#!/usr/bin/env python3
"""
Performance benchmark script for segmentation processing.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_segmentation import batch_extract_segmentation
from processing_configs import PROCESSING_CONFIGS


def run_benchmark(input_dir, gt_dir=None, test_files=None, configs=None):
    """
    Run performance benchmark with different configurations.
    
    Args:
        input_dir: Directory containing zarr files
        gt_dir: Ground truth directory
        test_files: Number of files to test (None for all)
        configs: List of config names to test (None for all)
    """
    if configs is None:
        configs = list(PROCESSING_CONFIGS.keys())
    
    print("="*80)
    print("PERFORMANCE BENCHMARK")
    print("="*80)
    print(f"Input directory: {input_dir}")
    print(f"Ground truth directory: {gt_dir}")
    print(f"Test configurations: {configs}")
    print("="*80)
    
    results = {}
    
    for config_name in configs:
        print(f"\nTesting configuration: {config_name}")
        print("-" * 40)
        
        config = PROCESSING_CONFIGS[config_name]
        print(f"Description: {config['description']}")
        print(f"Parameters: threshold={config['threshold']}, min_size={config['min_size']}")
        
        start_time = time.time()
        
        try:
            # Run processing
            result = batch_extract_segmentation(
                input_dir=input_dir,
                output_dir=input_dir,  # Use same directory for output
                threshold=config["threshold"],
                min_size=config["min_size"],
                max_instances=config["max_instances"],
                morphological_ops=config["morphological_ops"],
                gt_dir=gt_dir,
                save_results=False,  # Don't save results for benchmark
                n_workers=config["n_workers"],
                verbose=False  # Quiet mode for benchmark
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Store results
            results[config_name] = {
                "processing_time": processing_time,
                "total_files": result["total_files"],
                "successful": result["successful"],
                "failed": result["failed"],
                "evaluated": result["evaluated"],
                "average_dice": result["overall_stats"]["average_dice"],
                "total_instances": result["overall_stats"]["total_instances"],
                "time_per_file": processing_time / result["total_files"] if result["total_files"] > 0 else 0
            }
            
            print(f"✓ Completed in {processing_time:.2f} seconds")
            print(f"  Files: {result['total_files']} (successful: {result['successful']}, failed: {result['failed']})")
            print(f"  Time per file: {processing_time/result['total_files']:.2f} seconds")
            if result["evaluated"] > 0:
                print(f"  Average Dice: {result['overall_stats']['average_dice']:.4f}")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            results[config_name] = {"error": str(e)}
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Sort by processing time
    sorted_results = sorted(
        [(name, data) for name, data in results.items() if "error" not in data],
        key=lambda x: x[1]["processing_time"]
    )
    
    print(f"{'Configuration':<15} {'Time (s)':<10} {'Files':<8} {'Time/File':<12} {'Dice':<8} {'Instances':<10}")
    print("-" * 80)
    
    for config_name, data in sorted_results:
        dice_str = f"{data['average_dice']:.3f}" if data['evaluated'] > 0 else "N/A"
        print(f"{config_name:<15} {data['processing_time']:<10.2f} {data['total_files']:<8} "
              f"{data['time_per_file']:<12.2f} {dice_str:<8} {data['total_instances']:<10}")
    
    # Calculate speedup
    if len(sorted_results) > 1:
        baseline_time = sorted_results[0][1]["processing_time"]
        print(f"\nSpeedup relative to fastest configuration:")
        for config_name, data in sorted_results:
            speedup = data["processing_time"] / baseline_time
            print(f"  {config_name}: {speedup:.2f}x")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Performance benchmark for segmentation processing")
    
    parser.add_argument("--input-dir", "-i", required=True,
                       help="Input directory containing zarr files")
    parser.add_argument("--gt-dir", "-g",
                       help="Ground truth directory for evaluation")
    parser.add_argument("--configs", nargs="+", default=None,
                       help="Configurations to test (default: all)")
    parser.add_argument("--test-files", type=int, default=None,
                       help="Number of files to test (default: all)")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1
    
    # Run benchmark
    try:
        results = run_benchmark(
            input_dir=args.input_dir,
            gt_dir=args.gt_dir,
            test_files=args.test_files,
            configs=args.configs
        )
        
        print("\n✓ Benchmark completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

