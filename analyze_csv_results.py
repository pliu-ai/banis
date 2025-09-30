#!/usr/bin/env python3
"""
Analyze CSV evaluation results from segmentation processing.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_csv_results(csv_path):
    """
    Load CSV evaluation results.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        pandas.DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def analyze_summary_results(df):
    """
    Analyze summary results (case-level statistics).
    
    Args:
        df: DataFrame with evaluation results
    """
    print("\n" + "="*60)
    print("CASE-LEVEL ANALYSIS")
    print("="*60)
    
    # Filter summary rows
    summary_df = df[df['label_id'] == 'SUMMARY'].copy()
    
    if len(summary_df) == 0:
        print("No summary data found")
        return
    
    print(f"Total cases: {len(summary_df)}")
    print(f"Successful cases: {summary_df['success'].sum() if 'success' in summary_df.columns else 'N/A'}")
    print(f"Failed cases: {(~summary_df['success']).sum() if 'success' in summary_df.columns else 'N/A'}")
    
    # Dice statistics
    if 'dice' in summary_df.columns:
        dice_scores = summary_df['dice'].dropna()
        if len(dice_scores) > 0:
            print(f"\nDice Statistics:")
            print(f"  Mean: {dice_scores.mean():.4f}")
            print(f"  Std:  {dice_scores.std():.4f}")
            print(f"  Min:  {dice_scores.min():.4f}")
            print(f"  Max:  {dice_scores.max():.4f}")
            print(f"  Median: {dice_scores.median():.4f}")
            
            # Percentiles
            print(f"  25th percentile: {dice_scores.quantile(0.25):.4f}")
            print(f"  75th percentile: {dice_scores.quantile(0.75):.4f}")
    
    # Instance statistics
    if 'instances' in summary_df.columns:
        instances = summary_df['instances'].dropna()
        if len(instances) > 0:
            print(f"\nInstance Statistics:")
            print(f"  Mean: {instances.mean():.2f}")
            print(f"  Std:  {instances.std():.2f}")
            print(f"  Min:  {instances.min()}")
            print(f"  Max:  {instances.max()}")
    
    # Processing time statistics
    if 'processing_time' in summary_df.columns:
        times = summary_df['processing_time'].dropna()
        if len(times) > 0:
            print(f"\nProcessing Time Statistics:")
            print(f"  Mean: {times.mean():.2f} seconds")
            print(f"  Std:  {times.std():.2f} seconds")
            print(f"  Min:  {times.min():.2f} seconds")
            print(f"  Max:  {times.max():.2f} seconds")
            print(f"  Total: {times.sum():.2f} seconds")


def analyze_label_results(df):
    """
    Analyze individual label results.
    
    Args:
        df: DataFrame with evaluation results
    """
    print("\n" + "="*60)
    print("LABEL-LEVEL ANALYSIS")
    print("="*60)
    
    # Filter label rows (exclude summary)
    label_df = df[df['label_id'] != 'SUMMARY'].copy()
    
    if len(label_df) == 0:
        print("No individual label data found")
        return
    
    print(f"Total labels: {len(label_df)}")
    
    # Dice statistics for individual labels
    if 'dice' in label_df.columns:
        dice_scores = label_df['dice'].dropna()
        if len(dice_scores) > 0:
            print(f"\nLabel Dice Statistics:")
            print(f"  Mean: {dice_scores.mean():.4f}")
            print(f"  Std:  {dice_scores.std():.4f}")
            print(f"  Min:  {dice_scores.min():.4f}")
            print(f"  Max:  {dice_scores.max():.4f}")
            
            # Count of matched/unmatched labels
            if 'matched' in label_df.columns:
                matched = label_df['matched'].sum()
                total = len(label_df)
                print(f"  Matched labels: {matched}/{total} ({matched/total*100:.1f}%)")
    
    # Per-case analysis
    if 'case_name' in label_df.columns:
        case_stats = label_df.groupby('case_name').agg({
            'dice': ['count', 'mean', 'std', 'min', 'max'],
            'matched': 'sum'
        }).round(4)
        
        print(f"\nPer-case label statistics (first 10 cases):")
        print(case_stats.head(10))


def create_visualizations(df, output_dir):
    """
    Create visualization plots.
    
    Args:
        df: DataFrame with evaluation results
        output_dir: Directory to save plots
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter summary data
    summary_df = df[df['label_id'] == 'SUMMARY'].copy()
    
    if len(summary_df) == 0:
        print("No summary data for visualization")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Dice score distribution
    if 'dice' in summary_df.columns:
        plt.figure(figsize=(10, 6))
        dice_scores = summary_df['dice'].dropna()
        
        plt.subplot(1, 2, 1)
        plt.hist(dice_scores, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Dice Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Dice Scores')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(dice_scores)
        plt.ylabel('Dice Score')
        plt.title('Box Plot of Dice Scores')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dice_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved dice distribution plot")
    
    # 2. Instance count distribution
    if 'instances' in summary_df.columns:
        plt.figure(figsize=(8, 6))
        instances = summary_df['instances'].dropna()
        
        plt.hist(instances, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Number of Instances')
        plt.ylabel('Frequency')
        plt.title('Distribution of Instance Counts')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'instance_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved instance distribution plot")
    
    # 3. Processing time distribution
    if 'processing_time' in summary_df.columns:
        plt.figure(figsize=(8, 6))
        times = summary_df['processing_time'].dropna()
        
        plt.hist(times, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Processing Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Processing Times')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'processing_time_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved processing time distribution plot")
    
    # 4. Dice vs Instances scatter plot
    if 'dice' in summary_df.columns and 'instances' in summary_df.columns:
        plt.figure(figsize=(8, 6))
        dice_scores = summary_df['dice'].dropna()
        instances = summary_df['instances'].dropna()
        
        # Align the data
        common_idx = dice_scores.index.intersection(instances.index)
        if len(common_idx) > 0:
            plt.scatter(instances.loc[common_idx], dice_scores.loc[common_idx], alpha=0.6)
            plt.xlabel('Number of Instances')
            plt.ylabel('Dice Score')
            plt.title('Dice Score vs Instance Count')
            plt.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr = np.corrcoef(instances.loc[common_idx], dice_scores.loc[common_idx])[0, 1]
            plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'dice_vs_instances.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved dice vs instances plot")


def export_detailed_report(df, output_path):
    """
    Export detailed analysis report.
    
    Args:
        df: DataFrame with evaluation results
        output_path: Path to save report
    """
    print(f"\nExporting detailed report to: {output_path}")
    
    with open(output_path, 'w') as f:
        f.write("SEGMENTATION EVALUATION ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        # Summary statistics
        summary_df = df[df['label_id'] == 'SUMMARY'].copy()
        
        if len(summary_df) > 0:
            f.write("CASE-LEVEL SUMMARY:\n")
            f.write(f"Total cases: {len(summary_df)}\n")
            
            if 'success' in summary_df.columns:
                f.write(f"Successful cases: {summary_df['success'].sum()}\n")
                f.write(f"Failed cases: {(~summary_df['success']).sum()}\n")
            
            if 'dice' in summary_df.columns:
                dice_scores = summary_df['dice'].dropna()
                if len(dice_scores) > 0:
                    f.write(f"\nDice Statistics:\n")
                    f.write(f"  Mean: {dice_scores.mean():.4f}\n")
                    f.write(f"  Std:  {dice_scores.std():.4f}\n")
                    f.write(f"  Min:  {dice_scores.min():.4f}\n")
                    f.write(f"  Max:  {dice_scores.max():.4f}\n")
                    f.write(f"  Median: {dice_scores.median():.4f}\n")
        
        # Individual case results
        f.write(f"\nINDIVIDUAL CASE RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        for case_name in summary_df['case_name'].unique():
            case_data = summary_df[summary_df['case_name'] == case_name]
            if len(case_data) > 0:
                row = case_data.iloc[0]
                f.write(f"Case: {case_name}\n")
                f.write(f"  Dice: {row.get('dice', 'N/A'):.4f}\n")
                f.write(f"  Instances: {row.get('instances', 'N/A')}\n")
                f.write(f"  Success: {row.get('success', 'N/A')}\n")
                f.write(f"  Processing Time: {row.get('processing_time', 'N/A'):.2f}s\n")
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze CSV evaluation results")
    
    parser.add_argument("--csv", "-c", required=True,
                       help="Path to CSV evaluation results file")
    parser.add_argument("--output-dir", "-o", default="./analysis_output",
                       help="Output directory for analysis results")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip creating visualization plots")
    parser.add_argument("--export-report", action="store_true",
                       help="Export detailed text report")
    
    args = parser.parse_args()
    
    # Load CSV data
    df = load_csv_results(args.csv)
    if df is None:
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Perform analysis
    analyze_summary_results(df)
    analyze_label_results(df)
    
    # Create visualizations
    if not args.no_plots:
        create_visualizations(df, args.output_dir)
    
    # Export detailed report
    if args.export_report:
        report_path = os.path.join(args.output_dir, "detailed_report.txt")
        export_detailed_report(df, report_path)
    
    print(f"\n✓ Analysis completed! Results saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

