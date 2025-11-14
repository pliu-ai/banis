#!/usr/bin/env python
"""
Script for inference on specified dataset(s)
Supports selecting single or multiple datasets
"""
import argparse
from BANIS import BANIS

def main():
    parser = argparse.ArgumentParser(description="Generate affinities for specific dataset(s)")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--prediction_channels", type=int, default=3,
                       help="Number of prediction channels (3 or 6)")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Specific dataset to infer on (e.g., betaSeg). "
                            "If not provided, will infer on all datasets the model was trained on.")
    parser.add_argument("--mode", type=str, default="test", choices=["train", "val", "test"],
                       help="Which split to run inference on")
    parser.add_argument("--all_seeds", action="store_true",
                       help="Run inference on all seeds (default: only first seed)")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate thresholds (expensive)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Loading model...")
    print("="*80)
    model = BANIS.load_from_checkpoint(args.checkpoint_path)
    
    print("\nModel hyperparameters:")
    print(f"  Trained on: {model.hparams.data_setting}")
    print(f"  Base path: {model.hparams.base_data_path}")
    print(f"  Small size: {model.hparams.small_size}")
    
    if args.dataset:
        print(f"\nRunning inference on specified dataset: {args.dataset}")
    elif isinstance(model.hparams.data_setting, list):
        print(f"\nRunning inference on all {len(model.hparams.data_setting)} datasets")
    else:
        print(f"\nRunning inference on: {model.hparams.data_setting}")
    
    print("="*80)
    
    model.full_cube_inference(
        mode=args.mode,
        all_seeds=args.all_seeds,
        evaluate_thresholds=args.evaluate,
        prediction_channels=args.prediction_channels,
        data_setting=args.dataset
    )
    
    print("="*80)
    print("Inference completed!")
    print(f"Results saved to: {model.hparams.save_dir}")
    print("="*80)

if __name__ == "__main__":
    main()

