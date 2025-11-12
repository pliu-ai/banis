#!/usr/bin/env python3
"""Training script for BANIS.

This script provides a clean interface for training BANIS models
using the refactored codebase.

Example:
    # Train with default config
    python scripts/train.py --config examples/train_mito.yaml
    
    # Train with overrides
    python scripts/train.py --config examples/train_mito.yaml \
        --training.batch_size 16 --training.learning_rate 0.002
    
    # Resume from checkpoint
    python scripts/train.py --config examples/train_mito.yaml \
        --training.resume_from_checkpoint outputs/exp_001/checkpoints/last.ckpt
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config_loader import load_config, FullConfig
from banis.core.trainer import BANISTrainer
from banis.utils.logging import setup_logger
from data import load_data  # Using legacy data loading for now

import torch
from pytorch_lightning import seed_everything


def parse_args():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train BANIS model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/train_mito.yaml"),
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Allow config overrides via command line
    parser.add_argument(
        "--override",
        nargs='*',
        help=(
            "Override config values using dot notation, e.g., "
            "'training.batch_size=16 training.learning_rate=0.002'"
        )
    )
    
    return parser.parse_args()


def parse_overrides(override_list):
    """Parse override strings into a dictionary.
    
    Args:
        override_list: List of override strings like ['training.batch_size=16'].
        
    Returns:
        Dictionary of overrides.
    """
    overrides = {}
    if override_list:
        for override in override_list:
            key, value = override.split('=')
            # Try to convert to appropriate type
            try:
                value = eval(value)  # Handles int, float, bool, list, etc.
            except:
                pass  # Keep as string if eval fails
            overrides[key] = value
    return overrides


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger(
        name="banis.train",
        level=getattr(__import__('logging'), args.log_level),
    )
    logger.info("=" * 80)
    logger.info("BANIS Training")
    logger.info("=" * 80)
    
    # Parse overrides
    overrides = parse_overrides(args.override)
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config: FullConfig = load_config(args.config, **overrides)
    
    logger.info("Configuration loaded successfully")
    logger.info(f"Training configuration:")
    logger.info(f"  - Experiment: {config.training.exp_name or 'auto-generated'}")
    logger.info(f"  - Datasets: {[ds.name for ds in config.data.datasets]}")
    logger.info(f"  - Batch size: {config.training.batch_size}")
    logger.info(f"  - Learning rate: {config.training.learning_rate}")
    logger.info(f"  - Steps: {config.training.n_steps}")
    logger.info(f"  - Model: {config.model.model_id} (kernel size {config.model.kernel_size})")
    
    # Set seed
    seed_everything(config.training.seed, workers=True)
    logger.info(f"Random seed set to: {config.training.seed}")
    
    # Set PyTorch settings
    torch.set_float32_matmul_precision("medium")
    
    # Load data using legacy function (will be refactored later)
    logger.info("Loading training data...")
    # Convert config to argparse.Namespace for compatibility with legacy code
    from argparse import Namespace
    legacy_args = Namespace(
        data_setting=[ds.name for ds in config.data.datasets],
        base_data_path=str(config.data.base_data_path),
        small_size=config.data.small_size,
        long_range=config.data.long_range,
        augment=config.data.augment,
        drop_slice_prob=config.data.drop_slice_prob,
        shift_slice_prob=config.data.shift_slice_prob,
        shift_magnitude=config.data.shift_magnitude,
        intensity_aug=config.data.intensity_aug,
        noise_scale=config.data.noise_scale,
        affine=config.data.affine_prob,
        affine_scale=config.data.affine_scale,
        affine_shear=config.data.affine_shear,
        mul_int=config.data.mul_intensity,
        add_int=config.data.add_intensity,
        synthetic=config.data.synthetic_ratio,
        real_data_path=str(config.data.real_data_path) if config.data.real_data_path else "",
        workers=config.data.num_workers,
    )
    
    train_dataset, val_dataset, n_channels = load_data(legacy_args)
    config.model.num_input_channels = n_channels
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Input channels: {n_channels}")
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        shuffle=False,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else None,
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = BANISTrainer(
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
    )
    
    logger.info(f"Outputs will be saved to: {trainer.save_dir}")
    
    # Print model summary
    from banis.core.model import count_parameters
    n_params = count_parameters(trainer.lightning_module.model)
    logger.info(f"Model parameters: {n_params:,}")
    
    # Start training
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    try:
        trainer.train()
        
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Best checkpoint: {trainer.best_model_path}")
        logger.info(f"Last checkpoint: {trainer.last_model_path}")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        logger.info(f"Last checkpoint: {trainer.last_model_path}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

