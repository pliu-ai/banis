"""Data configuration classes."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional


@dataclass
class DatasetConfig:
    """Configuration for a single dataset.
    
    Attributes:
        name: Dataset name identifier.
        resolution: Voxel resolution in (z, y, x) order (nm or other units).
        path: Path to dataset directory.
        train_splits: List of training volume names.
        val_splits: List of validation volume names.
        test_splits: List of test volume names.
    """
    name: str
    resolution: Tuple[float, float, float]
    path: Path
    train_splits: List[str] = field(default_factory=list)
    val_splits: List[str] = field(default_factory=list)
    test_splits: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Convert path to Path object if it's a string."""
        if isinstance(self.path, str):
            self.path = Path(self.path)
        if isinstance(self.resolution, list):
            self.resolution = tuple(self.resolution)


@dataclass
class DataConfig:
    """Data loading and augmentation configuration.
    
    Attributes:
        # Dataset settings
        datasets: List of dataset configurations.
        base_data_path: Base path for all datasets.
        small_size: Size of patches to sample (voxels).
        long_range: Long-range affinity offset (voxels).
        
        # Data loading
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
        prefetch_factor: Number of batches to prefetch per worker.
        
        # Augmentation settings
        augment: Whether to apply data augmentation.
        drop_slice_prob: Probability of dropping a slice.
        shift_slice_prob: Probability of shifting a slice.
        shift_magnitude: Magnitude of shift augmentation (voxels).
        
        # Intensity augmentation
        intensity_aug: Whether to apply intensity augmentation.
        noise_scale: Scale of noise augmentation.
        mul_intensity: Multiplicative intensity augmentation range.
        add_intensity: Additive intensity augmentation range.
        
        # Geometric augmentation
        affine_prob: Probability of applying affine transformation.
        affine_scale: Scale range for affine augmentation.
        affine_shear: Shear range for affine augmentation.
        
        # Training data settings
        synthetic_ratio: Ratio of synthetic to real data [0, 1].
        real_data_path: Path to real data (if using mixed training).
    """
    
    # Dataset settings
    datasets: List[DatasetConfig] = field(default_factory=list)
    base_data_path: Path = Path("./data")
    small_size: int = 128
    long_range: int = 10
    
    # Data loading
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Augmentation settings
    augment: bool = True
    drop_slice_prob: float = 0.05
    shift_slice_prob: float = 0.05
    shift_magnitude: int = 10
    
    # Intensity augmentation
    intensity_aug: bool = True
    noise_scale: float = 0.5
    mul_intensity: float = 0.1
    add_intensity: float = 0.1
    
    # Geometric augmentation
    affine_prob: float = 0.5
    affine_scale: float = 0.2
    affine_shear: float = 0.5
    
    # Training data settings
    synthetic_ratio: float = 1.0
    real_data_path: Optional[Path] = None
    
    def __post_init__(self):
        """Validate and convert paths."""
        if isinstance(self.base_data_path, str):
            self.base_data_path = Path(self.base_data_path)
        if self.real_data_path is not None and isinstance(self.real_data_path, str):
            self.real_data_path = Path(self.real_data_path)
    
    def validate(self) -> None:
        """Validate data configuration.
        
        Raises:
            ValueError: If configuration is invalid.
        """
        if not 0 <= self.synthetic_ratio <= 1:
            raise ValueError(f"synthetic_ratio must be in [0, 1], got {self.synthetic_ratio}")
        
        if self.small_size < 32 or self.small_size > 512:
            raise ValueError(f"small_size should be in [32, 512], got {self.small_size}")
        
        if self.long_range < 1:
            raise ValueError(f"long_range must be positive, got {self.long_range}")
        
        if not self.datasets:
            raise ValueError("At least one dataset must be specified")
        
        for dataset in self.datasets:
            if not dataset.train_splits:
                raise ValueError(f"Dataset '{dataset.name}' has no training splits")

