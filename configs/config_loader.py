"""Configuration loading and saving utilities."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Any, Optional, Type, TypeVar
import yaml

from .base_config import BaseConfig
from .data_config import DataConfig, DatasetConfig
from .training_config import TrainingConfig
from .model_config import ModelConfig


T = TypeVar('T', bound=BaseConfig)


@dataclass
class FullConfig(BaseConfig):
    """Complete configuration for BANIS training.
    
    Attributes:
        data: Data loading and augmentation configuration.
        training: Training hyperparameters and settings.
        model: Model architecture configuration.
    """
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    
    def validate(self) -> None:
        """Validate all configuration components.
        
        Raises:
            ValueError: If any configuration is invalid.
        """
        self.data.validate()
        self.training.validate()
        self.model.validate()
        
        # Cross-validation between configs
        if self.training.batch_size > 32 and self.data.small_size > 256:
            import warnings
            warnings.warn(
                f"Large batch size ({self.training.batch_size}) with large patch size "
                f"({self.data.small_size}) may cause OOM errors."
            )


def load_config(
    config_path: Path,
    config_class: Type[T] = FullConfig,
    **overrides
) -> T:
    """Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to YAML configuration file.
        config_class: Configuration class to load into.
        **overrides: Key-value pairs to override configuration values.
        
    Returns:
        Configuration instance.
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist.
        ValueError: If configuration is invalid.
        
    Example:
        >>> config = load_config("configs/train.yaml", learning_rate=2e-3)
        >>> print(config.training.learning_rate)
        0.002
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if config_class == FullConfig:
        config = _load_full_config(config_dict)
    else:
        config = config_class.from_dict(config_dict)
    
    # Apply overrides
    if overrides:
        _apply_overrides(config, overrides)
    
    # Validate configuration
    config.validate()
    
    return config


def _load_full_config(config_dict: Dict[str, Any]) -> FullConfig:
    """Load full configuration from nested dictionary.
    
    Args:
        config_dict: Dictionary containing configuration.
        
    Returns:
        FullConfig instance.
    """
    # Parse datasets
    datasets = []
    if 'data' in config_dict and 'datasets' in config_dict['data']:
        for ds_dict in config_dict['data']['datasets']:
            datasets.append(DatasetConfig(**ds_dict))
    
    # Load data config
    data_dict = config_dict.get('data', {})
    data_dict['datasets'] = datasets
    data_config = DataConfig(**{k: v for k, v in data_dict.items() if k != 'datasets'})
    data_config.datasets = datasets
    
    # Load training config
    training_dict = config_dict.get('training', {})
    training_config = TrainingConfig(**training_dict)
    
    # Load model config
    model_dict = config_dict.get('model', {})
    model_config = ModelConfig(**model_dict)
    
    return FullConfig(
        data=data_config,
        training=training_config,
        model=model_config
    )


def _apply_overrides(config: Any, overrides: Dict[str, Any]) -> None:
    """Apply overrides to configuration recursively.
    
    Args:
        config: Configuration object to update.
        overrides: Dictionary of overrides with dotted keys (e.g., 'training.learning_rate').
    """
    for key, value in overrides.items():
        if '.' in key:
            # Handle nested keys like 'training.learning_rate'
            parts = key.split('.')
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            # Direct attribute
            if hasattr(config, key):
                setattr(config, key, value)


def save_config(
    config: BaseConfig,
    save_path: Path,
    include_defaults: bool = True
) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration instance to save.
        save_path: Path to save YAML file.
        include_defaults: Whether to include default values.
        
    Example:
        >>> config = FullConfig(...)
        >>> save_config(config, Path("outputs/config.yaml"))
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    with open(save_path, 'w') as f:
        yaml.dump(
            config_dict,
            f,
            default_flow_style=False,
            sort_keys=False,
            indent=2
        )


def load_legacy_config(legacy_config_path: Path) -> FullConfig:
    """Load legacy configuration and convert to new format.
    
    This function helps migrate from old YAML format to new dataclass-based config.
    
    Args:
        legacy_config_path: Path to legacy configuration file.
        
    Returns:
        FullConfig instance.
        
    Example:
        >>> config = load_legacy_config("config.yaml")
        >>> save_config(config, "configs/new_config.yaml")
    """
    with open(legacy_config_path, 'r') as f:
        legacy_dict = yaml.safe_load(f)
    
    # Convert legacy format to new format
    # This is project-specific and would need to be customized
    # based on the actual legacy format
    
    # For BANIS, the legacy format has 'params', 'mito', 'rib' sections
    params = legacy_dict.get('params', {})
    
    # Extract first value from lists in params (legacy format)
    def extract_value(val):
        return val[0] if isinstance(val, list) and len(val) > 0 else val
    
    params = {k: extract_value(v) for k, v in params.items()}
    
    # Build new config
    training_config = TrainingConfig(
        learning_rate=params.get('learning_rate', 1e-3),
        weight_decay=params.get('weight_decay', 1e-2),
        seed=params.get('seed', 0),
        batch_size=params.get('batch_size', 8),
        n_steps=params.get('n_steps', 50000),
        use_scheduler=params.get('scheduler', True),
    )
    
    model_config = ModelConfig(
        model_id=params.get('model_id', 'S'),
        kernel_size=params.get('kernel_size', 3),
    )
    
    # Parse dataset configs
    datasets = []
    for ds_section in ['mito', 'rib']:
        if ds_section in legacy_dict:
            for setting in legacy_dict[ds_section].get('settings', []):
                datasets.append(DatasetConfig(
                    name=setting['name'],
                    resolution=tuple(setting['resolution']),
                    path=Path(setting['path']),
                    train_splits=setting.get('train', []),
                    val_splits=setting.get('val', []),
                    test_splits=setting.get('test', []),
                ))
    
    data_config = DataConfig(
        datasets=datasets,
        base_data_path=Path(params.get('base_data_path', './data')),
        small_size=params.get('small_size', 128),
    )
    
    return FullConfig(
        data=data_config,
        training=training_config,
        model=model_config
    )

