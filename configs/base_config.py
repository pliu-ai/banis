"""Base configuration classes using dataclasses."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


@dataclass
class BaseConfig:
    """Base configuration class with common utilities.
    
    All configuration classes should inherit from this base class.
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return asdict(self)
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: Path to save the YAML file.
        """
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters.
            
        Returns:
            Configuration instance.
        """
        # Filter to only include fields that exist in the dataclass
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'BaseConfig':
        """Load configuration from YAML file.
        
        Args:
            path: Path to the YAML file.
            
        Returns:
            Configuration instance.
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def update(self, **kwargs) -> None:
        """Update configuration with new values.
        
        Args:
            **kwargs: Key-value pairs to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid.
        """
        # Override in subclasses for specific validation
        pass

