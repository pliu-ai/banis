#!/usr/bin/env python3
"""
Configuration presets for different processing scenarios.
"""

# Processing configurations for different use cases
PROCESSING_CONFIGS = {
    "fast": {
        "description": "Fast processing with minimal post-processing",
        "threshold": 0.5,
        "min_size": 100,
        "max_instances": None,
        "morphological_ops": None,
        "n_workers": None,  # Use all CPU cores
        "verbose": False
    },
    
    "balanced": {
        "description": "Balanced processing with moderate post-processing",
        "threshold": 0.5,
        "min_size": 200,
        "max_instances": 22,
        "morphological_ops": ["opening", "closing"],
        "n_workers": None,
        "verbose": True
    },
    
    "high_quality": {
        "description": "High quality processing with extensive post-processing",
        "threshold": 0.4,
        "min_size": 300,
        "max_instances": 22,
        "morphological_ops": ["opening", "closing"],
        "n_workers": None,
        "verbose": True
    },
    
    "rib_segmentation": {
        "description": "Optimized for rib segmentation",
        "threshold": 0.5,
        "min_size": 500,
        "max_instances": 22,
        "morphological_ops": ["opening", "closing"],
        "n_workers": None,
        "verbose": True
    },
    
    "debug": {
        "description": "Debug mode with verbose output",
        "threshold": 0.5,
        "min_size": 100,
        "max_instances": None,
        "morphological_ops": None,
        "n_workers": 1,  # Single worker for debugging
        "verbose": True
    }
}


def get_config(config_name):
    """
    Get processing configuration by name.
    
    Args:
        config_name: Name of the configuration preset
        
    Returns:
        dict: Configuration parameters
    """
    if config_name not in PROCESSING_CONFIGS:
        available_configs = list(PROCESSING_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available configs: {available_configs}")
    
    return PROCESSING_CONFIGS[config_name].copy()


def list_configs():
    """
    List all available configurations.
    
    Returns:
        dict: All available configurations with descriptions
    """
    return {name: config["description"] for name, config in PROCESSING_CONFIGS.items()}


def print_configs():
    """Print all available configurations."""
    print("Available processing configurations:")
    print("=" * 50)
    for name, config in PROCESSING_CONFIGS.items():
        print(f"{name:15} - {config['description']}")
        print(f"{'':15}   threshold: {config['threshold']}")
        print(f"{'':15}   min_size: {config['min_size']}")
        print(f"{'':15}   max_instances: {config['max_instances']}")
        print(f"{'':15}   morphological: {config['morphological_ops']}")
        print(f"{'':15}   workers: {config['n_workers'] or 'auto'}")
        print()


if __name__ == "__main__":
    print_configs()

