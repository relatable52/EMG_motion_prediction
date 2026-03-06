import json
import yaml
import os
from typing import Any, Dict, Optional
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration parameters from a YAML or JSON file.
    
    Args:
        config_path (str): The path to the configuration file (.yaml, .yml, or .json).
    
    Returns:
        dict: A dictionary containing the configuration parameters.
    
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the file format is not supported.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    suffix = config_path.suffix.lower()
    
    with open(config_path, 'r') as f:
        if suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json")
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that the configuration contains all required fields and valid values.
    
    Args:
        config (dict): Configuration dictionary to validate.
    
    Raises:
        ValueError: If required fields are missing or values are invalid.
    """
    # Required fields
    required_fields = [
        'experiment_name', 'target_angle_name', 'sample_window_length',
        'prediction_horizon', 'sample_stride', 'batch_size', 'learning_rate',
        'epochs', 'backbone', 'prediction_type', 'model_spec'
    ]
    
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Missing required config fields: {missing_fields}")
    
    # Validate value ranges
    if config['batch_size'] <= 0:
        raise ValueError(f"batch_size must be positive, got {config['batch_size']}")
    
    if config['learning_rate'] <= 0:
        raise ValueError(f"learning_rate must be positive, got {config['learning_rate']}")
    
    if config['epochs'] <= 0:
        raise ValueError(f"epochs must be positive, got {config['epochs']}")
    
    if config['sample_window_length'] <= 0:
        raise ValueError(f"sample_window_length must be positive, got {config['sample_window_length']}")
    
    if config['prediction_horizon'] <= 0:
        raise ValueError(f"prediction_horizon must be positive, got {config['prediction_horizon']}")
    
    if config['sample_stride'] <= 0:
        raise ValueError(f"sample_stride must be positive, got {config['sample_stride']}")
    
    # Validate categorical choices
    valid_backbones = ['LSTM', 'TCN', 'DualBackbone']
    if config['backbone'] not in valid_backbones:
        raise ValueError(f"backbone must be one of {valid_backbones}, got {config['backbone']}")
    
    valid_prediction_types = ['deterministic', 'probabilistic']
    if config['prediction_type'] not in valid_prediction_types:
        raise ValueError(f"prediction_type must be one of {valid_prediction_types}, got {config['prediction_type']}")
    
    # Validate target_angle_name is a list
    if not isinstance(config['target_angle_name'], list):
        raise ValueError(f"target_angle_name must be a list, got {type(config['target_angle_name'])}")
    
    if len(config['target_angle_name']) == 0:
        raise ValueError("target_angle_name cannot be empty")


def merge_config_with_args(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """
    Merge configuration with command-line arguments.
    CLI arguments override config file values.
    
    Args:
        config (dict): Base configuration dictionary.
        args: Parsed argparse arguments object.
    
    Returns:
        dict: Merged configuration dictionary.
    """
    # Create a copy to avoid modifying the original
    merged_config = config.copy()
    
    # Map of CLI argument names to config keys
    arg_to_config = {
        'experiment_name': 'experiment_name',
        'seed': 'seed',
        'device': 'device',
        'target_angle_name': 'target_angle_name',
        'use_cache': 'use_cache',
        'batch_size': 'batch_size',
        'learning_rate': 'learning_rate',
        'epochs': 'epochs',
        'backbone': 'backbone',
        'prediction_type': 'prediction_type',
        'sample_window_length': 'sample_window_length',
        'prediction_horizon': 'prediction_horizon',
        'sample_stride': 'sample_stride',
        'early_stopping_patience': 'early_stopping_patience',
        'log_interval': 'log_interval',
        'save_best_model': 'save_best_model',
        'save_last_model': 'save_last_model',
    }
    
    # Override with CLI arguments if provided
    for arg_name, config_key in arg_to_config.items():
        if hasattr(args, arg_name):
            arg_value = getattr(args, arg_name)
            # Only override if the argument was explicitly provided (not None)
            if arg_value is not None:
                merged_config[config_key] = arg_value
    
    # Handle target_angle_name specially - convert comma-separated string to list
    if hasattr(args, 'target_angle_name') and args.target_angle_name is not None:
        if isinstance(args.target_angle_name, str):
            merged_config['target_angle_name'] = [name.strip() for name in args.target_angle_name.split(',')]
    
    return merged_config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config (dict): Configuration dictionary to save.
        save_path (str): Path where to save the config file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
