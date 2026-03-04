"""
Configuration module for gait analysis project.

Usage:
    from config.load_config import load_config, validate_config
    
    config = load_config('path/to/config.yaml')
    validate_config(config)

DO NOT use a global CONFIG variable. Instead, load configs explicitly 
in scripts and pass them as needed.
"""

from config.load_config import (
    load_config,
    validate_config, 
    merge_config_with_args,
    save_config
)

__all__ = [
    'load_config',
    'validate_config',
    'merge_config_with_args',
    'save_config'
]