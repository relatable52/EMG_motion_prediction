"""
Experiment tracking utilities for managing training runs and results.
"""
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from config.load_config import save_config

load_dotenv()


def create_experiment_dir(experiment_name: str, results_base_dir: Optional[str] = None) -> Path:
    """
    Create a timestamped experiment directory.
    
    Args:
        experiment_name (str): Name of the experiment.
        results_base_dir (str, optional): Base directory for results. 
                                         Defaults to RESULTS_DIR from environment.
    
    Returns:
        Path: Path to the created experiment directory.
    """
    if results_base_dir is None:
        results_base_dir = os.getenv('RESULTS_DIR', './results')
    
    # Create timestamp: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory name
    exp_dir_name = f"{experiment_name}_{timestamp}"
    exp_dir = Path(results_base_dir) / exp_dir_name
    
    # Create directory
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def save_experiment_config(config: Dict[str, Any], experiment_dir: Path) -> None:
    """
    Save the experiment configuration as YAML.
    
    Args:
        config (dict): Configuration dictionary to save.
        experiment_dir (Path): Experiment directory path.
    """
    config_path = experiment_dir / "config.yaml"
    save_config(config, str(config_path))


def save_training_history(history: Dict[str, list], experiment_dir: Path) -> None:
    """
    Save training history as CSV and generate plots.
    
    Args:
        history (dict): Training history with keys like 'train_loss', 'val_loss', 'val_mae'.
        experiment_dir (Path): Experiment directory path.
    """
    # Save as CSV
    history_df = pd.DataFrame(history)
    history_df.index.name = 'epoch'
    history_path = experiment_dir / "training_history.csv"
    history_df.to_csv(history_path)
    
    # Generate plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # MAE plot
    if 'val_mae' in history:
        axes[1].plot(history['val_mae'], label='Val MAE', marker='o', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE (degrees)')
        axes[1].set_title('Validation Mean Absolute Error')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = experiment_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


def log_system_info(experiment_dir: Path) -> None:
    """
    Log system information (Python version, PyTorch version, GPU, git commit).
    
    Args:
        experiment_dir (Path): Experiment directory path.
    """
    import sys
    import torch
    
    info = []
    info.append("=" * 60)
    info.append("SYSTEM INFORMATION")
    info.append("=" * 60)
    
    # Python version
    info.append(f"Python Version: {sys.version}")
    
    # PyTorch version
    info.append(f"PyTorch Version: {torch.__version__}")
    
    # CUDA availability
    info.append(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        info.append(f"CUDA Version: {torch.version.cuda}")
        info.append(f"GPU Device: {torch.cuda.get_device_name(0)}")
        info.append(f"GPU Count: {torch.cuda.device_count()}")
    
    # Git information (if available)
    try:
        git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                            stderr=subprocess.DEVNULL).decode().strip()
        info.append(f"Git Branch: {git_branch}")
        
        git_commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                            stderr=subprocess.DEVNULL).decode().strip()
        info.append(f"Git Commit: {git_commit}")
        
        # Check for uncommitted changes
        git_status = subprocess.check_output(['git', 'status', '--porcelain'],
                                            stderr=subprocess.DEVNULL).decode().strip()
        if git_status:
            info.append("Git Status: UNCOMMITTED CHANGES PRESENT")
        else:
            info.append("Git Status: Clean")
    except (subprocess.CalledProcessError, FileNotFoundError):
        info.append("Git: Not available or not a git repository")
    
    # Timestamp
    info.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    info.append("=" * 60)
    
    # Write to file
    info_path = experiment_dir / "system_info.txt"
    with open(info_path, 'w') as f:
        f.write('\n'.join(info))


def copy_log_to_experiment(log_file: str, experiment_dir: Path) -> None:
    """
    Copy the log file to the experiment directory.
    
    Args:
        log_file (str): Path to the log file.
        experiment_dir (Path): Experiment directory path.
    """
    if os.path.exists(log_file):
        dest_path = experiment_dir / "experiment.log"
        shutil.copy2(log_file, dest_path)
