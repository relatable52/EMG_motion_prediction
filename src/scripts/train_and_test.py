"""
Train and test script for gait analysis prediction model.
Supports CLI arguments to override config file settings.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.load_config import load_config, validate_config, merge_config_with_args
from dataset.prediction_dataset import PredictionDataset
from model.backbone import EMGScalogramBackbone, AngleHistoryBackbone, DualBackbone
from model.predictor import DeterministicModel, ProbabilisticModel
from trainer.specific_trainer import DeterministicTrainer, ProbabilisticTrainer
from utils.experiment import (
    create_experiment_dir, 
    save_experiment_config, 
    save_training_history,
    log_system_info,
    copy_log_to_experiment
)
from utils.logger import logger

load_dotenv()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and test gait prediction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument('--config', type=str, default='src/config/default_config.yaml',
                       help='Path to config file (YAML or JSON)')
    
    # Experiment settings
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for the experiment (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                       help='Device to use for training')
    
    # Target configuration
    parser.add_argument('--target_angle_name', type=str, default=None,
                       help='Comma-separated target angles (e.g., "knee_angle_r,knee_angle_l")')
    
    # Data processing
    parser.add_argument('--use_cache', action='store_true', default=None,
                       help='Use cached processed data (default: from config)')
    parser.add_argument('--no_cache', dest='use_cache', action='store_false',
                       help='Disable data caching, reprocess from scratch')
    parser.add_argument('--sample_window_length', type=float, default=None,
                       help='Input window length in seconds')
    parser.add_argument('--prediction_horizon', type=float, default=None,
                       help='Prediction horizon in seconds')
    parser.add_argument('--sample_stride', type=float, default=None,
                       help='Stride between windows in seconds')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', '--lr', type=float, default=None, dest='learning_rate',
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                       help='Early stopping patience (0 to disable)')
    
    # Model architecture
    parser.add_argument('--backbone', type=str, default=None, choices=['LSTM', 'TCN'],
                       help='Backbone architecture')
    parser.add_argument('--prediction_type', type=str, default=None, 
                       choices=['deterministic', 'probabilistic'],
                       help='Prediction type')
    
    # Logging and model saving
    parser.add_argument('--log_interval', type=int, default=None,
                       help='Log metrics every N epochs')
    parser.add_argument('--save_best_model', action='store_true', default=None,
                       help='Save best model based on validation loss')
    parser.add_argument('--save_last_model', action='store_true', default=None,
                       help='Save final model state')
    
    return parser.parse_args()


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(config: dict, input_dim: int):
    """
    Legacy build function - now redirects to dual-backbone.
    For single-backbone models, use the new DualBackbone architecture.
    
    Args:
        config (dict): Configuration dictionary.
        input_dim (int): Input feature dimension (not used, kept for compatibility).
    
    Returns:
        nn.Module: The constructed model.
    """
    backbone_type = config['backbone']
    raise ValueError(
        f"Legacy backbone type '{backbone_type}' is no longer supported. "
        f"Please use 'DualBackbone' with the new wavelet-based architecture. "
        f"See config_dual_backbone.yaml for an example configuration."
    )


def build_dual_backbone_model(config: dict, dataset):
    """
    Build dual-backbone model for wavelet EMG + angle data.
    
    Args:
        config (dict): Configuration dictionary.
        dataset: PredictionDataset instance (to get data shapes).
    
    Returns:
        nn.Module: The constructed model.
    """
    # Get data dimensions from first sample
    emg_sample, angle_sample, _ = dataset[0]
    n_channels = emg_sample.shape[0]
    n_freq_scales = emg_sample.shape[2]
    n_angles = angle_sample.shape[0]
    
    model_spec = config['model_spec']
    feature_mode = config.get('feature_mode', 'both')
    
    logger.info(f"Building dual-backbone model:")
    logger.info(f"  EMG shape: ({n_channels} channels, {emg_sample.shape[1]} time, {n_freq_scales} freq)")
    logger.info(f"  Angle shape: ({n_angles} angles, {angle_sample.shape[1]} time)")
    logger.info(f"  Feature mode: {feature_mode}")
    
    # Build EMG backbone
    emg_backbone = EMGScalogramBackbone(
        n_channels=n_channels,
        n_freq_scales=n_freq_scales,
        hidden_dim=model_spec['emg_hidden_dim'],
        backbone_type=model_spec.get('emg_backbone_type', 'conv2d_lstm')
    )
    logger.info(f"  EMG backbone: {model_spec.get('emg_backbone_type', 'conv2d_lstm')} (hidden_dim={model_spec['emg_hidden_dim']})")
    
    # Build angle backbone
    angle_backbone = AngleHistoryBackbone(
        n_angles=n_angles,
        hidden_dim=model_spec['angle_hidden_dim'],
        backbone_type=model_spec.get('angle_backbone_type', 'lstm')
    )
    logger.info(f"  Angle backbone: {model_spec.get('angle_backbone_type', 'lstm')} (hidden_dim={model_spec['angle_hidden_dim']})")
    
    # Build dual backbone
    dual_backbone = DualBackbone(
        emg_backbone=emg_backbone,
        angle_backbone=angle_backbone,
        feature_mode=feature_mode,
        fusion_hidden_dim=model_spec['fusion_hidden_dim']
    )
    logger.info(f"  Fusion layer: hidden_dim={model_spec['fusion_hidden_dim']}")
    
    # Build predictor
    n_target_angles = len(config['target_angle_name'])
    prediction_type = config['prediction_type']
    if prediction_type == 'deterministic':
        model = DeterministicModel(dual_backbone, output_dim=n_target_angles)
    elif prediction_type == 'probabilistic':
        model = ProbabilisticModel(dual_backbone, output_dim=n_target_angles)
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")
    
    logger.info(f"  Prediction type: {prediction_type}")
    logger.info(f"  Output dimension: {n_target_angles} angles")
    
    return model


def main():
    """Main training and testing pipeline."""
    
    # Parse arguments
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("GAIT PREDICTION MODEL - TRAINING AND TESTING")
    logger.info("=" * 80)
    
    # Load base config
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Merge with CLI arguments
    config = merge_config_with_args(config, args)
    
    # Validate config
    logger.info("Validating configuration...")
    validate_config(config)
    logger.info("Configuration valid")
    
    # Set random seeds
    seed = config.get('seed', 42)
    set_random_seeds(seed)
    logger.info(f"Random seed set to: {seed}")
    
    # Determine device
    device_config = config.get('device', 'cuda')
    if device_config == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Using CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(device_config)
    logger.info(f"Using device: {device}")
    
    # Create experiment directory
    experiment_name = config['experiment_name']
    experiment_dir = create_experiment_dir(experiment_name)
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # Save system info
    log_system_info(experiment_dir)
    
    # Save config snapshot
    save_experiment_config(config, experiment_dir)
    logger.info("Configuration saved")
    
    # Create datasets
    logger.info("\nPreparing datasets...")
    use_cache = config.get('use_cache', True)
    
    # Get wavelet parameters from config if using DualBackbone
    dataset_kwargs = {
        'mode': 'train',
        'window_length': config['sample_window_length'],
        'stride': config['sample_stride'],
        'prediction_horizon': config['prediction_horizon'],
        'target_angle_name': config['target_angle_name'],
        'use_cache': use_cache
    }
    
    # Add wavelet parameters if they exist in config
    if 'output_fs' in config:
        dataset_kwargs['output_fs'] = config['output_fs']
    if 'freq_min' in config:
        dataset_kwargs['freq_min'] = config['freq_min']
    if 'freq_max' in config:
        dataset_kwargs['freq_max'] = config['freq_max']
    if 'n_scales' in config:
        dataset_kwargs['n_scales'] = config['n_scales']
    
    train_dataset = PredictionDataset(**dataset_kwargs)
    
    # Create validation dataset with same parameters
    val_dataset_kwargs = dataset_kwargs.copy()
    val_dataset_kwargs['mode'] = 'test'
    val_dataset = PredictionDataset(**val_dataset_kwargs)
    
    test_dataset = val_dataset  # Same as validation
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Determine input dimension
    # Features: EMG features + angle features
    # Get sample to determine dimensions
    sample_emg, sample_angle, _ = train_dataset[0]
    n_emg_features = sample_emg.shape[0]
    n_angle_features = sample_angle.shape[0]
    
    features_used = config.get('features_used', 'all')
    if features_used == 'all':
        input_dim = n_emg_features + n_angle_features
    elif features_used == 'emg_only':
        input_dim = n_emg_features
    elif features_used == 'angle_only':
        input_dim = n_angle_features
    else:
        raise ValueError(f"Unknown features_used: {features_used}")
    
    logger.info(f"Input dimension: {input_dim} ({n_emg_features} EMG + {n_angle_features} angle features)")
    
    # Build model
    logger.info("\nBuilding model...")
    backbone_type = config['backbone']
    
    if backbone_type == 'DualBackbone':
        # Use dual-backbone architecture for wavelet data
        model = build_dual_backbone_model(config, train_dataset)
    else:
        # Use traditional single-backbone architecture
        model = build_model(config, input_dim)
    
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {config['backbone']} + {config['prediction_type']}")
    logger.info(f"Total trainable parameters: {n_params:,}")
    
    # Create optimizer
    learning_rate = config['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logger.info(f"Optimizer: Adam (lr={learning_rate})")
    
    # Create trainer
    prediction_type = config['prediction_type']
    if prediction_type == 'deterministic':
        trainer = DeterministicTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            device=device,
            n_features=input_dim
        )
    else:
        trainer = ProbabilisticTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            device=device,
            n_features=input_dim
        )
    
    # Train
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    epochs = config['epochs']
    log_interval = config.get('log_interval', 1)
    save_best_model = config.get('save_best_model', True)
    save_last_model = config.get('save_last_model', True)
    
    history = trainer.train(
        epochs=epochs, 
        save_dir=str(experiment_dir),
        log_interval=log_interval,
        save_best_model=save_best_model,
        save_last_model=save_last_model
    )
    
    logger.info("\nTraining complete")
    
    # Save training history
    save_training_history(history, experiment_dir)
    logger.info("Training history saved")
    
    # Load best model for testing
    logger.info("\nLoading best model for testing...")
    best_model_path = experiment_dir / "best_model.pth"
    model.load_state_dict(torch.load(best_model_path))
    
    # Test
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TESTING")
    logger.info("=" * 80)
    
    test_results = trainer.test(
        save_dir=str(experiment_dir), 
        prefix="test",
        angle_names=config['target_angle_name']
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {experiment_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
