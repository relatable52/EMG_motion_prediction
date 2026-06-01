"""
Script to train a single model based on the provided configuration.
"""
import os
import json
from dataclasses import fields, asdict

import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from config.config import ExperimentConfig, ModelConfig, TrainConfig, DataConfig
from data.dataset import PredictionDataset
from model.factory import create_model, create_gp_model
from trainer.trainer import Trainer

def extract_full_dataset(dataloader, max_samples=None):
    """
    Extracts all samples from a DataLoader into full tensors (CPU).
    
    Args:
        dataloader: PyTorch DataLoader
        max_samples: If set, randomly subsamples to this many samples
        
    Returns:
        full_x: All EMG features of shape (Total_Samples, Channels, Time, Freq)
        full_y: All target labels of shape (Total_Samples, Output_Dim)
    """
    all_x = []
    all_y = []
    
    print("Extracting full dataset from DataLoader into RAM...")
    
    with torch.no_grad():
        for emg_sample, _, label in dataloader:
            all_x.append(emg_sample)
            all_y.append(label)
    
    full_x = torch.cat(all_x, dim=0)
    full_y = torch.cat(all_y, dim=0)
    
    # Optional subsampling to avoid memory issues
    if max_samples and full_x.size(0) > max_samples:
        idx = torch.randperm(full_x.size(0))[:max_samples]
        full_x = full_x[idx]
        full_y = full_y[idx]
        print(f"Subsampled to {max_samples} samples")
    
    print(f"Extraction complete! X shape: {full_x.shape}, Y shape: {full_y.shape}")
    
    return full_x, full_y

def parse_args():
    parser = ArgumentParser(description="Train a single model based on the provided configuration.")
    
    # Auto-add fields from ModelConfig
    for field in fields(ModelConfig):
        parser.add_argument(f'--model-{field.name.replace("_", "-")}', type=field.type, 
                          help=f'Model {field.name}')
    
    # Auto-add fields from TrainConfig
    for field in fields(TrainConfig):
        parser.add_argument(f'--train-{field.name.replace("_", "-")}', type=field.type, 
                          help=f'Train {field.name}')

    # Auto-add fields from DataConfig
    for field in fields(DataConfig):
        parser.add_argument(f'--data-{field.name.replace("_", "-")}', type=field.type, 
                          help=f'Data {field.name}')
    
    # Manually add experiment name
    parser.add_argument('--exp-name', type=str, default='default_experiment', 
                       help='Experiment name')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load experiment configuration
    config = ExperimentConfig(exp_name=args.exp_name)
    
    # Override ModelConfig values from command-line arguments
    for field in fields(ModelConfig):
        arg_name = f'model_{field.name}'
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            setattr(config.model, field.name, arg_value)
    
    # Override TrainConfig values from command-line arguments
    for field in fields(TrainConfig):
        arg_name = f'train_{field.name}'
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            setattr(config.train, field.name, arg_value)

    # Override DataConfig values from command-line arguments
    for field in fields(DataConfig):
        arg_name = f'data_{field.name}'
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            setattr(config.data, field.name, arg_value)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Create dataset and dataloader with optimized settings for GPU performance
    train_dataset = PredictionDataset(
        mode='train',
        window_length=config.data.window_length,
        stride=config.data.stride,
        prediction_horizon=config.data.prediction_horizon,
        target_angle_name=config.data.target_angle_name,
        use_cache=config.data.use_cache,
        cache_dir=config.env.cache_dir,
        output_fs=config.data.output_fs,
        freq_min=config.data.freq_min,
        freq_max=config.data.freq_max,
        n_scales=config.data.n_scales,
        split_strategy=config.data.split_strategy,
        n_folds=config.data.n_folds,
        fold_index=config.data.fold_index,
        test_subjects=config.data.test_subjects,
        test_activities=config.data.test_activities,
        split_random_state=config.data.split_random_state
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.train.batch_size, 
        shuffle=True,
        num_workers=8,              # Parallelize data loading across CPU cores
        pin_memory=True,            # Keep data in pinned memory for faster GPU transfer
        persistent_workers=True,    # Avoid worker restart overhead
        prefetch_factor=2           # Pre-load batches ahead of time
    )
    
    test_dataset = PredictionDataset(
        mode='test',
        window_length=config.data.window_length,
        stride=config.data.stride,
        prediction_horizon=config.data.prediction_horizon,
        target_angle_name=config.data.target_angle_name,
        use_cache=config.data.use_cache,
        cache_dir=config.env.cache_dir,
        output_fs=config.data.output_fs,
        freq_min=config.data.freq_min,
        freq_max=config.data.freq_max,
        n_scales=config.data.n_scales,
        split_strategy=config.data.split_strategy,
        n_folds=config.data.n_folds,
        fold_index=config.data.fold_index,
        test_subjects=config.data.test_subjects,
        test_activities=config.data.test_activities,
        split_random_state=config.data.split_random_state
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.train.batch_size, 
        shuffle=False,
        num_workers=4,              # Lower for inference
        pin_memory=True,
        persistent_workers=True
    )
    
    # Experiment save directory
    save_dir = os.path.join(config.env.results_dir, config.exp_name)
    
    # Create model based on paradigm
    if config.model.paradigm == 'gp':
        # For GP, need to extract training data first
        feature_extractor = create_model(config)
        
        # Extract full training data and subsample to avoid memory issues
        train_x, train_y = extract_full_dataset(train_loader, max_samples=8000)
        
        # Create GP model and likelihood
        model, likelihood = create_gp_model(config, train_x, train_y, feature_extractor)
        
        # Initialize trainer with GP model and likelihood
        trainer = Trainer(model=model, config=config, likelihood=likelihood)
        
        # Train using full-batch GP training
        train_history = trainer.train(gp_data=(train_x, train_y))
        
        # Run inference and save predictions
        inference_results = trainer.predict(test_loader, save_dir=save_dir)
    else:
        # For non-GP models, create model normally
        model = create_model(config).to(config.train.device)
        
        # Initialize trainer
        trainer = Trainer(model=model, config=config)
        
        # Train using mini-batch training
        train_history = trainer.train(train_loader)
        
        # Run inference and save predictions
        inference_results = trainer.predict(test_loader, save_dir=save_dir)
    
    # Save config and training history
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config as JSON
    config_dict = asdict(config)
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Config saved to: {config_path}")
    
    # Save training history as JSON
    history_path = os.path.join(save_dir, 'train_history.json')
    history_dict = {k: [float(v) for v in vs] for k, vs in train_history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    print(f"\nTraining and inference complete!")
    print(f"Results saved to: {save_dir}")

if __name__ == '__main__':
    main()