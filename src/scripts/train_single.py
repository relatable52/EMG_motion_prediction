"""
Script to train a single model based on the provided configuration.
"""
import os
from dataclasses import fields

import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from config.config import ExperimentConfig, ModelConfig, TrainConfig
from data.dataset import PredictionDataset
from model.factory import create_model
from trainer.trainer import Trainer

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
    for field in fields(ExperimentConfig.data):
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
    for field in fields(ExperimentConfig.data):
        arg_name = f'data_{field.name}'
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            setattr(config.data, field.name, arg_value)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Create dataset and dataloader
    train_dataset = PredictionDataset(mode='train', prediction_horizon=config.data.prediction_horizon, output_fs=config.data.output_fs)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
    
    # Create model based on configuration
    model = create_model(config).to(config.train.device)
    
    # Initialize trainer
    trainer = Trainer(model=model, config=config)
    
    # Train the model
    trainer.train(train_loader)