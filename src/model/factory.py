import torch
import torch.nn as nn
from config.config import ModelConfig, ExperimentConfig
from model.backbone import EMGScalogramBackbone
from model.predictor import (
    DeterministicRegressor,
    ProbabilisticRegressor,
    DeepEnsembleRegressor,
    GPFeatureExtractor,
    DKLGaussianProcess
)
import gpytorch


def create_model(config: ExperimentConfig) -> nn.Module:
    """
    Factory function to create model based on config.paradigm.
    
    Args:
        config: ExperimentConfig
    
    Returns:
        Initialized model
    """
    model_cfg = config.model
    paradigm = model_cfg.paradigm
    
    def make_backbone():
        return EMGScalogramBackbone(
            n_channels=model_cfg.n_channels,
            n_freq_scales=model_cfg.n_freq_scales,
            hidden_dim=model_cfg.hidden_dim
        )
    
    if paradigm == 'deterministic':
        return DeterministicRegressor(
            backbone=make_backbone(),
            hidden_dim=model_cfg.hidden_dim,
            output_dim=model_cfg.output_dim,
            dropout_rate=model_cfg.dropout_rate
        )
    
    elif paradigm == 'probabilistic':
        return ProbabilisticRegressor(
            backbone=make_backbone(),
            hidden_dim=model_cfg.hidden_dim,
            output_dim=model_cfg.output_dim
        )
    
    elif paradigm == 'ensemble':
        return DeepEnsembleRegressor(
            backbone_factory=make_backbone,
            num_models=model_cfg.ensemble_size,
            hidden_dim=model_cfg.hidden_dim,
            output_dim=model_cfg.output_dim
        )
    
    elif paradigm == 'mc_dropout':
        # MC-Dropout is just Deterministic with dropout enabled at test time
        return DeterministicRegressor(
            backbone=make_backbone(),
            hidden_dim=model_cfg.hidden_dim,
            output_dim=model_cfg.output_dim,
            dropout_rate=model_cfg.dropout_rate
        )
    
    elif paradigm == 'gp':
        # For GP, return feature extractor (model + likelihood created separately)
        feature_extractor = GPFeatureExtractor(
            backbone=make_backbone(),
            hidden_dim=model_cfg.hidden_dim,
            latent_dim=model_cfg.gp_latent_dim
        )
        return feature_extractor
    
    else:
        raise ValueError(f"Unknown paradigm: {paradigm}")


def create_gp_model(config: ExperimentConfig, train_x: torch.Tensor, 
                    train_y: torch.Tensor, feature_extractor: nn.Module):
    """
    Create GP model and likelihood for training.
    
    Args:
        config: ExperimentConfig
        train_x: Training input (batch, channels, time, freq_scales)
        train_y: Training targets (batch, output_dim)
        feature_extractor: Feature extractor backbone
    
    Returns:
        (model, likelihood)
    """
    model_cfg = config.model
    
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=model_cfg.output_dim
    )
    
    model = DKLGaussianProcess(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        feature_extractor=feature_extractor,
        latent_dim=model_cfg.gp_latent_dim,
        num_tasks=model_cfg.output_dim
    )
    
    return model, likelihood