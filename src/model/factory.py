from config.config import ModelConfig
from model.backbone import EMGScalogramBackbone
from model.predictor import DeterministicRegressor, ProbabilisticRegressor, DeepEnsembleRegressor, DKLGaussianProcessRegressor

def create_nn_model(model_config: ModelConfig):
    emg_backbone_factory = lambda: EMGScalogramBackbone(
        n_channels=model_config.n_emg_channels,
        hidden_dim=model_config.emg_hidden_dim,
        n_freq_scales=model_config.n_freq_scales
    )

    if model_config.model.paradigm == 'deterministic':
        return DeterministicRegressor(
            backbone=emg_backbone_factory(),
            hidden_dim=model_config.emg_hidden_dim,
            output_dim=model_config.output_dim,
            dropout_rate=model_config.dropout_rate
        )
    elif model_config.model.paradigm == 'probabilistic':
        return ProbabilisticRegressor(
            backbone=emg_backbone_factory(),
            hidden_dim=model_config.emg_hidden_dim,
            output_dim=model_config.output_dim
        )
    elif model_config.model.paradigm == 'ensemble':
        return DeepEnsembleRegressor(
            backbone_factory=emg_backbone_factory,
            num_models=model_config.model.ensemble_size,
            hidden_dim=model_config.emg_hidden_dim,
            output_dim=model_config.output_dim
        )
    
def create_gp_model(model_config: ModelConfig, train_x, train_y, likelihood):
    emg_backbone = EMGScalogramBackbone(
        n_channels=model_config.n_emg_channels,
        hidden_dim=model_config.model.gp_latent_dim,
        n_freq_scales=model_config.n_freq_scales
    )
    return DKLGaussianProcessRegressor(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        feature_extractor=emg_backbone,
        latent_dim=model_config.model.gp_latent_dim,
        num_tasks=model_config.output_dim
    )
