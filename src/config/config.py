import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv
import torch

load_dotenv()

@dataclass
class EnvConfig:
    data_dir: str = os.getenv('DATA_DIR', '/kaggle/input/ga-tech-emg-dataset')
    results_dir: str = os.getenv('RESULTS_DIR', '/kaggle/temp/results')
    log_dir: str = os.getenv('LOG_DIR', '/kaggle/working/logs')
    model_dir: str = os.getenv('MODEL_DIR', '/kaggle/working/models')
    cache_dir: str = os.getenv('CACHE_DIR', '/kaggle/working/cache')
    emg_freq: int = int(os.getenv('EMG_FREQUENCY', 2000))
    angle_freq: int = int(os.getenv('ANGLE_FREQUENCY', 200))

@dataclass
class DataConfig:
    window_length: float = 1.0
    stride: float = 0.05
    prediction_horizon: float = 0.05
    # Use default_factory for mutable types like lists
    target_angle_name: List[str] = field(default_factory=lambda: ['knee_angle_r', 'knee_angle_l'])
    use_cache: bool = True
    output_fs: int = 100
    freq_min: float = 5.0
    freq_max: float = 450.0
    n_scales: int = 40

@dataclass
class ModelConfig:
    paradigm: str = 'deterministic'  # ensemble | mc_dropout | probabilistic | gp
    hidden_dim: int = 128
    n_channels: int = 16
    n_freq_scales: int = 40  # Synced from data.n_scales in __post_init__
    output_dim: int = 12
    dropout_rate: float = 0.2
    ensemble_size: int = 20
    mc_dropout_passes: int = 20
    gp_latent_dim: int = 16
    gp_inducing_points: int = 100

@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_interval: int = 10

@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 42
    exp_name: str = "default_experiment"
    
    def __post_init__(self):
        # Auto-sync model parameters with data parameters to prevent shape mismatches!
        self.model.n_freq_scales = self.data.n_scales
        self.model.output_dim = len(self.data.target_angle_name)

    @classmethod
    def override(cls, **kwargs) -> 'ExperimentConfig':
        """Create config with overrides."""
        config = ExperimentConfig()
        for key, value in kwargs.items():
            section, field = key.split('.')
            setattr(getattr(config, section), field, value)
        config.__post_init__()
        return config