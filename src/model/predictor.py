import torch
import torch.nn as nn
import gpytorch

class DeterministicRegressor(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int, output_dim: int = 1, dropout_rate: float = 0.2):
        """
        Standard MSE model. Also serves as the MC Dropout model during inference.
        """
        super().__init__()
        self.backbone = backbone
        
        # Projection head to map LSTM features to kinematic angles
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # x shape: (batch, n_channels, time, freq_scales)
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions

class ProbabilisticRegressor(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int, output_dim: int = 1):
        """
        Heteroscedastic Aleatoric model trained via NLL.
        """
        super().__init__()
        self.backbone = backbone
        
        self.feature_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Outputs 2 * output_dim (one for mean, one for log-variance)
        self.output_layer = nn.Linear(hidden_dim // 2, output_dim * 2)

    def forward(self, x):
        features = self.backbone(x)
        hidden = self.feature_mlp(features)
        out = self.output_layer(hidden)
        
        # Split the output tensor into Mean and Log-Variance (unconstrained)
        mu, logvar = torch.chunk(out, 2, dim=-1)
        
        return mu, logvar

class DeepEnsembleRegressor(nn.Module):
    def __init__(self, backbone_factory, num_models: int, hidden_dim: int, output_dim: int = 1):
        """
        Ensemble of entirely independent Deterministic networks.
        """
        super().__init__()
        self.num_models = num_models
        
        # Initialize N separate models from scratch
        self.models = nn.ModuleList([
            DeterministicRegressor(
                backbone=backbone_factory(), 
                hidden_dim=hidden_dim, 
                output_dim=output_dim,
                dropout_rate=0.2 # Standard regularization during training
            ) 
            for _ in range(num_models)
        ])

    def forward(self, x):
        # Collect predictions from all independent models
        # Output shape: (num_models, batch_size, output_dim)
        ensemble_preds = torch.stack([model(x) for model in self.models])
        return ensemble_preds
    
    def predict_uncertainty(self, x):
        """Helper method for inference."""
        self.eval()
        with torch.no_grad():
            preds = self.forward(x)
            mean_pred = torch.mean(preds, dim=0)
            epistemic_var = torch.var(preds, dim=0)
            return mean_pred, epistemic_var

class GPFeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int, latent_dim: int = 16):
        """
        Wraps the backbone and adds the crucial GP bottleneck.
        """
        super().__init__()
        self.backbone = backbone
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.bottleneck(features)
        
class DKLGaussianProcessRegressor(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor: nn.Module, latent_dim: int = 16, num_tasks: int = 2):
        # 1. Save the image dimensions so we can reconstruct them later
        self.original_shape = train_x.shape[1:] # e.g., (16, 200, 40)
        
        # 2. Flatten train_x to 2D [Samples, Features] to bypass GPyTorch's shape-checker
        train_x_flat = train_x.view(train_x.size(0), -1)
        
        super().__init__(train_x_flat, train_y, likelihood)
        
        self.feature_extractor = feature_extractor
        
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        base_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim)
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            base_kernel, num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        # 3. Unflatten the 2D input back into the 4D tensor the CNN expects
        x_unflat = x.view(x.size(0), *self.original_shape)
        
        projected_x = self.feature_extractor(x_unflat)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)