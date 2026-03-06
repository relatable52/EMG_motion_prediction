import torch
import torch.nn as nn

class DeterministicModel(nn.Module):
    def __init__(self, backbone, output_dim=1):
        super().__init__()
        self.backbone = backbone
        # Standard regression head mapping to output_dim joint angles
        self.fc_out = nn.Linear(backbone.hidden_dim, output_dim)

    def forward(self, *args, **kwargs):
        features = self.backbone(*args, **kwargs)
        angle_prediction = self.fc_out(features)
        return angle_prediction
    
class ProbabilisticModel(nn.Module):
    def __init__(self, backbone, output_dim=1):
        super().__init__()
        self.backbone = backbone
        # Output 1: The predicted joint angle (Mean)
        self.fc_mean = nn.Linear(backbone.hidden_dim, output_dim)
        # Output 2: The uncertainty (Log-Variance)
        self.fc_logvar = nn.Linear(backbone.hidden_dim, output_dim)

    def forward(self, *args, **kwargs):
        features = self.backbone(*args, **kwargs)
        pred_mean = self.fc_mean(features)
        pred_logvar = self.fc_logvar(features)
        return pred_mean, pred_logvar