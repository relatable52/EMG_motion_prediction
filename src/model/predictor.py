import torch
import torch.nn as nn

class DeterministicModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # Standard regression head mapping to 1 joint angle
        self.fc_out = nn.Linear(backbone.hidden_dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        angle_prediction = self.fc_out(features)
        return angle_prediction
    
class ProbabilisticModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # Output 1: The predicted joint angle (Mean)
        self.fc_mean = nn.Linear(backbone.hidden_dim, 1)
        # Output 2: The uncertainty (Log-Variance)
        self.fc_logvar = nn.Linear(backbone.hidden_dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        pred_mean = self.fc_mean(features)
        pred_logvar = self.fc_logvar(features)
        return pred_mean, pred_logvar