import torch
import torch.nn as nn

from trainer.base import BaseTrainer
from model.backbone import DualBackbone

class DeterministicTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, device, n_features=32):
        super().__init__(model, train_loader, val_loader, test_loader, optimizer, device, n_features)
        self.criterion = nn.MSELoss()
        # Ensure model uses DualBackbone
        if not isinstance(model.backbone, DualBackbone):
            raise ValueError("DeterministicTrainer now requires a model with DualBackbone")

    def compute_loss(self, emg_data, angle_data, y):
        feature_mode = self.model.backbone.feature_mode
        if feature_mode == 'both':
            pred_angle = self.model(emg_data=emg_data, angle_data=angle_data)
        elif feature_mode == 'emg_only':
            pred_angle = self.model(emg_data=emg_data)
        elif feature_mode == 'angle_only':
            pred_angle = self.model(angle_data=angle_data)
        
        if pred_angle.shape != y.shape:
            pred_angle = pred_angle.view_as(y)
        return self.criterion(pred_angle, y)

    def get_predictions(self, emg_data, angle_data, return_std=False):
        feature_mode = self.model.backbone.feature_mode
        if feature_mode == 'both':
            pred_angle = self.model(emg_data=emg_data, angle_data=angle_data)
        elif feature_mode == 'emg_only':
            pred_angle = self.model(emg_data=emg_data)
        elif feature_mode == 'angle_only':
            pred_angle = self.model(angle_data=angle_data)
        
        return pred_angle
    
class ProbabilisticTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, device, n_features=32):
        super().__init__(model, train_loader, val_loader, test_loader, optimizer, device, n_features)
        self.criterion = nn.GaussianNLLLoss(full=True, eps=1e-6)
        # Ensure model uses DualBackbone
        if not isinstance(model.backbone, DualBackbone):
            raise ValueError("ProbabilisticTrainer now requires a model with DualBackbone")

    def compute_loss(self, emg_data, angle_data, y):
        feature_mode = self.model.backbone.feature_mode
        if feature_mode == 'both':
            pred_mean, pred_logvar = self.model(emg_data=emg_data, angle_data=angle_data)
        elif feature_mode == 'emg_only':
            pred_mean, pred_logvar = self.model(emg_data=emg_data)
        elif feature_mode == 'angle_only':
            pred_mean, pred_logvar = self.model(angle_data=angle_data)
        
        if pred_mean.shape != y.shape:
            pred_mean = pred_mean.view_as(y)
            pred_logvar = pred_logvar.view_as(y)
            
        return self.criterion(pred_mean, y, pred_logvar)

    def get_predictions(self, emg_data, angle_data, return_std=False):
        feature_mode = self.model.backbone.feature_mode
        if feature_mode == 'both':
            pred_mean, pred_logvar = self.model(emg_data=emg_data, angle_data=angle_data)
        elif feature_mode == 'emg_only':
            pred_mean, pred_logvar = self.model(emg_data=emg_data)
        elif feature_mode == 'angle_only':
            pred_mean, pred_logvar = self.model(angle_data=angle_data)
        
        if return_std:
            # Convert log variance to standard deviation
            std = torch.exp(0.5 * pred_logvar)
            return pred_mean, std
            
        return pred_mean