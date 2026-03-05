import torch
import torch.nn as nn

from trainer.base import BaseTrainer

class DeterministicTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, device, n_features=32):
        super().__init__(model, train_loader, val_loader, test_loader, optimizer, device, n_features)
        self.criterion = nn.MSELoss()

    def compute_loss(self, x, y):
        pred_angle = self.model(x)
        if pred_angle.shape != y.shape:
            pred_angle = pred_angle.view_as(y)
        return self.criterion(pred_angle, y)

    def get_predictions(self, x, return_std=False):
        pred_angle = self.model(x)
        return pred_angle
    
class ProbabilisticTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, device, n_features=32):
        super().__init__(model, train_loader, val_loader, test_loader, optimizer, device, n_features)
        self.criterion = nn.GaussianNLLLoss(full=True, eps=1e-6)

    def compute_loss(self, x, y):
        pred_mean, pred_logvar = self.model(x)
        if pred_mean.shape != y.shape:
            pred_mean = pred_mean.view_as(y)
            pred_logvar = pred_logvar.view_as(y)
            
        return self.criterion(pred_mean, y, pred_logvar)

    def get_predictions(self, x, return_std=False):
        pred_mean, pred_logvar = self.model(x)
        
        if return_std:
            # Convert log variance to standard deviation
            std = torch.exp(0.5 * pred_logvar)
            return pred_mean, std
            
        return pred_mean