import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, Optional, Tuple
import numpy as np
import gpytorch
from utils.logger import logger


class Trainer:
    """
    Trainer class to handle training and inference for all model paradigms (Deterministic, Probabilistic, Ensemble, MC-Dropout, Gaussian Process).
    """
    def __init__(self, model: nn.Module, config, likelihood=None):
        self.model = model
        self.config = config
        self.likelihood = likelihood
        self.device = torch.device(config.train.device)
        self.model = self.model.to(self.device)
        
        if self.likelihood is not None:
            self.likelihood = self.likelihood.to(self.device)
            
        self.model_type = self.config.model.model_type
        
        # 1. Optimizers
        if self.model_type == 'gaussian_process' and self.likelihood is not None:
            # GP needs gradients flowing through both the model and the likelihood
            self.optimizer = optim.Adam(
                list(self.model.parameters()) + list(self.likelihood.parameters()),
                lr=self.config.train.learning_rate
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.train.learning_rate,
                weight_decay=self.config.train.weight_decay
            )
            
        # 2. Built-in Loss Functions
        self.mse_loss = nn.MSELoss()
        self.gaussian_nll_loss = nn.GaussianNLLLoss(eps=1e-6, reduction='mean') # For Probabilistic target
        
        if self.model_type == 'gaussian_process':
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

    def train(self, train_loader: Optional[DataLoader] = None, 
              gp_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
              save_dir: Optional[str] = None) -> Dict[str, list]:
        """
        Master training loop. Routes to mini-batch or full-batch based on paradigm.
        """
        # Use results_dir from env config with exp_name
        if save_dir is None:
            save_dir = self.config.env.results_dir
        
        exp_save_dir = os.path.join(save_dir, self.config.exp_name)
        os.makedirs(exp_save_dir, exist_ok=True)
            
        logger.info(f"--- Starting Training: {self.model_type.upper()} ---")
        history = {'train_loss': []}
        
        for epoch in range(1, self.config.train.epochs + 1):
            # Route to correct epoch logic based on model type
            if self.model_type == 'gaussian_process':
                assert gp_data is not None, "GP requires full-batch gp_data tuple (X, Y)!"
                avg_loss = self._train_epoch_gp(gp_data, epoch)
            else:
                assert train_loader is not None, f"{self.model_type} requires testing data DataLoader!"
                avg_loss = self._train_epoch_nn(train_loader, epoch)
                
            history['train_loss'].append(avg_loss)
            
            # Logging
            if epoch % self.config.train.log_interval == 0 or epoch == 1:
                logger.info(f"Epoch [{epoch}/{self.config.train.epochs}] | Avg Loss: {avg_loss:.4f}")
                
        # Save final checkpoints
        if exp_save_dir:
            self._save_checkpoint(exp_save_dir)
            
        return history

    def _train_epoch_nn(self, train_loader: DataLoader, epoch: int) -> float:
        """Mini-batch training for standard NNs (Deterministic, Probabilistic, Ensemble, MC-Dropout)"""
        self.model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.train.epochs}", leave=False)
        for emg_sample, _, label in pbar:
            emg_sample, label = emg_sample.to(self.device), label.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Paradigm specific loss computations
            if self.model_type in ['deterministic', 'mc_dropout']:
                pred = self.model(emg_sample)
                loss = self.mse_loss(pred, label)
                
            elif self.model_type == 'probabilistic':
                mu, logvar = self.model(emg_sample)
                var = torch.exp(logvar) # Convert unconstrained log-variance to strict variance
                loss = self.gaussian_nll_loss(mu, label, var)
                
            elif self.model_type == 'ensemble':
                preds = self.model(emg_sample) # Shape: (M, batch_size, output_dim)
                target_expanded = label.unsqueeze(0).expand_as(preds)
                loss = self.mse_loss(preds, target_expanded)
                
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item() * emg_sample.size(0)
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        return train_loss / len(train_loader.dataset)

    def _train_epoch_gp(self, gp_data: Tuple[torch.Tensor, torch.Tensor], epoch: int) -> float:
        """Full-batch training specifically targeted for Exact Gaussian Processes"""
        self.model.train()
        self.likelihood.train() # Crucial: Gpytorch needs both in train mode
        
        x_train, y_train = gp_data
        x_train, y_train = x_train.to(self.device), y_train.to(self.device)
        
        # Flatten input to bypass shape checkers for GP input as in notebook
        x_train_flat = x_train.view(x_train.size(0), -1)
        
        self.optimizer.zero_grad()
        output = self.model(x_train_flat)
        loss = -self.mll(output, y_train) # Minimize negative marginal log likelihood
        
        loss.backward()
        self.optimizer.step()
        
        # In full batch, the loss is automatically averaged
        return loss.item()

    def predict(self, test_loader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Runs inference, standardizes outputs into (Mean, Variance),
        and safely moves data to CPU immediately to prevent GPU Memory Leaks.
        """
        self.model.eval()
        if self.model_type == 'gaussian_process':
            self.likelihood.eval()
        elif self.model_type == 'mc_dropout':
            self.model.train() # Must keep dropout layers active to introduce stochasticity!
            
        all_preds, all_vars, all_targets = [], [], []
        
        # Latency setup
        warmup_batches = 10
        batch_times = []
        
        logger.info(f"--- Running Inference: {self.model_type.upper()} ---")
        
        with torch.no_grad():
            for batch_idx, (emg_sample, _, label) in enumerate(tqdm(test_loader, desc="Inference")):
                emg_sample = emg_sample.to(self.device)
                
                # Start Latency Measurement
                start_time = time.time()
                
                # Paradigm routing logic for Inference standardizing into mean and epistemic/aleatoric variance
                if self.model_type == 'deterministic':
                    pred = self.model(emg_sample)
                    var = torch.zeros_like(pred)
                    
                elif self.model_type == 'probabilistic':
                    mu, logvar = self.model(emg_sample)
                    pred = mu
                    var = torch.exp(logvar) 
                    
                elif self.model_type == 'ensemble':
                    ensemble_preds = self.model(emg_sample)
                    pred = torch.mean(ensemble_preds, dim=0)
                    var = torch.var(ensemble_preds, dim=0) 
                    
                elif self.model_type == 'mc_dropout':
                    mc_samples = 30 
                    batch_passes = torch.stack([self.model(emg_sample) for _ in range(mc_samples)])
                    pred = torch.mean(batch_passes, dim=0)
                    var = torch.var(batch_passes, dim=0)
                    
                elif self.model_type == 'gaussian_process':
                    emg_sample_flat = emg_sample.view(emg_sample.size(0), -1)
                    predictive_dist = self.likelihood(self.model(emg_sample_flat))
                    pred = predictive_dist.mean
                    var = predictive_dist.variance
                    
                # End Latency Measurement
                if batch_idx >= warmup_batches:
                    batch_times.append((time.time() - start_time) / emg_sample.size(0))
                    
                # Prevent Memory Leaks by sending to CPU immediately
                all_preds.append(pred.cpu())
                all_vars.append(var.cpu())
                all_targets.append(label.cpu())
                
        # Record Inference Times
        avg_infer_ms = (np.mean(batch_times) * 1000) if batch_times else 0.0
        
        results = {
            'predictions': torch.cat(all_preds, dim=0),
            'variances': torch.cat(all_vars, dim=0),
            'targets': torch.cat(all_targets, dim=0),
            'inference_time_ms': avg_infer_ms,
            'model_type': self.model_type
        }
        
        logger.info(f"Inference complete! Average latency per sample: {avg_infer_ms:.3f} ms")
        return results

    def _save_checkpoint(self, save_dir: str):
        save_path = os.path.join(save_dir, f"{self.model_type}_final.pth")
        
        # GP likelihood covariance limits require it to be saved alongside CNN parameters 
        if self.model_type == 'gaussian_process':
            torch.save({
                'model_state': self.model.state_dict(),
                'likelihood_state': self.likelihood.state_dict()
            }, save_path)
        else:
            torch.save(self.model.state_dict(), save_path)
        
        logger.info(f"Checkpoint saved to: {save_path}")