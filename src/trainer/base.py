import os

import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

from utils.logger import logger

class BaseTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, device, n_features=32):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.n_features = n_features
        
        # Initialize history dictionary
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": []
        }

    def compute_loss(self, x, y):
        raise NotImplementedError("Subclasses must implement compute_loss")

    def get_predictions(self, x):
        raise NotImplementedError("Subclasses must implement get_predictions")

    def train(self, epochs, save_dir=".", log_interval=1, save_best_model=True, save_last_model=True):
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(epochs):
            # --- Training Phase ---
            self.model.train()
            train_loss = 0.0
            
            for x, y in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                loss = self.compute_loss(x, y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
            avg_train_loss = train_loss / len(self.train_loader)
            
            # --- Validation Phase ---
            self.model.eval()
            val_loss = 0.0
            val_mae = 0.0
            val_total = 0
            
            with torch.no_grad():
                for x, y in tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                    x, y = x.to(self.device), y.to(self.device)
                    
                    loss = self.compute_loss(x, y)
                    val_loss += loss.item()
                    
                    # Get deterministic prediction for MAE calculation
                    preds = self.get_predictions(x)
                    if preds.shape != y.shape:
                        preds = preds.view_as(y)
                        
                    val_mae += torch.abs(preds - y).sum().item()
                    val_total += y.size(0)
                    
            avg_val_loss = val_loss / len(self.val_loader)
            avg_val_mae = val_mae / val_total
            
            # Save history (always record, even if not logging)
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["val_mae"].append(avg_val_mae)
            
            # Log metrics according to log_interval
            if (epoch + 1) % log_interval == 0 or epoch == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.3f}°")
            
            # Save best model
            if save_best_model and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                best_model_path = os.path.join(save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), best_model_path)
        
        # Save last model
        if save_last_model:
            last_model_path = os.path.join(save_dir, "last_model.pth")
            torch.save(self.model.state_dict(), last_model_path)
            logger.info(f"\nLast model saved at epoch {epochs}")
                
        logger.info(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}")
        return self.history

    def test(self, save_dir="results", prefix="model"):
        self.model.eval()
        test_loss = 0.0
        test_mae = 0.0
        test_total = 0
        
        all_preds = []
        all_labels = []
        all_stds = [] # Only populated if using Uncertainty model
        
        os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            for x, y in tqdm(self.test_loader, desc="Testing", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                
                loss = self.compute_loss(x, y)
                test_loss += loss.item()
                
                # Retrieve predictions (and uncertainty if applicable)
                preds = self.get_predictions(x, return_std=True)
                
                if isinstance(preds, tuple):
                    out, std = preds
                    all_stds.append(std.cpu().numpy())
                else:
                    out = preds
                
                if out.shape != y.shape:
                    out = out.view_as(y)
                    
                test_mae += torch.abs(out - y).sum().item()
                
                all_preds.append(out.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                test_total += y.size(0)

        # Flatten arrays for global metrics
        all_preds = np.concatenate(all_preds).flatten()
        all_labels = np.concatenate(all_labels).flatten()

        # 1. Basic Error Metrics
        avg_test_loss = test_loss / len(self.test_loader)
        avg_test_rmse = np.sqrt(avg_test_loss)
        avg_test_mae = test_mae / test_total

        # 2. nRMSE (%)
        label_range = np.max(all_labels) - np.min(all_labels)
        nrmse_percent = (avg_test_rmse / label_range) * 100 if label_range > 0 else 0

        # 3. R^2 and Adjusted R^2
        ss_res = np.sum((all_labels - all_preds) ** 2)
        ss_tot = np.sum((all_labels - np.mean(all_labels)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        n = test_total
        p = self.n_features
        adj_r2 = 1 - ((1 - r2_score) * (n - 1) / (n - p - 1)) if n > p + 1 else r2_score

        # 4. Pearson Correlation Coefficient (r)
        corr_coeff, _ = pearsonr(all_labels, all_preds)

        logger.info(f"\nTest Results:")
        logger.info(f" MSE Loss:  {avg_test_loss:.4f}")
        logger.info(f" RMSE:      {avg_test_rmse:.4f}°")
        logger.info(f" nRMSE:     {nrmse_percent:.2f}%")
        logger.info(f" MAE:       {avg_test_mae:.4f}°")
        logger.info(f" Pearson r: {corr_coeff:.4f}")
        logger.info(f" R² Score:  {r2_score:.4f}")
        logger.info(f" Adj. R²:   {adj_r2:.4f}")
        
        # Save predictions to CSV
        save_df = pd.DataFrame({
            'True_Angle': all_labels,
            'Predicted_Angle': all_preds
        })
        
        if all_stds:
            all_stds = np.concatenate(all_stds).flatten()
            save_df['Uncertainty_Std'] = all_stds
            logger.info(f" Mean Std:  {np.mean(all_stds):.4f}°")
            
        csv_path = os.path.join(save_dir, f"{prefix}_predictions.csv")
        save_df.to_csv(csv_path, index=False)
        logger.info(f"Predictions saved to: {csv_path}")

        metrics = {
            "mse": avg_test_loss,
            "rmse": avg_test_rmse,
            "nrmse": nrmse_percent,
            "mae": avg_test_mae,
            "pearson_r": corr_coeff,
            "r2": r2_score,
            "adj_r2": adj_r2
        }
        
        return metrics, save_df