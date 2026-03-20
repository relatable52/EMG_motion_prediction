import os

import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

from utils.logger import logger
from utils.experiment import save_test_metrics

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

    def compute_loss(self, emg_data, angle_data, y):
        raise NotImplementedError("Subclasses must implement compute_loss")

    def get_predictions(self, emg_data, angle_data):
        raise NotImplementedError("Subclasses must implement get_predictions")

    def train(self, epochs, save_dir=".", log_interval=1, save_best_model=True, save_last_model=True):
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(epochs):
            # --- Training Phase ---
            self.model.train()
            train_loss = 0.0
            
            for emg_data, angle_data, y in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
                emg_data = emg_data.to(self.device)
                angle_data = angle_data.to(self.device)
                y = y.to(self.device)
                
                self.optimizer.zero_grad()
                loss = self.compute_loss(emg_data, angle_data, y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
            avg_train_loss = train_loss / len(self.train_loader)
            
            # --- Validation Phase ---
            self.model.eval()
            val_loss = 0.0
            val_mae = 0.0
            val_total_elements = 0
            
            with torch.no_grad():
                for emg_data, angle_data, y in tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                    emg_data = emg_data.to(self.device)
                    angle_data = angle_data.to(self.device)
                    y = y.to(self.device)
                    
                    loss = self.compute_loss(emg_data, angle_data, y)
                    val_loss += loss.item()
                    
                    # Get deterministic prediction for MAE calculation
                    preds = self.get_predictions(emg_data, angle_data)
                    if preds.shape != y.shape:
                        preds = preds.view_as(y)
                        
                    val_mae += torch.abs(preds - y).sum().item()
                    val_total_elements += y.numel()
                    
            avg_val_loss = val_loss / len(self.val_loader)
            avg_val_mae = val_mae / val_total_elements if val_total_elements > 0 else 0.0
            
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

    def test(self, save_dir="results", prefix="model", angle_names=None):
        self.model.eval()
        test_loss = 0.0
        
        all_preds = []
        all_labels = []
        all_stds = [] # Only populated if using Uncertainty model
        
        os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            for emg_data, angle_data, y in tqdm(self.test_loader, desc="Testing", leave=False):
                emg_data = emg_data.to(self.device)
                angle_data = angle_data.to(self.device)
                y = y.to(self.device)
                
                loss = self.compute_loss(emg_data, angle_data, y)
                test_loss += loss.item()
                
                # Retrieve predictions (and uncertainty if applicable)
                preds = self.get_predictions(emg_data, angle_data, return_std=True)
                
                if isinstance(preds, tuple):
                    out, std = preds
                    all_stds.append(std.cpu().numpy())
                else:
                    out = preds
                
                if out.shape != y.shape:
                    out = out.view_as(y)
                
                all_preds.append(out.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        # Keep predictions as 2D arrays: (n_samples, n_angles)
        all_preds = np.concatenate(all_preds)  # Shape: (n_samples, n_angles)
        all_labels = np.concatenate(all_labels)  # Shape: (n_samples, n_angles)
        
        n_samples, n_angles = all_preds.shape
        
        # Generate angle names if not provided
        if angle_names is None:
            angle_names = [f"angle_{i}" for i in range(n_angles)]

        # ============================================================
        # GLOBAL METRICS (across all angles, flattened)
        # ============================================================
        all_preds_flat = all_preds.flatten()
        all_labels_flat = all_labels.flatten()
        all_errors_flat = all_labels_flat - all_preds_flat
        
        # 1. Objective loss and prediction error metrics
        avg_test_objective_loss = test_loss / len(self.test_loader)
        global_mse = np.mean(all_errors_flat ** 2)
        global_rmse = np.sqrt(global_mse)
        # Average MAE over both samples and angles.
        global_mae = np.mean(np.abs(all_errors_flat))

        objective_name = "loss"
        if hasattr(self, 'criterion') and self.criterion is not None:
            objective_name = self.criterion.__class__.__name__

        # 2. nRMSE (%)
        label_range = np.max(all_labels_flat) - np.min(all_labels_flat)
        nrmse_percent = (global_rmse / label_range) * 100 if label_range > 0 else 0

        # 3. R^2 and Adjusted R^2
        ss_res = np.sum((all_labels_flat - all_preds_flat) ** 2)
        ss_tot = np.sum((all_labels_flat - np.mean(all_labels_flat)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        n = n_samples
        p = self.n_features
        adj_r2 = 1 - ((1 - r2_score) * (n - 1) / (n - p - 1)) if n > p + 1 else r2_score

        # 4. Pearson Correlation Coefficient (r)
        corr_coeff, _ = pearsonr(all_labels_flat, all_preds_flat)

        logger.info(f"\nTest Results (Global):")
        logger.info(f" {objective_name}: {avg_test_objective_loss:.4f}")
        logger.info(f" MSE:       {global_mse:.4f}")
        logger.info(f" RMSE:      {global_rmse:.4f}°")
        logger.info(f" nRMSE:     {nrmse_percent:.2f}%")
        logger.info(f" MAE:       {global_mae:.4f}°")
        logger.info(f" Pearson r: {corr_coeff:.4f}")
        logger.info(f" R² Score:  {r2_score:.4f}")
        logger.info(f" Adj. R²:   {adj_r2:.4f}")
        
        # ============================================================
        # PER-ANGLE METRICS
        # ============================================================
        per_angle_metrics = {}
        
        logger.info(f"\nPer-Angle Metrics:")
        for i, angle_name in enumerate(angle_names):
            angle_preds = all_preds[:, i]
            angle_labels = all_labels[:, i]
            
            # Compute metrics for this angle
            angle_mse = np.mean((angle_labels - angle_preds) ** 2)
            angle_rmse = np.sqrt(angle_mse)
            angle_mae = np.mean(np.abs(angle_labels - angle_preds))
            
            angle_range = np.max(angle_labels) - np.min(angle_labels)
            angle_nrmse = (angle_rmse / angle_range) * 100 if angle_range > 0 else 0
            
            angle_ss_res = np.sum((angle_labels - angle_preds) ** 2)
            angle_ss_tot = np.sum((angle_labels - np.mean(angle_labels)) ** 2)
            angle_r2 = 1 - (angle_ss_res / angle_ss_tot) if angle_ss_tot > 0 else 0
            
            angle_corr, _ = pearsonr(angle_labels, angle_preds)
            
            per_angle_metrics[angle_name] = {
                "mse": float(angle_mse),
                "rmse": float(angle_rmse),
                "nrmse": float(angle_nrmse),
                "mae": float(angle_mae),
                "pearson_r": float(angle_corr),
                "r2": float(angle_r2)
            }
            
            logger.info(f" {angle_name}: MAE={angle_mae:.4f}°, RMSE={angle_rmse:.4f}°, R²={angle_r2:.4f}")
        
        # ============================================================
        # BUILD DATAFRAME WITH TUPLE COLUMNS
        # ============================================================
        save_df = pd.DataFrame({
            'True_Angles': [tuple(row) for row in all_labels],
            'Pred_Angles': [tuple(row) for row in all_preds]
        })
        
        if all_stds:
            all_stds = np.concatenate(all_stds)  # Shape: (n_samples, n_angles)
            save_df['Uncertainty_Std'] = [tuple(row) for row in all_stds]
            mean_std_per_angle = np.mean(all_stds, axis=0)
            logger.info(f"\nMean Uncertainty Std per angle: {[f'{s:.4f}' for s in mean_std_per_angle]}")
            
        csv_path = os.path.join(save_dir, f"{prefix}_predictions.csv")
        save_df.to_csv(csv_path, index=False)
        logger.info(f"\nPredictions saved to: {csv_path}")

        # ============================================================
        # BUILD METRICS DICTIONARY AND SAVE TO YAML
        # ============================================================
        metrics = {
            "angle_names": angle_names,
            "n_samples": int(n_samples),
            "n_angles": int(n_angles),
            "global": {
                "objective_name": objective_name,
                "objective_loss": float(avg_test_objective_loss),
                "mse": float(global_mse),
                "rmse": float(global_rmse),
                "nrmse": float(nrmse_percent),
                "mae": float(global_mae),
                "pearson_r": float(corr_coeff),
                "r2": float(r2_score),
                "adj_r2": float(adj_r2)
            },
            "per_angle": per_angle_metrics
        }
        
        # Save metrics to YAML
        metrics_path = os.path.join(save_dir, f"{prefix}_metrics.yaml")
        save_test_metrics(metrics, metrics_path)
        logger.info(f"Metrics saved to: {metrics_path}")
        
        return metrics, save_df