import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from dotenv import load_dotenv

from dataset.utils import load_and_process_data, EMG_FREQUENCY

load_dotenv()

class PredictionDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and processing EMG and angle data for prediction tasks.
    Each sample in the dataset consists of a feature vector derived from EMG data additionally the corresponding angle data.
    The label is the angle in a future time step.
    """
    def __init__(self, mode='train', window_length: float=1, stride: float=0.1, 
                 prediction_horizon: float=0.2, target_angle_name: list=['knee_angle_r', 'knee_angle_l'],
                 use_cache: bool=True):
        """
        Initialize the PredictionDataset.
        Args:
            mode (str): 'train' or 'test' to specify which data split to use.
            window_length (float): The length of the window (in seconds) of EMG data to use as features.
            stride (float): The stride (in seconds) between consecutive windows.
            prediction_horizon (float): The time horizon (in seconds) into the future for which to predict the angle.
            target_angle_name (list): List of angle names to predict (e.g., ['knee_angle_r', 'knee_angle_l']).
            use_cache (bool): Whether to use cached processed data. Default True.
        """
        self.window_length = window_length
        self.stride = stride
        self.prediction_horizon = prediction_horizon
        self.target_angle_name = target_angle_name
        self.dataframes, self.emg_columns, self.angle_columns = load_and_process_data(mode=mode, use_cache=use_cache)
        self.emg_samples, self.angle_samples, self.labels = self._generate_samples()

    def _generate_samples(self):
        """
        Generate samples and labels from the combined dataframes.
        Each sample consists of EMG features at time t, and the label is the angle at time t + prediction_horizon.
        """
        angle_samples = []
        emg_samples = []
        labels = []
        for df in self.dataframes:
            for i in range(0, len(df) - int((self.prediction_horizon + self.window_length) * EMG_FREQUENCY), int(self.stride * EMG_FREQUENCY)):
                emg_features = df[self.emg_columns].iloc[i:i + int(self.window_length * EMG_FREQUENCY)].values.astype(np.float32)
                angle_features = df[self.angle_columns].iloc[i:i + int(self.window_length * EMG_FREQUENCY)].values.astype(np.float32)
                label = df.iloc[i + int((self.prediction_horizon + self.window_length) * EMG_FREQUENCY)][self.target_angle_name].values.astype(np.float32)
                emg_samples.append(emg_features.T)
                angle_samples.append(angle_features.T)
                labels.append(label)
        return emg_samples, angle_samples, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.emg_samples[idx], self.angle_samples[idx], self.labels[idx]
    
# Test the dataset and plot a sample
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = PredictionDataset(mode='train', prediction_horizon=1.0)
    sample_emg, sample_angle, sample_label = dataset[0]
    print("Sample emg shape:", sample_emg.shape)
    print("Sample angle shape:", sample_angle.shape)
    print("Sample label shape:", sample_label.shape)
    # Plot the first 10 features and the corresponding label
    plt.figure(figsize=(12, 6))
    plt.plot(sample_emg[0, :], label='EMG Channel 0')
    plt.plot(sample_emg[1, :], label='EMG Channel 1')

    plt.legend()
    plt.title('Sample EMG Features and Target Angle')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    save_path = os.path.join(os.getenv('RESULTS_DIR'), 'sample_plot.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)