import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from dataset.utils import load_and_process_data, EMG_FREQUENCY
from config import CONFIG

TARGET_ANGLE_NAME = CONFIG['TARGET_ANGLE_NAME']

class PredictionDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and processing EMG and angle data for prediction tasks.
    Each sample in the dataset consists of a feature vector derived from EMG data additionally the corresponding angle data.
    The label is the angle in a future time step.
    """
    def __init__(self, mode='train', window_length: float=1, stride: float=0.1, prediction_horizon: float=0.2):
        """
        Initialize the PredictionDataset.
        Args:
            mode (str): 'train' or 'test' to specify which data split to use.
            window_length (float): The length of the window (in seconds) of EMG data to use as features.
            stride (float): The stride (in seconds) between consecutive windows.
            prediction_horizon (float): The time horizon (in seconds) into the future for which to predict the angle.
        """
        self.window_length = window_length
        self.stride = stride
        self.prediction_horizon = prediction_horizon
        self.dataframes = load_and_process_data(mode=mode)
        self.data, self.labels = self._generate_samples()

    def _generate_samples(self):
        """
        Generate samples and labels from the combined dataframes.
        Each sample consists of EMG features at time t, and the label is the angle at time t + prediction_horizon.
        """
        samples = []
        labels = []
        for df in self.dataframes:
            for i in range(0, len(df) - int((self.prediction_horizon + self.window_length) * EMG_FREQUENCY), int(self.stride * EMG_FREQUENCY)):
                features = df.iloc[i:i + int(self.window_length * EMG_FREQUENCY)].drop('time', axis=1).values.astype(np.float32)
                label = df.iloc[i + int((self.prediction_horizon + self.window_length) * EMG_FREQUENCY)][TARGET_ANGLE_NAME].values.astype(np.float32)
                samples.append(features.flatten())
                labels.append(label)
        return samples, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
# Test the dataset and plot a sample
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = PredictionDataset(mode='train', prediction_horizon=1.0)
    sample_data, sample_label = dataset[0]
    print("Sample data shape:", sample_data.shape)
    print("Sample label shape:", sample_label.shape)
    # Plot the first 10 features and the corresponding label
    plt.figure(figsize=(12, 6))
    plt.plot(sample_data[:10], label='EMG Features')
    plt.plot(sample_label, label='Target Angle', marker='o')
    plt.legend()
    plt.title('Sample EMG Features and Target Angle')
    plt.xlabel('Feature Index')
    plt.ylabel('Value')
    plt.show()