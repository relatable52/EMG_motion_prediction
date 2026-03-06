import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from dotenv import load_dotenv

from dataset.utils import load_and_process_data

load_dotenv()

class PredictionDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and processing EMG wavelet scalogram and angle data for prediction tasks.
    Each sample consists of wavelet scalogram windows (time-frequency representations) and corresponding angle data.
    The label is the angle in a future time step.
    """
    def __init__(self, mode='train', window_length: float=1.0, stride: float=0.1, 
                 prediction_horizon: float=0.2, target_angle_name: list=['knee_angle_r', 'knee_angle_l'],
                 use_cache: bool=True, cache_dir: str=None, 
                 output_fs: int=100, freq_min: float=5, freq_max: float=450, n_scales: int=40):
        """
        Initialize the PredictionDataset.
        Args:
            mode (str): 'train' or 'test' to specify which data split to use.
            window_length (float): The length of the window (in seconds) of EMG data to use. Default 1.0.
            stride (float): The stride (in seconds) between consecutive windows. Default 0.1.
            prediction_horizon (float): The time horizon (in seconds) into the future for prediction. Default 0.2.
            target_angle_name (list): List of angle names to predict. Default ['knee_angle_r', 'knee_angle_l'].
            use_cache (bool): Whether to use cached processed data. Default True.
            cache_dir (str, optional): Custom cache directory path. If None, uses CACHE_DIR.
            output_fs (int): Output sampling frequency (Hz). Default 100.
            freq_min (float): Minimum frequency for wavelet transform (Hz). Default 5.
            freq_max (float): Maximum frequency for wavelet transform (Hz). Default 450.
            n_scales (int): Number of frequency scales for wavelet transform. Default 40.
        """
        self.window_length = window_length
        self.stride = stride
        self.prediction_horizon = prediction_horizon
        self.target_angle_name = target_angle_name
        self.output_fs = output_fs
        
        # Load wavelet-transformed data
        (self.combined_data, self.channel_names, self.angle_names, 
         self.frequencies, self.output_fs) = load_and_process_data(
            mode=mode, use_cache=use_cache, cache_dir=cache_dir,
            output_fs=output_fs, freq_min=freq_min, freq_max=freq_max, n_scales=n_scales
        )
        
        self.emg_samples, self.angle_samples, self.labels = self._generate_samples()

    def _generate_samples(self):
        """
        Generate samples and labels from the wavelet scalogram data.
        Each sample consists of EMG scalogram windows and angle history, 
        with the label being the angle at time t + prediction_horizon.
        """
        emg_samples = []
        angle_samples = []
        labels = []
        
        # Calculate window and stride sizes in samples (at output_fs rate)
        window_samples = int(self.window_length * self.output_fs)
        stride_samples = int(self.stride * self.output_fs)
        prediction_samples = int(self.prediction_horizon * self.output_fs)
        
        for data_dict in self.combined_data:
            emg_scalogram = data_dict['emg_scalogram']  # Shape: (n_channels, time, n_scales)
            angle_data = data_dict['angle_data']  # Shape: (n_angles, time)
            
            n_channels, n_time, n_scales = emg_scalogram.shape
            n_angles = angle_data.shape[0]
            
            # Find indices for target angles
            target_indices = [self.angle_names.index(name) for name in self.target_angle_name 
                            if name in self.angle_names]
            
            # Generate windowed samples
            for i in range(0, n_time - window_samples - prediction_samples + 1, stride_samples):
                # Extract EMG scalogram window: (n_channels, window_samples, n_scales)
                emg_window = emg_scalogram[:, i:i + window_samples, :].astype(np.float32)
                
                # Extract angle history window: (n_angles, window_samples)
                angle_window = angle_data[:, i:i + window_samples].astype(np.float32)
                
                # Extract target angle at future time point
                future_idx = i + window_samples + prediction_samples - 1
                if future_idx < n_time:
                    label = angle_data[target_indices, future_idx].astype(np.float32)
                    
                    emg_samples.append(emg_window)
                    angle_samples.append(angle_window)
                    labels.append(label)
        
        return emg_samples, angle_samples, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns:
            emg_sample: (n_channels, window_samples, n_scales) - Wavelet scalogram
            angle_sample: (n_angles, window_samples) - Angle history
            label: (n_target_angles,) - Future angle values
        """
        return self.emg_samples[idx], self.angle_samples[idx], self.labels[idx]
    
# Test the dataset and visualize wavelet scalogram
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Creating dataset...")
    dataset = PredictionDataset(mode='train', prediction_horizon=0.5, output_fs=100)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Channel names: {dataset.channel_names}")
    print(f"Angle names: {dataset.angle_names}")
    print(f"Frequencies: {dataset.frequencies.shape} (min: {dataset.frequencies.min():.1f} Hz, max: {dataset.frequencies.max():.1f} Hz)")
    print(f"Output sampling rate: {dataset.output_fs} Hz")
    
    sample_emg, sample_angle, sample_label = dataset[0]
    print(f"\nSample EMG scalogram shape: {sample_emg.shape}")  # (n_channels, time, freq_scales)
    print(f"Sample angle shape: {sample_angle.shape}")  # (n_angles, time)
    print(f"Sample label shape: {sample_label.shape}")  # (n_target_angles,)
    
    # Visualize first channel's scalogram
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot scalogram (time-frequency representation)
    ax1 = axes[0]
    scalogram_2d = sample_emg[0, :, :]  # First channel: (time, freq_scales)
    im = ax1.imshow(scalogram_2d.T, aspect='auto', origin='lower', cmap='viridis',
                    extent=[0, sample_emg.shape[1] / dataset.output_fs, 
                           dataset.frequencies.min(), dataset.frequencies.max()])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title(f'Wavelet Scalogram - {dataset.channel_names[0]}')
    plt.colorbar(im, ax=ax1, label='Magnitude')
    
    # Plot angle history and target
    ax2 = axes[1]
    time_axis = np.arange(sample_angle.shape[1]) / dataset.output_fs
    for i, angle_name in enumerate(dataset.angle_names):
        ax2.plot(time_axis, sample_angle[i, :], label=angle_name, alpha=0.7)
    ax2.axhline(y=sample_label[0], color='r', linestyle='--', label=f'Target: {sample_label[0]:.3f}')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Normalized Angle')
    ax2.set_title('Angle History and Prediction Target')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(os.getenv('RESULTS_DIR', './results'), 'wavelet_sample_plot.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")