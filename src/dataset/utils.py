import os

from dotenv import load_dotenv
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.logger import logger

load_dotenv()

DATA_DIR = os.getenv('DATA_DIR')
TARGET_ANGLE_NAME = [name.strip() for name in os.getenv('TARGET_ANGLE_NAME', 'knee_angle_r, knee_angle_l').split(',')]
EMG_FREQUENCY = int(os.getenv('EMG_FREQUENCY', 1000))
ANGLE_FREQUENCY = int(os.getenv('ANGLE_FREQUENCY', 100))

ACTIVITIES = [
    'normal_walk_1_0-6',
    'normal_walk_1_1-2',
    'normal_walk_1_1-8',
    'normal_walk_1_2-0',
    'normal_walk_1_2-5',
]
SUBJECTS = [f'AB{i:02}' for i in range(1, 14)]

def _get_data_files() -> dict:
    """
    Helper function to get the list of emg and angle files.
    Returns:
        dict: A dictionary with keys 'emg_files' and 'angle_files' containing lists of file paths.
    """
    files = []

    for subject in SUBJECTS:
        for activity in ACTIVITIES:
            emg_file = os.path.join(DATA_DIR, subject, activity, f'{subject}_{activity}_emg.csv')
            angle_file = os.path.join(DATA_DIR, subject, activity, f'{subject}_{activity}_angle.csv')

            if os.path.exists(emg_file) and os.path.exists(angle_file):
                files.append({
                    'subject': subject,
                    'activity': activity,
                    'emg_file': emg_file,
                    'angle_file': angle_file
                })
            else:
                logger.warning(f"Warning: Missing files for {subject} - {activity}")
    return files

DATA_FILES = _get_data_files()

TRAIN_FILES, TEST_FILES = train_test_split(DATA_FILES, test_size=0.2, random_state=42)

def _process_emg_file(emg_file, window_length: float = 0.1):
    """
    Process the EMG file to extract features using a sliding window approach.

    Args:
        emg_file (str): Path to the EMG file.
        window_length (float): Length of the window in seconds.
    Returns:
        pd.DataFrame: A dataframe containing the EMG data.
    """
    # Load the EMG data
    df = pd.read_csv(emg_file)
    time_stamps = df['time'].values

    muscle_columns = [col for col in df.columns if col.startswith('muscle')]
    emg_data = df[muscle_columns].values.T

    n_channels, n_samples = emg_data.shape
    window_size = int(window_length * EMG_FREQUENCY)
    n_windows = n_samples - window_size + 1

    # Design a bandpass Butterworth filter
    fs_emg = EMG_FREQUENCY
    nyquist_freq = fs_emg / 2
    low_cutoff = 20 / nyquist_freq
    high_cutoff = 450 / nyquist_freq
    b, a = butter(4, [low_cutoff, high_cutoff], btype='band')

    # Apply the filter to each channel
    filtered_emg = np.zeros_like(emg_data)
    for i in range(n_channels):
        filtered_emg[i] = filtfilt(b, a, emg_data[i])
        # Normalize the filtered signal
        filtered_emg[i] = (filtered_emg[i] - np.mean(filtered_emg[i])) / np.std(filtered_emg[i])
        # Remove ouliers (95th percentile)
        filtered_emg[i] = np.clip(filtered_emg[i], -np.percentile(np.abs(filtered_emg[i]), 95), np.percentile(np.abs(filtered_emg[i]), 95))

    # Sliding window feature extraction
    features = []
    for i in range(n_windows):
        window_data = filtered_emg[:, i:i + window_size]
        start_time = time_stamps[i]
        feature_vector = []
        for channel in window_data:
            feature_vector.extend(_calculate_features(channel))

        features.append([start_time] + feature_vector)

    columns = []
    for i in muscle_columns:
        columns.extend([f'{i}_{feature}' for feature in ['rms', 'mav', 'wl', 'zc', 'ssc']])

    return pd.DataFrame(features, columns=['time'] + columns)


def _calculate_features(channel_data):
    """
    Calculate features for a given window of data for a single EMG channel.
    The features calculated are:
    - Root Mean Square (RMS)
    - Mean Absolute Value (MAV)
    - Waveform Length (WL)
    - Variance (VAR)
    - Zero Crossing (ZC)
    - Slope Sign Changes (SSC)

    Args:
        channel_data (np.ndarray): The EMG data for a single channel within a window.
    Returns:
        list: A list of calculated features for the channel: rms, mav, wl, var, zc, ssc.
    """
    # Root Mean Square (RMS)
    rms = np.sqrt(np.mean(channel_data ** 2))

    # Mean Absolute Value (MAV)
    mav = np.mean(np.abs(channel_data))

    # Waveform Length (WL)
    wl = np.sum(np.abs(np.diff(channel_data)))

    # Zero Crossing (ZC)
    zc = np.sum((channel_data[:-1] * channel_data[1:] < 0) & (np.abs(channel_data[:-1] - channel_data[1:]) > 0.01))

    # Slope Sign Changes (SSC)
    ssc = np.sum(((channel_data[1:-1] - channel_data[:-2]) * (channel_data[2:] - channel_data[1:-1]) < 0) & (np.abs(channel_data[1:-1] - channel_data[:-2]) > 0.01) & (np.abs(channel_data[2:] - channel_data[1:-1]) > 0.01))

    return [rms, mav, wl, zc, ssc]

def _combine_emg_angle_data(emg_df, angle_df):
    """
    Interpolate the angle data to match the time stamps of the EMG features and combine them into a single dataframe.
    Args:
            emg_df (pd.DataFrame): Dataframe containing the EMG features with a 'time' column.
            angle_df (pd.DataFrame): Dataframe containing the angle data with a 'time' column.
    Returns:
        pd.DataFrame: A combined dataframe with EMG features and corresponding angle data.
    """
    # Interpolate angle data to match EMG time stamps
    angle_columns = [col for col in angle_df.columns if not col.startswith('time')]
    interpolated_angles = {}
    for col in angle_columns:
        interpolated_angles[col] = np.interp(emg_df['time'], angle_df['time'], angle_df[col])

    # Combine EMG features and interpolated angles
    combined_df = emg_df.copy()
    for col in angle_columns:
        combined_df[col] = interpolated_angles[col]

    return combined_df

def load_and_process_data(mode='train'):
    """
    Load and process the EMG and angle data for all subjects and activities, return a list of dataframes containing the combined data for each subject and activity.
    Returns:
        list: A list of dataframes containing EMG features and corresponding angle data for each subject and activity.
    """
    combined_data = []
    data_files = TRAIN_FILES if mode == 'train' else TEST_FILES
    for file_info in (loop := tqdm(data_files)):
        loop.set_description(f"Processing {file_info['subject']} - {file_info['activity']}")
        emg_df = _process_emg_file(file_info['emg_file'])
        angle_df = pd.read_csv(file_info['angle_file'])
        combined_df = _combine_emg_angle_data(emg_df, angle_df)
        combined_data.append(combined_df)

    return combined_data