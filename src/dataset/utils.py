import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, decimate
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.logger import logger

load_dotenv()

DATA_DIR = os.getenv('DATA_DIR')
CACHE_DIR = os.getenv('CACHE_DIR', './cache')
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

def _compute_wavelet_scalogram(channel_data, fs=1000, output_fs=100, freq_min=5, freq_max=450, n_scales=40):
    """
    Compute wavelet scalogram for a single EMG channel using Continuous Wavelet Transform.
    
    Args:
        channel_data (np.ndarray): 1D array of filtered EMG data.
        fs (int): Sampling frequency of input data (Hz). Default 1000.
        output_fs (int): Target output sampling frequency (Hz). Default 100.
        freq_min (float): Minimum frequency of interest (Hz). Default 5.
        freq_max (float): Maximum frequency of interest (Hz). Default 450.
        n_scales (int): Number of frequency scales. Default 40.
    
    Returns:
        np.ndarray: 2D scalogram array (downsampled_time, n_scales).
        np.ndarray: Frequency values corresponding to scales (Hz).
    """
    # Generate logarithmically-spaced frequencies
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), num=n_scales)
    
    # Convert frequencies to scales for Morlet wavelet
    # For pywt's 'morl' wavelet, center frequency fc ≈ 1.0
    fc = pywt.central_frequency('morl')
    scales = fc / (freqs * (1/fs))
    
    # Compute Continuous Wavelet Transform
    coefficients, _ = pywt.cwt(channel_data, scales, 'morl', sampling_period=1/fs)
    
    # Get magnitude (power) of complex coefficients
    scalogram = np.abs(coefficients).T  # Transpose to (time, freq_scales)
    
    # Downsample time axis if needed
    decimation_factor = int(fs / output_fs)
    if decimation_factor > 1:
        # Decimate along time axis (axis=0)
        scalogram = decimate(scalogram, decimation_factor, axis=0, zero_phase=True)
    
    return scalogram, freqs

def _process_emg_file(emg_file, output_fs=100, freq_min=5, freq_max=450, n_scales=40):
    """
    Process the EMG file using Continuous Wavelet Transform to generate scalograms.

    Args:
        emg_file (str): Path to the EMG file.
        output_fs (int): Target output sampling frequency (Hz). Default 100.
        freq_min (float): Minimum frequency of interest (Hz). Default 20.
        freq_max (float): Maximum frequency of interest (Hz). Default 450.
        n_scales (int): Number of frequency scales. Default 40.
    
    Returns:
        dict: Dictionary containing:
            - 'emg_scalogram': 3D array (n_channels, time, n_scales)
            - 'time': 1D array of time stamps at output_fs
            - 'channel_names': List of EMG channel names
            - 'frequencies': 1D array of frequency values
    """
    # Load the EMG data
    df = pd.read_csv(emg_file)
    time_stamps = df['time'].values

    muscle_columns = [col for col in df.columns if not col.startswith('time')]
    emg_data = df[muscle_columns].values.T

    n_channels, n_samples = emg_data.shape

    # Design a bandpass Butterworth filter
    fs_emg = EMG_FREQUENCY
    nyquist_freq = fs_emg / 2
    low_cutoff = freq_min / nyquist_freq
    high_cutoff = min(freq_max, nyquist_freq * 0.99) / nyquist_freq  # Ensure below Nyquist
    b, a = butter(4, [low_cutoff, high_cutoff], btype='band')

    # Apply the filter and compute scalograms for each channel
    scalograms = []
    freqs = None
    
    for i in range(n_channels):
        # Filter the signal
        filtered_signal = filtfilt(b, a, emg_data[i])
        
        # Normalize the filtered signal
        filtered_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
        
        # Remove outliers (95th percentile)
        filtered_signal = np.clip(filtered_signal, 
                                  -np.percentile(np.abs(filtered_signal), 95), 
                                  np.percentile(np.abs(filtered_signal), 95))
        
        # Compute wavelet scalogram
        scalogram, freqs = _compute_wavelet_scalogram(
            filtered_signal, 
            fs=fs_emg, 
            output_fs=output_fs,
            freq_min=freq_min,
            freq_max=freq_max,
            n_scales=n_scales
        )
        
        scalograms.append(scalogram)
    
    # Stack into 3D array (n_channels, time, n_scales)
    emg_scalogram = np.stack(scalograms, axis=0)
    
    # Generate downsampled time stamps
    decimation_factor = int(fs_emg / output_fs)
    downsampled_time = time_stamps[::decimation_factor][:emg_scalogram.shape[1]]
    
    return {
        'emg_scalogram': emg_scalogram,
        'time': downsampled_time,
        'channel_names': muscle_columns,
        'frequencies': freqs
    }


def _combine_emg_angle_data(emg_data_dict, angle_df, output_fs=100):
    """
    Resample angle data to match the EMG scalogram sampling rate and combine them.
    
    Args:
        emg_data_dict (dict): Dictionary from _process_emg_file containing scalogram data.
        angle_df (pd.DataFrame): Dataframe containing angle data with 'time' column.
        output_fs (int): Output sampling frequency (Hz). Default 100.
    
    Returns:
        dict: Combined data dictionary with:
            - 'emg_scalogram': 3D array (n_channels, time, n_scales)
            - 'angle_data': 2D array (n_angles, time)
            - 'time': 1D array of time stamps
            - 'channel_names': List of EMG channel names
            - 'angle_names': List of angle names
            - 'frequencies': 1D array of frequency values
    """
    # Get angle columns and data
    angle_columns = [col for col in angle_df.columns if not col.startswith('time')]
    
    # Resample angle data to match EMG time stamps
    emg_time = emg_data_dict['time']
    angle_time = angle_df['time'].values
    
    resampled_angles = []
    for col in angle_columns:
        # Interpolate and normalize to [-1, 1] range
        resampled_angle = np.interp(emg_time, angle_time, angle_df[col].values) / 90.0
        resampled_angles.append(resampled_angle)
    
    # Stack into 2D array (n_angles, time)
    angle_data = np.stack(resampled_angles, axis=0)
    
    return {
        'emg_scalogram': emg_data_dict['emg_scalogram'],
        'angle_data': angle_data,
        'time': emg_time,
        'channel_names': emg_data_dict['channel_names'],
        'angle_names': angle_columns,
        'frequencies': emg_data_dict['frequencies']
    }


def _get_cache_path(mode: str, cache_dir: str = None) -> Path:
    """
    Get the cache file path for the given mode.
    
    Args:
        mode (str): 'train' or 'test' mode.
        cache_dir (str, optional): Custom cache directory path. If None, uses CACHE_DIR.
    
    Returns:
        Path: Path to the cache file.
    """
    cache_path = Path(cache_dir if cache_dir is not None else CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path / f"processed_data_{mode}.pkl"


def _save_to_cache(cache_path: Path, combined_data: list, channel_names: list, angle_names: list, frequencies: np.ndarray, output_fs: int):
    """
    Save processed wavelet scalogram data to cache file.
    
    Args:
        cache_path (Path): Path to cache file.
        combined_data (list): List of processed data dictionaries.
        channel_names (list): List of EMG channel names.
        angle_names (list): List of angle names.
        frequencies (np.ndarray): Array of frequency values.
        output_fs (int): Output sampling frequency.
    """
    cache_data = {
        'combined_data': combined_data,
        'channel_names': channel_names,
        'angle_names': angle_names,
        'frequencies': frequencies,
        'output_fs': output_fs,
        'version': '2.0'  # Updated version for wavelet data
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"✓ Cached processed data to: {cache_path}")


def _load_from_cache(cache_path: Path):
    """
    Load processed wavelet scalogram data from cache file.
    
    Args:
        cache_path (Path): Path to cache file.
    
    Returns:
        tuple: (combined_data, channel_names, angle_names, frequencies, output_fs) or None if cache invalid.
    """
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Validate cache structure
        if not isinstance(cache_data, dict):
            logger.warning("Invalid cache format (not a dict)")
            return None
        
        # Check version - only accept version 2.0+ (wavelet data)
        version = cache_data.get('version', '1.0')
        if version.startswith('1.'):
            logger.warning(f"Incompatible cache version {version}. Expected 2.0+ for wavelet data. Please clear cache.")
            return None
        
        required_keys = ['combined_data', 'channel_names', 'angle_names', 'frequencies', 'output_fs']
        if not all(key in cache_data for key in required_keys):
            logger.warning("Invalid cache format (missing keys)")
            return None
        
        return (cache_data['combined_data'], cache_data['channel_names'], 
                cache_data['angle_names'], cache_data['frequencies'], 
                cache_data['output_fs'])
    
    except (pickle.UnpicklingError, EOFError, ImportError) as e:
        logger.warning(f"Failed to load cache: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error loading cache: {e}")
        return None


def load_and_process_data(mode='train', use_cache=True, cache_dir=None, output_fs=100, 
                          freq_min=20, freq_max=450, n_scales=40):
    """
    Load and process the EMG and angle data using wavelet transforms.
    Uses caching to avoid reprocessing on subsequent runs.
    
    Args:
        mode (str): 'train' or 'test' to specify which data split to use.
        use_cache (bool): Whether to use cached data if available. Default True.
        cache_dir (str, optional): Custom cache directory path. If None, uses CACHE_DIR.
        output_fs (int): Target output sampling frequency (Hz). Default 100.
        freq_min (float): Minimum frequency of interest (Hz). Default 20.
        freq_max (float): Maximum frequency of interest (Hz). Default 450.
        n_scales (int): Number of frequency scales for wavelet transform. Default 40.
    
    Returns:
        tuple: (combined_data, channel_names, angle_names, frequencies, output_fs)
            - combined_data (list): List of dicts with 'emg_scalogram', 'angle_data', 'time'
            - channel_names (list): List of EMG channel names
            - angle_names (list): List of angle names
            - frequencies (np.ndarray): Array of frequency values
            - output_fs (int): Output sampling frequency used
    """
    data_files = TRAIN_FILES if mode == 'train' else TEST_FILES
    
    # Try to load from cache
    if use_cache:
        cache_path = _get_cache_path(mode, cache_dir)
        
        if cache_path.exists():
            logger.info(f"Loading processed data from cache: {cache_path.name}")
            cached_result = _load_from_cache(cache_path)
            
            if cached_result is not None:
                logger.info(f"✓ Successfully loaded {len(cached_result[0])} files from cache")
                return cached_result
            else:
                logger.info("Cache invalid or incompatible version, reprocessing data...")
        else:
            logger.info(f"No cache found, processing data from scratch...")
    
    # Process data from scratch
    combined_data = []
    channel_names = None
    angle_names = None
    frequencies = None

    for file_info in (loop := tqdm(data_files, desc="Processing data")):
        loop.set_description(f"Processing {file_info['subject']} - {file_info['activity']}")
        
        # Process EMG with wavelet transform
        emg_data_dict = _process_emg_file(
            file_info['emg_file'],
            output_fs=output_fs,
            freq_min=freq_min,
            freq_max=freq_max,
            n_scales=n_scales
        )
        
        # Load and resample angle data
        angle_df = pd.read_csv(file_info['angle_file'])
        combined_dict = _combine_emg_angle_data(emg_data_dict, angle_df, output_fs=output_fs)
        
        combined_data.append(combined_dict)
        
        # Store metadata from first file
        if channel_names is None:
            channel_names = combined_dict['channel_names']
            angle_names = combined_dict['angle_names']
            frequencies = combined_dict['frequencies']
    
    # Save to cache
    if use_cache:
        cache_path = _get_cache_path(mode, cache_dir)
        _save_to_cache(cache_path, combined_data, channel_names, angle_names, frequencies, output_fs)

    return combined_data, channel_names, angle_names, frequencies, output_fs


def clear_cache(mode: str = None, cache_dir: str = None):
    """
    Clear cached processed data files.
    
    Args:
        mode (str, optional): 'train' or 'test' to clear specific mode cache.
                             If None, clears all cache files.
        cache_dir (str, optional): Custom cache directory path. If None, uses CACHE_DIR.
    """
    cache_path = Path(cache_dir if cache_dir is not None else CACHE_DIR)
    
    if not cache_path.exists():
        logger.info("Cache directory does not exist. Nothing to clear.")
        return
    
    if mode is not None:
        # Clear specific mode cache
        cache_file = cache_path / f"processed_data_{mode}.pkl"
        cache_files = [cache_file] if cache_file.exists() else []
    else:
        # Clear all cache files
        cache_files = list(cache_path.glob("processed_data_*.pkl"))
    
    if not cache_files:
        logger.info(f"No cache files found to clear{f' for mode: {mode}' if mode else ''}.")
        return
    
    for cache_file in cache_files:
        try:
            cache_file.unlink()
            logger.info(f"✓ Deleted cache file: {cache_file.name}")
        except Exception as e:
            logger.warning(f"Failed to delete {cache_file.name}: {e}")
    
    logger.info(f"✓ Cleared {len(cache_files)} cache file(s).")


if __name__ == "__main__":
    # Example: Clear all cache
    # clear_cache()
    
    # Example: Clear only train cache
    # clear_cache(mode='train')
    
    pass