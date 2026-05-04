import os
import kagglehub

def setup_environment():
    print("Starting Data Download & Environment Setup...")
    
   # Use RunPod's massive persistent storage volume
    runpod_workspace = '/workspace'
    
    # Create an explicit dataset directory there
    target_data_dir = os.path.join(runpod_workspace, 'gatech-emg-dataset')
    os.makedirs(target_data_dir, exist_ok=True)
    
    # 1. Download Dataset via kagglehub using output_dir
    print(f"Downloading dataset using kagglehub to {target_data_dir}...")
    
    # path will be exactly what you pass to output_dir
    dataset_path = kagglehub.dataset_download(
        'geeeeese/ga-tech-emg-dataset/versions/1', 
        output_dir=target_data_dir
    )
    print(f"\nDataset downloaded to: {dataset_path}")
    
    # 2. Generate the .env file automatically
    # Use absolute paths to ensure cache is found across different working directories
    cache_dir = os.path.join(runpod_workspace, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    env_content = f"""DATA_DIR={dataset_path}
RESULTS_DIR={os.path.join(runpod_workspace, 'results')}
LOG_DIR={os.path.join(runpod_workspace, 'logs')}
MODEL_DIR={os.path.join(runpod_workspace, 'models')}
CACHE_DIR={cache_dir}
EMG_FREQUENCY=2000
ANGLE_FREQUENCY=200
"""
    # Create the .env file in the root of the project
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    
    with open(env_path, "w") as f:
        f.write(env_content)
    print(f".env file successfully created at {env_path}.")

    print("\nSetup Complete! You can now run your experiments.")
    print("Run: uv run python src/scripts/run_experiments.py")

if __name__ == "__main__":
    setup_environment()