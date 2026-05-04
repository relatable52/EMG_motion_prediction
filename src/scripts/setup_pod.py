import os
import kagglehub

def setup_environment():
    print("Starting Data Download & Environment Setup...")
    
    # 1. Download Dataset via kagglehub
    print("Downloading dataset using kagglehub...")
    dataset_path = kagglehub.dataset_download('geeeeese/ga-tech-emg-dataset/versions/1')
    print(f"\nDataset downloaded to: {dataset_path}")
    
    # 2. Generate the .env file automatically
    env_content = f"""DATA_DIR={dataset_path}
RESULTS_DIR=results
LOG_DIR=logs
MODEL_DIR=models
CACHE_DIR=cache
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