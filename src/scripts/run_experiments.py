"""
Automated experiment orchestrator to answer Core RQs and run ablations.
Runs experiments sequentially to avoid memory overload.
"""
import os
import subprocess

# Path to the training script relative to the workspace root
TRAIN_SCRIPT = "src/scripts/train_single.py"

# Directory to save logs
LOG_DIR = "experiment_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def run_experiment(exp_name, kwargs):
    """Utility to run train_single.py with given kwargs sequentially."""
    print(f"\n{'='*10}\nStarting Experiment: {exp_name}\n{'='*10}")
    
    cmd = ["python", TRAIN_SCRIPT, "--exp-name", exp_name]
    
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])
        
    log_path = os.path.join(LOG_DIR, f"{exp_name}_log.txt")
    print(f"Logging terminal output to: {log_path}")
    
    # Run the subprocess synchronously (one at a time)
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        
    if result.returncode != 0:
        print(f"EXPERIMENT FAILED: {exp_name} - Check {log_path}")
    else:
        print(f"Finished {exp_name} successfully.")
        
    return result.returncode

def run_core_rq_suite():
    """RQ1, RQ2, RQ4: Compare core paradigms on identical data configs."""
    print("\n--- Running Core Comparison Suite ---")
    paradigms = ['deterministic', 'probabilistic', 'mc_dropout', 'ensemble', 'gp']
    
    for paradigm in paradigms:
        run_experiment(
            exp_name=f"core_comparison_{paradigm}",
            kwargs={
                "model-paradigm": paradigm,
                # Standardized settings for fair comparison
                "data-window-length": 1.0,
                "data-prediction-horizon": 0.05,
                "train-epochs": 30, 
            }
        )

def run_model_ablations():
    """Model Ablations: Test impact of UQ scaling"""
    print("\n--- Running Model Ablation Suite ---")
    
    # 1. Ensemble Size Sweep
    for size in [5, 10, 15, 20, 25, 30]:
        run_experiment(f"ablation_ensemble_n{size}", 
                      {"model-paradigm": "ensemble", "model-ensemble-size": size})
        
    # 2. MC Dropout Passes Sweep
    for passes in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
        run_experiment(f"ablation_mcd_p{passes}", 
                      {"model-paradigm": "mc_dropout", "model-mc-dropout-passes": passes})
                      
    # 3. GP Hidden Dim Sweep
    for dim in [8, 16, 24, 32, 40, 48, 56, 64]:
        run_experiment(f"ablation_gp_dim{dim}", 
                      {"model-paradigm": "gp", "model-gp-latent-dim": dim})
        
    # 4. Backbone Parameter Sweep 
    for hidden in [64, 128, 192, 256]:
        run_experiment(f"ablation_backbone_h{hidden}", 
                      {"model-paradigm": "deterministic", "model-hidden-dim": hidden})

def run_data_ablations():
    """Data ablations: Time windows, predictive horizons, and frequency resolution."""
    print("\n--- Running Data Ablation Suite ---")
    # Fix the model to a fast one like deterministic
    fixed_kwargs = {"model-paradigm": "probabilistic", "train-epochs": 20}  # Shorter epochs for ablations
    
    # Window Length Sweep
    for w_len in [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5]:
        kwargs = fixed_kwargs.copy()
        kwargs["data-window-length"] = w_len
        run_experiment(f"ablation_data_win_{w_len}s", kwargs)
        
    # Prediction Horizon
    for horizon in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        kwargs = fixed_kwargs.copy()
        kwargs["data-prediction-horizon"] = horizon
        run_experiment(f"ablation_data_horizon_{horizon}s", kwargs)

    # Frequency Scales
    for scales in [20, 30, 40, 50, 60]:
        kwargs = fixed_kwargs.copy()
        kwargs["data-n-scales"] = scales 
        run_experiment(f"ablation_data_scales_{scales}", kwargs)


if __name__ == "__main__":
    print("Starting Automated Experiment Pipeline...")
    
    # By default, runs the core subset. Uncomment others when ready.
    run_core_rq_suite()
    
    # Uncomment these when you're ready to run ablations!
    # run_model_ablations()
    # run_data_ablations()
    
    print("\nAll scheduled experiments finished!")
