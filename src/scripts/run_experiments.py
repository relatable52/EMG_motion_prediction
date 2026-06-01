"""
Automated experiment orchestrator to answer Core RQs and run ablations.
Runs experiments sequentially to avoid memory overload.
"""
import os
import subprocess
from utils.logger import logger

# 1. This is: /workspace/your-repo/src/scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Step up one level to get: /workspace/your-repo/src
SRC_DIR = os.path.dirname(SCRIPT_DIR)

# 3. Step up one more level to get the root: /workspace/your-repo
WORKSPACE_DIR = os.path.dirname(SRC_DIR)

def run_experiment(exp_name, kwargs):
    """Utility to run scripts.train_single with given kwargs sequentially."""
    print(f"\n{'='*10}\nStarting Experiment: {exp_name}\n{'='*10}")
    
    # 1. Changed to use module execution (-m scripts.train_single)
    cmd = ["uv", "run", "python", "-u", "-m", "scripts.train_single", "--exp-name", exp_name]
    
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])
        
    # 2. Run the subprocess synchronously from the SRC_DIR, directly to terminal
    result = subprocess.run(
        cmd, 
        cwd=SRC_DIR  # This forces the command to run from the 'src' folder
    )
        
    if result.returncode != 0:
        logger.error(f"EXPERIMENT FAILED: {exp_name}")
    else:
        logger.info(f"Finished {exp_name} successfully.")
        
    return result.returncode

def run_core_rq_suite():
    """RQ1, RQ2, RQ4: Compare core paradigms on identical data configs."""
    logger.info("\n--- Running Core Comparison Suite ---")
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
    logger.info("\n--- Running Model Ablation Suite ---")
    
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
    logger.info("\n--- Running Data Ablation Suite ---")
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


def run_subject_loso_suite():
    """Run leave-one-subject-out experiments by iterating fold_index across subjects."""
    logger.info("\n--- Running Subject LOSO Suite ---")
    # We don't import subjects here to avoid circular imports; iterate over plausible subject indices
    # The dataset contains AB01..AB13, so iterate 0..12
    for fold in range(13):
        run_experiment(
            exp_name=f"subject_loso_fold{fold}",
            kwargs={
                "data-split-strategy": "subject_loso",
                "data-fold-index": fold,
                "train-epochs": 30,
            }
        )


def run_activity_kfold_suite(n_folds=5):
    """Run activity k-fold experiments by iterating over fold indices."""
    logger.info("\n--- Running Activity KFold Suite ---")
    for fold in range(n_folds):
        run_experiment(
            exp_name=f"activity_kfold_f{fold}",
            kwargs={
                "data-split-strategy": "activity_kfold",
                "data-n-folds": n_folds,
                "data-fold-index": fold,
                "train-epochs": 30,
            }
        )


def run_subject_kfold_suite(n_folds=5):
    """Run subject k-fold experiments by partitioning subjects into k folds and running each fold."""
    logger.info("\n--- Running Subject KFold Suite ---")
    for fold in range(n_folds):
        run_experiment(
            exp_name=f"subject_kfold_f{fold}",
            kwargs={
                "data-split-strategy": "subject_kfold",
                "data-n-folds": n_folds,
                "data-fold-index": fold,
                "train-epochs": 30,
            }
        )


if __name__ == "__main__":
    logger.info("Starting Automated Experiment Pipeline...")
    
    # By default, runs the core subset. Uncomment others when ready.
    run_core_rq_suite()
    
    # Uncomment these when you're ready to run ablations!
    # run_model_ablations()
    # run_data_ablations()
    
    logger.info("\nAll scheduled experiments finished!")
