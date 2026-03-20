"""
Automated Experiment Runner for Multi-Configuration Testing

This script orchestrates multiple training experiments by systematically varying
configuration parameters to test feature importance and sample window length effects.

Usage:
    # Preview experiments without running
    python src/scripts/run_experiments.py --phase feature_importance --dry-run
    
    # Run feature importance experiments (18 runs)
    python src/scripts/run_experiments.py --phase feature_importance
    
    # Run sample length experiments (12 runs)
    python src/scripts/run_experiments.py --phase sample_length
    
    # Run all experiments (30 runs, ~7.5-15 hours estimated)
    python src/scripts/run_experiments.py --phase all
    
    # Resume from previous run (skip completed experiments)
    python src/scripts/run_experiments.py --phase all --resume
    
    # Use custom config file
    python src/scripts/run_experiments.py --phase all --config path/to/config.yaml

Experiment Phases:
    1. Feature Importance (18 runs):
       - Fixed: sample_window_length=1.0, default backbones
       - Vary: feature_mode ∈ {both, emg_only, angle_only}
               prediction_horizon ∈ {0.1, 0.2, 0.3, 0.5, 0.8, 1.0}
    
    2. Sample Length (12 runs):
       - Fixed: prediction_horizon=0.5, default backbones
       - Vary: feature_mode ∈ {both, emg_only, angle_only}
               sample_window_length ∈ {0.1, 0.2, 0.5, 1.0}

Naming Convention:
    Experiments are named: {feature_mode}_{sample_window_length}_{prediction_horizon}
    Example: "both_1.0_0.5", "emg_only_0.2_0.3"

Output:
    - Individual experiment results saved to: results/{experiment_name}_{timestamp}/
    - Experiment tracking CSV: experiment_tracker.csv
    - Each result directory contains: config.yaml, training_history.csv, test_metrics.yaml, etc.
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import yaml


def generate_feature_importance_experiments() -> List[Dict[str, Any]]:
    """
    Generate experiment configurations for testing feature importance.
    
    Fixed parameters:
        - sample_window_length: 1.0
        - emg_backbone_type: "conv2d_lstm"
        - angle_backbone_type: "tcn"
    
    Variable parameters:
        - feature_mode: {both, emg_only, angle_only}
        - prediction_horizon: {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
    
    Returns:
        List of 18 experiment configuration dictionaries
    """
    experiments = []
    feature_modes = ["both", "emg_only", "angle_only"]
    prediction_horizons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sample_window_length = 1.0
    
    for feature_mode in feature_modes:
        for prediction_horizon in prediction_horizons:
            exp_name = f"{feature_mode}_{sample_window_length}_{prediction_horizon}"
            experiments.append({
                "phase": "feature_importance",
                "experiment_name": exp_name,
                "feature_mode": feature_mode,
                "sample_window_length": sample_window_length,
                "prediction_horizon": prediction_horizon,
                "emg_backbone_type": "conv2d_lstm",
                "angle_backbone_type": "tcn",
            })
    
    return experiments


def generate_sample_length_experiments() -> List[Dict[str, Any]]:
    """
    Generate experiment configurations for testing sample window length importance.
    
    Fixed parameters:
        - prediction_horizon: 0.5
        - emg_backbone_type: "conv2d_lstm"
        - angle_backbone_type: "tcn"
    
    Variable parameters:
        - feature_mode: {both, emg_only, angle_only}
        - sample_window_length: {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
    
    Returns:
        List of 12 experiment configuration dictionaries
    """
    experiments = []
    feature_modes = ["both", "emg_only", "angle_only"]
    sample_window_lengths = [i*0.1 for i in range(1, 11)]  # 0.1 to 1.0 in steps of 0.1
    prediction_horizon = 0.5
    
    for feature_mode in feature_modes:
        for sample_window_length in sample_window_lengths:
            exp_name = f"{feature_mode}_{sample_window_length}_{prediction_horizon}"
            experiments.append({
                "phase": "sample_length",
                "experiment_name": exp_name,
                "feature_mode": feature_mode,
                "sample_window_length": sample_window_length,
                "prediction_horizon": prediction_horizon,
                "emg_backbone_type": "conv2d_lstm",
                "angle_backbone_type": "tcn",
            })
    
    return experiments


def get_all_experiments(phase: str) -> List[Dict[str, Any]]:
    """
    Get experiment configurations based on requested phase.
    
    Args:
        phase: One of "feature_importance", "sample_length", or "all"
    
    Returns:
        List of experiment configuration dictionaries
    """
    if phase == "feature_importance":
        return generate_feature_importance_experiments()
    elif phase == "sample_length":
        return generate_sample_length_experiments()
    elif phase == "all":
        experiments = generate_feature_importance_experiments()
        experiments.extend(generate_sample_length_experiments())
        return experiments
    else:
        raise ValueError(f"Unknown phase: {phase}. Must be one of: feature_importance, sample_length, all")


def check_experiment_completed(experiment_name: str, results_dir: Path) -> Optional[Path]:
    """
    Check if an experiment has already been completed.
    
    Args:
        experiment_name: Name of the experiment
        results_dir: Base results directory
    
    Returns:
        Path to completed experiment directory if exists and has test_metrics.yaml, else None
    """
    if not results_dir.exists():
        return None
    
    # Look for directories matching the experiment name pattern
    # Format: {experiment_name}_{timestamp}/
    for experiment_dir in results_dir.glob(f"{experiment_name}_*"):
        if experiment_dir.is_dir():
            metrics_file = experiment_dir / "test_metrics.yaml"
            if metrics_file.exists():
                return experiment_dir
    
    return None


def extract_metrics_from_result(result_dir: Path) -> Dict[str, Optional[float]]:
    """
    Extract key metrics from test_metrics.yaml file.
    
    Args:
        result_dir: Path to experiment result directory
    
    Returns:
        Dictionary with mae and rmse values (None if not found)
    """
    metrics_file = result_dir / "test_metrics.yaml"
    
    if not metrics_file.exists():
        return {"mae": None, "rmse": None}
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = yaml.safe_load(f)
        
        # Extract global metrics with backward compatibility.
        # New schema uses: {"global": {"mae": ..., "rmse": ...}}
        global_metrics = metrics.get("global")
        if not global_metrics:
            # Legacy schema fallback.
            global_metrics = metrics.get("global_metrics", {})

        mae = global_metrics.get("mae")
        rmse = global_metrics.get("rmse")

        if mae is None:
            mae = global_metrics.get("MAE")
        if rmse is None:
            rmse = global_metrics.get("RMSE")
        
        return {"mae": mae, "rmse": rmse}
    except Exception as e:
        print(f"Warning: Could not parse metrics from {metrics_file}: {e}")
        return {"mae": None, "rmse": None}


def build_train_command(experiment: Dict[str, Any], config_path: str, seed: int) -> List[str]:
    """
    Build the command to run train_and_test.py with appropriate arguments.
    
    Args:
        experiment: Experiment configuration dictionary
        config_path: Path to base config file
        seed: Random seed for reproducibility
    
    Returns:
        List of command arguments
    """
    cmd = [
        sys.executable,  # Use same Python interpreter
        "scripts/train_and_test.py",
        "--config", config_path,
        "--experiment_name", experiment["experiment_name"],
        "--feature_mode", experiment["feature_mode"],
        "--sample_window_length", str(experiment["sample_window_length"]),
        "--prediction_horizon", str(experiment["prediction_horizon"]),
        "--seed", str(seed),
    ]
    
    return cmd


def initialize_tracker_csv(tracker_path: Path):
    """
    Initialize the experiment tracking CSV file with headers.
    
    Args:
        tracker_path: Path to the tracking CSV file
    """
    headers = [
        "experiment_id",
        "phase",
        "feature_mode",
        "sample_window_length",
        "prediction_horizon",
        "emg_backbone_type",
        "angle_backbone_type",
        "status",
        "result_dir",
        "mae",
        "rmse",
        "start_time",
        "end_time",
        "duration_seconds",
    ]
    
    with open(tracker_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()


def append_to_tracker_csv(tracker_path: Path, experiment_id: int, experiment: Dict[str, Any], 
                          status: str, result_dir: Optional[Path], mae: Optional[float], 
                          rmse: Optional[float], start_time: str, end_time: str, 
                          duration_seconds: float):
    """
    Append experiment results to tracking CSV.
    
    Args:
        tracker_path: Path to the tracking CSV file
        experiment_id: Sequential experiment ID
        experiment: Experiment configuration dictionary
        status: "completed", "failed", or "skipped"
        result_dir: Path to result directory (None if failed/skipped)
        mae: Mean Absolute Error (None if not available)
        rmse: Root Mean Squared Error (None if not available)
        start_time: Timestamp when experiment started
        end_time: Timestamp when experiment ended
        duration_seconds: Duration in seconds
    """
    row = {
        "experiment_id": experiment_id,
        "phase": experiment["phase"],
        "feature_mode": experiment["feature_mode"],
        "sample_window_length": experiment["sample_window_length"],
        "prediction_horizon": experiment["prediction_horizon"],
        "emg_backbone_type": experiment["emg_backbone_type"],
        "angle_backbone_type": experiment["angle_backbone_type"],
        "status": status,
        "result_dir": str(result_dir) if result_dir else "",
        "mae": f"{mae:.6f}" if mae is not None else "",
        "rmse": f"{rmse:.6f}" if rmse is not None else "",
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": f"{duration_seconds:.2f}",
    }
    
    with open(tracker_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)


def run_experiments(phase: str, config_path: str, seed: int, results_dir: Path, 
                   tracker_path: Path, resume: bool, dry_run: bool):
    """
    Main experiment runner that orchestrates all experiments.
    
    Args:
        phase: Experiment phase to run ("feature_importance", "sample_length", or "all")
        config_path: Path to base configuration file
        seed: Random seed for reproducibility
        results_dir: Directory where results will be saved
        tracker_path: Path to experiment tracking CSV file
        resume: If True, skip already completed experiments
        dry_run: If True, only print commands without executing
    """
    # Get experiments for requested phase
    experiments = get_all_experiments(phase)
    total_experiments = len(experiments)
    
    print("=" * 80)
    print("AUTOMATED EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Phase: {phase}")
    print(f"Total experiments: {total_experiments}")
    print(f"Config file: {config_path}")
    print(f"Random seed: {seed}")
    print(f"Results directory: {results_dir}")
    print(f"Tracker CSV: {tracker_path}")
    print(f"Resume mode: {resume}")
    print(f"Dry run: {dry_run}")
    print("=" * 80)
    print()
    
    if dry_run:
        print("DRY RUN MODE - Commands will be printed but not executed")
        print()
    
    # Initialize tracker CSV if it doesn't exist
    if not tracker_path.exists() and not dry_run:
        initialize_tracker_csv(tracker_path)
        print(f"Initialized tracker CSV: {tracker_path}")
        print()
    
    completed_count = 0
    skipped_count = 0
    failed_count = 0
    total_duration = 0.0
    
    for idx, experiment in enumerate(experiments, start=1):
        exp_name = experiment["experiment_name"]
        print(f"\n[{idx}/{total_experiments}] Experiment: {exp_name}")
        print("-" * 80)
        
        # Check if experiment already completed
        if resume:
            existing_result = check_experiment_completed(exp_name, results_dir)
            if existing_result:
                print(f"✓ Already completed (found at: {existing_result})")
                metrics = extract_metrics_from_result(existing_result)
                
                if not dry_run:
                    append_to_tracker_csv(
                        tracker_path, idx, experiment, "skipped", 
                        existing_result, metrics["mae"], metrics["rmse"],
                        "", "", 0.0
                    )
                
                skipped_count += 1
                continue
        
        # Build command
        cmd = build_train_command(experiment, config_path, seed)
        
        print(f"Command: {' '.join(cmd)}")
        
        if dry_run:
            print("(Dry run - not executing)")
            continue
        
        # Run experiment
        start_time = datetime.now()
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Started at: {start_time_str}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            end_time = datetime.now()
            end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            duration = (end_time - start_time).total_seconds()
            total_duration += duration
            
            print(f"\n✓ Completed successfully!")
            print(f"Ended at: {end_time_str}")
            print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            
            # Find result directory and extract metrics
            result_dir = check_experiment_completed(exp_name, results_dir)
            metrics = {"mae": None, "rmse": None}
            if result_dir:
                metrics = extract_metrics_from_result(result_dir)
                if metrics["mae"] is not None:
                    print(f"MAE: {metrics['mae']:.6f}, RMSE: {metrics['rmse']:.6f}")
            
            # Log to tracker
            append_to_tracker_csv(
                tracker_path, idx, experiment, "completed",
                result_dir, metrics["mae"], metrics["rmse"],
                start_time_str, end_time_str, duration
            )
            
            completed_count += 1
            
            # Estimate remaining time
            if completed_count > 0:
                avg_duration = total_duration / completed_count
                remaining_experiments = total_experiments - idx
                estimated_remaining = avg_duration * remaining_experiments
                print(f"Estimated time remaining: {estimated_remaining/3600:.2f} hours")
            
        except subprocess.CalledProcessError as e:
            end_time = datetime.now()
            end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            duration = (end_time - start_time).total_seconds()
            
            print(f"\n✗ FAILED!")
            print(f"Error: {e}")
            print(f"Ended at: {end_time_str}")
            print(f"Duration: {duration:.2f} seconds")
            
            # Log failure to tracker
            append_to_tracker_csv(
                tracker_path, idx, experiment, "failed",
                None, None, None,
                start_time_str, end_time_str, duration
            )
            
            failed_count += 1
            print("Continuing to next experiment...")
        
        except KeyboardInterrupt:
            print("\n\nExperiment interrupted by user!")
            print(f"Progress: {completed_count} completed, {skipped_count} skipped, {failed_count} failed")
            sys.exit(1)
    
    # Final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Total experiments: {total_experiments}")
    print(f"Completed: {completed_count}")
    print(f"Skipped (already done): {skipped_count}")
    print(f"Failed: {failed_count}")
    
    if not dry_run and completed_count > 0:
        print(f"Total duration: {total_duration/3600:.2f} hours")
        print(f"Average duration per experiment: {total_duration/completed_count/60:.2f} minutes")
    
    if not dry_run:
        print(f"\nResults saved to: {results_dir}")
        print(f"Tracker CSV: {tracker_path}")
    
    print("=" * 80)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated experiment runner for multi-configuration testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview feature importance experiments
  python src/scripts/run_experiments.py --phase feature_importance --dry-run
  
  # Run feature importance experiments
  python src/scripts/run_experiments.py --phase feature_importance
  
  # Run all experiments with resume capability
  python src/scripts/run_experiments.py --phase all --resume
  
  # Use custom config file
  python src/scripts/run_experiments.py --phase all --config my_config.yaml
        """
    )
    
    parser.add_argument(
        '--phase',
        type=str,
        required=True,
        choices=['feature_importance', 'sample_length', 'all'],
        help='Which experiment phase to run'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config_dual_backbone.yaml',
        help='Path to base configuration file (default: src/config/config_dual_backbone.yaml)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory where results will be saved (default: results)'
    )
    
    parser.add_argument(
        '--tracker',
        type=str,
        default='experiment_tracker.csv',
        help='Path to experiment tracking CSV file (default: experiment_tracker.csv)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip experiments that have already been completed'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing them'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Convert paths to Path objects
    config_path = Path(args.config)
    results_dir = Path(args.results_dir)
    tracker_path = Path(args.tracker)
    
    # Validate config file exists
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Create results directory if it doesn't exist
    if not args.dry_run:
        results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    run_experiments(
        phase=args.phase,
        config_path=str(config_path),
        seed=args.seed,
        results_dir=results_dir,
        tracker_path=tracker_path,
        resume=args.resume,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
