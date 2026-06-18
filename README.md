# EMG Data Analysis

Uncertainty quantification benchmarks for EMG-based joint angle prediction. Compares deterministic, probabilistic, MC Dropout, ensemble, and Gaussian Process paradigms on the GA Tech EMG dataset.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [RunPod Setup](#2-runpod-setup)
3. [Data Download](#3-data-download)
4. [Running Experiments](#4-running-experiments)
5. [Project Structure](#5-project-structure)
6. [Configuration Reference](#6-configuration-reference)

---

## 1. Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management. Python ≥ 3.13 is required.

**Install `uv` (if not already installed):**

```bash
# Linux / macOS
curl -Lsf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

**Install dependencies:**

```bash
uv sync
```

This installs all packages listed in `pyproject.toml` (PyTorch, GPyTorch, scikit-learn, etc.) into a managed virtual environment.

---

## 2. RunPod Setup

The project is designed to run on [RunPod](https://runpod.io) with a persistent `/workspace` volume. The setup script automates data download and `.env` generation.

**Steps:**

1. Provision a RunPod pod with a GPU (recommended: RTX 3090 or better) and attach a persistent volume mounted at `/workspace`.
2. In the pod terminal, clone the repo into `/workspace`:

   ```bash
   cd /workspace
   git clone <your-repo-url> gatech_data_analysis
   cd gatech_data_analysis
   ```

3. Install `uv` and sync dependencies (see [Environment Setup](#1-environment-setup)).

4. Configure your Kaggle API credentials so `kagglehub` can authenticate. Either:
   - Place `~/.kaggle/kaggle.json` on the pod, or
   - Set environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.

---

## 3. Data Download

Run the setup script to download the dataset and auto-generate the `.env` file:

```bash
uv run python src/scripts/setup_pod.py
```

This will:
- Download the **GA Tech EMG Dataset** (Kaggle: `geeeeese/ga-tech-emg-dataset`) to `/workspace/gatech-emg-dataset/`
- Create a `.env` file in the project root with paths for `DATA_DIR`, `RESULTS_DIR`, `LOG_DIR`, `MODEL_DIR`, and `CACHE_DIR`

**Manual `.env` (local / non-RunPod):**

If running locally, create a `.env` file in the project root manually:

```dotenv
DATA_DIR=/path/to/ga-tech-emg-dataset
RESULTS_DIR=/path/to/results
LOG_DIR=/path/to/logs
MODEL_DIR=/path/to/models
CACHE_DIR=/path/to/cache
EMG_FREQUENCY=2000
ANGLE_FREQUENCY=200
```

---

## 4. Running Experiments

### Train a single model

Run `train_single.py` directly from the `src/` directory using module execution:

```bash
uv run python -u -m scripts.train_single \
    --exp-name my_experiment \
    --model-paradigm deterministic \
    --data-window-length 1.0 \
    --data-prediction-horizon 0.05 \
    --train-epochs 30
```

Available `--model-paradigm` values: `deterministic`, `probabilistic`, `mc_dropout`, `ensemble`, `gp`

### Run the full experiment pipeline

The orchestrator runs all experiments sequentially:

```bash
uv run python src/scripts/run_experiments.py
```

By default this runs the **core comparison suite** (one model per paradigm). To also run ablations, edit `src/scripts/run_experiments.py` and uncomment the relevant lines at the bottom:

```python
run_core_rq_suite()
# run_model_ablations()   # ensemble size, MC dropout passes, GP dim, backbone size
# run_data_ablations()    # window length, prediction horizon, frequency scales
```

### Ablation suites

| Suite | Function | Description |
|---|---|---|
| Core comparison | `run_core_rq_suite()` | One run per paradigm with fixed settings |
| Model ablations | `run_model_ablations()` | Sweep ensemble size, dropout passes, GP dim, backbone hidden dim |
| Data ablations | `run_data_ablations()` | Sweep window length, prediction horizon, frequency scales |
| Subject LOSO | `run_subject_loso_suite()` | Leave-one-subject-out (13 folds) |
| Activity k-fold | `run_activity_kfold_suite()` | Activity-based k-fold cross-validation |
| Subject k-fold | `run_subject_kfold_suite()` | Subject-based k-fold cross-validation |

### Analyse results

Open `results/analysis.ipynb` to visualise and compare experiment outputs stored under `results/`.

---

## 5. Project Structure

```
gatech_data_analysis/
├── src/
│   ├── config/
│   │   └── config.py          # EnvConfig, DataConfig, ModelConfig, TrainConfig dataclasses
│   ├── data/
│   │   ├── dataset.py         # PredictionDataset (EMG windowing + CWT features)
│   │   └── utils.py           # Data loading helpers
│   ├── model/
│   │   ├── backbone.py        # Shared CNN/TCN backbone
│   │   ├── factory.py         # create_model / create_gp_model factory functions
│   │   └── predictor.py       # Paradigm-specific prediction heads
│   ├── trainer/
│   │   └── trainer.py         # Training loop + evaluation
│   ├── utils/
│   │   └── logger.py          # Logging utility
│   └── scripts/
│       ├── setup_pod.py       # Data download + .env generation (RunPod)
│       ├── train_single.py    # Single-model training entry point
│       └── run_experiments.py # Automated experiment orchestrator
├── results/                   # Saved model checkpoints, predictions, and configs
├── logs/                      # Training logs
├── paper/                     # LaTeX manuscript
├── experiments.ipynb          # Interactive experimentation notebook
├── pyproject.toml             # Project dependencies (uv/pip)
└── .env                       # Local environment paths (not committed)
```

---

## 6. Configuration Reference

All config fields can be overridden via CLI flags when using `train_single.py`. Defaults are defined in `src/config/config.py`.

### Data (`--data-*`)

| Flag | Default | Description |
|---|---|---|
| `--data-window-length` | `1.0` | EMG window size in seconds |
| `--data-stride` | `0.05` | Sliding window stride in seconds |
| `--data-prediction-horizon` | `0.05` | Future prediction horizon in seconds |
| `--data-n-scales` | `40` | Number of CWT frequency scales |
| `--data-split-strategy` | `single_holdout` | `single_holdout`, `subject_loso`, `subject_kfold`, `activity_kfold` |
| `--data-n-folds` | `5` | Number of folds (k-fold strategies) |
| `--data-fold-index` | `0` | Which fold to use as the test set |

### Model (`--model-*`)

| Flag | Default | Description |
|---|---|---|
| `--model-paradigm` | `deterministic` | UQ paradigm |
| `--model-hidden-dim` | `128` | Backbone hidden dimension |
| `--model-dropout-rate` | `0.2` | Dropout rate |
| `--model-ensemble-size` | `20` | Number of ensemble members |
| `--model-mc-dropout-passes` | `20` | MC Dropout forward passes at inference |
| `--model-gp-latent-dim` | `16` | GP latent feature dimension |
| `--model-gp-inducing-points` | `100` | Number of GP inducing points |

### Training (`--train-*`)

| Flag | Default | Description |
|---|---|---|
| `--train-epochs` | `20` | Number of training epochs |
| `--train-batch-size` | `32` | Batch size |
| `--train-learning-rate` | `5e-4` | Adam learning rate |
| `--train-weight-decay` | `1e-5` | L2 regularisation |
