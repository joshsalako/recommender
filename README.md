# MovieLens ALS Recommender System

A high-performance recommender system implementing Alternating Least Squares (ALS) matrix factorization for the MovieLens 32M dataset. Optimized with Numba acceleration and parallel processing.

## Project Structure

```
recommender/
├── pyproject.toml          # Project configuration and dependencies (uv)
├── uv.lock                 # Locked dependencies
├── recommender.ipynb       # Original Jupyter Notebook (Reference)
├── recommender_system/     # Main package
│   ├── data/               # Data loading and processing
│   │   ├── loader.py       # Data download and loading
│   │   └── index.py        # Efficient data indexing (CSR format)
│   ├── models/             # ALS implementations
│   │   ├── als.py          # Optimized ALS model
│   │   ├── als_biases.py   # Bias-only model
│   │   └── als_latent.py   # Basic matrix factorization
│   ├── utils/              # Shared utilities
│   │   └── numba_ops.py    # Numba-accelerated operations
│   └── visualization/      # Visualization modules
│       ├── plots.py        # General plotting functions
│       └── vectors.py      # Latent vector visualization
├── data_files/             # Downloaded datasets (runtime)
├── results/                # Generated plots and models
└── main.py                 # Entry point script
```

## Note on Jupyter Notebook
The original implementation is available in `recommender.ipynb`. This notebook can be used for reference or exploratory analysis, but the main codebase has been refactored into the `recommender_system` package for better maintainability and performance.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable package management.

1. **Install uv** (if not installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Sync dependencies**:
   ```bash
   uv sync
   ```

## Usage

Run the main pipeline using `uv run`. The script supports several flags to control the workflow:

### 1. Full Pipeline
Download data, visualize, train model, and generate plots:
```bash
uv run python -m recommender_system.main --download --visualize --train --vectors
```

### 2. Individual Steps

**Download Data Only:**
```bash
uv run python -m recommender_system.main --download
```

**Run Visualizations:**
Generates rating distributions, genre analysis, and tag clouds in `results/`.
```bash
uv run python -m recommender_system.main --visualize
```

**Train Model:**
Runs grid search for hyperparameters and trains the model.
You can specify the model type using the `--model` argument (default: `als`).
Available options: `als`, `biases`, `latent`.

```bash
# Train default optimized ALS model
uv run python -m recommender_system.main --train

# Train bias-only model
uv run python -m recommender_system.main --train --model biases

# Train basic latent factor model
uv run python -m recommender_system.main --train --model latent
```

**Visualize Vectors:**
Visualizes movie latent vectors using PCA (requires trained model).
```bash
uv run python -m recommender_system.main --vectors
```

## Key Features

- **Scalable**: Handles MovieLens 32M (~32 million ratings)
- **Fast**: Core operations accelerated with Numba JIT compilation
- **Parallel**: Uses parallel processing for ALS updates
- **Modular**: Clean separation of data, model, and visualization concerns
- **Analysis**: comprehensive visualization of dataset statistics and model performance

## Models

1. **ALSBiases**: Baseline model using only user and item biases.
2. **ALSLatent**: Standard Matrix Factorization with latent factors.
3. **ALS (Optimized)**: Production-grade implementation with:
   - CSR (Compressed Sparse Row) data structures
   - Parallel coordinate descent updates
   - Checkpointing
   - Numba acceleration

## Configuration

Settings such as paths and hyperparameters can be modified in `recommender_system/config.py`.
