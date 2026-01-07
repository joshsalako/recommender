# MovieLens ALS Recommender System

A high-performance recommender system implementing Alternating Least Squares (ALS) matrix factorization for the MovieLens 32M dataset. Optimized with Numba acceleration and parallel processing.

**Live Demo**: [https://als-recommender.streamlit.app/](https://als-recommender.streamlit.app/)

## Project Structure

```
recommender/
├── pyproject.toml          # Project configuration and dependencies (uv)
├── uv.lock                 # Locked dependencies
├── .env.samaple            # Sample environment variable
├── recommender.ipynb       # Original Jupyter Notebook (Reference)
├── convert_model.py        # Convert trained models to lightweight inference format
├── recommender_system/     # Main package
│   ├── __init__.py
│   ├── app.py              # Streamlit web application for interactive recommendations
│   ├── config.py           # Configuration settings
│   ├── main.py             # Main pipeline script
│   ├── data/               # Data loading and processing
│   │   ├── __init__.py
│   │   ├── loader.py       # Data download and loading
│   │   └── index.py        # Efficient data indexing (CSR format)
│   ├── models/             # ALS implementations
│   │   ├── __init__.py
│   │   ├── als.py          # Optimized ALS model
│   │   ├── als_biases.py   # Bias-only model
│   │   └── als_latent.py   # Basic matrix factorization
│   ├── utils/              # Shared utilities
│   │   ├── __init__.py
│   │   ├── dummy_user.py   # Functions for new user recommendations
│   │   ├── numba_ops.py    # Numba-accelerated operations
│   │   └── posters.py      # TMDB poster fetching
│   └── visualization/      # Visualization modules
│       ├── __init__.py
│       ├── plots.py        # General plotting functions
│       └── vectors.py      # Latent vector visualization
├── data_files/             # Downloaded datasets (runtime)
├── results/                # Generated plots and models
├── inference/              # Lightweight inference files (generated)
└── .devcontainer/          # Development container configuration
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

3. **Set up TMDB API key**:
   ```bash
   cp .env.sample .env
   # Edit .env and add your TMDB API key from https://www.themoviedb.org/settings/api
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

**Run Streamlit Web Application:**
Interactive web interface for movie recommendations (requires trained model).
```bash
uv run streamlit run recommender_system/app.py
```

## Key Features

- **Scalable**: Handles MovieLens 32M (~32 million ratings)
- **Fast**: Core operations accelerated with Numba JIT compilation
- **Parallel**: Uses parallel processing for ALS updates
- **Modular**: Clean separation of data, model, and visualization concerns
- **Analysis**: Comprehensive visualization of dataset statistics and model performance
- **Interactive**: Streamlit web interface for real-time recommendations

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

## Web Interface

The project includes a Streamlit web application (`recommender_system/app.py`) for interactive movie recommendations:

1. **Prerequisite**: Train a model first using `uv run python -m recommender_system.main --train` (skip this step to use an already trained model)
2. Create an environment variable from `.env.sample` with your `TMDB_API_KEY`
3. **Run the app**: `uv run streamlit run recommender_system/app.py`
4. **Features**:
   - Rate movies to get personalized recommendations
   - Adjust alpha parameter to control bias weighting
   - View top recommendations with movie details

### Model Loading Options
The app supports two loading modes:
1. **Standard Loading**: Loads full `model.pkl` (slower startup)
2. **Memory-Efficient Loading**: Uses lightweight inference files from `inference/` directory (faster startup, requires running `convert_model.py` first)

The app automatically downloads the MovieLens dataset if needed and loads the trained model.
