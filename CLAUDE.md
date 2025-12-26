# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a recommender system project implementing Alternating Least Squares (ALS) matrix factorization for the MovieLens 32M dataset. The project includes data preprocessing, visualization, model training, hyperparameter tuning, and recommendation generation.

## Project Structure

The project has been modernized to use `uv` for Python package management:

```
recommender/
├── pyproject.toml          # Project configuration and dependencies (uv)
├── uv.lock                 # Locked dependencies
├── recommender.ipynb       # Original Jupyter Notebook (Reference)
├── recommender_system/     # Main package (refactored code)
│   ├── __init__.py
│   ├── app.py              # Streamlit web application for interactive recommendations
│   ├── config.py           # Configuration settings
│   ├── main.py             # Main pipeline script
│   ├── data/               # Data loading and processing
│   │   ├── __init__.py
│   │   ├── loader.py       # Data download and loading
│   │   └── index.py        # MovieIndex class for efficient data indexing
│   ├── models/             # ALS model implementations
│   │   ├── __init__.py
│   │   ├── als.py          # Optimized ALS implementation
│   │   ├── als_biases.py   # Bias-only model
│   │   └── als_latent.py   # Full matrix factorization
│   ├── utils/              # Utility functions
│   │   ├── __init__.py
│   │   ├── dummy_user.py   # Functions for new user recommendations
│   │   └── numba_ops.py    # Numba-accelerated operations
│   └── visualization/      # Visualization modules
│       ├── __init__.py
│       ├── plots.py        # Plotting functions
│       └── vectors.py      # Vector visualization
├── .venv/                  # Virtual environment (created by uv)
├── data_files/             # Downloaded datasets (created at runtime)
├── results/                # Generated plots and model checkpoints
├── README.md               # Project documentation
└── CLAUDE.md               # This file
```

## Development Commands

### Package Management with uv

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

**Install dependencies:**
```bash
uv sync
```

**Add new dependency:**
```bash
uv add <package-name>
```

**Run Python scripts:**
```bash
uv run python <script.py>
```

**Run the recommender system:**
```bash
uv run python -m recommender_system.main
```

**Run the Streamlit web application:**
```bash
uv run streamlit run recommender_system/app.py
```

### Running the Pipeline

**Full pipeline (download data, visualization, grid search, training):**
```bash
uv run python -m recommender_system.main --download --visualize --train --vectors
```

**Data loading and visualization only:**
```bash
uv run python -m recommender_system.main --download --visualize
```

**Model training only:**
```bash
uv run python -m recommender_system.main --train
```

### Key Dependencies
- **Data manipulation**: numpy, pandas, polars
- **Performance acceleration**: numba
- **Machine learning**: scikit-learn
- **Visualization**: matplotlib, seaborn, adjustText, wordcloud
- **Parallel processing**: joblib
- **Web interface**: streamlit

All dependencies are managed by `uv` and specified in `pyproject.toml`.

## Architecture

### Data Indexing (`MovieIndex` class)
The `MovieIndex` class efficiently indexes and structures the rating data:
- Maps user and movie IDs to internal indices
- Creates user-centric and movie-centric data structures using CSR-like formats
- Implements train/test split functionality
- Uses Numba-accelerated functions for performance

### ALS Model Classes
Three main ALS implementations with increasing complexity:

1. **`ALSBiases`**: Bias-only model with user and item biases
2. **`ALSLatent`**: Full matrix factorization with latent factors
3. **`ALS`**: Optimized implementation with Numba acceleration, parallel updates, and checkpointing

### Key Components

- **Data preprocessing**: Downloads MovieLens dataset, creates efficient indexing structures
- **Visualization**: Multiple plotting functions for rating distributions, genre analysis, degree distributions
- **Model training**: Alternating updates of user and item factors with regularization
- **Hyperparameter tuning**: Grid search over k (latent dimensions), λ (regularization), τ (bias regularization)
- **Evaluation**: RMSE, Precision@K, Recall@K metrics
- **Recommendation**: Functions for generating recommendations for new users

## Model Training

The main training workflow:
1. Initialize `MovieIndex` with rating data
2. Split into train/test sets
3. Train ALS model with chosen hyperparameters
4. Evaluate using RMSE and top-K metrics
5. Save model checkpoints

## Hyperparameter Tuning

Grid search over:
- `k`: Latent dimensions (2, 10, 50, 100)
- `lambda_reg`: Regularization strength (0.1, 0.5)
- `tau`: Bias regularization (0.05, 0.1, 0.25)

Results are saved to CSV files for analysis.

## Streamlit Web Application

The project includes a Streamlit web interface for interactive movie recommendations:

**Features:**
- Interactive movie rating interface
- Real-time personalized recommendations based on user ratings
- Alpha parameter tuning for bias weighting in recommendations
- Model loading from saved checkpoints (`model.pkl`)
- Integration with the trained ALS model for inference

**Usage:**
1. First train a model using the main pipeline: `uv run python -m recommender_system.main --train`
2. Run the Streamlit app: `uv run streamlit run recommender_system/app.py`
3. The app will automatically download data if needed and load the trained model

**Note:** The app requires a trained model checkpoint (`model.pkl`) in the project root. If not found, it will display instructions for training a model.

## Important Notes

1. **Data paths**: The script downloads MovieLens dataset to `data_files/` directory.
2. **Memory usage**: The MovieLens 32M dataset is large (~32 million ratings). Ensure sufficient memory.
3. **Checkpointing**: Models are saved as pickle files with timestamped names containing hyperparameters.
4. **Visualization**: All plots are saved as PDF files to the `results/` directory.
5. **Package management**: Use `uv` commands instead of `pip` for all package operations.
6. **Streamlit model requirement**: The web app requires a trained model checkpoint (`model.pkl`) in the project root.

## Model Loading and Inference

To load a trained model:
```python
model = ALS.load_checkpoint(train_user, test_user, train_movie, test_movie,
                           path="path/to/checkpoint.pkl", n_jobs=-1)
```

## Performance Optimizations

- **Numba JIT compilation**: Critical functions are decorated with `@njit` for CPU acceleration
- **CSR format**: Data stored in compressed sparse row format for efficient access
- **Parallel updates**: User and item updates can run in parallel using joblib
- **Vectorized operations**: NumPy vectorization used where possible