import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_files")
SAVE_DIR = os.path.join(BASE_DIR, "results")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# Data URLs
ML_32M_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
ML_LATEST_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Dataset selection
CURRENT_DATASET = 'ml-32m'  # or 'ml-latest-small'
DATASET_PATH = os.path.join(DATA_DIR, CURRENT_DATASET)

# Hyperparameters for Grid Search
K_VALUES = [2, 10, 50, 100]
TAU_VALUES = [0.05, 0.1, 0.25]
LAMBDA_VALUES = [0.1, 0.5]
N_EPOCHS = 20

# Model Paths
RESULTS_CSV_PATH = os.path.join(SAVE_DIR, "grid_search_results.csv")

# Model Selection
MODEL_CHOICES = ["als", "biases", "latent"]
DEFAULT_MODEL = "als"
