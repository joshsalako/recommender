import os
import zipfile
import urllib.request
import polars as pl
from .. import config

def download_and_extract_data():
    """
    Downloads and extracts the MovieLens datasets if not present.
    """
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)

    # Check if ML-32M is extracted
    ml_32m_path = os.path.join(config.DATA_DIR, "ml-32m")
    if not os.path.exists(ml_32m_path):
        zip_path = os.path.join(config.DATA_DIR, "ml-32m.zip")
        if not os.path.exists(zip_path):
            print(f"Downloading ML-32M from {config.ML_32M_URL}...")
            urllib.request.urlretrieve(config.ML_32M_URL, zip_path)
        print("Extracting ML-32M...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(config.DATA_DIR)
    else:
        print(f"ML-32M dataset found at {ml_32m_path}")

    # Check if ML-Latest-Small is extracted
    ml_small_path = os.path.join(config.DATA_DIR, "ml-latest-small")
    if not os.path.exists(ml_small_path):
        zip_path = os.path.join(config.DATA_DIR, "ml-latest-small.zip")
        if not os.path.exists(zip_path):
            print(f"Downloading ML-Latest-Small from {config.ML_LATEST_SMALL_URL}...")
            urllib.request.urlretrieve(config.ML_LATEST_SMALL_URL, zip_path)
        print("Extracting ML-Latest-Small...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(config.DATA_DIR)
    else:
        print(f"ML-Latest-Small dataset found at {ml_small_path}")


def load_data(data_dir=None):
    """
    Loads ratings, movies, and tags data into Polars DataFrames.
    """
    if data_dir is None:
        data_dir = config.DATASET_PATH

    print(f"Loading data from {data_dir}...")

    ratings_path = os.path.join(data_dir, 'ratings.csv')
    movies_path = os.path.join(data_dir, 'movies.csv')
    tags_path = os.path.join(data_dir, 'tags.csv')

    ratings_df = pl.read_csv(ratings_path)
    movies_df = pl.read_csv(movies_path)
    tags_df = pl.read_csv(tags_path)

    return ratings_df, movies_df, tags_df


def convert_to_numpy(ratings_df, movies_df, tags_df):
    """
    Converts Polars DataFrames to NumPy arrays.
    """
    ratings_np = ratings_df.to_numpy()
    movies_np = movies_df.to_numpy()
    tags_np = tags_df.to_numpy()

    return ratings_np, movies_np, tags_np
