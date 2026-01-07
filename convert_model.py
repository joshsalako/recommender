#!/usr/bin/env python3
"""
Convert existing model.pkl to lightweight inference files.

This script converts an existing trained model checkpoint (model.pkl) to the
memory-efficient inference format used by the Streamlit app.

Usage:
    uv run python convert_model.py --model model.pkl
    uv run python convert_model.py --model model.pkl --ratings data_files/ml-32m/ratings.csv --movies data_files/ml-32m/movies.csv
"""

import argparse
import os
import sys
import pickle
import numpy as np
import polars as pl

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from recommender_system.data.index import LightweightMovieIndex
from recommender_system.models.als import ALS


def load_checkpoint_data(model_path):
    """Load checkpoint data from model.pkl file."""
    print(f"Loading checkpoint from: {model_path}")
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)

    # Verify required keys are present
    required_keys = ['n_users', 'n_movies', 'k', 'lambda_reg', 'tau', 'mu',
                     'user_biases', 'item_biases', 'user_vector', 'item_vector']

    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(f"Checkpoint missing required key: {key}")

    print(f"Checkpoint loaded: n_users={checkpoint['n_users']}, n_movies={checkpoint['n_movies']}, k={checkpoint['k']}")
    return checkpoint


def extract_movie_data(movies_path):
    """Extract movie metadata from movies.csv file."""
    print(f"Loading movie metadata from: {movies_path}")

    # Load movies CSV with appropriate columns
    movies_df = pl.read_csv(movies_path)

    # Verify required columns
    required_cols = ['movieId', 'title', 'genres']
    for col in required_cols:
        if col not in movies_df.columns:
            raise ValueError(f"Movies CSV missing required column: {col}")

    print(f"Loaded {len(movies_df)} movies")
    return movies_df


def create_lightweight_index(ratings_path, checkpoint_n_movies):
    """
    Create LightweightMovieIndex from ratings CSV without loading full dataset.

    Uses memory-efficient Polars operations to extract:
    1. Unique movie IDs and create ID mappings
    2. Rating counts per movie
    """
    print(f"Processing ratings data from: {ratings_path}")

    # Load only the movieId column to minimize memory usage
    print("  Loading movieId column only...")
    ratings_df = pl.read_csv(ratings_path, columns=["movieId"])

    # Get unique movie IDs and create mappings
    print("  Creating ID mappings...")
    unique_movies = ratings_df["movieId"].unique().sort()
    n_movies_from_ratings = len(unique_movies)

    # Verify movie count matches checkpoint
    if n_movies_from_ratings != checkpoint_n_movies:
        print(f"  Warning: Ratings has {n_movies_from_ratings} movies, checkpoint has {checkpoint_n_movies}")
        print(f"  Using count from ratings: {n_movies_from_ratings}")

    # Create ID mappings
    movie_to_idx = {int(mid): i for i, mid in enumerate(unique_movies)}
    idx_to_movie = {i: int(mid) for i, mid in enumerate(unique_movies)}

    # Get rating counts per movie
    print("  Counting ratings per movie...")
    rating_counts = (
        ratings_df
        .group_by("movieId")
        .agg(pl.len().alias("count"))
        .sort("movieId")
    )

    # Create array of rating counts aligned with movie_to_idx ordering
    item_rating_counts = np.zeros(n_movies_from_ratings, dtype=np.int32)
    for row in rating_counts.iter_rows(named=True):
        movie_id = int(row['movieId'])
        if movie_id in movie_to_idx:
            idx = movie_to_idx[movie_id]
            item_rating_counts[idx] = row['count']

    print(f"  Created index for {n_movies_from_ratings} movies")
    print(f"  Total ratings processed: {len(ratings_df):,}")

    return LightweightMovieIndex(
        movie_to_idx=movie_to_idx,
        idx_to_movie=idx_to_movie,
        item_rating_counts=item_rating_counts
    )


def create_inference_model(checkpoint, inference_dir):
    """Create and save inference model from checkpoint data."""
    print("Creating inference model...")

    # Create a minimal ALS model for inference
    model = ALS(
        n_users=checkpoint['n_users'],
        n_movies=checkpoint['n_movies'],
        k=checkpoint['k'],
        lambda_reg=checkpoint['lambda_reg'],
        tau=checkpoint['tau'],
        train_user=None,
        test_user=None,
        train_movie=None,
        test_movie=None,
        n_jobs=1
    )

    # Copy parameters from checkpoint
    model.mu = checkpoint['mu']
    model.user_biases = checkpoint['user_biases'].astype(np.float32)
    model.item_biases = checkpoint['item_biases'].astype(np.float32)
    model.user_vector = checkpoint['user_vector'].astype(np.float32)
    model.item_vector = checkpoint['item_vector'].astype(np.float32)

    # Clear training history (not needed for inference)
    model.train_loss_history = []
    model.train_rmse_history = []
    model.test_rmse_history = []

    # Save for inference
    model_path = model.save_for_inference(inference_dir)
    print(f"  Saved inference model -> {model_path}")

    return model_path


def convert_model_to_inference(model_path, ratings_path, movies_path, inference_dir="inference"):
    """
    Convert existing model.pkl to lightweight inference files.

    Args:
        model_path: Path to existing model.pkl file
        ratings_path: Path to ratings.csv file
        movies_path: Path to movies.csv file
        inference_dir: Directory to save inference files
    """
    # Create inference directory
    os.makedirs(inference_dir, exist_ok=True)
    print(f"Saving inference files to: {inference_dir}/")

    # Step 1: Load checkpoint data
    checkpoint = load_checkpoint_data(model_path)

    # Step 2: Load movie metadata
    movies_df = extract_movie_data(movies_path)

    # Step 3: Create lightweight index from ratings
    lightweight_index = create_lightweight_index(ratings_path, checkpoint['n_movies'])

    # Step 4: Save movies metadata as parquet
    movies_parquet_path = os.path.join(inference_dir, "app_movies.parquet")
    movies_df.write_parquet(movies_parquet_path)
    print(f"Saved movies metadata -> {movies_parquet_path}")

    # Step 5: Save lightweight index
    index_path = os.path.join(inference_dir, "app_index.pkl")
    counts_path = os.path.join(inference_dir, "app_item_counts.npy")
    lightweight_index.save(index_path, counts_path)
    print(f"Saved lightweight index -> {index_path}, {counts_path}")

    # Step 6: Create and save inference model
    create_inference_model(checkpoint, inference_dir)

    print("\n" + "="*60)
    print("Conversion completed successfully!")
    print("="*60)
    print(f"\nGenerated files in '{inference_dir}/':")
    print("  - app_movies.parquet    : Movie metadata")
    print("  - app_model.npz         : Compressed model parameters")
    print("  - app_index.pkl         : ID mappings")
    print("  - app_item_counts.npy   : Rating counts per movie")
    print("\nYou can now use memory-efficient loading in the Streamlit app!")
    print("Set 'Use memory-efficient loading' to True in the sidebar.")


def find_default_data_paths():
    """Find default data paths for MovieLens dataset."""
    # Try different possible locations
    possible_locations = [
        "data_files/ml-32m",
        "data_files/ml-latest-small",
        "ml-32m",
        "ml-latest-small",
        "recommender_system/data_files/ml-32m",
        "recommender_system/data_files/ml-latest-small"
    ]

    for location in possible_locations:
        ratings_path = os.path.join(location, "ratings.csv")
        movies_path = os.path.join(location, "movies.csv")

        if os.path.exists(ratings_path) and os.path.exists(movies_path):
            return ratings_path, movies_path

    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Convert existing model.pkl to lightweight inference files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model.pkl",
        help="Path to existing model.pkl file (default: model.pkl)"
    )
    parser.add_argument(
        "--ratings",
        type=str,
        help="Path to ratings.csv file (default: auto-detect)"
    )
    parser.add_argument(
        "--movies",
        type=str,
        help="Path to movies.csv file (default: auto-detect)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference",
        help="Output directory for inference files (default: inference)"
    )

    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please provide a valid path to model.pkl")
        sys.exit(1)

    # Find data paths if not provided
    if not args.ratings or not args.movies:
        print("Auto-detecting data files...")
        ratings_path, movies_path = find_default_data_paths()

        if not ratings_path or not movies_path:
            print("Error: Could not auto-detect data files.")
            print("Please provide --ratings and --movies arguments.")
            print("\nExpected structure:")
            print("  data_files/ml-32m/ratings.csv")
            print("  data_files/ml-32m/movies.csv")
            sys.exit(1)
    else:
        ratings_path = args.ratings
        movies_path = args.movies

    # Verify data files exist
    if not os.path.exists(ratings_path):
        print(f"Error: Ratings file not found: {ratings_path}")
        sys.exit(1)

    if not os.path.exists(movies_path):
        print(f"Error: Movies file not found: {movies_path}")
        sys.exit(1)

    print("="*60)
    print("Converting existing model to inference format")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Ratings: {ratings_path}")
    print(f"Movies: {movies_path}")
    print(f"Output: {args.output}/")
    print()

    try:
        convert_model_to_inference(
            model_path=args.model,
            ratings_path=ratings_path,
            movies_path=movies_path,
            inference_dir=args.output
        )
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()