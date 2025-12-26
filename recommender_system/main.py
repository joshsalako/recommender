import argparse
import os
import pandas as pd
from itertools import product
import numpy as np

from recommender_system import config
from recommender_system.data.loader import download_and_extract_data, load_data, convert_to_numpy
from recommender_system.data.index import MovieIndex
from recommender_system.models.als import ALS
from recommender_system.models.als_biases import ALSBiases
from recommender_system.models.als_latent import ALSLatent
from recommender_system.visualization import plots, vectors

def run_visualization_pipeline(movies_df, ratings_df, tags_df):
    print("Running visualization pipeline...")
    plots.plot_rating_distribution(ratings_df)
    plots.analyze_movie_trends(movies_df, ratings_df)
    plots.plot_ratings_per_year(movies_df, ratings_df)
    plots.plot_degree_distribution(ratings_df)

    genre_counts = plots.plot_top_genres(movies_df.lazy())
    plots.plot_genre_pie_chart(genre_counts)

    plots.plot_genre_rating_distributions(movies_df.lazy(), ratings_df.lazy())
    plots.plot_genre_ratings_scatter(movies_df, ratings_df)

    tag_counts = plots.plot_top_tags(tags_df.lazy())
    plots.plot_wordcloud(tag_counts)
    print("Visualizations saved to results directory.")

def run_grid_search(train_user, test_user, train_movie, test_movie, n_users, n_movies, model_type="als"):
    print(f"Starting Grid Search for model: {model_type}...")
    results_data = []

    k_values = config.K_VALUES
    # ALSBiases doesn't use k, so we iterate once with a dummy value
    if model_type == "biases":
        k_values = [0]

    for k_, lambda_, tau_ in product(k_values, config.LAMBDA_VALUES, config.TAU_VALUES):
        print(f"Testing configuration: k={k_}, lambda={lambda_}, tau={tau_}")
        run_dir = os.path.join(config.SAVE_DIR, f"model={model_type}_k={k_}_lambda={lambda_}")

        if model_type == "biases":
            model = ALSBiases(
                train_user, test_user, train_movie, test_movie,
                n_users=n_users,
                n_movies=n_movies,
                lambda_reg=lambda_,
                tau=tau_
            )
        elif model_type == "latent":
            model = ALSLatent(
                train_user, test_user, train_movie, test_movie,
                n_users=n_users,
                n_movies=n_movies,
                k=k_,
                lambda_reg=lambda_,
                tau=tau_
            )
        else:  # als
            model = ALS(
                train_user, test_user, train_movie, test_movie,
                n_users=n_users,
                n_movies=n_movies,
                k=k_,
                lambda_reg=lambda_,
                tau=tau_,
                n_jobs=-1
            )

        model.train(n_epochs=config.N_EPOCHS, save_dir=run_dir)
        plots.plot_training_history(model)
        precision, recall = model.evaluate(k=k_ if k_ > 0 else 10, threshold=3.5)

        record = {
            "model": model_type,
            "k": k_,
            "lambda": lambda_,
            "tau": tau_,
            "precision": precision,
            "recall": recall,
            "final_train_rmse": model.train_rmse_history[-1],
            "final_test_rmse": model.test_rmse_history[-1],
            "final_loss": model.train_loss_history[-1]
        }
        results_data.append(record)

        # Save intermediate results
        pd.DataFrame(results_data).to_csv(config.RESULTS_CSV_PATH, index=False)

    print(f"Grid search completed. Results saved to {config.RESULTS_CSV_PATH}")
    return pd.DataFrame(results_data)

def main():
    parser = argparse.ArgumentParser(description="Recommender System Pipeline")
    parser.add_argument("--download", action="store_true", help="Download and extract datasets")
    parser.add_argument("--visualize", action="store_true", help="Run data visualizations")
    parser.add_argument("--train", action="store_true", help="Run training/grid search")
    parser.add_argument("--vectors", action="store_true", help="Visualize latent vectors (requires trained model)")
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL, choices=config.MODEL_CHOICES,
                        help=f"Model type to use (default: {config.DEFAULT_MODEL})")
    args = parser.parse_args()

    # 1. Setup Data
    if args.download:
        download_and_extract_data()

    # 2. Load Data
    try:
        ratings_df, movies_df, tags_df = load_data()
    except FileNotFoundError:
        print("Data not found. Please run with --download first.")
        return

    # 3. Visualization
    if args.visualize:
        run_visualization_pipeline(movies_df, ratings_df, tags_df)

    # 4. Prepare Data for Training
    if args.train or args.vectors:
        print("Converting data to NumPy and Indexing...")
        ratings_np, movies_np, tags_np = convert_to_numpy(ratings_df, movies_df, tags_df)

        movie_index = MovieIndex(ratings_np)
        train_user, test_user, train_movie, test_movie = movie_index.train_test_split()

        print(f"Data split stats: Train Users: {len(train_user)}, Test Users: {len(test_user)}")

    # 5. Training
    if args.train:
        df_results = run_grid_search(
            train_user, test_user, train_movie, test_movie,
            movie_index.n_users, movie_index.n_movies,
            model_type=args.model
        )

        plots.plot_precision_recall_comparison(df_results)
        plots.plot_rmse_heatmap(df_results)

    # 6. Vector Visualization (Example using a fresh model if no checkpoint provided)
    if args.vectors:
        # NOTE: In a real scenario, you'd load the best checkpoint.
        # Here we train a small model quickly for demonstration if not training fully.
        print("Visualizing vectors...")

        # Create a model instance (either load or train fresh)
        # For this script, let's just train a quick one if not already done
        model = ALS(
            train_user, test_user, train_movie, test_movie,
            n_users=movie_index.n_users,
            n_movies=movie_index.n_movies,
            k=10, lambda_reg=0.1, tau=0.1
        )
        model.train(n_epochs=5, save_dir=os.path.join(config.SAVE_DIR, "vector_viz"))

        vectors.visualize_vectors(model, movies_df, movie_index)

if __name__ == "__main__":
    main()
