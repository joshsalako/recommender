import streamlit as st
import numpy as np
import polars as pl
import pandas as pd
import os
import sys

# Add project root to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recommender_system.models.als import ALS
from recommender_system.utils.dummy_user import get_dummy_user_factors, recommend_for_dummy_user
from recommender_system.data.loader import load_data, download_and_extract_data
from recommender_system.data.index import MovieIndex

st.set_page_config(page_title="Movie Recommender", layout="wide")

@st.cache_resource
def load_resources():
    """
    Loads data and model. Cached to avoid reloading on every interaction.
    """
    # 1. Ensure data exists
    download_and_extract_data()

    # 2. Load DataFrames
    ratings_df, movies_df, tags_df = load_data()

    # 3. Create MovieIndex (needed for mapping IDs)
    # We need to convert to numpy for MovieIndex
    # Note: This might be memory intensive.
    # Ideally, we would save/load the MovieIndex or just the mappings,
    # but for now we reconstruct it to ensure compatibility with the trained model logic.
    ratings_np = ratings_df.select(["userId", "movieId", "rating"]).to_numpy()
    movie_index = MovieIndex(ratings_np)

    # 4. Split data (needed for model loading signature, though we might not use test set for inference)
    # The model.pkl expects us to pass these during load_checkpoint.
    # However, if we only need inference, maybe we can pass None?
    # Let's check ALS.load_checkpoint source again.
    # It passes them to __init__. And __init__ uses them to build CSR matrices if they are not None.
    # recommend_for_dummy_user NEEDS train_movie to check for min ratings count.
    # So we MUST have train_movie populated.

    train_user, test_user, train_movie, test_movie = movie_index.train_test_split()

    # 5. Load Model
    # Assumes model.pkl is in the project root or we can find it.
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please run training first.")
        return None, None, None, None

    model = ALS.load_checkpoint(train_user, test_user, train_movie, test_movie, model_path)

    return model, movies_df, movie_index, train_movie

# Load resources
with st.spinner("Loading model and data... This may take a minute."):
    model, movies_df, movie_index, train_movie = load_resources()

if model is None:
    st.stop()

# Sidebar controls
st.sidebar.title("Configuration")
alpha = st.sidebar.slider("Alpha (Bias Weight)", min_value=0.0, max_value=2.0, value=1.0, step=0.1, help="Controls how much weight is given to the global/item biases vs user specific preferences.")

st.title("Movie Recommender System")
st.markdown("Rate some movies to get personalized recommendations!")

# User Ratings Input
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}

# Movie selection for rating
# Create a searchable list of titles
# We want 'Title (Year)' format usually, which movies_df has in 'title' column
all_titles = movies_df['title'].to_list()

selected_movie_title = st.selectbox("Search for a movie to rate:", [""] + all_titles)

if selected_movie_title:
    # Find movieId
    movie_row = movies_df.filter(pl.col("title") == selected_movie_title)
    if not movie_row.is_empty():
        movie_id = movie_row['movieId'][0]

        # Rating input
        rating = st.slider(f"Rate '{selected_movie_title}'", 0.5, 5.0, 3.5, 0.5)

        if st.button("Add Rating"):
            st.session_state.user_ratings[selected_movie_title] = (movie_id, rating)
            st.success(f"Rated '{selected_movie_title}' as {rating} stars")

# Display current ratings
if st.session_state.user_ratings:
    st.subheader("Your Ratings")

    # Create a clean dataframe for display
    rated_movies_data = []
    movies_to_remove = []

    for title, (mid, score) in st.session_state.user_ratings.items():
        col1, col2, col3 = st.columns([6, 2, 2])
        with col1:
            st.write(title)
        with col2:
            st.write(f"{score} ⭐")
        with col3:
            if st.button("Remove", key=f"remove_{mid}"):
                movies_to_remove.append(title)

    # Process removals
    if movies_to_remove:
        for title in movies_to_remove:
            del st.session_state.user_ratings[title]
        st.rerun()

    # Get Recommendations Button
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            # Prepare dummy ratings list for the model: [(movie_idx, rating), ...]
            # IMPORTANT: We need internal movie indices, not raw movieIds

            dummy_ratings = []
            for title, (mid, score) in st.session_state.user_ratings.items():
                if mid in movie_index.movie_to_idx:
                    idx = movie_index.movie_to_idx[mid]
                    dummy_ratings.append((idx, score))
                else:
                    # Should unlikely happen if we source from movies_df which comes from same dataset,
                    # but possible if model trained on subset or different version.
                    pass

            if not dummy_ratings:
                st.warning("None of the rated movies are in the model's index.")
            else:
                # Get User Factors
                user_bias, user_vector = get_dummy_user_factors(model, dummy_ratings)

                # Get Recommendations
                # We need list of internal indices of rated items to exclude them
                rated_indices = [r[0] for r in dummy_ratings]

                recommendations = recommend_for_dummy_user(
                    model,
                    train_movie,
                    user_bias,
                    user_vector,
                    rated_items=rated_indices,
                    top_n=15,
                    alpha=alpha,
                    min_item_ratings=100
                )

                st.subheader("Recommended for You")

                # Display recommendations
                for item_idx, score in recommendations:
                    # Map back to title
                    if item_idx in movie_index.idx_to_movie:
                        original_id = movie_index.idx_to_movie[item_idx]

                        # Get details from movies_df
                        rec_movie_row = movies_df.filter(pl.col("movieId") == original_id)
                        if not rec_movie_row.is_empty():
                            title = rec_movie_row['title'][0]
                            genres = rec_movie_row['genres'][0]

                            with st.container():
                                st.markdown(f"### {title}")
                                st.text(f"Genres: {genres}")
                                st.caption(f"Score: {score:.2f}")
                                st.divider()

else:
    st.info("Start by searching and rating a few movies above!")
