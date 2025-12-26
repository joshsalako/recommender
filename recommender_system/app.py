import streamlit as st
import numpy as np
import polars as pl
import os
import sys

# Add project root to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recommender_system.models.als import ALS
from recommender_system.utils.dummy_user import get_dummy_user_factors, recommend_for_dummy_user
from recommender_system.data.loader import load_data, download_and_extract_data
from recommender_system.data.index import MovieIndex
from recommender_system.utils.posters import PosterFetcher

# Configure page with collapsed sidebar
st.set_page_config(
    page_title="Movie Recommender",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme and better spacing
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .movie-card {
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    """
    Loads data, model, and poster fetcher. Cached to avoid reloading.
    """
    download_and_extract_data()
    ratings_df, movies_df, tags_df = load_data()

    # Create MovieIndex
    ratings_np = ratings_df.select(["userId", "movieId", "rating"]).to_numpy()
    movie_index = MovieIndex(ratings_np)

    train_user, test_user, train_movie, test_movie = movie_index.train_test_split()

    # Initialize Poster Fetcher
    poster_fetcher = PosterFetcher()

    # Load Model
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        return None, None, None, None, None

    model = ALS.load_checkpoint(train_user, test_user, train_movie, test_movie, model_path)

    return model, movies_df, movie_index, train_movie, poster_fetcher

# Load resources
with st.spinner("Loading model and data... This may take a minute."):
    model, movies_df, movie_index, train_movie, poster_fetcher = load_resources()

if model is None:
    st.error("Model file (model.pkl) not found. Please train the model first using 'uv run python -m recommender_system.main --train'")
    st.stop()

# Sidebar controls (Hamburger menu)
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("### Recommendation Settings")
alpha = st.sidebar.slider(
    "Alpha (Bias Weight)",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.05,
    help="Controls how much weight is given to the global/item biases vs user specific preferences."
)

# Option to add API Key dynamically if not in env
api_key = st.sidebar.text_input("TMDB API Key (Optional)", type="password")
if api_key:
    # Update fetcher if key provided
    poster_fetcher.api_key = api_key
    poster_fetcher._init_tmdb()

st.sidebar.divider()
if st.sidebar.button("Clear All Ratings"):
    st.session_state.user_ratings = {}
    st.rerun()

st.title("🎬 Movie Recommender System")
st.markdown("Rate some movies to get personalized recommendations!")

# User Ratings Input
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}

# Movie selection for rating
col_search, col_rate, col_add = st.columns([3, 1, 1])

all_titles = movies_df['title'].to_list()

with col_search:
    selected_movie_title = st.selectbox("Search for a movie:", [""] + all_titles, label_visibility="collapsed")

rating = 3.5
if selected_movie_title:
    with col_rate:
        rating = st.slider("Rating", 0.5, 5.0, 3.5, 0.5, label_visibility="collapsed")

    with col_add:
        if st.button("Add Rating", type="primary"):
            # Find movieId
            movie_row = movies_df.filter(pl.col("title") == selected_movie_title)
            if not movie_row.is_empty():
                movie_id = movie_row['movieId'][0]
                st.session_state.user_ratings[selected_movie_title] = (movie_id, rating)
                st.success(f"Rated '{selected_movie_title}'")
                st.rerun()

# Display current ratings
if st.session_state.user_ratings:
    st.divider()
    st.subheader("Your Ratings")

    # Create a grid for rated movies
    rated_items = list(st.session_state.user_ratings.items())

    # Display in rows of 3
    for i in range(0, len(rated_items), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(rated_items):
                title, (mid, score) = rated_items[i + j]
                with cols[j]:
                    st.info(f"**{title}**\n\nRating: {score} ⭐")
                    if st.button("Remove", key=f"remove_{mid}"):
                        del st.session_state.user_ratings[title]
                        st.rerun()

    st.divider()

    # Get Recommendations
    if st.button("✨ Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Generating personalized recommendations..."):
            dummy_ratings = []
            for title, (mid, score) in st.session_state.user_ratings.items():
                if mid in movie_index.movie_to_idx:
                    idx = movie_index.movie_to_idx[mid]
                    dummy_ratings.append((idx, score))

            if not dummy_ratings:
                st.warning("None of the rated movies are in the model's index.")
            else:
                user_bias, user_vector = get_dummy_user_factors(model, dummy_ratings)
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

                # Grid View Implementation
                rec_cols = st.columns(3)

                for idx, (item_idx, score) in enumerate(recommendations):
                    col = rec_cols[idx % 3]

                    with col:
                        if item_idx in movie_index.idx_to_movie:
                            original_id = movie_index.idx_to_movie[item_idx]
                            rec_movie_row = movies_df.filter(pl.col("movieId") == original_id)

                            if not rec_movie_row.is_empty():
                                title = rec_movie_row['title'][0]
                                genres = rec_movie_row['genres'][0]

                                # Fetch Poster
                                poster_url = poster_fetcher.get_poster_url(original_id, title)

                                st.image(poster_url, use_column_width=True)
                                st.markdown(f"**{title}**")
                                st.caption(f"_{genres}_")
                                # Score hidden as requested
                                # st.caption(f"Score: {score:.2f}")
                                st.markdown("---")

else:
    st.info("Start by searching and rating a few movies above to see recommendations!")
