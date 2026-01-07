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
from recommender_system.data.index import MovieIndex, LightweightMovieIndex
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
    /* Ensure consistent poster image sizing */
    [data-testid="stImage"] img {
        width: 100% !important;
        height: 400px !important;
        object-fit: cover !important;
        border-radius: 8px !important;
    }
    /* Center the title */
    h1[data-testid="stMarkdownContainer"] {
        text-align: center;
    }
    /* Center buttons */
    .centered-button {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    /* Fix for dark theme contrast */
    .dark-theme-fix {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    /* Better rating cards for dark theme */
    .rating-card {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-left: 4px solid #4CAF50 !important;
    }
    /* Fix grid alignment for long titles */
    .movie-grid-item {
        display: flex !important;
        flex-direction: column !important;
        height: 100% !important;
    }
    .movie-title {
        font-size: 20px !important;
        line-height: 1.3 !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        display: -webkit-box !important;
        -webkit-line-clamp: 2 !important;
        -webkit-box-orient: vertical !important;
        min-height: 42px !important;
        margin-bottom: 5px !important;
        font-weight: 600 !important;
    }
    /* Consistent card heights */
    .movie-card-container {
        height: 550px !important;
        display: flex !important;
        flex-direction: column !important;
    }
    .movie-poster-container {
        flex: 0 0 400px !important;
        overflow: hidden !important;
    }
    .movie-info-container {
        flex: 1 !important;
        padding: 10px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    """
    Loads data, model, and poster fetcher. Cached to avoid reloading.
    Uses full dataset loading (memory intensive).
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

@st.cache_resource
def load_resources_light():
    """
    Loads data, model, and poster fetcher using lightweight pre-computed files.
    Memory-efficient alternative to load_resources().
    """
    # Check if inference files exist
    inference_dir = "inference"
    model_path = os.path.join(inference_dir, "app_model.npz")
    index_path = os.path.join(inference_dir, "app_index.pkl")
    counts_path = os.path.join(inference_dir, "app_item_counts.npy")
    movies_path = os.path.join(inference_dir, "app_movies.parquet")

    if not all(os.path.exists(p) for p in [model_path, index_path, counts_path, movies_path]):
        st.warning("Inference files not found. Please run model training with --inference flag first.")
        return None, None, None, None, None

    # Load lightweight resources
    try:
        # Load model for inference
        model = ALS.load_for_inference(model_path)

        # Load movies metadata
        movies_df = pl.read_parquet(movies_path)

        # Load lightweight index
        movie_index = LightweightMovieIndex.load(index_path, counts_path)

        # Initialize Poster Fetcher
        poster_fetcher = PosterFetcher()

        # train_movie is replaced with item_rating_counts in the lightweight index
        # We pass the index itself as the item_rating_counts parameter
        return model, movies_df, movie_index, movie_index, poster_fetcher

    except Exception as e:
        st.error(f"Error loading lightweight resources: {e}")
        return None, None, None, None, None

# Configuration option for loading method
st.sidebar.markdown("### Memory Settings")
use_lightweight = st.sidebar.checkbox(
    "Use memory-efficient loading (recommended for low-memory devices)",
    value=True,
    help="Uses pre-computed inference files instead of loading full dataset"
)

# Load resources
with st.spinner("Loading model and data... This may take a minute."):
    if use_lightweight:
        model, movies_df, movie_index, train_movie, poster_fetcher = load_resources_light()
    else:
        model, movies_df, movie_index, train_movie, poster_fetcher = load_resources()

if model is None:
    if use_lightweight:
        st.error("""
        Inference files not found. Please either:
        1. Train the model with inference file generation: `uv run python -m recommender_system.main --train --inference`
        2. Or disable memory-efficient loading and use the full dataset
        """)
    else:
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

st.markdown("<h1 style='text-align: center;'>🎬 Movie Recommender System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Rate some movies to get personalized recommendations!</p>", unsafe_allow_html=True)

# User Ratings Input
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}

# Movie selection for rating
st.markdown("<p style='color: #666; margin-bottom: 10px;'>Start typing to search from 32,000+ movies:</p>", unsafe_allow_html=True)

col_search, col_rate, col_add = st.columns([3, 1, 1])

all_titles = movies_df['title'].to_list()

with col_search:
    selected_movie_title = st.selectbox(
        "Search for a movie:",
        [""] + all_titles,
        label_visibility="collapsed",
        help="Type to search for movies. You can search by title, year, or keywords."
    )

rating = 3.5
if selected_movie_title:
    with col_rate:
        rating = st.slider(
            "Rating",
            0.5, 5.0, 3.5, 0.5,
            label_visibility="collapsed",
            help="Drag to set your rating from 0.5 (worst) to 5.0 (best)"
        )

    with col_add:
        if st.button("Add Rating", type="primary", use_container_width=True):
            # Find movieId
            movie_row = movies_df.filter(pl.col("title") == selected_movie_title)
            if not movie_row.is_empty():
                movie_id = movie_row['movieId'][0]
                st.session_state.user_ratings[selected_movie_title] = (movie_id, rating)
                st.success(f"Rated '{selected_movie_title}' with {rating} ⭐")
                st.rerun()
else:
    # Show placeholder when no movie is selected
    with col_rate:
        st.markdown("<div style='height: 38px; display: flex; align-items: center; justify-content: center; color: #999;'>Select a movie first</div>", unsafe_allow_html=True)
    with col_add:
        st.button("Add Rating", disabled=True, use_container_width=True)

# Display current ratings
if st.session_state.user_ratings:
    st.divider()

    # Add rating summary
    num_ratings = len(st.session_state.user_ratings)
    avg_rating = np.mean([score for _, (_, score) in st.session_state.user_ratings.items()])

    col_summary1, col_summary2 = st.columns(2)
    with col_summary1:
        st.metric("Movies Rated", num_ratings)
    with col_summary2:
        st.metric("Average Rating", f"{avg_rating:.1f} ⭐")

    # Create a grid for rated movies
    rated_items = list(st.session_state.user_ratings.items())

    # Display in rows of 3
    for i in range(0, len(rated_items), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(rated_items):
                title, (mid, score) = rated_items[i + j]
                with cols[j]:
                    # Create a nicer rating card with dark theme support
                    st.markdown(f"""
                    <div class="rating-card" style='padding: 15px; border-radius: 8px;'>
                    <h4 style='margin-top: 0; margin-bottom: 8px; color: var(--text-color);'>{title}</h4>
                    <p style='margin-bottom: 5px; color: var(--text-color);'><b>Rating:</b> {score} ⭐</p>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("🗑️ Delete", key=f"remove_{mid}", use_container_width=True):
                        del st.session_state.user_ratings[title]
                        st.rerun()

    st.divider()

    # Get Recommendations - centered button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✨ Get Personalized Recommendations", type="primary", use_container_width=True):
            # Generate recommendations and store in session state
            with st.spinner("Generating personalized recommendations..."):
                dummy_ratings = []
                for title, (mid, score) in st.session_state.user_ratings.items():
                    if mid in movie_index.movie_to_idx:
                        idx = movie_index.movie_to_idx[mid]
                        dummy_ratings.append((idx, score))

                if not dummy_ratings:
                    st.warning("None of the rated movies are in the model's index.")
                    st.session_state.recommendations = None
                else:
                    user_bias, user_vector = get_dummy_user_factors(model, dummy_ratings)
                    rated_indices = [r[0] for r in dummy_ratings]

                    recommendations = recommend_for_dummy_user(
                        model,
                        train_movie,
                        user_bias,
                        user_vector,
                        rated_items=rated_indices,
                        top_n=20,
                        alpha=alpha,
                        min_item_ratings=100
                    )
                    st.session_state.recommendations = recommendations
                    st.rerun()

    # Display recommendations at full width (outside column constraint)
    if 'recommendations' in st.session_state and st.session_state.recommendations is not None:
        st.markdown("""
        <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #4CAF50;'>
        <h3 style='color: #2e7d32; margin-top: 0;'>Recommended for You</h3>
        <p style='color: #555;'>Based on your ratings and preferences, here are movies you might enjoy:</p>
        </div>
        """, unsafe_allow_html=True)

        # Grid View Implementation
        rec_cols = st.columns(4)

        for idx, (item_idx, score) in enumerate(st.session_state.recommendations):
            col = rec_cols[idx % 4]

            with col:
                if item_idx in movie_index.idx_to_movie:
                    original_id = movie_index.idx_to_movie[item_idx]
                    rec_movie_row = movies_df.filter(pl.col("movieId") == original_id)

                    if not rec_movie_row.is_empty():
                        title = rec_movie_row['title'][0]
                        genres = rec_movie_row['genres'][0]

                        # Fetch Poster
                        poster_url = poster_fetcher.get_poster_url(original_id, title)

                        # Create a movie card with consistent height and dark theme support
                        # All content must be inside a single markdown block to stay within the div
                        if genres and genres != "(no genres listed)":
                            # Format genres: replace pipes with commas, limit to 3 genres
                            genre_list = genres.split('|')
                            # Take first 3 genres and format nicely
                            if len(genre_list) > 3:
                                formatted_genres = ', '.join(genre_list[:3]) + '...'
                            else:
                                formatted_genres = ', '.join(genre_list)
                            genres_html = f"<p style='color: #888; font-size: 12px; margin-top: 5px;'>🎭 {formatted_genres}</p>"
                        else:
                            genres_html = "<p style='color: #888; font-size: 12px; margin-top: 5px;'></p>"

                        st.markdown(f"""
                        <div class="movie-card-container" style='background-color: var(--background-color); border-radius: 8px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; border: 1px solid rgba(255, 255, 255, 0.1);'>
                            <div class="movie-poster-container">
                                <img src="{poster_url}" style="width: 100%; height: 400px; object-fit: cover; border-radius: 8px;">
                            </div>
                            <div class="movie-info-container">
                                <h4 class="movie-title" style='color: var(--text-color); margin-top: 10px; margin-bottom: 5px;'>{title}</h4>
                                {genres_html}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

else:
    # Show a more engaging empty state
    st.markdown("""
    <div style='background-color: #fff3cd; padding: 30px; border-radius: 10px; border: 1px solid #ffeaa7; text-align: center; margin-top: 30px;'>
    <h3 style='color: #856404; margin-top: 0;'>📝 No Ratings Yet</h3>
    <p style='color: #856404; font-size: 16px;'>
    You haven't rated any movies yet. <b>Start by searching and rating 3-5 movies</b> above to get personalized recommendations!
    </p>
    <p style='color: #856404; font-size: 14px; margin-bottom: 0;'>
    💡 <b>Tip:</b> Rate movies you've seen and enjoyed to get better suggestions.
    </p>
    </div>
    """, unsafe_allow_html=True)
