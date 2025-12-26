import os
import requests
import polars as pl
from tmdbv3api import TMDb, Movie
import streamlit as st
from ..config import DATASET_PATH

class PosterFetcher:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("TMDB_API_KEY")
        self.tmdb = None
        self.movie_api = None
        self.links_df = None
        self.poster_cache = {}  # {movieId: poster_url}
        self.use_api = False

        if self.api_key:
            self._init_tmdb()
            self.use_api = True

        self._load_links()

    def _init_tmdb(self):
        try:
            self.tmdb = TMDb()
            self.tmdb.api_key = self.api_key
            self.tmdb.language = 'en'
            self.movie_api = Movie()
        except Exception as e:
            print(f"Error initializing TMDB: {e}")
            self.use_api = False

    def _load_links(self):
        links_path = os.path.join(DATASET_PATH, 'links.csv')
        try:
            if os.path.exists(links_path):
                self.links_df = pl.read_csv(links_path)
            else:
                # Fallback path if DATASET_PATH is weird (e.g. not expanded)
                # Try to find it relative to current working directory if not found
                cwd_path = os.path.join("data_files", "ml-32m", "links.csv")
                if os.path.exists(cwd_path):
                    self.links_df = pl.read_csv(cwd_path)
                else:
                    self.links_df = None
        except Exception as e:
            print(f"Error loading links.csv: {e}")
            self.links_df = None

    def get_poster_url(self, movie_id, title=None):
        """
        Get poster URL for a given movie ID.
        1. Check cache.
        2. If API key exists, try TMDB.
        3. Fallback to IMDbOT (free, no key).
        4. Fallback to placeholder.
        """
        if movie_id in self.poster_cache:
            return self.poster_cache[movie_id]

        url = None

        # 1. Try TMDB if configured
        if self.use_api and self.links_df is not None:
            tmdb_id = self._get_tmdb_id(movie_id)
            if tmdb_id:
                url = self._fetch_from_tmdb(tmdb_id)

        # 2. Try IMDbOT (fallback) if no API key or TMDB failed
        if not url and title:
            url = self._fetch_from_imdbot(title)

        # 3. Final fallback
        if not url:
            url = "https://via.placeholder.com/300x450?text=No+Poster"

        self.poster_cache[movie_id] = url
        return url

    def _get_tmdb_id(self, movie_id):
        if self.links_df is None:
            return None
        try:
            row = self.links_df.filter(pl.col("movieId") == movie_id)
            if not row.is_empty():
                return row["tmdbId"][0]
        except Exception:
            return None
        return None

    def _fetch_from_tmdb(self, tmdb_id):
        try:
            # tmdb_id might be float/int in dataframe, ensure it's handled
            if not tmdb_id:
                return None

            movie = self.movie_api.details(int(tmdb_id))
            if hasattr(movie, 'poster_path') and movie.poster_path:
                return f"https://image.tmdb.org/t/p/w500{movie.poster_path}"
        except Exception:
            pass
        return None

    def _fetch_from_imdbot(self, title):
        """
        Fetch from IMDbOT (unofficial free API).
        Docs: https://github.com/BullyWiiPlaza/imdb-api
        """
        try:
            # Clean title for better search (remove year)
            clean_title = title.split('(')[0].strip()
            response = requests.get(
                f"https://imdb.iamidiotareyoutoo.com/search?q={clean_title}",
                timeout=2
            )
            if response.status_code == 200:
                data = response.json()
                if data and 'description' in data and len(data['description']) > 0:
                    return data['description'][0].get('#IMG_POSTER')
        except Exception:
            pass
        return None
