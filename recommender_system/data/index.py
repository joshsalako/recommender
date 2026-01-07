import numpy as np
import gc
import pickle
from ..utils.numba_ops import build_grouped_data, split_user_data

class MovieIndex:
    def __init__(self, data):
        data = np.asarray(data, dtype=np.float32)

        unique_users, user_indices = np.unique(data[:, 0].astype(np.int32), return_inverse=True)
        unique_movies, movie_indices = np.unique(data[:, 1].astype(np.int32), return_inverse=True)
        ratings = data[:, 2].astype(np.float32)

        del data
        gc.collect()

        self.n_users = len(unique_users)
        self.n_movies = len(unique_movies)

        # Build Lookup Dicts
        self.user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.movie_to_idx = {mid: i for i, mid in enumerate(unique_movies)}
        # Reverse lookups
        self.idx_to_user = {i: uid for i, uid in enumerate(unique_users)}
        self.idx_to_movie = {i: mid for i, mid in enumerate(unique_movies)}

        del unique_users, unique_movies

        # --- Build Data Structures using Numba ---

        # 1. Build User-Centric Data
        # This returns flat sorted arrays and offsets (CSR format)
        self._u_offsets, self._u_movies, self._u_ratings = build_grouped_data(
            user_indices, movie_indices, ratings, self.n_users
        )

        # 2. Build Movie-Centric Data
        self._m_offsets, self._m_users, self._m_ratings = build_grouped_data(
            movie_indices, user_indices, ratings, self.n_movies
        )

        # 3. Convert to Object Arrays (Ragged Arrays)
        self.data_by_user = np.empty(self.n_users, dtype=object)
        for i in range(self.n_users):
            start, end = self._u_offsets[i], self._u_offsets[i+1]
            if end > start:
                m_col = self._u_movies[start:end].astype(np.float32)
                r_col = self._u_ratings[start:end]
                self.data_by_user[i] = np.column_stack((m_col, r_col))
            else:
                self.data_by_user[i] = np.array([], dtype=np.float32)

        self.data_by_movie = np.empty(self.n_movies, dtype=object)
        for i in range(self.n_movies):
            start, end = self._m_offsets[i], self._m_offsets[i+1]
            if end > start:
                u_col = self._m_users[start:end].astype(np.float32)
                r_col = self._m_ratings[start:end]
                self.data_by_movie[i] = np.column_stack((u_col, r_col))
            else:
                self.data_by_movie[i] = np.array([], dtype=np.float32)

        gc.collect()

    def get_by_user(self, user_id):
        """Return all (movie_idx, rating) pairs for a user."""
        if user_id not in self.user_to_idx:
            return []
        idx = self.user_to_idx[user_id]
        result = self.data_by_user[idx]
        return [(int(row[0]), float(row[1])) for row in result] if len(result) > 0 else []

    def get_by_movie(self, movie_id):
        """Return all (user_idx, rating) pairs for a movie."""
        if movie_id not in self.movie_to_idx:
            return []
        idx = self.movie_to_idx[movie_id]
        result = self.data_by_movie[idx]
        return [(int(row[0]), float(row[1])) for row in result] if len(result) > 0 else []

    def train_test_split(self, test_size=0.2):
        """Split data into train/test by user while preserving structure."""

        # 1. Perform Split using Numba
        (tr_u, tr_m, tr_r), (te_u, te_m, te_r) = split_user_data(
            self._u_offsets, self._u_movies, self._u_ratings, self.n_users, test_size
        )

        # 2. Build Object Arrays for Users (Train/Test)
        tr_u_offsets, tr_u_m_vals, tr_u_r_vals = build_grouped_data(tr_u, tr_m, tr_r, self.n_users)
        te_u_offsets, te_u_m_vals, te_u_r_vals = build_grouped_data(te_u, te_m, te_r, self.n_users)

        train_user_data = self._flat_to_object_array(tr_u_offsets, tr_u_m_vals, tr_u_r_vals, self.n_users)
        test_user_data = self._flat_to_object_array(te_u_offsets, te_u_m_vals, te_u_r_vals, self.n_users)

        # 3. Build Object Arrays for Movies (Train/Test)
        tr_m_offsets, tr_m_u_vals, tr_m_r_vals = build_grouped_data(tr_m, tr_u, tr_r, self.n_movies)
        te_m_offsets, te_m_u_vals, te_m_r_vals = build_grouped_data(te_m, te_u, te_r, self.n_movies)

        train_movie_data = self._flat_to_object_array(tr_m_offsets, tr_m_u_vals, tr_m_r_vals, self.n_movies)
        test_movie_data = self._flat_to_object_array(te_m_offsets, te_m_u_vals, te_m_r_vals, self.n_movies)

        gc.collect()
        return (train_user_data, test_user_data, train_movie_data, test_movie_data)

    def get_item_rating_counts(self):
        """Get array of rating counts for each movie."""
        counts = np.zeros(self.n_movies, dtype=np.int32)
        for i in range(self.n_movies):
            counts[i] = len(self.data_by_movie[i])
        return counts

    def _flat_to_object_array(self, offsets, col_ids, ratings, n_groups):
        """Helper to convert flat CSR arrays back to the legacy object-array format."""
        out = np.empty(n_groups, dtype=object)
        for i in range(n_groups):
            start, end = offsets[i], offsets[i+1]
            if end > start:
                c_dat = col_ids[start:end].astype(np.float32)
                r_dat = ratings[start:end]
                out[i] = np.column_stack((c_dat, r_dat))
            else:
                out[i] = np.array([], dtype=np.float32)
        return out


class LightweightMovieIndex:
    """Lightweight index for inference-only use cases."""

    def __init__(self, movie_to_idx=None, idx_to_movie=None, item_rating_counts=None):
        """
        Initialize with minimal data needed for inference.

        Args:
            movie_to_idx: Dictionary mapping movieId to internal index
            idx_to_movie: Dictionary mapping internal index to movieId
            item_rating_counts: Array of rating counts for each movie
        """
        self.movie_to_idx = movie_to_idx if movie_to_idx is not None else {}
        self.idx_to_movie = idx_to_movie if idx_to_movie is not None else {}
        self.item_rating_counts = item_rating_counts if item_rating_counts is not None else np.array([], dtype=np.int32)
        self.n_movies = len(self.movie_to_idx)

    @classmethod
    def from_full_index(cls, full_index):
        """Create lightweight index from a full MovieIndex."""
        return cls(
            movie_to_idx=full_index.movie_to_idx.copy(),
            idx_to_movie=full_index.idx_to_movie.copy(),
            item_rating_counts=full_index.get_item_rating_counts()
        )

    @classmethod
    def load(cls, index_path, counts_path):
        """Load lightweight index from saved files."""
        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)

        item_rating_counts = np.load(counts_path, mmap_mode='r')

        return cls(
            movie_to_idx=index_data['movie_to_idx'],
            idx_to_movie=index_data['idx_to_movie'],
            item_rating_counts=item_rating_counts
        )

    def save(self, index_path, counts_path):
        """Save lightweight index to files."""
        # Save index dictionaries
        index_data = {
            'movie_to_idx': self.movie_to_idx,
            'idx_to_movie': self.idx_to_movie
        }

        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)

        # Save rating counts as numpy array
        np.save(counts_path, self.item_rating_counts)

    def get_item_rating_count(self, movie_idx):
        """Get rating count for a specific movie index."""
        if 0 <= movie_idx < len(self.item_rating_counts):
            return self.item_rating_counts[movie_idx]
        return 0
