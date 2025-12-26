import numpy as np
import pickle
import time
import os
from ..utils.numba_ops import (
    update_stage,
    compute_rmse_loss,
    compute_topk_metrics_subset,
    convert_ragged_to_csr,
)


class ALS:
    def __init__(self, train_user, test_user, train_movie,
                 test_movie, n_users, n_movies, k=10, lambda_reg=2.5,
                 tau=0.1, n_jobs=-1):

        self.n_users = n_users
        self.n_movies = n_movies
        self.k = k
        self.lambda_reg = lambda_reg
        self.tau = tau
        self.n_jobs = n_jobs

        self.user_biases = np.zeros(self.n_users, dtype=np.float32)
        self.item_biases = np.zeros(self.n_movies, dtype=np.float32)
        self.user_vector = np.random.normal(0, 1 / np.sqrt(self.k), size=(self.n_users, self.k)).astype(np.float32)
        self.item_vector = np.random.normal(0, 1 / np.sqrt(self.k), size=(self.n_movies, self.k)).astype(np.float32)

        self.train_rmse_history = []
        self.test_rmse_history = []
        self.train_loss_history = []

        if train_user is not None:
            self.tr_u_offsets, self.tr_u_indices, self.tr_u_ratings = convert_ragged_to_csr(train_user, n_users)

            total_rating = np.sum(self.tr_u_ratings)
            num_ratings = len(self.tr_u_ratings)
            self.mu = float(total_rating / num_ratings) if num_ratings > 0 else 0.0
        else:
            self.mu = 0.0
            self.tr_u_offsets = np.zeros(n_users + 1, dtype=np.int32)
            self.tr_u_indices = np.zeros(0, dtype=np.int32)
            self.tr_u_ratings = np.zeros(0, dtype=np.float32)

        if train_movie is not None:
            self.tr_m_offsets, self.tr_m_indices, self.tr_m_ratings = convert_ragged_to_csr(train_movie, n_movies)

        if test_user is not None:
            self.te_u_offsets, self.te_u_indices, self.te_u_ratings = convert_ragged_to_csr(test_user, n_users)

    def predict(self, user_idx, item_idx):
        return (self.mu + self.user_biases[user_idx] + self.item_biases[item_idx] +
                np.dot(self.user_vector[user_idx], self.item_vector[item_idx]))

    def train(self, n_epochs=10, save_dir="", prefix="ALS"):
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nStarting training for {n_epochs} epochs...")
        print(f"Hyperparameters: lambda={self.lambda_reg}, mu={self.mu:.4f}, tau = {self.tau:.4f}, k={self.k}")

        start_epoch = len(self.train_loss_history)

        k = int(self.k)
        lam = float(self.lambda_reg)
        tau = float(self.tau)
        mu = float(self.mu)

        for epoch in range(start_epoch, start_epoch + n_epochs):
            # Update Users
            update_stage(
                self.tr_u_offsets, self.tr_u_indices, self.tr_u_ratings,
                self.item_vector, self.item_biases,
                self.user_vector, self.user_biases,
                self.n_users, k, lam, tau, mu
            )

            # Update Items
            update_stage(
                self.tr_m_offsets, self.tr_m_indices, self.tr_m_ratings,
                self.user_vector, self.user_biases,
                self.item_vector, self.item_biases,
                self.n_movies, k, lam, tau, mu
            )

            # Compute Metrics
            train_sse, train_count = compute_rmse_loss(
                self.tr_u_offsets, self.tr_u_indices, self.tr_u_ratings,
                self.user_vector, self.user_biases,
                self.item_vector, self.item_biases,
                mu, self.n_users, lam, tau
            )

            reg_term = tau * (
                np.sum(self.user_biases ** 2) +
                np.sum(self.item_biases ** 2) +
                np.sum(self.user_vector ** 2) +
                np.sum(self.item_vector ** 2)
            )

            train_rmse = np.sqrt(train_sse / train_count) if train_count > 0 else 0.0
            train_loss = (lam * train_sse) + reg_term

            test_sse, test_count = compute_rmse_loss(
                self.te_u_offsets, self.te_u_indices, self.te_u_ratings,
                self.user_vector, self.user_biases,
                self.item_vector, self.item_biases,
                mu, self.n_users, lam, tau
            )
            test_rmse = np.sqrt(test_sse / test_count) if test_count > 0 else 0.0

            self.train_loss_history.append(train_loss)
            self.train_rmse_history.append(train_rmse)
            self.test_rmse_history.append(test_rmse)

            if (epoch + 1) % 5 == 0 or epoch == start_epoch + n_epochs - 1:
                print(
                    f"Epoch {epoch + 1}/{start_epoch + n_epochs} | "
                    f"Loss: {train_loss:.2f} | "
                    f"Train RMSE: {train_rmse:.4f} | "
                    f"Test RMSE: {test_rmse:.4f}"
                )

                self._save_checkpoint(save_dir, prefix, epoch, timestamp=time.strftime("%Y%m%d-%H%M%S"))

    def evaluate(self, k=10, threshold=3.5, n_eval_users=3000):
        """
        Calculates Precision@K and Recall@K.
        n_eval_users: If set, only evaluates on a random subset of users.
        """

        if n_eval_users is not None and n_eval_users < self.n_users:
            rng = np.random.default_rng(42)
            target_users = rng.choice(self.n_users, size=n_eval_users, replace=False).astype(np.int32)
        else:
            target_users = np.arange(self.n_users, dtype=np.int32)

        p_sum, r_sum, count = compute_topk_metrics_subset(
            target_users, self.n_movies, int(k), float(threshold),
            self.tr_u_offsets, self.tr_u_indices,
            self.te_u_offsets, self.te_u_indices, self.te_u_ratings,
            self.user_vector, self.user_biases,
            self.item_vector, self.item_biases, self.mu
        )

        precision = p_sum / count if count > 0 else 0.0
        recall = r_sum / count if count > 0 else 0.0

        print(f"Precision@{k}: {precision:.4f} | Recall@{k}: {recall:.4f}")

        return precision, recall

    def _save_checkpoint(self, save_dir, prefix, epoch, timestamp):
        filename = (
            f"{prefix}_epoch={epoch+1}_k={self.k}_lambda={self.lambda_reg}"
            f"_tau={self.tau}_{self.mu:.4f}_{timestamp}.pkl"
        )
        filepath = os.path.join(save_dir, filename)

        checkpoint = {
            "epoch": epoch + 1,
            "k": self.k,
            "tau": self.tau,
            "lambda_reg": self.lambda_reg,
            "mu": self.mu,
            "user_biases": self.user_biases,
            "item_biases": self.item_biases,
            "user_vector": self.user_vector,
            "item_vector": self.item_vector,
            "train_loss_history": self.train_loss_history.copy(),
            "train_rmse_history": self.train_rmse_history.copy(),
            "test_rmse_history": self.test_rmse_history.copy(),
            "n_users": self.n_users,
            "n_movies": self.n_movies,
        }
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved model checkpoint -> {filepath}\n")

    @classmethod
    def load_checkpoint(cls, train_user, test_user, train_movie, test_movie, path, n_jobs=1):
        with open(path, "rb") as f:
            ckpt = pickle.load(f)

        model = cls(
            n_users=ckpt["n_users"],
            n_movies=ckpt["n_movies"],
            k=ckpt["k"],
            lambda_reg=ckpt["lambda_reg"],
            tau=ckpt["tau"],
            train_user=train_user,
            test_user=test_user,
            train_movie=train_movie,
            test_movie=test_movie,
            n_jobs=n_jobs
        )

        model.mu = ckpt["mu"]
        model.user_biases = ckpt["user_biases"].astype(np.float32)
        model.item_biases = ckpt["item_biases"].astype(np.float32)
        model.user_vector = ckpt["user_vector"].astype(np.float32)
        model.item_vector = ckpt["item_vector"].astype(np.float32)
        model.train_loss_history = ckpt.get("train_loss_history", [])
        model.train_rmse_history = ckpt.get("train_rmse_history", [])
        model.test_rmse_history = ckpt.get("test_rmse_history", [])

        return model
