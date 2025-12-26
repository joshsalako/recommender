import numpy as np
import os
import pickle
import time

class ALSBiases:
    def __init__(self, train_user, test_user, train_movie,
                 test_movie, n_users, n_movies, lambda_reg=25, tau=5.0):

        self.train_user = train_user
        self.test_user = test_user
        self.train_movie = train_movie
        self.test_movie = test_movie
        self.n_users = n_users
        self.n_movies = n_movies
        self.tau = tau

        self.lambda_reg = lambda_reg
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_movies)

        self.train_rmse_history = []
        self.test_rmse_history = []
        self.train_loss_history = []

        total_rating = sum(r for u_ratings in self.train_user for _, r in u_ratings)
        num_ratings = sum(len(u_ratings) for u_ratings in self.train_user)
        self.mu = total_rating / num_ratings if num_ratings > 0 else 0

    def predict(self, user_idx, item_idx):
        return self.mu + self.user_biases[user_idx] + self.item_biases[item_idx]

    def _calculate_rmse(self, data_by_user):
        sse = 0
        count = 0
        for user_idx, ratings in enumerate(data_by_user):
            for item_idx, true_rating in ratings:
                prediction = self.predict(user_idx, item_idx)
                sse += (true_rating - prediction) ** 2
                count += 1
        return np.sqrt(sse / count) if count > 0 else 0.0

    def _calculate_loss(self):
        sse = 0
        for user_idx, ratings in enumerate(self.train_user):
            for item_idx, true_rating in ratings:
                prediction = self.predict(user_idx, item_idx)
                sse += (true_rating - prediction) ** 2

        reg_term = self.tau * (np.sum(self.user_biases**2) + np.sum(self.item_biases**2))
        return self.lambda_reg * sse + reg_term

    def train(self, n_epochs=10, save_dir="", prefix="ALSBiases"):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        print(f"\nStarting training for {n_epochs} epochs...")
        print(f"Hyperparameters: lambda={self.lambda_reg}, tau={self.tau}")

        start_epoch = len(self.train_loss_history)

        for epoch in range(start_epoch, start_epoch + n_epochs):
            for m in range(self.n_users):
                residuals_sum = sum(r - self.mu - self.item_biases[i] for i, r in self.train_user[m])
                count = len(self.train_user[m])
                self.user_biases[m] = (self.lambda_reg * residuals_sum) / (self.tau + self.lambda_reg * count)

            for n in range(self.n_movies):
                residuals_sum = sum(r - self.mu - self.user_biases[u] for u, r in self.train_movie[n])
                count = len(self.train_movie[n])
                self.item_biases[n] = (self.lambda_reg * residuals_sum) / (self.tau + self.lambda_reg * count)

            train_loss = self._calculate_loss()
            train_rmse = self._calculate_rmse(self.train_user)
            test_rmse = self._calculate_rmse(self.test_user)

            self.train_loss_history.append(train_loss)
            self.train_rmse_history.append(train_rmse)
            self.test_rmse_history.append(test_rmse)

            print(f"Epoch {epoch+1}/{start_epoch + n_epochs} | "
                  f"Loss: {train_loss:.2f} | "
                  f"Train RMSE: {train_rmse:.4f} | "
                  f"Test RMSE: {test_rmse:.4f}")

            if save_dir and ((epoch + 1) % 5 == 0 or epoch == start_epoch + n_epochs - 1):
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                self._save_checkpoint(save_dir, prefix, epoch, timestamp)

    def evaluate(self, k=10, threshold=3.5, n_eval_users=3000):
        """
        Evaluate the model using Top-K metrics (Precision and Recall).
        Note: ALSBiases is a bias-only model, so ranking is based on item biases.
        """
        # For bias-only models, "recommendation" is just ranking items by bias.
        # This is computationally expensive to do for all users if not careful,
        # but we follow the same interface as ALS class.

        # NOTE: A more efficient implementation would be good here, but for consistency
        # we can iterate. However, since user bias is constant for all items for a user,
        # the ranking of items is purely determined by item_biases.

        # Simplified evaluation for bias model:
        # Sort all items by item_bias (descending)
        top_items = np.argsort(self.item_biases)[::-1][:k]

        precision_sum = 0.0
        recall_sum = 0.0
        valid_user_count = 0

        if n_eval_users is not None and n_eval_users < self.n_users:
            rng = np.random.default_rng(42)
            target_users = rng.choice(self.n_users, size=n_eval_users, replace=False).astype(np.int32)
        else:
            target_users = np.arange(self.n_users, dtype=np.int32)

        for u in target_users:
            # Ground Truth
            # Note: We need to check self.test_user which is ragged array
            # test_user structure: array of arrays of (item_idx, rating)
            if len(self.test_user[u]) == 0:
                continue

            relevant_items = {int(item) for item, rating in self.test_user[u] if rating >= threshold}
            if not relevant_items:
                continue

            n_relevant = len(relevant_items)

            # Hits
            hits = sum(1 for item in top_items if item in relevant_items)

            precision_sum += hits / k
            recall_sum += hits / n_relevant
            valid_user_count += 1

        precision = precision_sum / valid_user_count if valid_user_count > 0 else 0.0
        recall = recall_sum / valid_user_count if valid_user_count > 0 else 0.0

        print(f"Precision@{k}: {precision:.4f} | Recall@{k}: {recall:.4f}")
        return precision, recall

    def _save_checkpoint(self, save_dir, prefix, epoch, timestamp):
        filename = (
            f"{prefix}_epoch={epoch+1}_lambda={self.lambda_reg}"
            f"_tau={self.tau}_{self.mu:.4f}_{timestamp}.pkl"
        )
        filepath = os.path.join(save_dir, filename)

        checkpoint = {
            "epoch": epoch + 1,
            "tau": self.tau,
            "lambda_reg": self.lambda_reg,
            "mu": self.mu,
            "user_biases": self.user_biases,
            "item_biases": self.item_biases,
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
            lambda_reg=ckpt["lambda_reg"],
            tau=ckpt["tau"],
            train_user=train_user,
            test_user=test_user,
            train_movie=train_movie,
            test_movie=test_movie,
        )

        model.mu = ckpt["mu"]
        model.user_biases = ckpt["user_biases"]
        model.item_biases = ckpt["item_biases"]
        model.train_loss_history = ckpt.get("train_loss_history", [])
        model.train_rmse_history = ckpt.get("train_rmse_history", [])
        model.test_rmse_history = ckpt.get("test_rmse_history", [])

        return model
