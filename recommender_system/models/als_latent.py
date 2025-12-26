import numpy as np
import os
import pickle
import time

class ALSLatent:
    def __init__(self, train_user, test_user, train_movie,
                 test_movie, n_users, n_movies, k=10, lambda_reg=2.5, tau=0.05):

        self.train_user = train_user
        self.test_user = test_user
        self.train_movie = train_movie
        self.test_movie = test_movie
        self.n_users = n_users
        self.n_movies = n_movies
        self.k = k
        self.tau = tau

        self.lambda_reg = lambda_reg
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_movies)
        self.user_vector = np.random.normal(0, 1 / np.sqrt(self.k), size=(self.n_users, self.k))
        self.item_vector = np.random.normal(0, 1 / np.sqrt(self.k), size=(self.n_movies, self.k))

        self.train_rmse_history = []
        self.test_rmse_history = []
        self.train_loss_history = []

        total_rating = sum(r for u_ratings in self.train_user for _, r in u_ratings)
        num_ratings = sum(len(u_ratings) for u_ratings in self.train_user)
        self.mu = total_rating / num_ratings if num_ratings > 0 else 0

    def predict(self, user_idx, item_idx):
        return self.mu + self.user_biases[user_idx] + self.item_biases[item_idx] + \
        np.dot(self.user_vector[user_idx], self.item_vector[item_idx])

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

        reg_term = self.tau * (np.sum(self.user_biases**2) +
                                      np.sum(self.item_biases**2) +
                                      np.sum(self.user_vector**2) +
                                      np.sum(self.item_vector**2)
                                      )
        return self.lambda_reg * sse + reg_term

    def train(self, n_epochs=10, save_dir="", prefix="ALSLatent"):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        print(f"\nStarting training for {n_epochs} epochs...")
        print(f"Hyperparameters: lambda={self.lambda_reg}, mu={self.mu:.4f}, tau={self.tau:.4f}")

        start_epoch = len(self.train_loss_history)

        for epoch in range(start_epoch, start_epoch + n_epochs):
            for m in range(self.n_users):
                # Update user bias
                residuals_sum = sum(r - self.mu - self.item_biases[i] -
                                    np.dot(self.user_vector[m], self.item_vector[i])
                                    for i, r in self.train_user[m])
                count = len(self.train_user[m])
                self.user_biases[m] = residuals_sum / (self.lambda_reg + count)

                # Update user latent
                A = np.zeros((self.k, self.k))
                Y = np.zeros(self.k)
                I = np.identity(self.k)

                # Loop over items rated by user
                for i, r in self.train_user[m]:
                    item_vec_i = self.item_vector[i]

                    A += self.lambda_reg * np.outer(item_vec_i, item_vec_i)
                    residual_for_vector = r - self.mu - self.user_biases[m] - self.item_biases[i]
                    Y += self.lambda_reg * item_vec_i * residual_for_vector

                A += self.tau * I
                self.user_vector[m] = np.linalg.solve(A, Y)

            for n in range(self.n_movies):
                # Update item bias
                residuals_sum = sum(r - self.mu - self.user_biases[u] -
                                    np.dot(self.user_vector[u], self.item_vector[n])
                                    for u, r in self.train_movie[n])
                count = len(self.train_movie[n])
                self.item_biases[n] = residuals_sum / (self.lambda_reg + count)

                # Update item latent
                A = np.zeros((self.k, self.k))
                Y = np.zeros(self.k)
                I = np.identity(self.k)

                # Loop over users that rated the movie
                for u, r in self.train_movie[n]:
                    user_vec_u = self.user_vector[u]

                    A += self.lambda_reg * np.outer(user_vec_u, user_vec_u)
                    residual_for_vector = r - self.mu - self.user_biases[u] - self.item_biases[n]
                    Y += self.lambda_reg * user_vec_u * residual_for_vector

                A += self.tau * I
                self.item_vector[n] = np.linalg.solve(A, Y)

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
        """
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
            # Check test_user ragged array
            if len(self.test_user[u]) == 0:
                continue

            relevant_items = {int(item) for item, rating in self.test_user[u] if rating >= threshold}
            if not relevant_items:
                continue

            n_relevant = len(relevant_items)

            # Predict scores for all items
            # Optim: Vectorized prediction for all items for a single user
            # score = mu + user_bias + item_biases + user_vec . item_vecs
            scores = (self.mu + self.user_biases[u] + self.item_biases +
                      self.item_vector.dot(self.user_vector[u]))

            # Mask training items
            train_items = {int(item) for item, _ in self.train_user[u]}
            scores[list(train_items)] = -np.inf

            # Top-K
            top_k_indices = np.argsort(scores)[::-1][:k]

            # Hits
            hits = sum(1 for item in top_k_indices if item in relevant_items)

            precision_sum += hits / k
            recall_sum += hits / n_relevant
            valid_user_count += 1

        precision = precision_sum / valid_user_count if valid_user_count > 0 else 0.0
        recall = recall_sum / valid_user_count if valid_user_count > 0 else 0.0

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
        )

        model.mu = ckpt["mu"]
        model.user_biases = ckpt["user_biases"]
        model.item_biases = ckpt["item_biases"]
        model.user_vector = ckpt["user_vector"]
        model.item_vector = ckpt["item_vector"]
        model.train_loss_history = ckpt.get("train_loss_history", [])
        model.train_rmse_history = ckpt.get("train_rmse_history", [])
        model.test_rmse_history = ckpt.get("test_rmse_history", [])

        return model
