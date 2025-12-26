import numpy as np

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
        return np.sqrt(sse / count)

    def _calculate_loss(self):
        sse = 0
        for user_idx, ratings in enumerate(self.train_user):
            for item_idx, true_rating in ratings:
                prediction = self.predict(user_idx, item_idx)
                sse += (true_rating - prediction) ** 2

        reg_term = self.tau * (np.sum(self.user_biases**2) + np.sum(self.item_biases**2))
        return self.lambda_reg * sse + reg_term

    def train(self, n_epochs=10):
        print(f"\nStarting training for {n_epochs} epochs...")
        print(f"Hyperparameters: lambda={self.lambda_reg}, tau={self.tau}")

        for epoch in range(n_epochs):
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

            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Loss: {train_loss:.2f} | "
                  f"Train RMSE: {train_rmse:.4f} | "
                  f"Test RMSE: {test_rmse:.4f}")
