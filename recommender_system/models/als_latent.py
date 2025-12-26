import numpy as np

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
        return np.sqrt(sse / count)

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

    def train(self, n_epochs=10):
        print(f"\nStarting training for {n_epochs} epochs...")
        print(f"Hyperparameters: lambda={self.lambda_reg}, mu={self.mu:.4f}, tau={self.tau:.4f}")

        for epoch in range(n_epochs):
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

            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Loss: {train_loss:.2f} | "
                  f"Train RMSE: {train_rmse:.4f} | "
                  f"Test RMSE: {test_rmse:.4f}")
