import numpy as np

def get_dummy_user_factors(model, dummy_ratings):
    """
    Computes the user bias and latent vector for a dummy user based on given ratings.
    Simulates one iteration of the ALS user update for a new user.
    """

    item_indices, true_ratings = zip(*dummy_ratings)
    item_indices = np.array(list(item_indices), dtype=np.int32)
    true_ratings = np.array(list(true_ratings), dtype=np.float32)

    # Get existing item data from the trained model
    item_vecs = model.item_vector[item_indices]
    item_biases = model.item_biases[item_indices]

    # Initialize dummy user's bias and vector for calculation
    dummy_user_bias_temp = 0.0
    dummy_user_vector_temp = np.zeros(model.k)

    # Calculate dummy user bias (one iteration)
    residuals_for_bias = true_ratings - model.mu - item_biases - np.dot(item_vecs, dummy_user_vector_temp)
    dummy_user_bias = (model.lambda_reg * np.sum(residuals_for_bias)) / (model.tau + model.lambda_reg * len(true_ratings))

    # Calculate dummy user latent vector (one iteration)
    residuals_for_latent = true_ratings - model.mu - dummy_user_bias - item_biases
    A = model.lambda_reg * (item_vecs.T @ item_vecs) + model.tau * np.eye(model.k)
    Y = model.lambda_reg * (item_vecs.T @ residuals_for_latent)
    dummy_user_vector = np.linalg.solve(A, Y)

    return dummy_user_bias, dummy_user_vector

def recommend_for_dummy_user(model, train_movie,
                             user_bias, user_vector,
                             rated_items, top_n=10, alpha=1,
                             min_item_ratings=100):
    """
    Generates recommendations for a dummy user:
    - excluding items already rated by that user
    - excluding items with fewer than `min_item_ratings` ratings in the dataset
    """

    scores = alpha * (model.mu + model.item_biases) + np.dot(model.item_vector, user_vector)
    rated_items = set(rated_items)

    # Mask items with insufficient global ratings (< 100)
    for item_idx in range(model.n_movies):
        if len(train_movie[item_idx]) < min_item_ratings or item_idx in rated_items:
            scores[item_idx] = -np.inf

    top_items = np.argpartition(scores, -top_n)[-top_n:]
    top_items = top_items[np.argsort(scores[top_items])[::-1]]

    return [(int(i), float(scores[i])) for i in top_items]
