import numpy as np
from numba import njit, prange


@njit
def build_grouped_data(primary_ids, secondary_ids, ratings, n_groups):
    """
    Constructs a CSR-like structure (grouped data) from flat arrays.
    Returns:
        offsets: Starting index for each group in the sorted arrays.
        sorted_secondary: secondary_ids sorted by primary_ids.
        sorted_ratings: ratings sorted by primary_ids.
    """
    # Count items per group
    counts = np.zeros(n_groups, dtype=np.int32)
    n_items = len(primary_ids)

    for i in range(n_items):
        counts[primary_ids[i]] += 1

    # Calculate offsets (Cumulative Sum)
    offsets = np.zeros(n_groups + 1, dtype=np.int32)
    current_idx = 0
    for i in range(n_groups):
        offsets[i] = current_idx
        current_idx += counts[i]
    offsets[n_groups] = current_idx

    # Fill the output arrays
    insert_pos = offsets.copy()

    sorted_secondary = np.empty(n_items, dtype=np.int32)
    sorted_ratings = np.empty(n_items, dtype=np.float32)

    for i in range(n_items):
        p_id = primary_ids[i]
        pos = insert_pos[p_id]

        sorted_secondary[pos] = secondary_ids[i]
        sorted_ratings[pos] = ratings[i]

        insert_pos[p_id] += 1

    return offsets, sorted_secondary, sorted_ratings


@njit
def split_user_data(user_offsets, movie_ids, ratings, n_users, test_size):
    """
    Splits the user-grouped data into flat train and test arrays.
    Replicates logic: shuffle per user, ensure at least 1 train item if items > 0.
    """
    # First pass: Calculate exact sizes for train and test to pre-allocate
    total_train = 0
    total_test = 0

    for u in range(n_users):
        start = user_offsets[u]
        end = user_offsets[u + 1]
        n = end - start

        if n == 0:
            continue

        split_idx = int(n * (1 - test_size))
        if split_idx == 0 and n > 0:
            split_idx = 1

        total_train += split_idx
        total_test += (n - split_idx)

    # Train output
    train_u = np.empty(total_train, dtype=np.int32)
    train_m = np.empty(total_train, dtype=np.int32)
    train_r = np.empty(total_train, dtype=np.float32)

    # Test output
    test_u = np.empty(total_test, dtype=np.int32)
    test_m = np.empty(total_test, dtype=np.int32)
    test_r = np.empty(total_test, dtype=np.float32)

    train_ptr = 0
    test_ptr = 0

    # Second pass: Shuffle and Fill
    for u in range(n_users):
        start = user_offsets[u]
        end = user_offsets[u + 1]
        n = end - start

        if n == 0:
            continue

        # Get indices for this user's block
        indices = np.arange(n)
        np.random.shuffle(indices)

        split_idx = int(n * (1 - test_size))
        if split_idx == 0 and n > 0:
            split_idx = 1

        # Fill Train
        for k in range(split_idx):
            real_idx = start + indices[k]
            train_u[train_ptr] = u
            train_m[train_ptr] = movie_ids[real_idx]
            train_r[train_ptr] = ratings[real_idx]
            train_ptr += 1

        # Fill Test
        for k in range(split_idx, n):
            real_idx = start + indices[k]
            test_u[test_ptr] = u
            test_m[test_ptr] = movie_ids[real_idx]
            test_r[test_ptr] = ratings[real_idx]
            test_ptr += 1

    return (train_u, train_m, train_r), (test_u, test_m, test_r)


@njit(parallel=True)
def update_stage(offsets, indices, ratings,
                 fixed_vecs, fixed_biases,
                 target_vecs, target_biases,
                 n_targets, k, lambda_reg, tau, mu):
    """
    Updates Users or Items.
    """
    # Loop over every User or Item
    for i in prange(n_targets):
        start = offsets[i]
        end = offsets[i + 1]
        n_ratings = end - start

        if n_ratings == 0:
            continue

        # Slice data for this user/item
        idx_block = indices[start:end]
        rating_block = ratings[start:end]

        # Update Bias
        current_vec = target_vecs[i]
        sum_residuals = 0.0

        for j in range(n_ratings):
            other_idx = idx_block[j]
            r_ui = rating_block[j]

            # dot product
            dot_val = 0.0
            for d in range(k):
                dot_val += fixed_vecs[other_idx, d] * current_vec[d]

            pred = mu + fixed_biases[other_idx] + dot_val
            sum_residuals += (r_ui - pred)

        new_bias = (lambda_reg * sum_residuals) / (tau + lambda_reg * n_ratings)
        target_biases[i] = new_bias

        # Update Vector
        A = np.zeros((k, k), dtype=np.float32)
        Y = np.zeros(k, dtype=np.float32)

        # Add regularization to diagonal of A
        for d in range(k):
            A[d, d] = tau

        for j in range(n_ratings):
            other_idx = idx_block[j]
            r_ui = rating_block[j]

            residual = r_ui - mu - new_bias - fixed_biases[other_idx]
            weighted_residual = lambda_reg * residual

            fixed_v = fixed_vecs[other_idx]

            for r_idx in range(k):
                val_r = fixed_v[r_idx]
                Y[r_idx] += val_r * weighted_residual

                lam_val_r = lambda_reg * val_r
                for c in range(k):
                    A[r_idx, c] += lam_val_r * fixed_v[c]

        new_vec = np.linalg.solve(A, Y)
        target_vecs[i] = new_vec


@njit(parallel=True)
def compute_rmse_loss(offsets, indices, ratings,
                      user_vecs, user_biases,
                      item_vecs, item_biases,
                      mu, n_users, lambda_reg, tau):
    """
    Computes SSE and Regularization term.
    """
    sse = 0.0
    count = 0

    for u in prange(n_users):
        start = offsets[u]
        end = offsets[u + 1]
        if start == end:
            continue

        u_bias = user_biases[u]
        u_vec = user_vecs[u]

        for j in range(start, end):
            m = indices[j]
            r = ratings[j]

            # Dot product
            dot_val = 0.0
            for d in range(len(u_vec)):
                dot_val += u_vec[d] * item_vecs[m, d]

            pred = mu + u_bias + item_biases[m] + dot_val
            diff = r - pred
            sse += diff * diff
            count += 1

    return sse, count


@njit
def compute_topk_metrics_subset(target_users, n_items, k, threshold,
                                tr_offsets, tr_indices,
                                te_offsets, te_indices, te_ratings,
                                user_vecs, user_biases,
                                item_vecs, item_biases, mu):

    precision_sum = 0.0
    recall_sum = 0.0
    valid_user_count = 0

    n_targets = len(target_users)

    # Loop only over the sampled users
    for i in range(n_targets):
        u = target_users[i]

        # Check Ground Truth
        start_te = te_offsets[u]
        end_te = te_offsets[u + 1]
        if start_te == end_te:
            continue

        n_relevant = 0
        for idx in range(start_te, end_te):
            if te_ratings[idx] >= threshold:
                n_relevant += 1
        if n_relevant == 0:
            continue

        # Calculate Scores
        u_vec = user_vecs[u]
        u_bias = user_biases[u]
        base_score = mu + u_bias

        scores = np.empty(n_items, dtype=np.float32)

        for m in range(n_items):
            dot = 0.0
            for d in range(len(u_vec)):
                dot += u_vec[d] * item_vecs[m, d]
            scores[m] = base_score + item_biases[m] + dot

        # Mask Training Items (Set to -infinity)
        start_tr = tr_offsets[u]
        end_tr = tr_offsets[u + 1]
        for idx in range(start_tr, end_tr):
            m = tr_indices[idx]
            scores[m] = -99999.0

        # Fast Top-K Selection (Argpartition)
        top_k_indices = np.argsort(scores)[::-1][:k]

        # Calculate Hits
        hits = 0
        for rec_idx in top_k_indices:
            for idx in range(start_te, end_te):
                if te_indices[idx] == rec_idx and te_ratings[idx] >= threshold:
                    hits += 1
                    break

        precision_sum += hits / k
        recall_sum += hits / n_relevant
        valid_user_count += 1

    return precision_sum, recall_sum, valid_user_count


def convert_ragged_to_csr(ragged_data, n_rows):
    """
    Converts list-of-lists to CSR format (offsets, indices, values).
    """
    lengths = np.array([len(x) for x in ragged_data], dtype=np.int32)

    offsets = np.zeros(n_rows + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(lengths)

    valid_rows = [row for row in ragged_data if len(row) > 0]

    if len(valid_rows) == 0:
        indices = np.zeros(0, dtype=np.int32)
        values = np.zeros(0, dtype=np.float32)
    else:
        flat_data = np.concatenate(valid_rows, axis=0)

        indices = flat_data[:, 0].astype(np.int32)
        values = flat_data[:, 1].astype(np.float32)

    return offsets, indices, values
