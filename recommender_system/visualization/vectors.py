import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import pandas as pd
import numpy as np
import re
import os
from sklearn.decomposition import PCA
from .. import config

def visualize_vectors(model, movies_df, movie_index):
    movie_title_list = [
        "Toy Story (1995)",
        "Toy Story 3 (2010)",
        "Lion King, The (1994)",
        "Finding Nemo (2003)",
        "Shrek (2001)",
        "Star Wars: Episode IV - A New Hope (1977)",
        "Blade Runner (1982)",
        "Matrix, The (1999)",
        "Lord of the Rings: The Return of the King, The (2003)",
        "Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)",
        "Pulp Fiction (1994)",
        "Goodfellas (1990)",
        "The Shawshank Redemption (1994)",
        "Forrest Gump (1994)",
        "Eternal Sunshine of the Spotless Mind (2004)",
        "Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)",
        "Spirited Away (Sen to Chihiro no kamikakushi) (2001)",
        "Princess Mononoke (Mononoke-hime) (1997)",
        "Interstellar (2014)",
        "Inception (2010)",
        "Dark Knight, The (2008)",
        "Gladiator (2000)",
        "Saving Private Ryan (1998)",
        "Maltese Falcon, The (1941)",
        "Batman (1989)",
        "Batman Returns (1992)",
    ]

    selected_movie_latent_vectors = []
    selected_movie_labels = []
    selected_movie_genres = []

    specific_movies_df = movies_df.filter(pl.col('title').is_in(movie_title_list))

    for movie_row in specific_movies_df.iter_rows(named=True):
        movie_id = movie_row['movieId']
        title = movie_row['title']
        genres = movie_row['genres']

        if movie_id in movie_index.movie_to_idx:
            item_idx = movie_index.movie_to_idx[movie_id]
            latent_vector = model.item_vector[item_idx]
            selected_movie_latent_vectors.append(latent_vector)

            cleaned_title = re.sub(r'\(.*?\)', '', title)
            cleaned_title = re.sub(r',.*', '', cleaned_title).strip()
            selected_movie_labels.append(cleaned_title)

            if genres and genres != '(no genres listed)':
                selected_movie_genres.append(genres.split('|')[0])
            else:
                selected_movie_genres.append('Unknown')
        else:
            print(f"Warning: Movie '{title}' (ID: {movie_id}) not found in movie_index. Skipping.")

    selected_movie_latent_vectors = np.array(selected_movie_latent_vectors)

    print(f"Number of selected movies for visualization: {len(selected_movie_labels)}")
    print(f"Shape of selected_movie_latent_vectors: {selected_movie_latent_vectors.shape}")

    # Perform PCA if the latent vectors have more than 2 dimensions
    if selected_movie_latent_vectors.shape[1] > 2:
        pca = PCA(n_components=2)
        selected_movie_latent_vectors = pca.fit_transform(selected_movie_latent_vectors)
        print(f"Reduced latent vectors to 2 dimensions using PCA. New shape: {selected_movie_latent_vectors.shape}")

    plot_data = pd.DataFrame({
        'x': selected_movie_latent_vectors[:, 0],
        'y': selected_movie_latent_vectors[:, 1],
        'label': selected_movie_labels,
        'genre': selected_movie_genres
    })

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        data=plot_data,
        x='x',
        y='y',
        hue='genre',
        style='genre',
        s=200,
    )

    for i, row in plot_data.iterrows():
        plt.annotate(row['label'], (row['x'], row['y']), fontsize=12,
                     va='bottom', ha='center',
                     xytext=(5, 5),
                     textcoords='offset points')

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='best',
               borderaxespad=0., title='Genre')
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(config.SAVE_DIR, "movie_latent_vectors_by_genre.pdf"))
    # plt.show()

def plot_density(tau, r, lam=1.0, ax=None):
    u = np.arange(-5, 5, 0.25)
    v = np.arange(-5, 5, 0.25)
    U, V = np.meshgrid(u, v)
    # Unnormalized posterior density p(u,v|r) ∝ exp(-0.5 * tau * (u² + v²) - 0.5 * lam * (r - u*v)²)
    log_P = -0.5 * tau * (U**2 + V**2) - 0.5 * lam * (r - U * V)**2
    P = np.exp(log_P)
    if ax is None:
        fig, ax = plt.subplots()
    surf = ax.contourf(U, V, P, levels=10)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_title(f'τ={tau}, r={r}, λ={lam}')
    return surf
