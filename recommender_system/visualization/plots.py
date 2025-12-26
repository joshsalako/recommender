import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import pandas as pd
import numpy as np
import os
from wordcloud import WordCloud
from .. import config

def plot_rating_distribution(ratings_df):
    """Plots the distribution of rating scores."""
    ratings_dist = (
        ratings_df
        .group_by('rating')
        .agg(pl.len().alias('count'))
        .sort('rating')
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(x='rating', y='count',
                data=ratings_dist.to_pandas())
    plt.title('Distribution of Movie Ratings', fontsize=16)
    plt.xlabel('Ratings')
    plt.ylabel('Number of Ratings (in millions)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(config.SAVE_DIR, "rating_distribution.pdf"))
    # plt.show() # Commented out to avoid blocking execution

def analyze_movie_trends(movies_df, ratings_df):

    movies_df = movies_df.with_columns(
        pl.col('title').str.extract(r'\((\d{4})\)').cast(pl.Int64).alias('year')
    )

    merged_df = movies_df.join(ratings_df, on='movieId')

    average_rating_per_year = (
        merged_df.group_by('year')
        .agg(pl.mean('rating').alias('average_rating'))
        .sort('year')
    )

    movies_per_year = (
        movies_df.group_by('year')
        .agg(pl.count('movieId').alias('movie_count'))
        .sort('year')
    )

    # Convert to pandas for plotting
    avg_rating_pd = average_rating_per_year.to_pandas()
    movies_count_pd = movies_per_year.to_pandas()

    plt.figure(figsize=(9, 5))
    sns.lineplot(data=avg_rating_pd, x='year', y='average_rating')
    plt.title('Average Movie Rating Over Time')
    plt.xlabel('Year')
    plt.ylabel('Average Rating')
    plt.tight_layout()
    plt.ylim(0, 5)
    plt.xlim(1874, 2025)
    plt.savefig(os.path.join(config.SAVE_DIR, "average_rating_over_time.pdf"))
    # plt.show()

    plt.figure(figsize=(9, 5))
    sns.lineplot(data=movies_count_pd, x='year', y='movie_count', color='orange')
    plt.title('Number of Movies Released Per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Movies')
    plt.tight_layout()
    plt.xlim(1874, 2025)
    plt.savefig(os.path.join(config.SAVE_DIR, "movies_per_year.pdf"))
    # plt.show()

def plot_ratings_per_year(movies_df, ratings_df):
    movies_with_year = movies_df.with_columns(
        pl.col('title').str.extract(r'\((\d{4})\)').cast(pl.Int64).alias('year')
    )

    merged_df = movies_with_year.join(ratings_df, on='movieId', how='inner')

    ratings_count_per_year = (
        merged_df.group_by('year')
        .agg(pl.len().alias('rating_count'))
        .sort('year')
        .filter(pl.col('year').is_not_null())
    )

    ratings_count_pd = ratings_count_per_year.to_pandas()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=ratings_count_pd, x='year', y='rating_count', color='purple')
    plt.title('Number of Ratings Per Year', fontsize=12)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Ratings', fontsize=12)
    plt.tight_layout()
    plt.xlim(1874, 2025)
    plt.savefig(os.path.join(config.SAVE_DIR, "ratings_per_year.pdf"))
    # plt.show()

def plot_degree_distribution(df):
    """
    Calculates and plots the degree distributions for both movies and users
    on a single log-log scale plot with different colors and a legend.
    """
    # Convert Polars DataFrame to Pandas DataFrame if necessary
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    movie_degrees = df['movieId'].value_counts()
    movie_degree_counts = movie_degrees.value_counts().sort_index()
    k_movies = movie_degree_counts.index
    freq_movies = movie_degree_counts.values

    user_degrees = df['userId'].value_counts()
    user_degree_counts = user_degrees.value_counts().sort_index()
    k_users = user_degree_counts.index
    freq_users = user_degree_counts.values

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(k_movies, freq_movies, marker='.',
               alpha=0.8, color='blue',label='Movie')
    ax.scatter(k_users, freq_users, marker='.',
               alpha=0.8, color='green', label='User')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Movie Popularity vs. User Activity', fontsize=18)
    ax.set_xlabel('Degree', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(config.SAVE_DIR, "degree_distribution.pdf"))
    # plt.show()

def plot_top_genres(movies_lazy):
    """Finds and plots the most common movie genres."""

    genre_counts = (
        movies_lazy
        .filter(pl.col('genres') != '(no genres listed)')
        .select(pl.col('genres').str.split('|'))
        .explode('genres')
        .group_by('genres')
        .agg(pl.len().alias('count'))
        .sort('count', descending=True)
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(x='count', y='genres',
                data=genre_counts.limit(15).collect().to_pandas())
    plt.title('Top Movie Genres', fontsize=16)
    plt.xlabel('Number of Movies')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.savefig(os.path.join(config.SAVE_DIR, "top_genres.pdf"))
    # plt.show()

    return genre_counts

def plot_genre_pie_chart(genre_counts):
    top_8_genres = genre_counts.head(8).select(pl.sum('count')).collect().item()
    total_genres_count = genre_counts.select(pl.sum('count')).collect().item()
    others_count = total_genres_count - top_8_genres

    pie_data = genre_counts.head(12).collect().to_pandas()

    others_row = pd.DataFrame({'genres': ['Others'], 'count': [others_count]})
    pie_data = pd.concat([pie_data, others_row], ignore_index=True)

    plt.figure(figsize=(8, 8))
    plt.pie(pie_data['count'], labels=pie_data['genres'], rotatelabels=True,
            normalize=True, autopct='%1.1f%%', labeldistance=1.05,
            startangle=140, colors=sns.color_palette('pastel', len(pie_data)))
    plt.axis('equal')
    plt.savefig(os.path.join(config.SAVE_DIR, "genre_distribution.pdf"))
    # plt.show()

def plot_genre_rating_distributions(movies_pl, ratings_pl):
    """
    Plots the distribution of ratings for the top 12 movie genres.
    """
    # Get top 12 genres
    genre_counts = (
        movies_pl
        .filter(pl.col('genres') != '(no genres listed)')
        .select(pl.col('genres').str.split(by='|'))
        .explode('genres')
        .group_by('genres')
        .agg(pl.len().alias('count'))
        .sort('count', descending=True)
        .head(10)
        .collect()
    )
    top_genres = genre_counts.select('genres').to_series().to_list()

    genre_ratings = (
        movies_pl
        .join(ratings_pl, on='movieId')
        .filter(pl.col('genres') != '(no genres listed)')
        .select(['genres', pl.col('rating').cast(pl.Float64).alias('rating')])
        .with_columns(pl.col('genres').str.split(by='|'))
        .explode('genres')
        .filter(pl.col('genres').is_in(top_genres))
        .collect()
    )

    genre_ratings_pd = genre_ratings.to_pandas()

    sampled_pd = genre_ratings_pd.groupby('genres', group_keys=False).apply(
        lambda x: x.sample(min(len(x), 10000), random_state=42)
    ).reset_index(drop=True)

    plt.figure(figsize=(12, 8))
    sns.boxplot(x='genres', y='rating', data=sampled_pd,
                   palette='pastel', order=top_genres)
    plt.title('Rating Distribution for Top Movie Genres', fontsize=16)
    plt.xlabel('Genre')
    plt.ylabel('Rating')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(config.SAVE_DIR, "genre_rating_distribution.pdf"))
    # plt.show()

def plot_genre_ratings_scatter(movies_df, ratings_df):
    """Analyzes the average rating and popularity of each genre."""

    genre_ratings = (
        movies_df
        .join(ratings_df, on='movieId')
        .filter(pl.col('genres') != '(no genres listed)')
        .select(['genres', 'rating'])
        .with_columns(pl.col('genres').str.split('|'))
        .explode('genres')
        .group_by('genres')
        .agg(
            pl.mean('rating').alias('average_rating'),
            pl.count('rating').alias('number_of_ratings')
        )
        .sort('average_rating', descending=True)
    )

    genre_ratings_filtered = genre_ratings.filter(pl.col('number_of_ratings') > 50000)

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.scatterplot(
        data=genre_ratings_filtered.to_pandas(),
        x="number_of_ratings",
        y="average_rating",
        legend=False,
        color="red",
        s=100,
        ax=ax
    )

    texts = []
    for i, row in genre_ratings_filtered.to_pandas().iterrows():
        texts.append(ax.annotate(
            f" {row['genres']}",
            (row['number_of_ratings'], row['average_rating']), fontsize=12,
            xytext=(5, 5), textcoords='offset points', va='bottom', ha='center'
        ))

    ax.set_xscale('log')
    plt.title('Average Rating vs. Popularity for Movie Genres', fontsize=12)
    plt.xlabel('Number of Ratings (Log Scale)', fontsize=12)
    plt.ylabel('Average Rating', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(config.SAVE_DIR, "genre_rating_popularity.pdf"))
    # plt.show()

def plot_top_tags(tags_lazy):
    """Finds and plots the most common user-generated tags."""
    tag_counts = (
        tags_lazy
        .select(pl.col('tag').str.to_lowercase().alias('tag'))
        .group_by('tag')
        .agg(pl.len().alias('count'))
        .sort('count', descending=True)
        .limit(60)
    )

    return tag_counts

def plot_wordcloud(tag_counts):
    tag_frequencies = tag_counts.collect().to_pandas().set_index('tag')['count'].to_dict()

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(tag_frequencies)

    plt.figure(figsize=(16, 9))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join(config.SAVE_DIR, "tag_wordcloud.pdf"))
    # plt.show()

def plot_training_history(model):
    sns.set_style("white")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    epochs = range(1, len(model.train_loss_history) + 1)

    ax1.plot(epochs, model.train_loss_history, 'b', label='Training Loss')
    ax1.set_title('Negative Log-Likelihood loss', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_xticks(epochs)

    ax2.plot(epochs, model.train_rmse_history, 'g', label='Train')
    ax2.plot(epochs, model.test_rmse_history, 'r', label='Test')
    ax2.set_title('Train vs. Test RMSE', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE')
    ax2.legend()
    ax2.set_xticks(epochs)

    plt.tight_layout()

    model_name = model.__class__.__name__
    lambda_str = f"lambda={model.lambda_reg}"
    tau_str = f"tau={model.tau}"
    mu_str = f"mu={model.mu:.4f}"

    k_str = ""
    if hasattr(model, 'k'):
        k_str = f"k={model.k}_"

    filename = f"training_history_{model_name}_{k_str}{lambda_str}_{tau_str}_{mu_str}.pdf"
    filepath = os.path.join(config.SAVE_DIR, filename)
    plt.savefig(filepath)
    # plt.show()

def plot_precision_recall_comparison(df):
    sns.set_theme(style="white")
    plt.figure(figsize=(16, 9))
    sns.scatterplot(
        data=df,
        x="recall",
        y="precision",
        hue="k",
        size="lambda",
        style="tau",
        palette="viridis",
        sizes=(50, 200),
        alpha=0.8
    )

    plt.title("Precision vs Recall", fontsize=18)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
               borderaxespad=0., fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(config.SAVE_DIR, "precision_vs_recall.pdf"))
    # plt.show()

def plot_rmse_heatmap(df):
    best_tau_df = df.loc[df.groupby(['k', 'lambda'])['final_test_rmse'].idxmin()]

    pivot_table = best_tau_df.pivot(index="k", columns="lambda", values="final_test_rmse")

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="coolwarm_r")
    plt.title("Test RMSE", fontsize=15)
    plt.xlabel("λ", fontsize=14)
    plt.ylabel("k", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(config.SAVE_DIR, "test_rmse_heatmap.pdf"))
    # plt.show()
