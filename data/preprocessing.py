
"""Preprocess MovieLens data for recommender-system training."""

from dataclasses import dataclass

import pandas as pd

from data.data_loader import (
    load_movies,
    load_ratings,
    load_users,
    validate_ratings,
)


@dataclass
class ProcessedData:
    """Store processed interactions and original-ID mappings."""

    ratings: pd.DataFrame
    movies: pd.DataFrame
    users: pd.DataFrame
    user_id_map: dict[int, int]
    movie_id_map: dict[int, int]
    num_users: int
    num_movies: int


def preprocess_movielens() -> ProcessedData:
    """Load, validate, and remap MovieLens IDs to zero-based indices."""

    ratings = load_ratings()
    movies = load_movies()
    users = load_users()

    validate_ratings(ratings)

    # Get the unique IDs as regular Python integers.
    user_ids = sorted(int(value) for value in ratings["user_id"].unique())
    movie_ids = sorted(int(value) for value in ratings["movie_id"].unique())

    # Explicitly type the dictionaries to satisfy PyCharm.
    user_id_map: dict[int, int] = {
        original_id: index
        for index, original_id in enumerate(user_ids)
    }

    movie_id_map: dict[int, int] = {
        original_id: index
        for index, original_id in enumerate(movie_ids)
    }

    # Preserve original IDs and add zero-based model indices.
    ratings = ratings.copy()
    ratings["user_idx"] = ratings["user_id"].map(user_id_map)
    ratings["movie_idx"] = ratings["movie_id"].map(movie_id_map)

    movies = movies.copy()
    movies["movie_idx"] = movies["movie_id"].map(movie_id_map)

    users = users.copy()
    users["user_idx"] = users["user_id"].map(user_id_map)

    # Ensure every interaction has a valid mapped index.
    if ratings[["user_idx", "movie_idx"]].isnull().any().any():
        raise ValueError("Some ratings could not be mapped to valid indices.")

    return ProcessedData(
        ratings=ratings,
        movies=movies,
        users=users,
        user_id_map=user_id_map,
        movie_id_map=movie_id_map,
        num_users=len(user_id_map),
        num_movies=len(movie_id_map),
    )
