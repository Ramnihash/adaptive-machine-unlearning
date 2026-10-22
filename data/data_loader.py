"""Load and validate the MovieLens-1M dataset."""

from pathlib import Path
from typing import cast
from collections.abc import Hashable, Sequence
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml-1m"

RATINGS_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
MOVIES_COLUMNS = ["movie_id", "title", "genres"]
USERS_COLUMNS = ["user_id", "gender", "age", "occupation", "zip_code"]


# noinspection argument-list
def read_movielens_file(
    path: Path,
    columns: Sequence[Hashable],
) -> pd.DataFrame:
    """Read a MovieLens .dat file into a DataFrame."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    dataframe = pd.read_csv(  # type: ignore[call-overload]
        str(path),
        sep="::",
        engine="python",
        names=columns,
        encoding="latin-1",
    )

    return cast(pd.DataFrame, dataframe)


def load_ratings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load user-movie ratings from ratings.dat."""
    path = data_dir / "ratings.dat"
    ratings = read_movielens_file(path, RATINGS_COLUMNS)

    ratings["user_id"] = ratings["user_id"].astype("int64")
    ratings["movie_id"] = ratings["movie_id"].astype("int64")
    ratings["rating"] = ratings["rating"].astype("int8")
    ratings["timestamp"] = ratings["timestamp"].astype("int64")

    return ratings


def load_movies(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load movie metadata from movies.dat."""
    path = data_dir / "movies.dat"
    movies = read_movielens_file(path, MOVIES_COLUMNS)

    movies["movie_id"] = movies["movie_id"].astype("int64")

    return movies


def load_users(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load user metadata from users.dat."""
    path = data_dir / "users.dat"
    users = read_movielens_file(path, USERS_COLUMNS)

    users["user_id"] = users["user_id"].astype("int64")
    users["age"] = users["age"].astype("int64")
    users["occupation"] = users["occupation"].astype("int64")

    return users


def validate_ratings(ratings: pd.DataFrame) -> None:
    """Raise ValueError if the ratings table violates basic constraints."""
    missing = set(RATINGS_COLUMNS).difference(ratings.columns)

    if missing:
        raise ValueError(f"Missing ratings columns: {sorted(missing)}")

    if ratings.empty:
        raise ValueError("Ratings table is empty.")

    if ratings[RATINGS_COLUMNS].isnull().any().any():
        raise ValueError("Ratings table contains missing required values.")

    if not ratings["rating"].between(1, 5).all():
        raise ValueError("Ratings must be between 1 and 5.")

    if ratings.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("Duplicate user-movie rating pairs found.")