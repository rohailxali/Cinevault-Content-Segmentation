"""
CineVault Content Segmentation — ML Preprocessing Pipeline

Handles:
- Placeholder string detection and treatment
- Duration field parsing (mixed Movie/TV-Show semantics)
- Genre multi-hot encoding (top N from listed_in)
- Country binarization (top N countries)
- Rating one-hot encoding
- Type binary encoding
- Release year scaling
- Duration scaling (separate for movies/TV)
"""

from __future__ import annotations

import re
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from typing import Tuple

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Constants ────────────────────────────────────────────────────────────────
PLACEHOLDER_NOT_GIVEN = "Not Given"
TOP_N_GENRES = 25         # top genre dimensions from listed_in
TOP_N_COUNTRIES = 15      # top country dimensions
RANDOM_STATE = 42


# ─── Duration Parsing ─────────────────────────────────────────────────────────
def _parse_duration(row: pd.Series) -> Tuple[float, float]:
    """
    Parse the mixed-type duration field.
    Returns (movie_minutes, tv_seasons).
    For Movies: extract integer minutes, set tv_seasons=0.
    For TV Shows: extract integer seasons, set movie_minutes=0.
    """
    duration: str = str(row["duration"]).strip()
    content_type: str = str(row["type"]).strip()

    if content_type == "Movie":
        match = re.search(r"(\d+)\s*min", duration, re.IGNORECASE)
        if match:
            return float(match.group(1)), 0.0
        return 0.0, 0.0
    else:  # TV Show
        match = re.search(r"(\d+)\s*season", duration, re.IGNORECASE)
        if match:
            return 0.0, float(match.group(1))
        return 0.0, 0.0


# ─── Genre Extraction ─────────────────────────────────────────────────────────
def _extract_all_genres(series: pd.Series) -> list[str]:
    """Explode listed_in comma-separated strings and return all unique genre tokens."""
    genres: list[str] = []
    for val in series.dropna():
        for g in val.split(","):
            g = g.strip()
            if g:
                genres.append(g)
    return genres


def _get_top_genres(series: pd.Series, n: int = TOP_N_GENRES) -> list[str]:
    """Return the top-N most frequent genre labels."""
    from collections import Counter
    all_genres = _extract_all_genres(series)
    counter = Counter(all_genres)
    return [g for g, _ in counter.most_common(n)]


def _multi_hot_genres(df: pd.DataFrame, top_genres: list[str]) -> pd.DataFrame:
    """Create binary columns for each top genre."""
    rows = []
    for val in df["listed_in"]:
        present = set(g.strip() for g in str(val).split(","))
        rows.append({f"genre_{g.replace(' ', '_').replace('&', 'and').replace(',', '')}": 1 if g in present else 0 for g in top_genres})
    return pd.DataFrame(rows, index=df.index)


# ─── Country Binarization ─────────────────────────────────────────────────────
def _get_top_countries(series: pd.Series, n: int = TOP_N_COUNTRIES) -> list[str]:
    """Return top-N most frequent countries (excluding placeholder)."""
    from collections import Counter
    counts = Counter(
        c.strip()
        for c in series
        if c.strip().lower() != "not given"
    )
    return [c for c, _ in counts.most_common(n)]


def _binarize_countries(series: pd.Series, top_countries: list[str]) -> pd.DataFrame:
    """Create binary indicator columns for top countries."""
    rows = []
    for val in series:
        rows.append({f"country_{c.replace(' ', '_')}": 1 if val.strip() == c else 0 for c in top_countries})
    return pd.DataFrame(rows, index=series.index)


# ─── Rating Encoding ──────────────────────────────────────────────────────────
# Map all known rating values to a canonical set
RATING_MAP = {
    "TV-MA": "TV-MA",
    "TV-14": "TV-14",
    "TV-PG": "TV-PG",
    "TV-G": "TV-G",
    "TV-Y": "TV-Y",
    "TV-Y7": "TV-Y7",
    "TV-Y7-FV": "TV-Y7",
    "R": "R",
    "PG-13": "PG-13",
    "PG": "PG",
    "G": "G",
    "NC-17": "NC-17",
    "NR": "NR",
    "UR": "NR",  # Unrated → NR
}
CANONICAL_RATINGS = ["TV-MA", "TV-14", "TV-PG", "TV-G", "TV-Y", "TV-Y7", "R", "PG-13", "PG", "G", "NC-17", "NR"]


def _encode_rating(series: pd.Series) -> pd.DataFrame:
    canonical = series.map(RATING_MAP).fillna("NR")
    rows = []
    for val in canonical:
        rows.append({f"rating_{r}": 1 if val == r else 0 for r in CANONICAL_RATINGS})
    return pd.DataFrame(rows, index=series.index)


# ─── Main Feature Builder ──────────────────────────────────────────────────────
class CineVaultFeatureBuilder:
    """
    Stateful feature builder that fits on training data and transforms
    consistently. Separates feature engineering from scikit-learn Pipeline
    for clarity, then wraps numerical features in a StandardScaler Pipeline.
    """

    def __init__(
        self,
        top_n_genres: int = TOP_N_GENRES,
        top_n_countries: int = TOP_N_COUNTRIES,
    ):
        self.top_n_genres = top_n_genres
        self.top_n_countries = top_n_countries
        self.top_genres_: list[str] = []
        self.top_countries_: list[str] = []
        self.genre_cols_: list[str] = []
        self.country_cols_: list[str] = []
        self.rating_cols_: list[str] = []
        self.numerical_cols_: list[str] = []
        self.all_feature_cols_: list[str] = []
        self.scaler_: StandardScaler = StandardScaler()
        self.is_fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "CineVaultFeatureBuilder":
        self.top_genres_ = _get_top_genres(df["listed_in"], self.top_n_genres)
        self.top_countries_ = _get_top_countries(df["country"], self.top_n_countries)
        self.genre_cols_ = [
            f"genre_{g.replace(' ', '_').replace('&', 'and').replace(',', '')}"
            for g in self.top_genres_
        ]
        self.country_cols_ = [f"country_{c.replace(' ', '_')}" for c in self.top_countries_]
        self.rating_cols_ = [f"rating_{r}" for r in CANONICAL_RATINGS]
        self.numerical_cols_ = ["release_year", "movie_duration_min", "tv_seasons"]

        # Fit scaler
        raw = self._build_raw(df)
        scaler_input = raw[self.numerical_cols_].values.astype(float)
        self.scaler_.fit(scaler_input)

        self.all_feature_cols_ = (
            self.numerical_cols_
            + ["is_movie"]
            + self.rating_cols_
            + self.country_cols_
            + self.genre_cols_
        )
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self.is_fitted_, "Call fit() before transform()"
        raw = self._build_raw(df)

        # Scale numerical
        num_scaled = self.scaler_.transform(raw[self.numerical_cols_].values.astype(float))
        num_df = pd.DataFrame(num_scaled, columns=self.numerical_cols_, index=df.index)

        # Binary type
        is_movie = (df["type"].str.strip() == "Movie").astype(int).values.reshape(-1, 1)

        # Rating
        rating_df = _encode_rating(df["rating"])[self.rating_cols_]

        # Country
        country_df = _binarize_countries(df["country"], self.top_countries_)[self.country_cols_]

        # Genre
        genre_df = _multi_hot_genres(df, self.top_genres_)[self.genre_cols_]

        X = np.hstack([
            num_df.values,
            is_movie,
            rating_df.values,
            country_df.values,
            genre_df.values,
        ])
        return X.astype(float)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def _build_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the raw feature dataframe before scaling."""
        durations = df.apply(_parse_duration, axis=1, result_type="expand")
        durations.columns = ["movie_duration_min", "tv_seasons"]
        raw = pd.DataFrame(index=df.index)
        raw["release_year"] = df["release_year"].astype(float)
        raw["movie_duration_min"] = durations["movie_duration_min"]
        raw["tv_seasons"] = durations["tv_seasons"]
        return raw

    def get_feature_names(self) -> list[str]:
        return list(self.all_feature_cols_)


# ─── Data Loader ──────────────────────────────────────────────────────────────
def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Load the dataset and apply cleaning:
    - Remove rows with 'unknown' title
    - Normalize placeholder strings (NOT removing them — handled by features)
    """
    df = pd.read_csv(csv_path)

    # Remove single "unknown" title row
    df = df[df["title"].str.lower().str.strip() != "unknown"].reset_index(drop=True)

    # Normalize string whitespace
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    return df
