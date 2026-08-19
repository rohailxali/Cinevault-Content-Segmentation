"""
CineVault ML Tests — pytest suite

Tests the ML pipeline components without requiring model artifacts.
Run: pytest tests/test_ml.py -v
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ml.pipeline import (
    CineVaultFeatureBuilder,
    load_and_clean,
    _parse_duration,
    _get_top_genres,
    _multi_hot_genres,
    _encode_rating,
    _binarize_countries,
)
from ml.clustering import evaluate_k_range, select_best_k, train_final_model, compute_final_metrics
from ml.profiles import generate_cluster_profiles
from ml.visualize import generate_pca_2d

CSV_PATH = ROOT / "Dataset.csv"


# ─── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def raw_df():
    return pd.read_csv(str(CSV_PATH))


@pytest.fixture(scope="module")
def clean_df():
    return load_and_clean(str(CSV_PATH))


@pytest.fixture(scope="module")
def feature_builder(clean_df):
    builder = CineVaultFeatureBuilder(top_n_genres=10, top_n_countries=5)
    builder.fit(clean_df)
    return builder


@pytest.fixture(scope="module")
def X(feature_builder, clean_df):
    return feature_builder.transform(clean_df)


@pytest.fixture(scope="module")
def small_X():
    """A small synthetic feature matrix for fast clustering tests."""
    np.random.seed(42)
    # 4 clearly separated clusters
    centers = [[0, 0], [10, 0], [0, 10], [10, 10]]
    data = []
    for c in centers:
        data.append(np.random.randn(50, 2) + c)
    return np.vstack(data)


# ─── Data Loading Tests ───────────────────────────────────────────────────────
class TestDataLoading:
    def test_csv_loads(self, raw_df):
        assert raw_df is not None
        assert len(raw_df) > 0

    def test_expected_columns(self, raw_df):
        expected = {"show_id", "type", "title", "director", "country",
                    "date_added", "release_year", "rating", "duration", "listed_in"}
        assert expected.issubset(set(raw_df.columns))

    def test_no_true_nulls(self, clean_df):
        # After cleaning, no column should have nulls
        assert clean_df.isnull().sum().sum() == 0

    def test_clean_removes_unknown_title(self, clean_df):
        assert "unknown" not in clean_df["title"].str.lower().values

    def test_shape_reasonable(self, clean_df):
        # Should have 8700+ rows after removing 1 unknown
        assert len(clean_df) >= 8700


# ─── Placeholder Handling Tests ───────────────────────────────────────────────
class TestPlaceholders:
    def test_not_given_director_exists(self, clean_df):
        not_given_count = (clean_df["director"].str.lower() == "not given").sum()
        assert not_given_count > 2000  # known ~2588

    def test_not_given_country_exists(self, clean_df):
        not_given_count = (clean_df["country"].str.lower() == "not given").sum()
        assert not_given_count > 200


# ─── Duration Parsing Tests ───────────────────────────────────────────────────
class TestDurationParsing:
    def test_movie_duration_parsed(self):
        row = pd.Series({"type": "Movie", "duration": "90 min"})
        mins, seasons = _parse_duration(row)
        assert mins == 90.0
        assert seasons == 0.0

    def test_tv_seasons_parsed(self):
        row = pd.Series({"type": "TV Show", "duration": "3 Seasons"})
        mins, seasons = _parse_duration(row)
        assert mins == 0.0
        assert seasons == 3.0

    def test_tv_single_season(self):
        row = pd.Series({"type": "TV Show", "duration": "1 Season"})
        mins, seasons = _parse_duration(row)
        assert seasons == 1.0

    def test_movie_returns_zero_seasons(self):
        row = pd.Series({"type": "Movie", "duration": "120 min"})
        mins, seasons = _parse_duration(row)
        assert seasons == 0.0

    def test_tv_returns_zero_minutes(self):
        row = pd.Series({"type": "TV Show", "duration": "2 Seasons"})
        mins, seasons = _parse_duration(row)
        assert mins == 0.0


# ─── Genre Extraction Tests ───────────────────────────────────────────────────
class TestGenreExtraction:
    def test_top_genres_length(self, clean_df):
        genres = _get_top_genres(clean_df["listed_in"], n=20)
        assert len(genres) == 20

    def test_genres_are_strings(self, clean_df):
        genres = _get_top_genres(clean_df["listed_in"], n=5)
        assert all(isinstance(g, str) for g in genres)

    def test_multi_hot_shape(self, clean_df):
        top = _get_top_genres(clean_df["listed_in"], n=10)
        mh = _multi_hot_genres(clean_df, top)
        assert mh.shape[0] == len(clean_df)
        assert mh.shape[1] == 10

    def test_multi_hot_binary(self, clean_df):
        top = _get_top_genres(clean_df["listed_in"], n=5)
        mh = _multi_hot_genres(clean_df, top)
        assert set(mh.values.flatten().tolist()).issubset({0, 1})


# ─── Rating Encoding Tests ─────────────────────────────────────────────────────
class TestRatingEncoding:
    def test_rating_encoding_shape(self, clean_df):
        enc = _encode_rating(clean_df["rating"])
        assert enc.shape[0] == len(clean_df)
        assert enc.shape[1] == 12  # 12 canonical ratings

    def test_rating_binary_values(self, clean_df):
        enc = _encode_rating(clean_df["rating"])
        vals = set(enc.values.flatten().tolist())
        assert vals.issubset({0, 1})

    def test_ur_mapped_to_nr(self):
        test_series = pd.Series(["UR", "NR", "TV-MA"])
        enc = _encode_rating(test_series)
        # Both UR and NR should map to NR column
        assert enc["rating_NR"].iloc[0] == 1
        assert enc["rating_NR"].iloc[1] == 1


# ─── Feature Builder Tests ─────────────────────────────────────────────────────
class TestFeatureBuilder:
    def test_fit_produces_feature_names(self, feature_builder):
        names = feature_builder.get_feature_names()
        assert len(names) > 10
        assert "release_year" in names
        assert "movie_duration_min" in names
        assert "tv_seasons" in names
        assert "is_movie" in names

    def test_transform_shape(self, X, clean_df):
        assert X.shape[0] == len(clean_df)
        assert X.shape[1] > 10

    def test_transform_no_nan(self, X):
        assert not np.isnan(X).any()

    def test_transform_no_inf(self, X):
        assert not np.isinf(X).any()

    def test_identifiers_not_in_features(self, feature_builder):
        names = feature_builder.get_feature_names()
        assert "show_id" not in names
        assert "title" not in names

    def test_director_not_in_features(self, feature_builder):
        names = feature_builder.get_feature_names()
        assert not any("director" in n for n in names)

    def test_reproducible_transform(self, feature_builder, clean_df):
        X1 = feature_builder.transform(clean_df)
        X2 = feature_builder.transform(clean_df)
        assert np.allclose(X1, X2)


# ─── Clustering Tests ─────────────────────────────────────────────────────────
class TestClustering:
    def test_evaluate_k_range_returns_all_k(self, small_X):
        results = evaluate_k_range(small_X, k_range=range(2, 6))
        assert len(results) == 4

    def test_evaluation_keys(self, small_X):
        results = evaluate_k_range(small_X, k_range=range(2, 4))
        for r in results:
            assert "k" in r
            assert "inertia" in r
            assert "silhouette_score" in r
            assert "davies_bouldin_score" in r
            assert "calinski_harabasz_score" in r

    def test_select_best_k_range(self, small_X):
        results = evaluate_k_range(small_X, k_range=range(2, 8))
        best_k = select_best_k(results)
        assert 2 <= best_k <= 7

    def test_select_best_k_clear_clusters(self, small_X):
        # For our clearly separated 4-cluster data, K should be 4
        results = evaluate_k_range(small_X, k_range=range(2, 8))
        best_k = select_best_k(results)
        assert best_k == 4

    def test_train_final_model_labels(self, small_X):
        model = train_final_model(small_X, k=4)
        labels = model.labels_
        assert len(labels) == len(small_X)
        assert set(labels).issubset(set(range(4)))

    def test_deterministic_with_seed(self, small_X):
        model1 = train_final_model(small_X, k=3)
        model2 = train_final_model(small_X, k=3)
        assert np.array_equal(model1.inertia_, model2.inertia_)

    def test_compute_metrics_keys(self, small_X):
        model = train_final_model(small_X, k=4)
        metrics = compute_final_metrics(small_X, model.labels_)
        assert "silhouette_score" in metrics
        assert "davies_bouldin_score" in metrics
        assert "calinski_harabasz_score" in metrics

    def test_silhouette_between_neg1_and_1(self, small_X):
        model = train_final_model(small_X, k=4)
        metrics = compute_final_metrics(small_X, model.labels_)
        assert -1 <= metrics["silhouette_score"] <= 1


# ─── Leakage / Correctness Tests ─────────────────────────────────────────────
class TestLeakageCorrectness:
    def test_release_year_not_raw(self, X):
        # After StandardScaler, release_year should be centered around 0
        # (raw values would be 1925-2021, so mean ~0 after scaling)
        # First column is release_year
        col_mean = X[:, 0].mean()
        assert abs(col_mean) < 1.0, f"release_year not properly scaled: mean={col_mean}"

    def test_movie_duration_only_for_movies(self, feature_builder, clean_df):
        X = feature_builder.transform(clean_df)
        feature_names = feature_builder.get_feature_names()
        if "movie_duration_min" in feature_names:
            col_idx = feature_names.index("movie_duration_min")
            tv_mask = clean_df["type"].str.strip() == "TV Show"
            # TV shows should have 0 for movie_duration_min (before scaling)
            # After scaling, they will be a single value (the scaled 0)
            # We verify that TV shows all have the same value in this column
            tv_vals = X[tv_mask.values, col_idx]
            assert len(set(tv_vals.round(4).tolist())) == 1, "TV Shows should all have same scaled 0 for movie_duration_min"

    def test_tv_seasons_only_for_tv_shows(self, feature_builder, clean_df):
        X = feature_builder.transform(clean_df)
        feature_names = feature_builder.get_feature_names()
        if "tv_seasons" in feature_names:
            col_idx = feature_names.index("tv_seasons")
            movie_mask = clean_df["type"].str.strip() == "Movie"
            movie_vals = X[movie_mask.values, col_idx]
            assert len(set(movie_vals.round(4).tolist())) == 1, "Movies should all have same scaled 0 for tv_seasons"


# ─── Cluster Profile Tests ─────────────────────────────────────────────────────
class TestClusterProfiles:
    @pytest.fixture(scope="class")
    def profiles_and_data(self, clean_df, feature_builder):
        X = feature_builder.transform(clean_df)
        model = train_final_model(X, k=4)
        labels = model.labels_
        profiles = generate_cluster_profiles(clean_df, X, labels, model)
        return profiles, X, labels

    def test_profile_count(self, profiles_and_data, clean_df):
        profiles, _, _ = profiles_and_data
        assert len(profiles) == 4

    def test_profile_keys(self, profiles_and_data):
        profiles, _, _ = profiles_and_data
        required_keys = {"cluster_id", "n_titles", "pct_of_dataset", "dominant_type",
                        "top_genres", "top_ratings", "representative_titles", "label"}
        for p in profiles:
            assert required_keys.issubset(set(p.keys()))

    def test_profile_sizes_sum_to_dataset(self, profiles_and_data, clean_df):
        profiles, _, _ = profiles_and_data
        total = sum(p["n_titles"] for p in profiles)
        assert total == len(clean_df)

    def test_representative_titles_exist(self, profiles_and_data):
        profiles, _, _ = profiles_and_data
        for p in profiles:
            assert len(p["representative_titles"]) > 0

    def test_label_is_non_empty_string(self, profiles_and_data):
        profiles, _, _ = profiles_and_data
        for p in profiles:
            assert isinstance(p["label"], str)
            assert len(p["label"]) > 0


# ─── PCA Visualization Tests ──────────────────────────────────────────────────
class TestPCAVisualization:
    def test_pca_2d_returns_all_points(self, X, clean_df):
        labels = np.zeros(len(X), dtype=int)
        result = generate_pca_2d(X, clean_df, labels)
        assert len(result["points"]) == len(clean_df)

    def test_pca_2d_point_keys(self, X, clean_df):
        labels = np.zeros(len(X), dtype=int)
        result = generate_pca_2d(X, clean_df, labels)
        required = {"show_id", "title", "pc1", "pc2", "cluster_id"}
        assert required.issubset(set(result["points"][0].keys()))

    def test_pca_explained_variance_between_0_and_1(self, X, clean_df):
        labels = np.zeros(len(X), dtype=int)
        result = generate_pca_2d(X, clean_df, labels)
        for v in result["explained_variance_ratio"]:
            assert 0 <= v <= 1

    def test_pca_coordinates_are_finite(self, X, clean_df):
        labels = np.zeros(len(X), dtype=int)
        result = generate_pca_2d(X, clean_df, labels)
        for pt in result["points"][:100]:  # check a sample
            assert np.isfinite(pt["pc1"])
            assert np.isfinite(pt["pc2"])
