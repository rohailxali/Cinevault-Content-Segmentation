"""
CineVault Content Segmentation — Cluster Profile Generator

For each cluster, computes:
- Size statistics
- Dominant type
- Top genres (with lift ratios)
- Top ratings (with lift ratios)
- Top countries (with lift ratios)
- Release year tendencies
- Duration tendencies
- Representative titles (by centroid proximity)
- Descriptive label derived from statistics
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans


# ─── Lift Computation ─────────────────────────────────────────────────────────
def _compute_lift(cluster_vals: list, global_vals: list, top_n: int = 5) -> list[dict]:
    """Compute top-N items by lift (cluster freq / global freq)."""
    cluster_count = Counter(cluster_vals)
    global_count = Counter(global_vals)
    total_global = len(global_vals)
    total_cluster = len(cluster_vals)

    lifts = []
    for item, c_count in cluster_count.items():
        c_freq = c_count / total_cluster
        g_freq = global_count.get(item, 0) / total_global
        lift = c_freq / g_freq if g_freq > 0 else 0.0
        lifts.append({
            "value": item,
            "cluster_pct": round(c_freq * 100, 1),
            "global_pct": round(g_freq * 100, 1),
            "lift": round(lift, 2),
            "count": c_count,
        })

    lifts.sort(key=lambda x: x["cluster_pct"], reverse=True)
    return lifts[:top_n]


# ─── Genre Extraction ─────────────────────────────────────────────────────────
def _extract_genres(series: pd.Series) -> list[str]:
    genres = []
    for val in series:
        for g in str(val).split(","):
            g = g.strip()
            if g:
                genres.append(g)
    return genres


# ─── Duration Parsing ─────────────────────────────────────────────────────────
def _parse_movie_minutes(duration: str) -> float | None:
    match = re.search(r"(\d+)\s*min", str(duration), re.IGNORECASE)
    return float(match.group(1)) if match else None


def _parse_tv_seasons(duration: str) -> float | None:
    match = re.search(r"(\d+)\s*season", str(duration), re.IGNORECASE)
    return float(match.group(1)) if match else None


# ─── Representative Titles ─────────────────────────────────────────────────────
def _get_representative_titles(
    cluster_df: pd.DataFrame,
    X_cluster: np.ndarray,
    centroid: np.ndarray,
    n: int = 5,
) -> list[dict]:
    """Return titles closest to the cluster centroid."""
    dists = np.linalg.norm(X_cluster - centroid, axis=1)
    closest_idx = np.argsort(dists)[:n]
    titles = []
    for i in closest_idx:
        row = cluster_df.iloc[i]
        titles.append({
            "title": str(row["title"]),
            "type": str(row["type"]),
            "rating": str(row["rating"]),
            "release_year": int(row["release_year"]),
            "country": str(row["country"]),
            "listed_in": str(row["listed_in"]),
            "duration": str(row["duration"]),
            "distance_to_centroid": round(float(dists[i]), 4),
        })
    return titles


# ─── Label Generation ─────────────────────────────────────────────────────────
def _generate_label(profile: dict) -> str:
    """Generate a descriptive label from actual cluster statistics."""
    parts = []

    # Content type dominance
    dominant_type = profile.get("dominant_type", "")
    if dominant_type:
        type_pct = profile.get("dominant_type_pct", 0)
        if type_pct >= 80:
            parts.append(f"Mostly {dominant_type}s")
        elif type_pct >= 60:
            parts.append(f"Predominantly {dominant_type}s")
        else:
            parts.append("Mixed Content")

    # Top 2 genres
    top_genres = profile.get("top_genres", [])
    if top_genres:
        genre_names = [g["value"].replace(" TV Shows", "").replace(" Movies", "").strip() for g in top_genres[:2]]
        parts.append(" & ".join(genre_names))

    # Rating character
    top_ratings = profile.get("top_ratings", [])
    if top_ratings:
        top_rating = top_ratings[0]["value"]
        mature_ratings = {"TV-MA", "R", "NC-17"}
        family_ratings = {"TV-G", "TV-Y", "TV-Y7", "G", "PG"}
        if top_rating in mature_ratings:
            parts.append("Mature")
        elif top_rating in family_ratings:
            parts.append("Family-Friendly")
        else:
            parts.append(f"Rated {top_rating}")

    # Era
    median_year = profile.get("release_year_median")
    if median_year:
        if median_year >= 2018:
            parts.append("Recent")
        elif median_year >= 2010:
            parts.append("2010s")
        else:
            parts.append("Classic")

    return " · ".join(parts[:4]) if parts else f"Cluster {profile.get('cluster_id', '?')}"


# ─── Main Profile Builder ──────────────────────────────────────────────────────
def generate_cluster_profiles(
    df: pd.DataFrame,
    X: np.ndarray,
    labels: np.ndarray,
    model: KMeans,
) -> list[dict]:
    """
    Generate a comprehensive profile for each cluster.
    
    Parameters
    ----------
    df : original cleaned dataframe
    X : scaled feature matrix used for clustering
    labels : cluster assignment array
    model : fitted KMeans model
    """
    n_clusters = model.n_clusters
    total_titles = len(df)
    profiles = []

    # Global stats for lift
    global_genres = _extract_genres(df["listed_in"])
    global_ratings = df["rating"].tolist()
    global_countries = [c for c in df["country"].tolist() if c.lower() != "not given"]
    global_types = df["type"].tolist()

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_df = df[mask].reset_index(drop=True)
        X_cluster = X[mask]
        centroid = model.cluster_centers_[cluster_id]

        n_titles = int(mask.sum())
        pct = round(n_titles / total_titles * 100, 1)

        # Type dominance
        type_counts = Counter(cluster_df["type"].tolist())
        dominant_type = type_counts.most_common(1)[0][0] if type_counts else "Unknown"
        dominant_type_pct = round(type_counts.get(dominant_type, 0) / n_titles * 100, 1)

        # Genres
        cluster_genres = _extract_genres(cluster_df["listed_in"])
        top_genres = _compute_lift(cluster_genres, global_genres, top_n=5)

        # Ratings
        top_ratings = _compute_lift(cluster_df["rating"].tolist(), global_ratings, top_n=5)

        # Countries
        cluster_countries = [c for c in cluster_df["country"].tolist() if c.lower() != "not given"]
        top_countries = _compute_lift(cluster_countries, global_countries, top_n=5)

        # Release year
        years = cluster_df["release_year"].dropna()
        release_year_mean = round(float(years.mean()), 1) if len(years) > 0 else None
        release_year_median = round(float(years.median()), 1) if len(years) > 0 else None
        release_year_min = int(years.min()) if len(years) > 0 else None
        release_year_max = int(years.max()) if len(years) > 0 else None

        # Duration by type
        movies_df = cluster_df[cluster_df["type"] == "Movie"]
        tv_df = cluster_df[cluster_df["type"] == "TV Show"]

        movie_minutes = [_parse_movie_minutes(d) for d in movies_df["duration"] if _parse_movie_minutes(d) is not None]
        tv_seasons_list = [_parse_tv_seasons(d) for d in tv_df["duration"] if _parse_tv_seasons(d) is not None]

        movie_duration_median = round(float(np.median(movie_minutes)), 1) if movie_minutes else None
        tv_seasons_median = round(float(np.median(tv_seasons_list)), 1) if tv_seasons_list else None

        # Representative titles
        rep_titles = _get_representative_titles(cluster_df, X_cluster, centroid, n=5)

        profile = {
            "cluster_id": cluster_id,
            "n_titles": n_titles,
            "pct_of_dataset": pct,
            "dominant_type": dominant_type,
            "dominant_type_pct": dominant_type_pct,
            "n_movies": int(len(movies_df)),
            "n_tv_shows": int(len(tv_df)),
            "top_genres": top_genres,
            "top_ratings": top_ratings,
            "top_countries": top_countries,
            "release_year_mean": release_year_mean,
            "release_year_median": release_year_median,
            "release_year_min": release_year_min,
            "release_year_max": release_year_max,
            "movie_duration_median_min": movie_duration_median,
            "tv_seasons_median": tv_seasons_median,
            "representative_titles": rep_titles,
        }

        profile["label"] = _generate_label(profile)
        profiles.append(profile)

    return profiles
