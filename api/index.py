"""
CineVault Content Segmentation — FastAPI Backend

Serves precomputed model artifacts to the frontend.
All expensive computation is done at startup (pipeline must run first).
"""

from __future__ import annotations

import json
import csv
import os
import math
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ─── Config ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts"

app = FastAPI(
    title="CineVault Content Segmentation API",
    description="Serves K-Means clustering results for the Netflix catalog dataset.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ─── Artifact Loaders (cached at startup) ────────────────────────────────────
@lru_cache(maxsize=1)
def _load_overview() -> dict:
    with open(ARTIFACTS / "overview.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_profiles() -> list[dict]:
    with open(ARTIFACTS / "cluster_profiles.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_k_evaluation() -> list[dict]:
    with open(ARTIFACTS / "k_evaluation.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_feature_meta() -> dict:
    with open(ARTIFACTS / "feature_meta.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_pca_points_2d() -> list[dict]:
    with open(ARTIFACTS / "pca_points_2d.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_pca_visualization_meta() -> dict:
    with open(ARTIFACTS / "pca_visualization.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_titles() -> list[dict]:
    """Load cluster assignments CSV into memory as list of dicts."""
    titles = []
    with open(ARTIFACTS / "cluster_assignments.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            titles.append({
                "show_id": row["show_id"],
                "title": row["title"],
                "type": row["type"],
                "rating": row["rating"],
                "release_year": int(row["release_year"]) if row["release_year"] else None,
                "country": row["country"],
                "duration": row["duration"],
                "listed_in": row["listed_in"],
                "director": row["director"],
                "date_added": row["date_added"],
                "cluster_id": int(row["cluster_id"]),
                "pc1": float(row["pc1"]) if row.get("pc1") else None,
                "pc2": float(row["pc2"]) if row.get("pc2") else None,
            })
    return titles


def _artifacts_available() -> bool:
    required = ["overview.json", "cluster_profiles.json", "k_evaluation.json",
                "feature_meta.json", "pca_points_2d.json", "cluster_assignments.csv"]
    return all((ARTIFACTS / f).exists() for f in required)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "artifacts_available": _artifacts_available(),
    }


@app.get("/api/overview")
def overview():
    if not _artifacts_available():
        raise HTTPException(503, "Model artifacts not found. Run ml/run_pipeline.py first.")
    return _load_overview()


@app.get("/api/clusters")
def clusters():
    if not _artifacts_available():
        raise HTTPException(503, "Artifacts not available.")
    profiles = _load_profiles()
    # Return summary list (without representative_titles for lighter payload)
    return [
        {
            "cluster_id": p["cluster_id"],
            "label": p["label"],
            "n_titles": p["n_titles"],
            "pct_of_dataset": p["pct_of_dataset"],
            "dominant_type": p["dominant_type"],
            "dominant_type_pct": p["dominant_type_pct"],
            "n_movies": p["n_movies"],
            "n_tv_shows": p["n_tv_shows"],
            "top_genres": p["top_genres"][:3],
            "top_ratings": p["top_ratings"][:3],
            "top_countries": p["top_countries"][:3],
            "release_year_median": p["release_year_median"],
            "movie_duration_median_min": p.get("movie_duration_median_min"),
            "tv_seasons_median": p.get("tv_seasons_median"),
        }
        for p in profiles
    ]


@app.get("/api/clusters/{cluster_id}")
def cluster_detail(cluster_id: int):
    if not _artifacts_available():
        raise HTTPException(503, "Artifacts not available.")
    profiles = _load_profiles()
    for p in profiles:
        if p["cluster_id"] == cluster_id:
            return p
    raise HTTPException(404, f"Cluster {cluster_id} not found.")


@app.get("/api/titles")
def titles(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None),
    cluster: Optional[int] = Query(None),
    content_type: Optional[str] = Query(None, alias="type"),
    rating: Optional[str] = Query(None),
    year_min: Optional[int] = Query(None),
    year_max: Optional[int] = Query(None),
    genre: Optional[str] = Query(None),
    sort_by: str = Query("title"),
    sort_order: str = Query("asc"),
):
    if not _artifacts_available():
        raise HTTPException(503, "Artifacts not available.")

    all_titles = _load_titles()
    filtered = all_titles

    # Search
    if search:
        q = search.strip().lower()
        filtered = [t for t in filtered if q in t["title"].lower() or q in t["director"].lower()]

    # Cluster filter
    if cluster is not None:
        filtered = [t for t in filtered if t["cluster_id"] == cluster]

    # Type filter
    if content_type:
        filtered = [t for t in filtered if t["type"].lower() == content_type.lower()]

    # Rating filter
    if rating:
        filtered = [t for t in filtered if t["rating"].lower() == rating.lower()]

    # Year range
    if year_min is not None:
        filtered = [t for t in filtered if t["release_year"] and t["release_year"] >= year_min]
    if year_max is not None:
        filtered = [t for t in filtered if t["release_year"] and t["release_year"] <= year_max]

    # Genre filter (substring match in listed_in)
    if genre:
        g_lower = genre.lower()
        filtered = [t for t in filtered if g_lower in t["listed_in"].lower()]

    # Sort
    reverse = sort_order.lower() == "desc"
    valid_sorts = {"title", "release_year", "cluster_id", "rating", "type"}
    if sort_by in valid_sorts:
        filtered = sorted(
            filtered,
            key=lambda t: (t.get(sort_by) or ""),
            reverse=reverse,
        )

    # Pagination
    total = len(filtered)
    total_pages = math.ceil(total / limit) if total > 0 else 1
    start = (page - 1) * limit
    end = start + limit
    page_data = filtered[start:end]

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
        "results": page_data,
    }


@app.get("/api/titles/{show_id}")
def title_detail(show_id: str):
    if not _artifacts_available():
        raise HTTPException(503, "Artifacts not available.")
    all_titles = _load_titles()
    for t in all_titles:
        if t["show_id"] == show_id:
            return t
    raise HTTPException(404, f"Title {show_id} not found.")


@app.get("/api/filters")
def filters():
    if not _artifacts_available():
        raise HTTPException(503, "Artifacts not available.")
    all_titles = _load_titles()
    overview = _load_overview()

    ratings = sorted(set(t["rating"] for t in all_titles if t["rating"]))
    years = sorted(set(t["release_year"] for t in all_titles if t["release_year"]))
    clusters = sorted(set(t["cluster_id"] for t in all_titles))
    cluster_labels = overview.get("cluster_labels", {})

    return {
        "ratings": ratings,
        "types": ["Movie", "TV Show"],
        "year_range": [min(years), max(years)],
        "clusters": [{"id": c, "label": cluster_labels.get(str(c), f"Cluster {c}")} for c in clusters],
        "top_genres": list(overview.get("top_genres_global", {}).keys()),
    }


@app.get("/api/visualization")
def visualization(mode: str = Query("2d", regex="^(2d|3d)$")):
    if not _artifacts_available():
        raise HTTPException(503, "Artifacts not available.")
    
    meta = _load_pca_visualization_meta()
    points = _load_pca_points_2d()
    
    # Return downsampled for performance (max 5000 points for initial load)
    if len(points) > 5000:
        import random
        random.seed(42)
        sampled = random.sample(points, 5000)
    else:
        sampled = points

    return {
        "mode": "2d",
        "explained_variance_ratio": meta["explained_variance_ratio"],
        "total_explained_variance": meta["total_explained_variance"],
        "points": sampled,
        "total_points": len(points),
        "sampled": len(points) > 5000,
    }


@app.get("/api/visualization/full")
def visualization_full():
    """Full dataset visualization (all points)."""
    if not _artifacts_available():
        raise HTTPException(503, "Artifacts not available.")
    meta = _load_pca_visualization_meta()
    points = _load_pca_points_2d()
    return {
        "mode": "2d",
        "explained_variance_ratio": meta["explained_variance_ratio"],
        "total_explained_variance": meta["total_explained_variance"],
        "points": points,
        "total_points": len(points),
        "sampled": False,
    }


@app.get("/api/evaluation")
def evaluation():
    if not _artifacts_available():
        raise HTTPException(503, "Artifacts not available.")
    meta = _load_feature_meta()
    k_eval = _load_k_evaluation()
    return {
        "k_evaluation": k_eval,
        "selected_k": meta["k"],
        "final_metrics": meta["final_metrics"],
        "reasoning": meta.get("selected_k_reasoning", ""),
        "n_features": meta["n_features"],
        "feature_names": meta["feature_names"],
        "dataset_rows": meta["dataset_rows"],
    }
