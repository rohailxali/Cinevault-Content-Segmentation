"""
CineVault Content Segmentation — PCA Visualization Generator

Generates 2D and 3D PCA representations of the clustered feature matrix.
PCA is fit on the full scaled feature matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def generate_pca_2d(
    X: np.ndarray,
    df: pd.DataFrame,
    labels: np.ndarray,
) -> dict:
    """
    Fit PCA with 2 components and return visualization data.

    Returns
    -------
    dict with:
      - points: list of dicts (show_id, title, type, rating, listed_in, pc1, pc2, cluster_id)
      - explained_variance_ratio: [pc1_var, pc2_var]
      - total_explained_variance: sum
    """
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    points = []
    for i, (row_idx, row) in enumerate(df.iterrows()):
        points.append({
            "show_id": str(row["show_id"]),
            "title": str(row["title"]),
            "type": str(row["type"]),
            "rating": str(row["rating"]),
            "listed_in": str(row["listed_in"]),
            "release_year": int(row["release_year"]),
            "country": str(row["country"]),
            "pc1": round(float(coords[i, 0]), 4),
            "pc2": round(float(coords[i, 1]), 4),
            "cluster_id": int(labels[i]),
        })

    ev = pca.explained_variance_ratio_
    return {
        "points": points,
        "explained_variance_ratio": [round(float(ev[0]), 4), round(float(ev[1]), 4)],
        "total_explained_variance": round(float(ev.sum()), 4),
        "pca_components": pca.components_.tolist(),
    }


def generate_pca_3d(
    X: np.ndarray,
    df: pd.DataFrame,
    labels: np.ndarray,
) -> dict:
    """
    Fit PCA with 3 components and return visualization data.
    """
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(X)

    points = []
    for i, (row_idx, row) in enumerate(df.iterrows()):
        points.append({
            "show_id": str(row["show_id"]),
            "title": str(row["title"]),
            "type": str(row["type"]),
            "rating": str(row["rating"]),
            "listed_in": str(row["listed_in"]),
            "release_year": int(row["release_year"]),
            "pc1": round(float(coords[i, 0]), 4),
            "pc2": round(float(coords[i, 1]), 4),
            "pc3": round(float(coords[i, 2]), 4),
            "cluster_id": int(labels[i]),
        })

    ev = pca.explained_variance_ratio_
    return {
        "points": points,
        "explained_variance_ratio": [round(float(v), 4) for v in ev],
        "total_explained_variance": round(float(ev.sum()), 4),
    }
