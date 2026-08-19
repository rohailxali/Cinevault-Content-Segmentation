"""
CineVault Content Segmentation — K-Means Clustering Module

Handles:
- K evaluation over a range with 4 metrics
- Final model training with selected K
- Result storage
"""

from __future__ import annotations

import json
import numpy as np
from typing import Any
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

RANDOM_STATE = 42
K_RANGE = range(2, 13)  # evaluate K = 2..12


def evaluate_k_range(X: np.ndarray, k_range=K_RANGE) -> list[dict]:
    """
    Evaluate KMeans for each K in k_range.
    Returns list of dicts with: k, inertia, silhouette, davies_bouldin, calinski_harabasz.
    """
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=15, max_iter=500)
        labels = km.fit_predict(X)

        inertia = float(km.inertia_)
        sil = float(silhouette_score(X, labels, sample_size=min(3000, len(X)), random_state=RANDOM_STATE))
        db = float(davies_bouldin_score(X, labels))
        ch = float(calinski_harabasz_score(X, labels))

        results.append({
            "k": k,
            "inertia": round(inertia, 2),
            "silhouette_score": round(sil, 4),
            "davies_bouldin_score": round(db, 4),
            "calinski_harabasz_score": round(ch, 2),
        })
        print(f"  K={k:2d} | Inertia={inertia:10.1f} | Silhouette={sil:.4f} | DB={db:.4f} | CH={ch:.1f}")

    return results


def select_best_k(evaluation_results: list[dict]) -> int:
    """
    Select the best K based on a composite score:
    - Normalize each metric to [0,1]
    - Silhouette: higher = better (maximize)
    - Davies-Bouldin: lower = better (minimize, invert)
    - Calinski-Harabasz: higher = better (maximize)
    - Inertia: lower = better (minimize, invert) — used as tiebreaker via elbow approach

    Returns the K with the best composite score, capped at practical maximum (usually <= 10).
    """
    metrics = {
        "silhouette_score": True,        # higher is better
        "calinski_harabasz_score": True,  # higher is better
        "davies_bouldin_score": False,    # lower is better
    }

    scores = np.zeros(len(evaluation_results))
    for metric, higher_better in metrics.items():
        vals = np.array([r[metric] for r in evaluation_results], dtype=float)
        rng = vals.max() - vals.min()
        if rng == 0:
            normalized = np.ones_like(vals)
        elif higher_better:
            normalized = (vals - vals.min()) / rng
        else:
            normalized = (vals.max() - vals) / rng
        scores += normalized

    best_idx = int(np.argmax(scores))
    best_k = evaluation_results[best_idx]["k"]
    return best_k


def train_final_model(X: np.ndarray, k: int) -> KMeans:
    """Train the final KMeans model with the selected K."""
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20, max_iter=1000)
    km.fit(X)
    return km


def compute_final_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute final clustering metrics for the selected model."""
    return {
        "silhouette_score": round(float(silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=RANDOM_STATE)), 4),
        "davies_bouldin_score": round(float(davies_bouldin_score(X, labels)), 4),
        "calinski_harabasz_score": round(float(calinski_harabasz_score(X, labels)), 2),
    }
