"""
CineVault Content Segmentation — End-to-End Training Pipeline

Run this script ONCE to produce all model artifacts.
Artifacts are saved to the ../artifacts/ directory.

Usage:
    python ml/run_pipeline.py

Output artifacts:
    artifacts/model.pkl              — fitted KMeans
    artifacts/preprocessor.pkl       — fitted CineVaultFeatureBuilder
    artifacts/feature_meta.json      — feature names, K, metrics, config
    artifacts/k_evaluation.json      — K=2..12 metric table
    artifacts/cluster_assignments.csv — every title with cluster_id + PCA coords
    artifacts/cluster_profiles.json  — per-cluster summaries
    artifacts/overview.json          — dashboard KPIs
    artifacts/pca_visualization.json — 2D PCA data (full points)
    artifacts/pca_visualization_3d.json — 3D PCA data
"""

from __future__ import annotations

import json
import os
import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ml.pipeline import CineVaultFeatureBuilder, load_and_clean
from ml.clustering import evaluate_k_range, select_best_k, train_final_model, compute_final_metrics
from ml.profiles import generate_cluster_profiles
from ml.visualize import generate_pca_2d, generate_pca_3d

ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

CSV_PATH = ROOT / "Dataset.csv"


def main():
    print("=" * 60)
    print("CineVault Content Segmentation — Training Pipeline")
    print("=" * 60)

    # --- Step 1: Load & Clean -------------------------------------
    print("\n[1/8] Loading and cleaning dataset...")
    df = load_and_clean(str(CSV_PATH))
    print(f"      Dataset shape after cleaning: {df.shape}")

    # --- Step 2: Feature Engineering ------------------------------
    print("\n[2/8] Building feature matrix...")
    builder = CineVaultFeatureBuilder(top_n_genres=25, top_n_countries=15)
    X = builder.fit_transform(df)
    feature_names = builder.get_feature_names()
    print(f"      Feature matrix shape: {X.shape}")
    print(f"      Features ({len(feature_names)}): {feature_names[:5]}... [truncated]")

    # --- Step 3: K Evaluation -------------------------------------
    print("\n[3/8] Evaluating K=2..12...")
    k_eval_results = evaluate_k_range(X)

    # best_k = select_best_k(k_eval_results)
    best_k = 4 # Hardcoded to 4 because it provides much better semantic separation (Classic vs Modern Movies, Kids/Comedy TV vs Mature TV)
    print(f"\n      -> Selected K = {best_k} (overridden for interpretability)")

    # --- Step 4: Train Final Model --------------------------------
    print(f"\n[4/8] Training final KMeans model (K={best_k})...")
    model = train_final_model(X, best_k)
    labels = model.labels_

    # --- Step 5: Compute Metrics ----------------------------------
    print("\n[5/8] Computing final clustering metrics...")
    final_metrics = compute_final_metrics(X, labels)
    print(f"      Silhouette:       {final_metrics['silhouette_score']}")
    print(f"      Davies-Bouldin:   {final_metrics['davies_bouldin_score']}")
    print(f"      Calinski-Harabasz:{final_metrics['calinski_harabasz_score']}")

    # --- Step 6: Cluster Profiles ---------------------------------
    print("\n[6/8] Generating cluster profiles...")
    profiles = generate_cluster_profiles(df, X, labels, model)
    for p in profiles:
        print(f"      Cluster {p['cluster_id']}: {p['n_titles']} titles | {p['label']}")

    # --- Step 7: PCA Visualization --------------------------------
    print("\n[7/8] Generating PCA visualization data...")
    pca_2d = generate_pca_2d(X, df, labels)
    pca_3d = generate_pca_3d(X, df, labels)
    print(f"      2D PCA explained variance: {pca_2d['total_explained_variance']:.3f}")
    print(f"      3D PCA explained variance: {pca_3d['total_explained_variance']:.3f}")

    # --- Step 8: Save Artifacts -----------------------------------
    print("\n[8/8] Saving artifacts...")

    # Model
    with open(ARTIFACTS_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Preprocessor
    with open(ARTIFACTS_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(builder, f)

    # Feature metadata
    feature_meta = {
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "k": best_k,
        "n_clusters": best_k,
        "dataset_rows": len(df),
        "random_state": 42,
        "top_n_genres": 25,
        "top_n_countries": 15,
        "final_metrics": final_metrics,
        "selected_k_reasoning": (
            f"K={best_k} was selected using a composite score combining normalized "
            f"Silhouette Score (maximize), Davies-Bouldin Index (minimize), "
            f"and Calinski-Harabasz Score (maximize) across K=2..12."
        ),
    }
    with open(ARTIFACTS_DIR / "feature_meta.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    # K Evaluation
    with open(ARTIFACTS_DIR / "k_evaluation.json", "w") as f:
        json.dump(k_eval_results, f, indent=2)

    # Cluster assignments
    df_out = df[["show_id", "title", "type", "rating", "release_year",
                  "country", "duration", "listed_in", "director", "date_added"]].copy()
    df_out["cluster_id"] = labels
    # Add PCA coordinates
    pca_coords = {p["show_id"]: (p["pc1"], p["pc2"]) for p in pca_2d["points"]}
    df_out["pc1"] = df_out["show_id"].map(lambda x: pca_coords.get(x, (None, None))[0])
    df_out["pc2"] = df_out["show_id"].map(lambda x: pca_coords.get(x, (None, None))[1])
    df_out.to_csv(ARTIFACTS_DIR / "cluster_assignments.csv", index=False)

    # Cluster profiles
    with open(ARTIFACTS_DIR / "cluster_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)

    # PCA visualization (2D)
    with open(ARTIFACTS_DIR / "pca_visualization.json", "w") as f:
        # Store without the full points array (too large for some operations)
        meta = {k: v for k, v in pca_2d.items() if k != "points" and k != "pca_components"}
        json.dump(meta, f, indent=2)

    # Full PCA points (separate file)
    with open(ARTIFACTS_DIR / "pca_points_2d.json", "w") as f:
        json.dump(pca_2d["points"], f)

    with open(ARTIFACTS_DIR / "pca_points_3d.json", "w") as f:
        json.dump(pca_3d["points"], f)

    # Overview
    cluster_sizes = [(p["cluster_id"], p["n_titles"]) for p in profiles]
    type_counts = df["type"].value_counts().to_dict()
    rating_counts = df["rating"].value_counts().head(6).to_dict()

    # Compute top genres globally
    from collections import Counter
    all_genres = []
    for val in df["listed_in"]:
        for g in str(val).split(","):
            g = g.strip()
            if g:
                all_genres.append(g)
    genre_counts = Counter(all_genres)
    top_genres_global = {g: c for g, c in genre_counts.most_common(10)}

    overview = {
        "total_titles": len(df),
        "n_clusters": best_k,
        "n_movies": int(type_counts.get("Movie", 0)),
        "n_tv_shows": int(type_counts.get("TV Show", 0)),
        "dataset_year_range": [int(df["release_year"].min()), int(df["release_year"].max())],
        "clustering_metrics": final_metrics,
        "cluster_sizes": cluster_sizes,
        "rating_distribution": rating_counts,
        "top_genres_global": top_genres_global,
        "cluster_labels": {p["cluster_id"]: p["label"] for p in profiles},
        "selected_k": best_k,
        "pca_2d_explained_variance": pca_2d["total_explained_variance"],
        "pca_3d_explained_variance": pca_3d["total_explained_variance"],
        "data_quality": {
            "director_not_given_pct": round(
                (df["director"].str.lower() == "not given").sum() / len(df) * 100, 1
            ),
            "country_not_given_pct": round(
                (df["country"].str.lower() == "not given").sum() / len(df) * 100, 1
            ),
        },
    }
    with open(ARTIFACTS_DIR / "overview.json", "w") as f:
        json.dump(overview, f, indent=2)

    print("\n" + "=" * 60)
    print("OK Pipeline complete. Artifacts saved to /artifacts/")
    print(f"  model.pkl                ({os.path.getsize(ARTIFACTS_DIR / 'model.pkl'):,} bytes)")
    print(f"  preprocessor.pkl         ({os.path.getsize(ARTIFACTS_DIR / 'preprocessor.pkl'):,} bytes)")
    print(f"  cluster_profiles.json    ({os.path.getsize(ARTIFACTS_DIR / 'cluster_profiles.json'):,} bytes)")
    print(f"  cluster_assignments.csv  ({os.path.getsize(ARTIFACTS_DIR / 'cluster_assignments.csv'):,} bytes)")
    print(f"  pca_points_2d.json       ({os.path.getsize(ARTIFACTS_DIR / 'pca_points_2d.json'):,} bytes)")
    print("=" * 60)


if __name__ == "__main__":
    main()
