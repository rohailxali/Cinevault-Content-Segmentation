"""
Check K=4 cluster quality vs K=2 to determine whether to override.
The composite score picked K=2, but K=4 may be more analytically valuable.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from ml.pipeline import CineVaultFeatureBuilder, load_and_clean
from ml.clustering import train_final_model, compute_final_metrics
from ml.profiles import generate_cluster_profiles

df = load_and_clean("Dataset.csv")
builder = CineVaultFeatureBuilder(top_n_genres=25, top_n_countries=15)
X = builder.fit_transform(df)

# Load existing K eval
k_eval = json.load(open("artifacts/k_evaluation.json"))
print("=== K Evaluation Summary ===")
for r in k_eval:
    print(f"K={r['k']:2d} | Sil={r['silhouette_score']:.4f} | DB={r['davies_bouldin_score']:.4f} | CH={r['calinski_harabasz_score']:.1f}")

print()
print("=== K=2 Cluster Profile ===")
model2 = train_final_model(X, 2)
profiles2 = generate_cluster_profiles(df, X, model2.labels_, model2)
for p in profiles2:
    print(f"  Cluster {p['cluster_id']}: {p['n_titles']} titles | dominant={p['dominant_type']} ({p['dominant_type_pct']}%) | {p['label']}")
    print(f"    Top genres: {[g['value'] for g in p['top_genres'][:3]]}")

print()
print("=== K=4 Cluster Profile ===")
model4 = train_final_model(X, 4)
m4 = compute_final_metrics(X, model4.labels_)
print(f"  K=4 metrics: Sil={m4['silhouette_score']:.4f} DB={m4['davies_bouldin_score']:.4f} CH={m4['calinski_harabasz_score']:.1f}")
profiles4 = generate_cluster_profiles(df, X, model4.labels_, model4)
for p in profiles4:
    print(f"  Cluster {p['cluster_id']}: {p['n_titles']} titles | dominant={p['dominant_type']} ({p['dominant_type_pct']}%) | {p['label']}")
    print(f"    Top genres: {[g['value'] for g in p['top_genres'][:3]]}")
    print(f"    Top ratings: {[g['value'] for g in p['top_ratings'][:2]]}")
    print(f"    Median year: {p['release_year_median']}")
    print(f"    Top countries: {[g['value'] for g in p['top_countries'][:3]]}")
