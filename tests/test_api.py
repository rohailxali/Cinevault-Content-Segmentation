"""
CineVault API Tests — pytest suite using FastAPI TestClient

Prerequisites: Model artifacts must exist (run ml/run_pipeline.py first).
Run: pytest tests/test_api.py -v
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Check artifacts before importing
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS_EXIST = (ARTIFACTS / "overview.json").exists()

if ARTIFACTS_EXIST:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
else:
    client = None


# ─── Skip marker if artifacts not ready ───────────────────────────────────────
requires_artifacts = pytest.mark.skipif(
    not ARTIFACTS_EXIST,
    reason="Model artifacts not found. Run ml/run_pipeline.py first."
)


# ─── Health Tests ─────────────────────────────────────────────────────────────
class TestHealth:
    @requires_artifacts
    def test_health_returns_200(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    @requires_artifacts
    def test_health_structure(self):
        resp = client.get("/api/health")
        data = resp.json()
        assert "status" in data
        assert "artifacts_available" in data
        assert data["status"] == "ok"

    @requires_artifacts
    def test_health_artifacts_available(self):
        resp = client.get("/api/health")
        data = resp.json()
        assert data["artifacts_available"] is True


# ─── Overview Tests ───────────────────────────────────────────────────────────
class TestOverview:
    @requires_artifacts
    def test_overview_returns_200(self):
        resp = client.get("/api/overview")
        assert resp.status_code == 200

    @requires_artifacts
    def test_overview_total_titles(self):
        resp = client.get("/api/overview")
        data = resp.json()
        assert "total_titles" in data
        assert data["total_titles"] >= 8700

    @requires_artifacts
    def test_overview_has_clustering_metrics(self):
        resp = client.get("/api/overview")
        data = resp.json()
        assert "clustering_metrics" in data
        metrics = data["clustering_metrics"]
        assert "silhouette_score" in metrics
        assert "davies_bouldin_score" in metrics
        assert "calinski_harabasz_score" in metrics

    @requires_artifacts
    def test_overview_cluster_count_positive(self):
        resp = client.get("/api/overview")
        data = resp.json()
        assert data["n_clusters"] >= 2

    @requires_artifacts
    def test_overview_type_counts_sum_to_total(self):
        resp = client.get("/api/overview")
        data = resp.json()
        total = data.get("total_titles", 0)
        movies = data.get("n_movies", 0)
        tv_shows = data.get("n_tv_shows", 0)
        assert movies + tv_shows == total


# ─── Clusters Tests ───────────────────────────────────────────────────────────
class TestClusters:
    @requires_artifacts
    def test_clusters_returns_200(self):
        resp = client.get("/api/clusters")
        assert resp.status_code == 200

    @requires_artifacts
    def test_clusters_is_list(self):
        resp = client.get("/api/clusters")
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 2

    @requires_artifacts
    def test_cluster_has_required_fields(self):
        resp = client.get("/api/clusters")
        data = resp.json()
        required = {"cluster_id", "label", "n_titles", "pct_of_dataset",
                    "dominant_type", "top_genres", "top_ratings"}
        for cluster in data:
            assert required.issubset(set(cluster.keys()))

    @requires_artifacts
    def test_cluster_sizes_sum_to_total(self):
        clusters_resp = client.get("/api/clusters")
        overview_resp = client.get("/api/overview")
        clusters = clusters_resp.json()
        total = overview_resp.json()["total_titles"]
        assert sum(c["n_titles"] for c in clusters) == total


# ─── Cluster Detail Tests ──────────────────────────────────────────────────────
class TestClusterDetail:
    @requires_artifacts
    def test_cluster_0_detail_returns_200(self):
        resp = client.get("/api/clusters/0")
        assert resp.status_code == 200

    @requires_artifacts
    def test_cluster_detail_has_representative_titles(self):
        resp = client.get("/api/clusters/0")
        data = resp.json()
        assert "representative_titles" in data
        assert len(data["representative_titles"]) > 0

    @requires_artifacts
    def test_invalid_cluster_returns_404(self):
        resp = client.get("/api/clusters/9999")
        assert resp.status_code == 404

    @requires_artifacts
    def test_cluster_detail_has_lift_data(self):
        resp = client.get("/api/clusters/0")
        data = resp.json()
        for genre in data.get("top_genres", []):
            assert "value" in genre
            assert "cluster_pct" in genre
            assert "lift" in genre


# ─── Titles Tests ─────────────────────────────────────────────────────────────
class TestTitles:
    @requires_artifacts
    def test_titles_returns_200(self):
        resp = client.get("/api/titles")
        assert resp.status_code == 200

    @requires_artifacts
    def test_titles_default_pagination(self):
        resp = client.get("/api/titles")
        data = resp.json()
        assert "total" in data
        assert "page" in data
        assert "results" in data
        assert data["page"] == 1
        assert len(data["results"]) == 20

    @requires_artifacts
    def test_titles_search(self):
        resp = client.get("/api/titles?search=movie")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 0

    @requires_artifacts
    def test_titles_search_specific_title(self):
        resp = client.get("/api/titles?search=Dick+Johnson")
        data = resp.json()
        assert data["total"] >= 1
        titles_lower = [r["title"].lower() for r in data["results"]]
        assert any("dick johnson" in t for t in titles_lower)

    @requires_artifacts
    def test_titles_cluster_filter(self):
        resp = client.get("/api/titles?cluster=0")
        data = resp.json()
        assert data["total"] > 0
        for result in data["results"]:
            assert result["cluster_id"] == 0

    @requires_artifacts
    def test_titles_type_filter_movie(self):
        resp = client.get("/api/titles?type=Movie")
        data = resp.json()
        assert data["total"] > 0
        for result in data["results"]:
            assert result["type"] == "Movie"

    @requires_artifacts
    def test_titles_type_filter_tv(self):
        resp = client.get("/api/titles?type=TV+Show")
        data = resp.json()
        assert data["total"] > 0
        for result in data["results"]:
            assert result["type"] == "TV Show"

    @requires_artifacts
    def test_titles_year_range_filter(self):
        resp = client.get("/api/titles?year_min=2020&year_max=2021")
        data = resp.json()
        assert data["total"] > 0
        for result in data["results"]:
            assert 2020 <= result["release_year"] <= 2021

    @requires_artifacts
    def test_titles_custom_page_limit(self):
        resp = client.get("/api/titles?page=2&limit=10")
        data = resp.json()
        assert data["page"] == 2
        assert len(data["results"]) <= 10

    @requires_artifacts
    def test_titles_limit_out_of_range_clamped(self):
        resp = client.get("/api/titles?limit=200")
        # Should clamp to max=100 or return validation error
        assert resp.status_code in (200, 422)

    @requires_artifacts
    def test_empty_search_returns_valid(self):
        resp = client.get("/api/titles?search=xyzunlikelytitle123456")
        data = resp.json()
        assert data["total"] == 0
        assert data["results"] == []


# ─── Title Detail Tests ────────────────────────────────────────────────────────
class TestTitleDetail:
    @requires_artifacts
    def test_title_s1_returns_200(self):
        resp = client.get("/api/titles/s1")
        assert resp.status_code == 200

    @requires_artifacts
    def test_title_has_cluster_id(self):
        resp = client.get("/api/titles/s1")
        data = resp.json()
        assert "cluster_id" in data
        assert isinstance(data["cluster_id"], int)

    @requires_artifacts
    def test_nonexistent_title_returns_404(self):
        resp = client.get("/api/titles/s999999")
        assert resp.status_code == 404


# ─── Filters Tests ────────────────────────────────────────────────────────────
class TestFilters:
    @requires_artifacts
    def test_filters_returns_200(self):
        resp = client.get("/api/filters")
        assert resp.status_code == 200

    @requires_artifacts
    def test_filters_structure(self):
        resp = client.get("/api/filters")
        data = resp.json()
        assert "ratings" in data
        assert "types" in data
        assert "year_range" in data
        assert "clusters" in data
        assert "top_genres" in data

    @requires_artifacts
    def test_filters_types_correct(self):
        resp = client.get("/api/filters")
        data = resp.json()
        types = data["types"]
        assert "Movie" in types
        assert "TV Show" in types

    @requires_artifacts
    def test_filters_year_range_valid(self):
        resp = client.get("/api/filters")
        data = resp.json()
        year_range = data["year_range"]
        assert len(year_range) == 2
        assert year_range[0] < year_range[1]
        assert year_range[0] >= 1900  # oldest reasonable content
        assert year_range[1] <= 2025


# ─── Visualization Tests ──────────────────────────────────────────────────────
class TestVisualization:
    @requires_artifacts
    def test_visualization_returns_200(self):
        resp = client.get("/api/visualization")
        assert resp.status_code == 200

    @requires_artifacts
    def test_visualization_has_points(self):
        resp = client.get("/api/visualization")
        data = resp.json()
        assert "points" in data
        assert len(data["points"]) > 0

    @requires_artifacts
    def test_visualization_explained_variance(self):
        resp = client.get("/api/visualization")
        data = resp.json()
        assert "explained_variance_ratio" in data
        for v in data["explained_variance_ratio"]:
            assert 0 <= v <= 1

    @requires_artifacts
    def test_visualization_point_has_coordinates(self):
        resp = client.get("/api/visualization")
        data = resp.json()
        pt = data["points"][0]
        assert "pc1" in pt
        assert "pc2" in pt
        assert "cluster_id" in pt


# ─── Evaluation Tests ─────────────────────────────────────────────────────────
class TestEvaluation:
    @requires_artifacts
    def test_evaluation_returns_200(self):
        resp = client.get("/api/evaluation")
        assert resp.status_code == 200

    @requires_artifacts
    def test_evaluation_has_k_range(self):
        resp = client.get("/api/evaluation")
        data = resp.json()
        assert "k_evaluation" in data
        assert len(data["k_evaluation"]) >= 5

    @requires_artifacts
    def test_evaluation_selected_k_positive(self):
        resp = client.get("/api/evaluation")
        data = resp.json()
        assert data["selected_k"] >= 2

    @requires_artifacts
    def test_evaluation_has_feature_names(self):
        resp = client.get("/api/evaluation")
        data = resp.json()
        assert "feature_names" in data
        assert len(data["feature_names"]) > 0

    @requires_artifacts
    def test_evaluation_no_identifier_features(self):
        resp = client.get("/api/evaluation")
        data = resp.json()
        names = [n.lower() for n in data["feature_names"]]
        assert "show_id" not in names
        assert "title" not in names
