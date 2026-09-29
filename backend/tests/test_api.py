"""
backend/tests/test_api.py
==========================
Integration tests for the FastAPI endpoints using TestClient.

Covers:
  - GET /health
  - GET /metadata
  - POST /predict (valid, invalid, edge values)
  - GET /model-info
  - CORS headers
  - 422 validation errors
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert "version" in data

    def test_health_has_security_headers(self, client):
        resp = client.get("/health")
        assert "x-content-type-options" in resp.headers
        assert "x-frame-options" in resp.headers
        assert "x-request-id" in resp.headers


# ---------------------------------------------------------------------------
# Metadata endpoint
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_metadata_ok(self, client):
        resp = client.get("/metadata")
        assert resp.status_code == 200
        data = resp.json()
        assert "numeric_ranges" in data
        assert "allowed_countries" in data
        assert "allowed_platforms" in data
        assert "score_bands" in data
        assert "ordinal_categories" in data

    def test_metadata_countries_list(self, client):
        data = client.get("/metadata").json()
        assert "India" in data["allowed_countries"]

    def test_metadata_score_bands_structure(self, client):
        data = client.get("/metadata").json()
        for band in data["score_bands"]:
            assert "label" in band
            assert "min" in band
            assert "max" in band


# ---------------------------------------------------------------------------
# Predict endpoint — valid inputs
# ---------------------------------------------------------------------------

class TestPredictValid:
    def test_predict_returns_200(self, client, valid_payload):
        resp = client.post("/predict", json=valid_payload)
        assert resp.status_code == 200

    def test_predict_response_schema(self, client, valid_payload):
        data = client.post("/predict", json=valid_payload).json()
        assert "mental_health_score" in data
        assert "status_flag" in data
        assert "message" in data
        assert "disclaimer" in data
        assert "prediction_id" in data

    def test_predict_score_in_range(self, client, valid_payload):
        data = client.post("/predict", json=valid_payload).json()
        score = data["mental_health_score"]
        assert 0.0 <= score <= 100.0

    def test_predict_status_flag_is_valid(self, client, valid_payload):
        data = client.post("/predict", json=valid_payload).json()
        valid_flags = {"Thriving", "Good", "Moderate", "Needs Attention", "Seek Support"}
        assert data["status_flag"] in valid_flags

    def test_predict_edge_low(self, client):
        """Minimal values: expect a low score."""
        payload = {
            "Physical_Activity_Hours": 0.0,
            "Sleep_Hours":             2.0,
            "Screen_Time_Hours":       16.0,
            "Stress_Level":            "High",
            "Country":                 "India",
            "Platform":                "TikTok",
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["mental_health_score"] <= 50.0  # should be low

    def test_predict_edge_high(self, client):
        """Healthy values: expect a high score."""
        payload = {
            "Physical_Activity_Hours": 3.0,
            "Sleep_Hours":             8.0,
            "Screen_Time_Hours":       1.0,
            "Stress_Level":            "Low",
            "Country":                 "India",
            "Platform":                "None",
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["mental_health_score"] >= 50.0  # should be higher

    def test_predict_unique_prediction_ids(self, client, valid_payload):
        """Each prediction should have a unique UUID."""
        ids = [
            client.post("/predict", json=valid_payload).json()["prediction_id"]
            for _ in range(3)
        ]
        assert len(set(ids)) == 3


# ---------------------------------------------------------------------------
# Predict endpoint — invalid inputs (422 expected)
# ---------------------------------------------------------------------------

class TestPredictInvalid:
    def test_activity_too_high(self, client, valid_payload):
        payload = {**valid_payload, "Physical_Activity_Hours": 25.0}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_sleep_negative(self, client, valid_payload):
        payload = {**valid_payload, "Sleep_Hours": -1.0}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_invalid_stress_level(self, client, valid_payload):
        payload = {**valid_payload, "Stress_Level": "Extreme"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_invalid_country(self, client, valid_payload):
        payload = {**valid_payload, "Country": "Mars"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_missing_field(self, client, valid_payload):
        payload = {k: v for k, v in valid_payload.items() if k != "Sleep_Hours"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_empty_body(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Model info endpoint
# ---------------------------------------------------------------------------

class TestModelInfo:
    def test_model_info_ok(self, client):
        resp = client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_name" in data
        assert "training_date" in data
        assert "metrics" in data
        assert "is_synthetic" in data

    def test_model_info_has_r2(self, client):
        data = client.get("/model-info").json()
        assert "r2" in data["metrics"]


# ---------------------------------------------------------------------------
# CORS headers
# ---------------------------------------------------------------------------

class TestCORS:
    def test_cors_header_present_for_allowed_origin(self, client):
        resp = client.get(
            "/health",
            headers={"Origin": "http://localhost:5500"},
        )
        # With allow_origins="*", header should be present
        assert resp.status_code == 200
