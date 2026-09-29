"""
backend/tests/conftest.py
==========================
Shared fixtures for backend tests.
The app is set up with the real trained model (models/pipeline.joblib).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture(scope="session")
def client():
    """
    Create a TestClient with the full FastAPI app.
    The lifespan loads the real model artifact.
    """
    from backend.app.main import app

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def valid_payload() -> dict:
    """A known-good prediction request payload."""
    return {
        "Physical_Activity_Hours": 1.5,
        "Sleep_Hours":             7.0,
        "Screen_Time_Hours":       4.0,
        "Stress_Level":            "Medium",
        "Country":                 "India",
        "Platform":                "Instagram",
    }
