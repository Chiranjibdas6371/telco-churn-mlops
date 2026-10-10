
import sys
from pathlib import Path

import pytest

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app as flask_app


@pytest.fixture
def client():
    flask_app.app.config["TESTING"] = True

    with flask_app.app.test_client() as test_client:
        yield test_client


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_metrics_endpoint(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert b"api_requests_total" in response.data



def test_prediction_rejects_missing_fields(client):
    response = client.post("/predict", json={})

    assert response.status_code == 400

    data = response.get_json()
    assert "error" in data
    assert "Missing fields:" in data["error"]

