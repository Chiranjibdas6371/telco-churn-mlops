
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app as flask_app


class MockModel:
    def predict(self, X):
        return [0]

    def predict_proba(self, X):
        return [[0.8, 0.2]]


@pytest.fixture
def client(monkeypatch):
    # Use a mock model so tests do not require best_model.pkl.
    monkeypatch.setattr(flask_app, "model", MockModel())

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
    assert data["model_loaded"] is True


def test_metrics_endpoint(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert b"api_requests_total" in response.data


def test_prediction_rejects_missing_fields(client):
    response = client.post("/predict", json={})

    assert response.status_code == 400
    assert "error" in response.get_json()
