import pytest
import sys
import os

# Make sure app/ is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.predict import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"

def test_predict_valid(client):
    response = client.post("/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert "label" in data
    assert "confidence" in data
    assert data["label"] in ["setosa", "versicolor", "virginica"]

def test_predict_missing_features(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data

def test_predict_setosa(client):
    # Setosa has small petal dimensions
    response = client.post("/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]})
    data = response.get_json()
    assert data["label"] == "setosa"

def test_predict_virginica(client):
    # Virginica has large petal dimensions
    response = client.post("/predict",
        json={"features": [6.7, 3.0, 5.2, 2.3]})
    data = response.get_json()
    assert data["label"] == "virginica"