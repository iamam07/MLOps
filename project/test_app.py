# project/test_app.py
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={
        "sentence1": "This is a test.",
        "sentence2": "This is a trial."
    })
    assert response.status_code == 200
    assert "similarity" in response.json()
