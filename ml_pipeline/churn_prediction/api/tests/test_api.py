import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json

from ..main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["service"] == "Churn Prediction API"
    assert response.json()["version"] == "1.0.0"

def test_model_health():
    response = client.get("/health/model")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "message" in response.json()

def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "model_type" in data
    assert "metrics" in data
    assert "feature_importance" in data

def test_predict_endpoint():
    # Test data
    test_request = {
        "customers": [{
            "CustomerID": "C1234567",
            "CustomerDOB": "1/1/80",
            "CustGender": "M",
            "CustLocation": "MUMBAI",
            "CustAccountBalance": 50000.0,
            "TransactionID": "T123",
            "TransactionAmount": 1000.0,
            "TransactionTime": 14.5,
            "TransactionDate": "2024-03-01"
        }]
    }
    
    response = client.post("/predict", json=test_request)
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "predictions" in data
    assert "model_version" in data
    assert "prediction_timestamp" in data
    assert "request_id" in data
    
    # Check predictions
    predictions = data["predictions"]
    assert len(predictions) == 1
    prediction = predictions[0]
    assert "customer_id" in prediction
    assert "churn_prediction" in prediction
    assert "churn_probability" in prediction
    assert isinstance(prediction["churn_prediction"], bool)
    assert isinstance(prediction["churn_probability"], float)
    assert 0 <= prediction["churn_probability"] <= 1

def test_predict_endpoint_invalid_data():
    # Test with missing required fields
    invalid_request = {
        "customers": [{
            "CustomerID": "C1234567"  # Missing other required fields
        }]
    }
    
    response = client.post("/predict", json=invalid_request)
    assert response.status_code == 422  # Validation error

def test_predict_endpoint_empty_request():
    # Test with empty customer list
    empty_request = {
        "customers": []
    }
    
    response = client.post("/predict", json=empty_request)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 0

def test_predict_endpoint_multiple_customers():
    # Test with multiple customers
    test_request = {
        "customers": [
            {
                "CustomerID": "C1234567",
                "CustomerDOB": "1/1/80",
                "CustGender": "M",
                "CustLocation": "MUMBAI",
                "CustAccountBalance": 50000.0,
                "TransactionID": "T123",
                "TransactionAmount": 1000.0,
                "TransactionTime": 14.5,
                "TransactionDate": "2024-03-01"
            },
            {
                "CustomerID": "C7654321",
                "CustomerDOB": "1/1/85",
                "CustGender": "F",
                "CustLocation": "DELHI",
                "CustAccountBalance": 75000.0,
                "TransactionID": "T456",
                "TransactionAmount": 2000.0,
                "TransactionTime": 16.5,
                "TransactionDate": "2024-03-01"
            }
        ]
    }
    
    response = client.post("/predict", json=test_request)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2 