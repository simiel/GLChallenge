import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_health():
    print("\nTesting health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure the FastAPI server is running.")
        print("Run 'uvicorn app.main:app --reload' in the backend directory first.")
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Response text: {response.text if 'response' in locals() else 'No response'}")

def test_model_health():
    print("\nTesting model health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health/model")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure the FastAPI server is running.")
        print("Run 'uvicorn app.main:app --reload' in the backend directory first.")
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Response text: {response.text if 'response' in locals() else 'No response'}")

def test_model_info():
    print("\nTesting model info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure the FastAPI server is running.")
        print("Run 'uvicorn app.main:app --reload' in the backend directory first.")
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Response text: {response.text if 'response' in locals() else 'No response'}")

def test_predict():
    print("\nTesting prediction endpoint...")
    test_data = {
        "customers": [
            {
                "CustomerID": "C1234567",
                "CustomerDOB": "1/1/80",
                "CustGender": "M",
                "CustLocation": "MUMBAI",
                "CustAccountBalance": 50000.0,
                "TransactionID": "T123",
                "TransactionAmount_INR": 1000.0,
                "TransactionTime": 14.5,
                "TransactionDate": "2024-03-01"
            }
        ]
    }
    try:
        response = requests.post(f"{BASE_URL}/predict", json=test_data)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 500:
            print("Error Details:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure the FastAPI server is running.")
        print("Run 'uvicorn app.main:app --reload' in the backend directory first.")
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Response text: {response.text if 'response' in locals() else 'No response'}")

if __name__ == "__main__":
    print("Starting API tests...")
    test_health()
    test_model_health()
    test_model_info()
    test_predict() 