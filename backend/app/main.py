from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from typing import List, Dict, Optional
import json
from datetime import datetime

from .api.predict import PredictionPipeline

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn risk",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction pipeline
pipeline = PredictionPipeline(model_dir='app/models')

class CustomerData(BaseModel):
    CustomerID: str
    CustomerDOB: str
    CustGender: str
    CustLocation: str
    CustAccountBalance: float
    TransactionID: str
    TransactionAmount_INR: float = 0.0  # Default value for backward compatibility
    TransactionTime: float
    TransactionDate: str

    class Config:
        json_schema_extra = {
            "example": {
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
        }

class PredictionRequest(BaseModel):
    customers: List[CustomerData]

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, any]]
    model_version: str
    prediction_timestamp: str
    request_id: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Churn Prediction API",
        "version": "1.0.0"
    }

@app.get("/model/info")
async def model_info():
    """Get model information and metrics"""
    try:
        feature_config = pipeline._load_json('feature_columns')
        artifacts = pipeline._load_artifacts('artifacts')
        
        return {
            "model_version": datetime.now().strftime('%Y%m%d'),
            "model_type": "Random Forest Classifier",
            "feature_columns": feature_config,
            "preprocessing_info": {
                "numerical_columns": artifacts.get('numerical_cols', []),
                "categorical_columns": artifacts.get('categorical_cols', [])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model info: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate churn predictions for a list of customers"""
    try:
        # Convert input data to DataFrame
        customers_data = []
        for customer in request.customers:
            customer_dict = customer.dict()
            # Rename TransactionAmount_INR to TransactionAmount (INR)
            if 'TransactionAmount_INR' in customer_dict:
                customer_dict['TransactionAmount (INR)'] = customer_dict.pop('TransactionAmount_INR')
            customers_data.append(customer_dict)
        
        df = pd.DataFrame(customers_data)
        
        # Generate predictions
        _, predictions_df = pipeline.predict_dataframe(df)
        
        # Format response
        predictions = []
        for _, row in predictions_df.iterrows():
            predictions.append({
                "customer_id": str(row['CustomerID']),
                "churn_prediction": bool(row['churn_prediction']),
                "churn_probability": float(row['churn_probability'])
            })
        
        return {
            "predictions": predictions,
            "model_version": datetime.now().strftime('%Y%m%d'),
            "prediction_timestamp": datetime.now().isoformat(),
            "request_id": datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        }
    except Exception as e:
        import traceback
        error_details = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        raise HTTPException(
            status_code=500,
            detail=error_details
        )

@app.get("/health/model")
async def model_health():
    """Check if model and all required artifacts are available"""
    try:
        # Check all required artifacts
        pipeline._load_json('feature_columns')
        pipeline._load_artifacts('artifacts')
        pipeline._load_artifacts('model')
        
        return {
            "status": "healthy",
            "message": "All model artifacts available"
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready: {str(e)}"
        ) 