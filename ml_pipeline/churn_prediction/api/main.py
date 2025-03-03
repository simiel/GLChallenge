from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from typing import List, Dict, Optional
import json
from datetime import datetime

from ..predict_pipeline import PredictionPipeline

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
pipeline = PredictionPipeline(model_dir='models')

class CustomerData(BaseModel):
    CustomerID: str
    CustomerDOB: str
    CustGender: str
    CustLocation: str
    CustAccountBalance: float
    TransactionID: str
    TransactionAmount: float
    TransactionTime: float
    TransactionDate: str

class PredictionRequest(BaseModel):
    customers: List[CustomerData]

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, float]]
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
        metadata = pipeline._load_json('pipeline_metadata')
        metrics = pipeline._load_json('model_metrics')
        
        return {
            "model_version": metadata['training_date'],
            "model_type": metadata['model_type'],
            "metrics": metadata['metrics'],
            "feature_importance": metadata['feature_importance']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model info: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate churn predictions for a list of customers"""
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([customer.dict() for customer in request.customers])
        
        # Generate predictions
        _, predictions_df = pipeline.predict_dataframe(df)
        
        # Format response
        predictions = []
        for _, row in predictions_df.iterrows():
            predictions.append({
                "customer_id": row['CustomerID'],
                "churn_prediction": bool(row['churn_prediction']),
                "churn_probability": float(row['churn_probability'])
            })
        
        # Get model metadata
        metadata = pipeline._load_json('pipeline_metadata')
        
        return {
            "predictions": predictions,
            "model_version": metadata['training_date'],
            "prediction_timestamp": datetime.now().isoformat(),
            "request_id": datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health/model")
async def model_health():
    """Check if model and all required artifacts are available"""
    try:
        # Check all required artifacts
        pipeline._load_json('feature_config')
        pipeline._load_json('pipeline_metadata')
        pipeline._load_json('model_metrics')
        pipeline._load_artifacts('preprocessing')
        
        return {
            "status": "healthy",
            "message": "All model artifacts available"
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready: {str(e)}"
        ) 