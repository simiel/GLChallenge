from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any
import numpy as np
import joblib
import os
from ..data.data_processor import DataProcessor
from ..models.ensemble_model import EnsembleModel

app = FastAPI(title="Customer Churn Prediction API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data processor and model
data_processor = DataProcessor()
model = EnsembleModel()

class CustomerData(BaseModel):
    features: Dict[str, Any]

class BatchPredictionRequest(BaseModel):
    customers: List[CustomerData]

@app.get("/")
async def root():
    return {"message": "Welcome to the Customer Churn Prediction API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_churn(customer: CustomerData):
    try:
        # Convert input data to numpy array
        features = np.array(list(customer.features.values())).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = float(model.predict_proba(features)[0][1])
        
        return {
            "churn_prediction": bool(prediction),
            "churn_probability": probability
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    try:
        # Convert input data to numpy array
        features = np.array([list(customer.features.values()) 
                           for customer in request.customers])
        
        # Make predictions
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)[:, 1]
        
        return {
            "predictions": [bool(pred) for pred in predictions],
            "probabilities": [float(prob) for prob in probabilities]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model/features")
async def get_feature_importance():
    try:
        # Get feature names from the data processor
        feature_names = data_processor.get_feature_names()
        
        # Get feature importance
        importance = model.get_feature_importance(feature_names)
        
        return {
            "feature_importance": importance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/summary")
async def get_data_summary():
    try:
        # Load and process data
        df = data_processor.load_data()
        df = data_processor.preprocess_data(df)
        
        # Generate summary statistics
        summary = data_processor.generate_summary_stats(df)
        
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/metrics")
async def get_model_metrics():
    try:
        # Load and process data
        df = data_processor.load_data()
        df = data_processor.preprocess_data(df)
        X, y = data_processor.prepare_features(df)
        X_train, X_test, y_train, y_test = data_processor.split_data(X, y)
        
        # Train model if not already trained
        if not hasattr(model, 'is_trained') or not model.is_trained:
            model.train_models(X_train, y_train)
            model.optimize_weights(X_test, y_test)
            model.is_trained = True
        
        # Get model metrics
        metrics = model.evaluate(X_test, y_test)
        
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 