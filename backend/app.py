from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set up paths
MODELS_DIR = Path(__file__).parent / "models"

# Load model artifacts
def load_model_artifacts():
    """Load the trained model and its artifacts."""
    try:
        logger.info("Loading model artifacts...")
        
        # Load the model
        model = joblib.load(MODELS_DIR / "churn_model.joblib")
        logger.info("Model loaded successfully")
        
        # Load the scaler
        scaler = joblib.load(MODELS_DIR / "scaler.joblib")
        logger.info("Scaler loaded successfully")
        
        # Load feature names
        with open(MODELS_DIR / "feature_names.json", 'r') as f:
            feature_names = json.load(f)
        logger.info("Feature names loaded successfully")
        
        # Load label encoders
        label_encoders = joblib.load(MODELS_DIR / "label_encoders.joblib")
        logger.info("Label encoders loaded successfully")
        
        return model, scaler, feature_names, label_encoders
    
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        raise

# Load model artifacts at startup
model, scaler, feature_names, label_encoders = load_model_artifacts()

def prepare_features(data, feature_names, label_encoders):
    """Prepare features for prediction."""
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([data])
        
        # Encode categorical features
        for col, encoder in label_encoders.items():
            # Handle missing categories
            if data.get(col) not in encoder.classes_:
                logger.warning(f"Unknown category '{data.get(col)}' in {col}. Using most frequent category.")
                data[col] = encoder.classes_[0]
            df[col + '_encoded'] = encoder.transform(df[col])
        
        # Select only the features used by the model
        X = df[feature_names]
        
        return X
    
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict_churn():
    """Endpoint for churn prediction."""
    try:
        # Get input data
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'TransactionCount', 'AverageTransactionAmount', 'TotalTransactionAmount',
            'TransactionAmountStd', 'Age', 'AccountBalance', 'DaysSinceLastTransaction',
            'CustomerTenure', 'TransactionsPerMonth', 'Gender', 'Location'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Prepare features
        X = prepare_features(data, feature_names, label_encoders)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        # Prepare response
        response = {
            'prediction': bool(prediction),
            'churn_probability': float(probability),
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low',
            'input_data': data
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}")
        return jsonify({
            'error': 'Error processing prediction request',
            'details': str(e)
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Endpoint for model information."""
    try:
        # Get available categories for categorical features
        categories = {
            col: encoder.classes_.tolist()
            for col, encoder in label_encoders.items()
        }
        
        return jsonify({
            'feature_names': feature_names,
            'categorical_features': categories,
            'model_type': type(model).__name__
        })
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'error': 'Error getting model info',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    # For development
    app.run(host='0.0.0.0', port=5000) 