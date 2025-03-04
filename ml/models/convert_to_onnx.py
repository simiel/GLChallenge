import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import onnx
import onnxruntime
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType, Int64TensorType
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
ONNX_DIR = MODELS_DIR / "onnx"
ONNX_DIR.mkdir(exist_ok=True)

def load_model_artifacts():
    """Load the trained model and its artifacts."""
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

def create_initial_types(feature_names, label_encoders):
    """Create initial types for ONNX conversion."""
    initial_types = []
    
    # Add numerical features
    numerical_features = [name for name in feature_names if not name.endswith('_encoded')]
    for name in numerical_features:
        initial_types.append((name, FloatTensorType([None, 1])))
    
    # Add categorical features
    for col, encoder in label_encoders.items():
        initial_types.append((col, StringTensorType([None, 1])))
    
    return initial_types

def convert_to_onnx(model, scaler, feature_names, label_encoders):
    """Convert the model pipeline to ONNX format."""
    logger.info("Converting model to ONNX format...")
    
    # Create initial types
    initial_types = create_initial_types(feature_names, label_encoders)
    
    # Create preprocessing pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # Create preprocessing steps
    numerical_features = [name for name in feature_names if not name.endswith('_encoded')]
    categorical_features = [name for name in feature_names if name.endswith('_encoded')]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', 'passthrough', categorical_features)
        ]
    )
    
    # Create full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Convert to ONNX
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        options={'zipmap': False}
    )
    
    # Save the ONNX model
    onnx_path = ONNX_DIR / "churn_model.onnx"
    onnx.save(onnx_model, onnx_path)
    logger.info(f"ONNX model saved to {onnx_path}")
    
    return onnx_path

def verify_onnx_model(onnx_path, feature_names, label_encoders):
    """Verify the ONNX model with sample data."""
    logger.info("Verifying ONNX model...")
    
    # Create sample data
    sample_data = {
        'TransactionCount': np.array([[10]], dtype=np.float32),
        'AverageTransactionAmount': np.array([[100.50]], dtype=np.float32),
        'TotalTransactionAmount': np.array([[1005.00]], dtype=np.float32),
        'TransactionAmountStd': np.array([[25.30]], dtype=np.float32),
        'Age': np.array([[35]], dtype=np.float32),
        'AccountBalance': np.array([[5000.00]], dtype=np.float32),
        'DaysSinceLastTransaction': np.array([[5]], dtype=np.float32),
        'CustomerTenure': np.array([[12]], dtype=np.float32),
        'TransactionsPerMonth': np.array([[2.5]], dtype=np.float32),
        'Gender': np.array([['Female']], dtype=str),
        'Location': np.array([['Rural']], dtype=str)
    }
    
    # Create ONNX Runtime session
    session = onnxruntime.InferenceSession(str(onnx_path))
    
    # Get input names
    input_names = [input.name for input in session.get_inputs()]
    
    # Prepare input data
    inputs = {name: sample_data[name] for name in input_names}
    
    # Run inference
    outputs = session.run(None, inputs)
    
    # Print results
    logger.info("ONNX Model Verification Results:")
    logger.info(f"Prediction: {outputs[0]}")
    logger.info(f"Probabilities: {outputs[1]}")
    
    return outputs

def main():
    """Main function to convert model to ONNX format."""
    try:
        # Load model artifacts
        model, scaler, feature_names, label_encoders = load_model_artifacts()
        
        # Convert to ONNX
        onnx_path = convert_to_onnx(model, scaler, feature_names, label_encoders)
        
        # Verify the ONNX model
        verify_onnx_model(onnx_path, feature_names, label_encoders)
        
        logger.info("Model conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model conversion: {str(e)}")
        raise

if __name__ == "__main__":
    main() 