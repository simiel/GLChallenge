import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def train_model(
    X: pd.DataFrame,
    y: np.ndarray,
    model_type: str = 'rf',
    random_state: int = 42
) -> Tuple[Any, Dict]:
    """
    Train a supervised model for churn prediction
    """
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train model
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred)),
        'recall': float(recall_score(y, y_pred)),
        'f1': float(f1_score(y, y_pred)),
        'roc_auc': float(roc_auc_score(y, y_prob))
    }
    
    # Get feature importance
    feature_importance = {
        'features': X.columns.tolist(),
        'importance': model.feature_importances_.tolist()
    }
    
    artifacts = {
        'metrics': metrics,
        'feature_importance': feature_importance,
        'model_type': model_type,
        'model_params': model.get_params()
    }
    
    return model, artifacts

def save_model(model: Any, artifacts: Dict, model_path: str, artifacts_path: str):
    """
    Save the trained model and artifacts
    """
    joblib.dump(model, model_path)
    joblib.dump(artifacts, artifacts_path)

def load_model(model_path: str, artifacts_path: str) -> Tuple[Any, Dict]:
    """
    Load a trained model and artifacts
    """
    model = joblib.load(model_path)
    artifacts = joblib.load(artifacts_path)
    return model, artifacts

def predict_churn(model: Any, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using a trained model
    Returns predictions and probabilities
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return predictions, probabilities

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
    """
    Evaluate predictions using various metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    } 