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
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Any, Dict]:
    """
    Train a supervised model for churn prediction
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize model
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=random_state
        )
    elif model_type == 'gb':
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        feature_importance = None
    
    artifacts = {
        'metrics': metrics,
        'feature_importance': feature_importance,
        'test_indices': X_test.index,
        'model_type': model_type
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