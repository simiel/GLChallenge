import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Any
import optuna

class EnsembleModel:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42),
            'xgboost': XGBClassifier(random_state=42),
            'lightgbm': LGBMClassifier(random_state=42),
            'catboost': CatBoostClassifier(random_state=42, verbose=False)
        }
        self.weights = None
        
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train all models in the ensemble."""
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
    
    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Optimize model weights using Optuna."""
        def objective(trial):
            weights = []
            for _ in range(len(self.models)):
                weights.append(trial.suggest_float(f'weight_{_}', 0, 1))
            weights = np.array(weights) / np.sum(weights)
            
            predictions = np.zeros((X_val.shape[0], len(self.models)))
            for i, model in enumerate(self.models.values()):
                predictions[:, i] = model.predict_proba(X_val)[:, 1]
            
            ensemble_pred = np.sum(predictions * weights, axis=1)
            ensemble_pred = (ensemble_pred > 0.5).astype(int)
            
            return f1_score(y_val, ensemble_pred)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        
        # Get the best weights
        self.weights = []
        for i in range(len(self.models)):
            self.weights.append(study.best_params[f'weight_{i}'])
        self.weights = np.array(self.weights) / np.sum(self.weights)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the weighted ensemble."""
        if self.weights is None:
            raise ValueError("Model weights not optimized. Run optimize_weights first.")
        
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models.values()):
            predictions[:, i] = model.predict_proba(X)[:, 1]
        
        ensemble_pred = np.sum(predictions * self.weights, axis=1)
        return (ensemble_pred > 0.5).astype(int)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the ensemble model."""
        y_pred = self.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get weighted feature importance from all models."""
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                continue
                
            for i, imp in enumerate(importance):
                feature = feature_names[i]
                if feature not in importance_dict:
                    importance_dict[feature] = 0
                importance_dict[feature] += imp
        
        # Normalize importance scores
        total = sum(importance_dict.values())
        return {k: v/total for k, v in importance_dict.items()} 