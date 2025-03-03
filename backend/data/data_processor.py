import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

class DataProcessor:
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        
    def load_data(self) -> pd.DataFrame:
        """Load the customer churn dataset from Kaggle."""
        # Note: You'll need to download the dataset first using Kaggle API
        # or place it in the data/raw directory
        df = pd.read_csv(os.path.join(self.data_path, "customer_churn.csv"))
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for model training."""
        # Handle missing values
        df = df.fillna({
            'TotalCharges': df['TotalCharges'].median(),
            'MonthlyCharges': df['MonthlyCharges'].median(),
            'tenure': df['tenure'].median()
        })
        
        # Convert categorical variables to numeric
        categorical_columns = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for model training."""
        # Assuming 'Churn' is the target column
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                  test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        return X_train, X_test, y_train, y_test
    
    def get_feature_importance(self, model: Any, feature_names: list) -> Dict[str, float]:
        """Get feature importance scores from the model."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            raise ValueError("Model does not have feature importance attribute")
            
        return dict(zip(feature_names, importance))
    
    def generate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the dataset."""
        return {
            'total_customers': len(df),
            'churn_rate': df['Churn'].mean(),
            'numeric_features': df.select_dtypes(include=['float64', 'int64']).columns.tolist(),
            'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
            'missing_values': df.isnull().sum().to_dict()
        } 