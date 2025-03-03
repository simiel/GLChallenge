import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import joblib
import json
from datetime import datetime

class PredictionPipeline:
    def __init__(self, model_dir: str = 'app/models'):
        """Initialize the prediction pipeline"""
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory {model_dir} does not exist. Please train the model first.")
    
    def _load_artifacts(self, name: str) -> Dict:
        """Helper method to load artifacts"""
        path = os.path.join(self.model_dir, f'{name}.joblib')
        if not os.path.exists(path):
            raise ValueError(f"Required artifact {name} not found in {self.model_dir}")
        return joblib.load(path)
    
    def _load_json(self, name: str) -> Dict:
        """Helper method to load JSON files"""
        path = os.path.join(self.model_dir, f'{name}.json')
        if not os.path.exists(path):
            raise ValueError(f"Required file {name}.json not found in {self.model_dir}")
        with open(path, 'r') as f:
            return json.load(f)
    
    def _validate_input_data(self, df: pd.DataFrame, feature_config: Dict) -> List[str]:
        """Validate input data against training feature configuration"""
        missing_cols = []
        for col_type in ['numerical', 'categorical']:
            for col in feature_config[col_type]:
                if col not in df.columns:
                    missing_cols.append(col)
        return missing_cols
    
    def _preprocess_data(self, df: pd.DataFrame, preprocessing_artifacts: Dict, model) -> pd.DataFrame:
        """Preprocess input data using saved artifacts"""
        # Convert date columns to datetime
        date_cols = ['CustomerDOB', 'TransactionDate']
        for col in date_cols:
            df[col] = pd.to_datetime(df[col])
        
        # Extract features from dates
        df['CustomerAge'] = (datetime.now() - df['CustomerDOB']).dt.days / 365.25
        df['TransactionHour'] = df['TransactionTime'].apply(lambda x: int(x))
        df['TransactionDayOfWeek'] = df['TransactionDate'].dt.dayofweek
        
        # Get feature configuration
        feature_config = self._load_json('feature_columns')
        
        # Handle column name variations
        column_mapping = {
            'TransactionAmount_INR': 'TransactionAmount (INR)'
        }
        df = df.rename(columns=column_mapping)
        
        # Apply saved preprocessing steps
        numerical_cols = feature_config['numerical']
        categorical_cols = feature_config['categorical']
        
        # Scale numerical features
        if 'scaler' in preprocessing_artifacts:
            scaler = preprocessing_artifacts['scaler']
            df[numerical_cols] = scaler.transform(df[numerical_cols])
        
        # One-hot encode categorical features
        expected_features = model.feature_names_in_
        for col in categorical_cols:
            # Create one-hot encoded DataFrame
            encoded_df = pd.get_dummies(df[col], prefix=col)
            
            # Get the expected columns for this categorical feature
            col_prefix = f"{col}_"
            expected_cols = [f for f in expected_features if f.startswith(col_prefix)]
            
            # Ensure all expected columns exist
            for expected_col in expected_cols:
                if expected_col not in encoded_df.columns:
                    encoded_df[expected_col] = 0
            
            # Select only the expected columns
            encoded_df = encoded_df[expected_cols]
            
            # Drop the original column and concatenate encoded columns
            df = df.drop(columns=[col])
            df = pd.concat([df, encoded_df], axis=1)
        
        # Ensure all expected feature columns exist
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select only the expected features
        df = df[expected_features]
        
        return df
    
    def predict_dataframe(self, df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
        """Make predictions on a DataFrame directly"""
        print("\n=== Starting Prediction Pipeline ===")
        
        # Step 1: Load all necessary artifacts
        print("\nStep 1: Loading model and artifacts...")
        try:
            # Load configurations and artifacts
            feature_config = self._load_json('feature_columns')
            preprocessing_artifacts = self._load_artifacts('artifacts')
            model = self._load_artifacts('model')
            
            # Print model feature names for debugging
            print("\nModel feature names:")
            print(model.feature_names_in_)
        except Exception as e:
            raise Exception(f"Error loading model artifacts: {str(e)}")
        
        # Step 2: Validate input data
        print("\nStep 2: Validating input data...")
        print(f"Input data shape: {df.shape}")
        
        # Validate required columns
        missing_cols = self._validate_input_data(df, feature_config)
        if missing_cols:
            raise ValueError(f"Missing required columns in input data: {', '.join(missing_cols)}")
        
        # Step 3: Preprocess input data
        print("\nStep 3: Preprocessing input data...")
        df_processed = self._preprocess_data(df, preprocessing_artifacts, model)
        print(f"Processed data shape: {df_processed.shape}")
        print("\nProcessed feature names:")
        print(df_processed.columns.tolist())
        
        # Step 4: Generate predictions
        print("\nStep 4: Generating predictions...")
        predictions = model.predict(df_processed)
        probabilities = model.predict_proba(df_processed)[:, 1]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'churn_prediction': predictions,
            'churn_probability': probabilities
        }, index=df.index)
        
        # Add metadata
        results['model_version'] = datetime.now().strftime('%Y%m%d')
        results['prediction_date'] = datetime.now().isoformat()
        
        # Combine with original data
        final_results = pd.concat([df, results], axis=1)
        
        return None, final_results  # Return None for file path since we're not saving 