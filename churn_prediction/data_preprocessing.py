import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file
    """
    return pd.read_csv(file_path)

def preprocess_numerical_features(df: pd.DataFrame, numerical_columns: list) -> Tuple[pd.DataFrame, Dict]:
    """
    Preprocess numerical features with scaling and imputation
    Returns preprocessed data and fitted preprocessors
    """
    # Initialize preprocessors
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    # Fit and transform
    df_num = df[numerical_columns].copy()
    df_num_imputed = pd.DataFrame(
        imputer.fit_transform(df_num),
        columns=df_num.columns,
        index=df_num.index
    )
    df_num_scaled = pd.DataFrame(
        scaler.fit_transform(df_num_imputed),
        columns=df_num.columns,
        index=df_num.index
    )
    
    preprocessors = {
        'imputer': imputer,
        'scaler': scaler
    }
    
    return df_num_scaled, preprocessors

def preprocess_categorical_features(df: pd.DataFrame, categorical_columns: list) -> Tuple[pd.DataFrame, Dict]:
    """
    Preprocess categorical features with one-hot encoding
    Returns preprocessed data and encoding mappings
    """
    df_cat = df[categorical_columns].copy()
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df_cat, prefix_sep='_')
    
    # Store category mappings and column names
    category_mappings = {
        col: df_cat[col].unique().tolist()
        for col in categorical_columns
    }
    
    # Store encoded column names
    encoded_columns = df_encoded.columns.tolist()
    
    mappings = {
        'categories': category_mappings,
        'encoded_columns': encoded_columns
    }
    
    return df_encoded, mappings

def prepare_features(
    df: pd.DataFrame,
    numerical_columns: List[str],
    categorical_columns: List[str]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Prepare features for modeling and return preprocessing artifacts
    """
    # Initialize preprocessing artifacts
    artifacts = {
        'numerical_columns': numerical_columns,
        'categorical_columns': categorical_columns,
        'scaler': StandardScaler(),
        'categorical_values': {},
        'feature_names': []  # Store feature names in order
    }
    
    # Process numerical features
    if numerical_columns:
        numerical_data = df[numerical_columns].copy()
        # Fill missing values with median
        for col in numerical_columns:
            median = numerical_data[col].median()
            numerical_data[col] = numerical_data[col].fillna(median)
            artifacts[f'{col}_median'] = median
        
        # Scale numerical features
        numerical_scaled = pd.DataFrame(
            artifacts['scaler'].fit_transform(numerical_data),
            columns=numerical_columns,
            index=df.index
        )
        artifacts['feature_names'].extend(numerical_columns)
    else:
        numerical_scaled = pd.DataFrame(index=df.index)
    
    # Process categorical features
    if categorical_columns:
        categorical_encoded_dfs = []
        for col in categorical_columns:
            # Get unique values and store in artifacts
            unique_values = sorted(df[col].dropna().unique())
            artifacts['categorical_values'][col] = unique_values
            
            # Create dummy variables
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
            categorical_encoded_dfs.append(dummies)
            
            # Store feature names
            artifacts['feature_names'].extend(dummies.columns.tolist())
        
        # Combine all encoded categorical features
        categorical_encoded = pd.concat(categorical_encoded_dfs, axis=1)
    else:
        categorical_encoded = pd.DataFrame(index=df.index)
    
    # Combine numerical and categorical features
    processed_df = pd.concat([numerical_scaled, categorical_encoded], axis=1)
    
    return processed_df, artifacts

def transform_new_data(
    df: pd.DataFrame,
    artifacts: Dict
) -> pd.DataFrame:
    """
    Transform new data using saved preprocessing artifacts
    """
    numerical_columns = artifacts['numerical_columns']
    categorical_columns = artifacts['categorical_columns']
    feature_names = artifacts['feature_names']
    
    # Process numerical features
    if numerical_columns:
        numerical_data = df[numerical_columns].copy()
        # Fill missing values with stored medians
        for col in numerical_columns:
            numerical_data[col] = numerical_data[col].fillna(artifacts[f'{col}_median'])
        
        # Scale numerical features
        numerical_scaled = pd.DataFrame(
            artifacts['scaler'].transform(numerical_data),
            columns=numerical_columns,
            index=df.index
        )
    else:
        numerical_scaled = pd.DataFrame(index=df.index)
    
    # Process categorical features
    if categorical_columns:
        categorical_encoded_dfs = []
        for col in categorical_columns:
            # Create dummy variables for known categories
            dummies = pd.DataFrame(0, index=df.index, columns=[
                f"{col}_{value}" for value in artifacts['categorical_values'][col]
            ])
            
            # Set values for known categories
            for value in artifacts['categorical_values'][col]:
                col_name = f"{col}_{value}"
                dummies[col_name] = (df[col] == value).astype(float)
            
            # Handle unknown values
            dummies[f"{col}_nan"] = df[col].isna().astype(float)
            categorical_encoded_dfs.append(dummies)
        
        # Combine all encoded categorical features
        categorical_encoded = pd.concat(categorical_encoded_dfs, axis=1)
    else:
        categorical_encoded = pd.DataFrame(index=df.index)
    
    # Combine numerical and categorical features
    processed_df = pd.concat([numerical_scaled, categorical_encoded], axis=1)
    
    # Ensure columns are in the same order as during training
    processed_df = processed_df.reindex(columns=feature_names, fill_value=0)
    
    return processed_df 