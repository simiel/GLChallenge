import pandas as pd
import numpy as np
from typing import Tuple, Dict
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
    
    # Store category mappings
    category_mappings = {
        col: df_cat[col].unique().tolist()
        for col in categorical_columns
    }
    
    return df_encoded, category_mappings

def prepare_features(
    df: pd.DataFrame,
    numerical_columns: list,
    categorical_columns: list
) -> Tuple[pd.DataFrame, Dict]:
    """
    Prepare all features for modeling
    Returns preprocessed data and preprocessing artifacts
    """
    # Preprocess numerical features
    df_num, num_preprocessors = preprocess_numerical_features(df, numerical_columns)
    
    # Preprocess categorical features
    df_cat, cat_mappings = preprocess_categorical_features(df, categorical_columns)
    
    # Combine features
    df_processed = pd.concat([df_num, df_cat], axis=1)
    
    preprocessing_artifacts = {
        'numerical_preprocessors': num_preprocessors,
        'categorical_mappings': cat_mappings,
        'feature_columns': {
            'numerical': numerical_columns,
            'categorical': categorical_columns
        }
    }
    
    return df_processed, preprocessing_artifacts

def transform_new_data(
    df: pd.DataFrame,
    preprocessing_artifacts: Dict
) -> pd.DataFrame:
    """
    Transform new data using fitted preprocessors
    """
    num_cols = preprocessing_artifacts['feature_columns']['numerical']
    cat_cols = preprocessing_artifacts['feature_columns']['categorical']
    
    # Transform numerical features
    df_num = df[num_cols].copy()
    imputer = preprocessing_artifacts['numerical_preprocessors']['imputer']
    scaler = preprocessing_artifacts['numerical_preprocessors']['scaler']
    
    df_num_imputed = pd.DataFrame(
        imputer.transform(df_num),
        columns=df_num.columns,
        index=df_num.index
    )
    df_num_scaled = pd.DataFrame(
        scaler.transform(df_num_imputed),
        columns=df_num.columns,
        index=df_num.index
    )
    
    # Transform categorical features
    df_cat = df[cat_cols].copy()
    df_encoded = pd.get_dummies(df_cat, prefix_sep='_')
    
    # Ensure all columns from training are present
    for col in preprocessing_artifacts['categorical_mappings']:
        expected_cols = [f"{col}_{val}" for val in preprocessing_artifacts['categorical_mappings'][col]]
        for exp_col in expected_cols:
            if exp_col not in df_encoded.columns:
                df_encoded[exp_col] = 0
    
    return pd.concat([df_num_scaled, df_encoded], axis=1) 