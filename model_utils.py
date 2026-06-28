"""
Utility functions for model serialization and deserialization for federated learning
"""
import pickle
import numpy as np
import xgboost as xgb
from typing import Dict, List, Any
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_label_encoders():
    """Load label encoders"""
    with open('label_encoders.pkl', 'rb') as f:
        return pickle.load(f)

def get_model_params(model: xgb.XGBRegressor) -> Dict[str, Any]:
    """Extract model parameters for federated learning"""
    # Get the booster object
    booster = model.get_booster()
    
    # Get tree structure as JSON
    model_json = booster.get_dump(dump_format='json')
    
    # Get model parameters
    params = model.get_params()
    
    # Get feature importance to understand feature mapping
    feature_importance = model.feature_importances_
    
    return {
        'model_json': model_json,
        'params': params,
        'feature_importance': feature_importance.tolist(),
        'n_estimators': params.get('n_estimators', 100),
        'max_depth': params.get('max_depth', 6),
        'learning_rate': params.get('learning_rate', 0.1)
    }

def create_model_from_params(model_params: Dict[str, Any]) -> xgb.XGBRegressor:
    """Create XGBoost model from parameters"""
    params = model_params['params'].copy()
    
    # Create model with same parameters
    model = xgb.XGBRegressor(**params)
    
    # We can't directly restore tree structure from JSON in XGBoost,
    # so we'll return a model initialized with the same parameters
    # The actual training will happen on clients
    return model

def prepare_features_for_training(df: pd.DataFrame, label_encoders: Dict[str, LabelEncoder]) -> tuple:
    """Prepare features for training"""
    # Drop unnecessary columns
    columns_to_drop = ['Variety', 'Grade', 'Commodity_Code']
    existing_drops = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_drops)
    
    # Convert date
    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Arrival_Date'])
    
    df['Year'] = df['Arrival_Date'].dt.year
    df['Month'] = df['Arrival_Date'].dt.month
    df['Day'] = df['Arrival_Date'].dt.day
    df['Day_of_Week'] = df['Arrival_Date'].dt.dayofweek
    df['Week_of_Year'] = df['Arrival_Date'].dt.isocalendar().week
    
    # Encode categorical variables
    categorical_cols = ['State', 'District', 'Market', 'Commodity']
    for col in categorical_cols:
        if col in df.columns and col in label_encoders:
            df[col] = df[col].astype(str)
            # Handle unknown categories
            known_categories = set(label_encoders[col].classes_)
            df[col] = df[col].apply(
                lambda x: x if x in known_categories else label_encoders[col].classes_[0]
            )
            df[col] = label_encoders[col].transform(df[col])
    
    # Handle missing values in price columns
    price_cols = ['Min_Price', 'Max_Price', 'Modal_Price']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Sort by date
    df = df.sort_values(by='Arrival_Date').reset_index(drop=True)
    
    # Create lag and moving average features
    df['Modal_Price_Lag_1'] = df.groupby(['State', 'District', 'Market', 'Commodity'])['Modal_Price'].shift(1)
    df['Modal_Price_Lag_7'] = df.groupby(['State', 'District', 'Market', 'Commodity'])['Modal_Price'].shift(7)
    df['Modal_Price_Lag_30'] = df.groupby(['State', 'District', 'Market', 'Commodity'])['Modal_Price'].shift(30)
    
    df['Modal_Price_MA_7'] = df.groupby(['State', 'District', 'Market', 'Commodity'])['Modal_Price'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df['Modal_Price_MA_30'] = df.groupby(['State', 'District', 'Market', 'Commodity'])['Modal_Price'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )
    
    # Fill NaN values
    lag_cols = ['Modal_Price_Lag_1', 'Modal_Price_Lag_7', 'Modal_Price_Lag_30',
               'Modal_Price_MA_7', 'Modal_Price_MA_30']
    df[lag_cols] = df[lag_cols].ffill().bfill().fillna(df['Modal_Price'].mean())
    
    # Feature columns
    feature_cols = ['State', 'District', 'Market', 'Commodity', 'Min_Price', 'Max_Price',
                   'Year', 'Month', 'Day', 'Day_of_Week', 'Week_of_Year',
                   'Modal_Price_Lag_1', 'Modal_Price_Lag_7', 'Modal_Price_Lag_30',
                   'Modal_Price_MA_7', 'Modal_Price_MA_30']
    
    X = df[feature_cols]
    y = df['Modal_Price']
    
    # Remove any remaining NaN values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    return X, y, feature_cols

