"""
Feature Builder
Concatenates all feature sets into a final feature matrix.
"""

import pandas as pd
from src.features.returns import build_return_features
from src.features.volatility import build_volatility_features
from src.features.trend import build_trend_features
from src.features.distribution import build_distribution_features

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build and concatenate all features for the given dataframe.
    
    Args:
        df: Input dataframe with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame containing all features
    """
    # 1. Return-based features
    return_features = build_return_features(df)
    
    # 2. Volatility-based features
    volatility_features = build_volatility_features(df)
    
    # 3. Trend-based features
    trend_features = build_trend_features(df)
    
    # 4. Distribution-based features
    distribution_features = build_distribution_features(df)
    
    # Concatenate all features
    # Axis=1 to concat columns (features)
    # Join='inner' or 'outer' - since they share the same index (from df), it should be fine.
    # We use pd.concat
    
    all_features = pd.concat([return_features, volatility_features, trend_features, distribution_features], axis=1)
    
    return all_features
