"""
Compute distribution-based features for 4h BTC OHLCV data.

Features included:
- rolling_skew: Rolling skewness of log returns
- rolling_kurtosis: Rolling kurtosis of log returns
"""

import numpy as np
import pandas as pd

# Default rolling windows (consistent with other modules)
# Note: Skew and Kurtosis require more data points to be stable
SHORT_WINDOW = 24    # ~4 days (increased from 8)
MEDIUM_WINDOW = 64   # ~10 days
LONG_WINDOW = 128    # ~21 days

def compute_log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """
    Compute log returns from close prices
    """
    return np.log(df[price_col] / df[price_col].shift(1))

def compute_rolling_skew(log_returns: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling skewness of returns.
    Skewness measures asymmetry of the distribution.
    Positive skew: frequent small losses, few large gains.
    Negative skew: frequent small gains, few large losses.
    """
    return log_returns.rolling(window=window).skew()

def compute_rolling_kurtosis(log_returns: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling kurtosis of returns.
    Kurtosis measures tail thickness (fat tails).
    High kurtosis: higher probability of extreme events (both positive and negative).
    """
    return log_returns.rolling(window=window).kurt()

def build_distribution_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all distribution-based features for a dataframe
    """
    features = pd.DataFrame(index=df.index)
    
    log_ret = compute_log_returns(df)
    
    # 1. Rolling Skewness
    features['dist_skew_short'] = compute_rolling_skew(log_ret, SHORT_WINDOW)
    features['dist_skew_medium'] = compute_rolling_skew(log_ret, MEDIUM_WINDOW)
    features['dist_skew_long'] = compute_rolling_skew(log_ret, LONG_WINDOW)
    
    # 2. Rolling Kurtosis
    features['dist_kurt_short'] = compute_rolling_kurtosis(log_ret, SHORT_WINDOW)
    features['dist_kurt_medium'] = compute_rolling_kurtosis(log_ret, MEDIUM_WINDOW)
    features['dist_kurt_long'] = compute_rolling_kurtosis(log_ret, LONG_WINDOW)
    
    return features
