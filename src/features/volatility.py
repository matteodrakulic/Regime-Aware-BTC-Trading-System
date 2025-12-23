"""
Compute volatility-based features for 4h BTC OHLCV data.

Features included:
- realized_volatility: Rolling standard deviation of log returns
- parkinson_volatility: High-Low based volatility estimator
- garman_klass_volatility: Open-High-Low-Close based volatility estimator
- vol_of_vol: Volatility of the realized volatility
"""

import numpy as np
import pandas as pd

# Default rolling windows (consistent with returns.py)
SHORT_WINDOW = 8     # ~1.5 days
MEDIUM_WINDOW = 32   # ~5 days
LONG_WINDOW = 64     # ~10 days

def compute_log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """
    Compute log returns from close prices
    """
    return np.log(df[price_col] / df[price_col].shift(1))

def compute_realized_volatility(log_returns: pd.Series, window: int = SHORT_WINDOW) -> pd.Series:
    """
    Rolling standard deviation of log returns.
    """
    return log_returns.rolling(window=window).std()

def compute_parkinson_volatility(df: pd.DataFrame, window: int = SHORT_WINDOW) -> pd.Series:
    """
    Parkinson volatility estimator based on High and Low prices.
    
    Formula: sigma = sqrt( (1 / (4 * ln(2))) * mean( ln(H/L)^2 ) )
    """
    # log(High / Low)^2
    hl_ratio_sq = np.log(df['high'] / df['low']) ** 2
    
    # Scaling factor
    const = 1.0 / (4.0 * np.log(2.0))
    
    # Rolling mean of hl_ratio_sq
    rolling_val = hl_ratio_sq.rolling(window=window).mean() * const
    
    return np.sqrt(rolling_val)

def compute_garman_klass_volatility(df: pd.DataFrame, window: int = SHORT_WINDOW) -> pd.Series:
    """
    Garman-Klass volatility estimator using OHLC.
    
    Formula includes opening jumps and high-low range.
    """
    # 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2
    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])
    
    term1 = 0.5 * (log_hl ** 2)
    term2 = (2 * np.log(2) - 1) * (log_co ** 2)
    
    daily_est = term1 - term2
    
    # Rolling mean then sqrt
    return np.sqrt(daily_est.rolling(window=window).mean())

def compute_vol_of_vol(realized_vol: pd.Series, window: int = SHORT_WINDOW) -> pd.Series:
    """
    Rolling standard deviation of the realized volatility.
    """
    return realized_vol.rolling(window=window).std()

def build_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all volatility-based features for a dataframe
    """
    features = pd.DataFrame(index=df.index)
    
    log_ret = compute_log_returns(df)
    
    # 1. Realized Volatility (Std Dev of Returns)
    features['vol_realized_short'] = compute_realized_volatility(log_ret, SHORT_WINDOW)
    features['vol_realized_medium'] = compute_realized_volatility(log_ret, MEDIUM_WINDOW)
    features['vol_realized_long'] = compute_realized_volatility(log_ret, LONG_WINDOW)
    
    # 2. High-Low Estimators
    features['vol_parkinson_short'] = compute_parkinson_volatility(df, SHORT_WINDOW)
    features['vol_parkinson_medium'] = compute_parkinson_volatility(df, MEDIUM_WINDOW)
    
    features['vol_gk_short'] = compute_garman_klass_volatility(df, SHORT_WINDOW)
    features['vol_gk_medium'] = compute_garman_klass_volatility(df, MEDIUM_WINDOW)
    
    # 3. Vol of Vol (Stability of volatility)
    # How volatile is the short-term volatility?
    features['vol_of_vol_short'] = compute_vol_of_vol(features['vol_realized_short'], MEDIUM_WINDOW)
    features['vol_of_vol_medium'] = compute_vol_of_vol(features['vol_realized_medium'], LONG_WINDOW)
    
    return features
