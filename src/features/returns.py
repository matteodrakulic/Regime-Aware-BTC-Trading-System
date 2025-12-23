"""
Compute return-based features for 4h BTC OHLCV data.

Features included:
- log_returns: log(C_t / C_{t-1})
- rolling_mean_return: rolling mean of log returns
- rolling_std_return: rolling volatility of log returns
- autocorr_lag1: autocorrelation with lag 1
- additional lags can be added easily
"""

import pandas as pd
import numpy as np

# Default rolling windows (number of candles)
SHORT_WINDOW = 8     # ~1.5 days
MEDIUM_WINDOW = 32   # ~5 days
LONG_WINDOW = 64     # ~10 days

def compute_log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """
    Compute log returns from close prices
    """
    return np.log(df[price_col] / df[price_col].shift(1))


def rolling_mean_return(log_returns: pd.Series, window: int = SHORT_WINDOW) -> pd.Series:
    """
    Rolling mean of log returns
    """
    return log_returns.rolling(window=window).mean()


def rolling_std_return(log_returns: pd.Series, window: int = SHORT_WINDOW) -> pd.Series:
    """
    Rolling standard deviation (volatility) of log returns
    """
    return log_returns.rolling(window=window).std()


def autocorr_lag1(log_returns: pd.Series, window: int = SHORT_WINDOW) -> pd.Series:
    """
    Rolling autocorrelation with lag 1
    """
    return log_returns.rolling(window=window).apply(lambda x: x.autocorr(lag=1), raw=False)


def build_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all return-based features for a dataframe
    """
    features = pd.DataFrame(index=df.index)

    log_ret = compute_log_returns(df)
    features["log_return"] = log_ret

    # Rolling mean returns
    features["rolling_mean_short"] = rolling_mean_return(log_ret, SHORT_WINDOW)
    features["rolling_mean_medium"] = rolling_mean_return(log_ret, MEDIUM_WINDOW)
    features["rolling_mean_long"] = rolling_mean_return(log_ret, LONG_WINDOW)

    # Rolling std / volatility
    features["rolling_std_short"] = rolling_std_return(log_ret, SHORT_WINDOW)
    features["rolling_std_medium"] = rolling_std_return(log_ret, MEDIUM_WINDOW)
    features["rolling_std_long"] = rolling_std_return(log_ret, LONG_WINDOW)

    # Autocorrelation lag1 (short window)
    features["autocorr_lag1_short"] = autocorr_lag1(log_ret, SHORT_WINDOW)

    return features
