"""
Compute trend-based features for 4h BTC OHLCV data.

Features included:
- sma: Simple Moving Average
- ema: Exponential Moving Average
- macd: Moving Average Convergence Divergence
- rsi: Relative Strength Index
- adx: Average Directional Index (simplified or full if feasible)
"""

import numpy as np
import pandas as pd

# Default rolling windows (consistent with returns.py)
SHORT_WINDOW = 8     # ~1.5 days
MEDIUM_WINDOW = 32   # ~5 days
LONG_WINDOW = 64     # ~10 days

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """
    Compute Simple Moving Average
    """
    return series.rolling(window=window).mean()

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """
    Compute Exponential Moving Average
    """
    return series.ewm(span=span, adjust=False).mean()

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Compute MACD, Signal line, and Histogram
    """
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd_line': macd_line,
        'macd_signal': signal_line,
        'macd_hist': histogram
    })

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI)
    """
    delta = series.diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    # Calculate exponential moving average of gains and losses
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def compute_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Compute Average Directional Index (ADX)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # 1. True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 2. Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    
    # 3. Smoothed TR and DM (Wilder's Smoothing)
    # Wilder's smoothing is roughly alpha = 1/n. Pandas ewm(com=n-1) or ewm(alpha=1/n)
    # Wilder uses alpha = 1/window
    
    tr_smooth = tr.ewm(alpha=1/window, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/window, adjust=False).mean() / tr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1/window, adjust=False).mean() / tr_smooth)
    
    # 4. DX and ADX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.ewm(alpha=1/window, adjust=False).mean()
    
    return adx

def build_trend_features(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Build all trend-based features for a dataframe
    """
    features = pd.DataFrame(index=df.index)
    prices = df[price_col]
    
    # 1. Moving Averages
    # SMAs
    features['trend_sma_short'] = compute_sma(prices, SHORT_WINDOW)
    features['trend_sma_medium'] = compute_sma(prices, MEDIUM_WINDOW)
    features['trend_sma_long'] = compute_sma(prices, LONG_WINDOW)
    
    # EMAs
    features['trend_ema_short'] = compute_ema(prices, SHORT_WINDOW)
    features['trend_ema_medium'] = compute_ema(prices, MEDIUM_WINDOW)
    features['trend_ema_long'] = compute_ema(prices, LONG_WINDOW)
    
    # Price distance from MAs (normalized)
    # (Price - MA) / MA
    features['trend_dist_sma_short'] = (prices - features['trend_sma_short']) / features['trend_sma_short']
    features['trend_dist_sma_medium'] = (prices - features['trend_sma_medium']) / features['trend_sma_medium']
    features['trend_dist_sma_long'] = (prices - features['trend_sma_long']) / features['trend_sma_long']
    
    features['trend_dist_ema_short'] = (prices - features['trend_ema_short']) / features['trend_ema_short']
    features['trend_dist_ema_medium'] = (prices - features['trend_ema_medium']) / features['trend_ema_medium']
    features['trend_dist_ema_long'] = (prices - features['trend_ema_long']) / features['trend_ema_long']
    
    # 2. MACD (Standard settings 12, 26, 9)
    macd_df = compute_macd(prices, fast=12, slow=26, signal=9)
    features['trend_macd_line'] = macd_df['macd_line']
    features['trend_macd_signal'] = macd_df['macd_signal']
    features['trend_macd_hist'] = macd_df['macd_hist']
    
    # 3. RSI
    features['trend_rsi_14'] = compute_rsi(prices, window=14)
    
    # 4. ADX
    features['trend_adx_14'] = compute_adx(df, window=14)
    
    return features
