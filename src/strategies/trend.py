import pandas as pd
import numpy as np
from .base import BaseStrategy

class RegimeTrendStrategy(BaseStrategy):
    """
    Strategy 1: Regime-Conditional Trend Following
    
    Logic:
    - If Regime == MEDIUM_VOL (1):
        - Long if Price > EMA(fast)
        - Short if Price < EMA(fast)
    - Else:
        - Flat
    """
    
    def __init__(self, ema_window: int = 50, trend_regime: int = 1):
        super().__init__(f"RegimeTrend_EMA{ema_window}")
        self.ema_window = ema_window
        self.trend_regime = trend_regime
        
    def generate_signals(self, df: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        # Calculate Trend Indicator
        ema = df['close'].ewm(span=self.ema_window, adjust=False).mean()
        
        # Determine Trend Direction
        trend_long = df['close'] > ema
        trend_short = df['close'] < ema
        
        # Filter by Regime
        is_trend_regime = (regimes == self.trend_regime)
        
        # Apply Logic
        signals[is_trend_regime & trend_long] = 1
        signals[is_trend_regime & trend_short] = -1
        
        return signals
