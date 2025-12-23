import pandas as pd
import numpy as np
from .base import BaseStrategy

class RobustTrendStrategy(BaseStrategy):
    """
    Robust Trend Strategy using EMA Crossover and ADX Filter.
    
    Logic:
    - Compute Fast EMA and Slow EMA.
    - Compute ADX.
    - Long if Fast > Slow AND ADX > threshold.
    - Short if Fast < Slow AND ADX > threshold.
    - Flat otherwise (choppy market).
    """
    
    def __init__(self, fast_span: int = 20, slow_span: int = 50, adx_threshold: float = 20.0, trend_regime: int = 1, long_only: bool = False, macro_trend_window: int = 1200):
        super().__init__(f"RobustTrend_{fast_span}_{slow_span}_ADX{adx_threshold}")
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.adx_threshold = adx_threshold
        self.trend_regime = trend_regime
        self.long_only = long_only
        self.macro_trend_window = macro_trend_window
        
    def generate_signals(self, df: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        # We need to rebuild indicators here or pass them in?
        # Ideally, strategies should calculate their own indicators to be self-contained
        # given the raw OHLCV.
        
        # 1. EMAs
        ema_fast = df['close'].ewm(span=self.fast_span, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.slow_span, adjust=False).mean()
        
        # 2. ADX (Re-calculate here to be safe, or assume it's in df? Backtester passes raw df usually)
        # But wait, run_backtest passes `df` which might NOT have features if I loaded raw data.
        # But in run_backtest.py I did: features = build_features(df); df = df.loc[features.index]
        # The `df` passed to generate_signals is just OHLCV usually.
        # Let's re-calculate ADX using the helper we just wrote? 
        # Or duplicate logic. Duplicating is safer for standalone.
        
        adx = self._compute_adx(df)
        
        # 3. Macro Trend Filter (e.g. 200-day SMA = 1200 4h bars)
        if self.macro_trend_window > 0:
            macro_ma = df['close'].rolling(window=self.macro_trend_window).mean()
            # If Price < Macro MA, we are in Bear Market -> Allow Shorts
            # If Price > Macro MA, we are in Bull Market -> Longs Only (typically safer)
            # Or we can use it to FILTER shorts.
            
            # Implementation: 
            # - Always allow Longs (if signal present)
            # - Allow Shorts ONLY if Price < Macro MA (Bear Market)
            
            bear_market = df['close'] < macro_ma
        else:
            bear_market = pd.Series(True, index=df.index) # Always allow shorts if no filter
        
        # 3. Logic
        signals = pd.Series(0, index=df.index)
        
        # Vectorized conditions
        long_cond = (ema_fast > ema_slow) & (adx > self.adx_threshold)
        short_cond = (ema_fast < ema_slow) & (adx > self.adx_threshold)
        
        # Filter by Regime
        is_trend_regime = (regimes == self.trend_regime)
        
        # Apply signals only in regime
        # If not in regime -> Flat (0)
        
        signals[is_trend_regime & long_cond] = 1
        
        # Short Logic:
        # If long_only=True, NEVER short.
        # If long_only=False, Short if signal AND (no macro filter OR bear market)
        
        if not self.long_only:
            # We add the bear_market condition to shorting
            # If macro_trend_window is set, bear_market is True only when below MA
            signals[is_trend_regime & short_cond & bear_market] = -1
        
        return signals

    def _compute_adx(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)
        
        tr_smooth = tr.ewm(alpha=1/window, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/window, adjust=False).mean() / tr_smooth)
        minus_di = 100 * (minus_dm.ewm(alpha=1/window, adjust=False).mean() / tr_smooth)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.ewm(alpha=1/window, adjust=False).mean()
        
        return adx
