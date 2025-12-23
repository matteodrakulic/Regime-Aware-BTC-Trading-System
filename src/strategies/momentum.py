import pandas as pd
import numpy as np
from .base import BaseStrategy

class RegimeBreakoutStrategy(BaseStrategy):
    """
    Strategy 3: High-Vol Breakout
    
    Logic:
    - If Regime == HIGH_VOL (2):
        - Long if Price > rolling_max(window)
        - Short if Price < rolling_min(window)
        - Exit if Price crosses rolling_mean
    - Else:
        - Flat (Capital Preservation)
    """
    
    def __init__(self, window: int = 20, breakout_regime: int = 2):
        super().__init__(f"RegimeBreakout_W{window}")
        self.window = window
        self.breakout_regime = breakout_regime
        
    def generate_signals(self, df: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        # Donchian Channel
        rolling_max = df['high'].rolling(window=self.window).max().shift(1)
        rolling_min = df['low'].rolling(window=self.window).min().shift(1)
        rolling_mean = df['close'].rolling(window=self.window).mean()
        
        is_breakout_regime = (regimes == self.breakout_regime)
        
        current_pos = 0
        pos_history = []
        
        for i in range(len(df)):
            if not is_breakout_regime.iloc[i]:
                current_pos = 0 # Force exit
            else:
                price = df['close'].iloc[i]
                r_max = rolling_max.iloc[i]
                r_min = rolling_min.iloc[i]
                r_mean = rolling_mean.iloc[i]
                
                if current_pos == 0:
                    if price > r_max:
                        current_pos = 1
                    elif price < r_min:
                        current_pos = -1
                elif current_pos == 1:
                    if price < r_mean: # Trailing stop at mean
                        current_pos = 0
                elif current_pos == -1:
                    if price > r_mean: # Trailing stop at mean
                        current_pos = 0
                        
            pos_history.append(current_pos)
            
        return pd.Series(pos_history, index=df.index)
