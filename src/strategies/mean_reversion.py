import pandas as pd
import numpy as np
from .base import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    """
    Strategy 2: Low-Vol Mean Reversion
    
    Logic:
    - If Regime == LOW_VOL (0):
        - Long if Z-Score < -threshold
        - Short if Z-Score > +threshold
        - Exit if Z-Score crosses 0
    - Else:
        - Flat
    """
    
    def __init__(self, window: int = 20, z_threshold: float = 2.0, reversion_regime: int = 0):
        super().__init__(f"MeanReversion_Z{z_threshold}")
        self.window = window
        self.z_threshold = z_threshold
        self.reversion_regime = reversion_regime
        
    def generate_signals(self, df: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        # Calculate Rolling Z-Score of Price
        rolling_mean = df['close'].rolling(window=self.window).mean()
        rolling_std = df['close'].rolling(window=self.window).std()
        z_score = (df['close'] - rolling_mean) / rolling_std
        
        # Filter by Regime
        is_reversion_regime = (regimes == self.reversion_regime)
        
        # We need to maintain state (long/short/flat) to handle exits
        # This is harder to vectorize perfectly if exit is state-dependent (cross 0)
        # But we can approximate vectorization or iterate
        
        # Iterative approach for correctness on exits
        current_pos = 0
        pos_history = []
        
        for i in range(len(df)):
            if not is_reversion_regime.iloc[i]:
                current_pos = 0 # Force exit if regime changes
            else:
                z = z_score.iloc[i]
                
                if current_pos == 0:
                    if z < -self.z_threshold:
                        current_pos = 1 # Long
                    elif z > self.z_threshold:
                        current_pos = -1 # Short
                elif current_pos == 1:
                    if z >= 0: # Exit condition
                        current_pos = 0
                elif current_pos == -1:
                    if z <= 0: # Exit condition
                        current_pos = 0
            
            pos_history.append(current_pos)
            
        return pd.Series(pos_history, index=df.index)
