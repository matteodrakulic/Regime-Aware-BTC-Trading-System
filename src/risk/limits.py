import pandas as pd
import numpy as np

class RiskLimits:
    """
    Enforces hard risk limits on positions.
    """
    
    def __init__(self, max_leverage: float = 1.0, max_position_size: float = 1.0):
        """
        Args:
            max_leverage: Absolute maximum leverage allowed (e.g., 1.0 = no leverage).
            max_position_size: Maximum size for a single position (percentage of equity).
        """
        self.max_leverage = max_leverage
        self.max_position_size = max_position_size
        
    def apply_limits(self, signals: pd.Series) -> pd.Series:
        """
        Clips signals to respect hard limits.
        
        Args:
            signals: Target position sizes (e.g., 1.2, -0.5, 2.0).
            
        Returns:
            pd.Series: Signals clipped to [-max_leverage, max_leverage].
        """
        # 1. Clip absolute size to max_leverage
        clipped_signals = signals.clip(lower=-self.max_leverage, upper=self.max_leverage)
        
        # 2. (Optional) In a multi-asset system, we would check max_position_size per asset.
        # Since this is single-asset (BTC), max_leverage and max_position_size are effectively the same.
        # But for correctness:
        clipped_signals = clipped_signals.clip(lower=-self.max_position_size, upper=self.max_position_size)
        
        return clipped_signals
