import pandas as pd
import numpy as np

class DrawdownControl:
    """
    Implements a Circuit Breaker that halts trading if drawdown exceeds a threshold.
    """
    
    def __init__(self, max_drawdown_limit: float = 0.20, cooldown_bars: int = 42):
        """
        Args:
            max_drawdown_limit: Stop trading if DD > this (e.g. 0.20 for 20%).
            cooldown_bars: Number of bars to stay flat after breach (e.g. 42 bars = 1 week on 4h).
        """
        self.max_drawdown_limit = max_drawdown_limit
        self.cooldown_bars = cooldown_bars
        
    def apply_circuit_breaker(self, signals: pd.Series, equity_curve: pd.Series) -> pd.Series:
        """
        Modifies signals to force 0 (FLAT) when in deep drawdown or cooldown.
        
        Args:
            signals: The intended position signals.
            equity_curve: The current equity curve (simulated or real).
            
        Returns:
            pd.Series: Signals with circuit breaker applied.
        """
        # Calculate Drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        
        # Identify breaches
        breach_mask = drawdown < -self.max_drawdown_limit
        
        # Apply cooldown logic
        # If breach happens at t, we want to be flat for [t+1, t+cooldown]
        # This is iterative, hard to vectorize perfectly without a loop, 
        # but we can try a forward fill approach or just a simple loop.
        
        # Let's use a loop for safety and clarity as this is critical logic.
        
        modified_signals = signals.copy()
        cooldown_counter = 0
        
        for i in range(len(signals)):
            # Check if we are in cooldown from previous breach
            if cooldown_counter > 0:
                modified_signals.iloc[i] = 0.0
                cooldown_counter -= 1
                continue
            
            # Check for new breach at current step
            # Note: In a real backtest, we'd check yesterday's close equity.
            # Here we assume equity_curve is aligned with signals (known at t).
            if breach_mask.iloc[i]:
                modified_signals.iloc[i] = 0.0
                cooldown_counter = self.cooldown_bars
                
        return modified_signals
