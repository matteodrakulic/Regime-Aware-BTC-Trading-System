import pandas as pd
from typing import List
from .base import BaseStrategy

class StrategySelector(BaseStrategy):
    """
    Composite Strategy: Combines multiple strategies based on regimes.
    
    Logic:
    - If Regime 0: Use Mean Reversion Strategy
    - If Regime 1: Use Trend Strategy
    - If Regime 2: Use Breakout Strategy (or Flat)
    """
    
    def __init__(self, strategies: List[BaseStrategy]):
        super().__init__("Composite_Selector")
        self.strategies = strategies
        
    def generate_signals(self, df: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        final_signals = pd.Series(0, index=df.index)
        
        # We assume strategies handle their own regime logic internally
        # But a selector could also strictly enforce it
        
        # Here we just sum them (assuming they are mutually exclusive by regime design)
        # If they overlap, this might lead to leverage > 1 (e.g. 1 + 1 = 2)
        # For safety, we can clip to [-1, 1]
        
        for strategy in self.strategies:
            s_signals = strategy.generate_signals(df, regimes)
            final_signals = final_signals + s_signals
            
        return final_signals.clip(-1, 1)
