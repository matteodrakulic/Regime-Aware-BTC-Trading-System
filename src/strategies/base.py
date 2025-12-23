import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """
    Abstract base class for all strategies.
    Ensures a consistent interface for the backtester.
    """
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        """
        Generate trading signals based on price data and regimes.
        
        Args:
            df: DataFrame with OHLCV data and features
            regimes: Series containing regime labels (0, 1, 2) aligned with df
            
        Returns:
            pd.Series: Signal series (-1 for Short, 0 for Flat, 1 for Long)
        """
        pass
