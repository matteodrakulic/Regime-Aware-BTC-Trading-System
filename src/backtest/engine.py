import pandas as pd
import numpy as np
from src.backtest.metrics import compute_performance_metrics

class BacktestEngine:
    """
    The central engine for running simulations.
    Orchestrates Data -> Signals -> Risk -> Execution -> Metrics.
    """
    
    def __init__(self, initial_capital: float = 10000.0, fee_rate: float = 0.0005, idle_apy: float = 0.0):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.idle_apy = idle_apy
        
    def run(self, df: pd.DataFrame, signals: pd.Series, price_col: str = "close") -> dict:
        """
        Execute the backtest.
        
        Args:
            df: Price DataFrame.
            signals: Position sizing signals (0.0 to 1.0, or negative for short).
            price_col: Column to use for execution.
        """
        # 1. Setup Data
        data = df[[price_col]].copy()
        
        # 2. Shift Signals (Avoid Lookahead)
        # The signal calculated at time t is applied at time t+1 (or close of t, effectively next bar)
        data['signal'] = signals.shift(1).fillna(0)
        
        # 3. Calculate Raw Returns
        data['returns'] = data[price_col].pct_change().fillna(0)
        
        # 4. Strategy Returns (Gross)
        data['strategy_returns'] = data['returns'] * data['signal']
        
        # 5. Yield on Idle Capital
        # If signal is 0.6, then 0.4 is idle.
        candles_per_year = 6 * 365
        yield_per_bar = (1 + self.idle_apy) ** (1 / candles_per_year) - 1
        data['idle_capital'] = (1 - data['signal'].abs()).clip(lower=0)
        data['yield_returns'] = data['idle_capital'] * yield_per_bar
        
        # 6. Transaction Costs
        # Turnover = change in position size
        data['position_change'] = data['signal'].diff().abs().fillna(0)
        data['fees'] = data['position_change'] * self.fee_rate
        
        # 7. Net Returns
        data['net_returns'] = data['strategy_returns'] + data['yield_returns'] - data['fees']
        
        # 8. Equity Curve
        data['equity'] = self.initial_capital * (1 + data['net_returns']).cumprod()
        
        # 9. Compute Metrics
        metrics = compute_performance_metrics(data['net_returns'], data['equity'], self.initial_capital)
        
        # Add Time in Market metric
        time_in_market = (data['signal'].abs() > 0).mean()
        metrics['time_in_market'] = time_in_market
        
        return {
            'data': data,
            'metrics': metrics
        }
