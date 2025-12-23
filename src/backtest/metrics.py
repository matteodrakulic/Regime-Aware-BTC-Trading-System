import pandas as pd
import numpy as np

def compute_performance_metrics(returns: pd.Series, equity: pd.Series, initial_capital: float = 10000.0, candles_per_year: int = 2190) -> dict:
    """
    Compute comprehensive performance metrics.
    
    Args:
        returns: Series of net returns per bar.
        equity: Series of equity curve values.
        initial_capital: Starting capital.
        candles_per_year: Number of bars in a year (6 * 365 = 2190 for 4h).
        
    Returns:
        dict: Key metrics (CAGR, Sharpe, Drawdown, Calmar, Sortino, etc.)
    """
    total_return = (equity.iloc[-1] / initial_capital) - 1
    
    # Time factor
    n_years = len(returns) / candles_per_year
    
    # CAGR
    cagr = (equity.iloc[-1] / initial_capital) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Volatility (Annualized)
    mean_ret = returns.mean() * candles_per_year
    vol = returns.std() * np.sqrt(candles_per_year)
    
    # Sharpe Ratio
    sharpe = mean_ret / vol if vol > 0 else 0
    
    # Sortino Ratio (Downside Deviation only)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(candles_per_year)
    sortino = mean_ret / downside_vol if downside_vol > 0 else 0
    
    # Max Drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Calmar Ratio (CAGR / Max Drawdown)
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'volatility': vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar
    }
