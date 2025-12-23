import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class VectorizedBacktester:
    """
    A fast, vectorized backtester for strategy evaluation.
    Handles PnL calculation, transaction costs, and performance metrics.
    """
    
    def __init__(self, initial_capital: float = 10000.0, fee_rate: float = 0.0005, idle_apy: float = 0.0):
        """
        Args:
            initial_capital: Starting portfolio value
            fee_rate: Transaction fee rate per trade (e.g., 0.0005 = 0.05%)
            idle_apy: Annual Percentage Yield earned on idle cash (e.g., 0.06 = 6%)
        """
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.idle_apy = idle_apy
        
    def run(self, df: pd.DataFrame, signals: pd.Series, price_col: str = "close") -> dict:
        """
        Run the backtest.
        
        Args:
            df: DataFrame containing price data
            signals: Series of -1, 0, 1 indicating target position
            price_col: Column name for execution price
            
        Returns:
            dict: Performance metrics and equity curve
        """
        # Align signals with price
        data = df[[price_col]].copy()
        data['signal'] = signals.shift(1).fillna(0) # Shift 1 to avoid lookahead bias
        
        # Calculate returns
        data['returns'] = data[price_col].pct_change().fillna(0)
        data['strategy_returns'] = data['returns'] * data['signal']
        
        # Calculate Idle Yield (Yield Farming / Funding Rate)
        # We assume 6 candles per day (4h data) -> 2190 candles per year
        candles_per_year = 6 * 365
        yield_per_bar = (1 + self.idle_apy) ** (1 / candles_per_year) - 1
        
        # Apply yield when signal is 0 (Flat)
        # Note: If we are partial size (e.g. 0.5), we should earn yield on the remaining 0.5.
        # Logic: Yield on (1 - abs(signal))
        data['idle_capital'] = (1 - data['signal'].abs()).clip(lower=0)
        data['yield_returns'] = data['idle_capital'] * yield_per_bar
        
        # Calculate transaction costs
        # Fee is paid on every position change
        # position change = abs(current_signal - prev_signal)
        # e.g., 0 -> 1 (Buy) = 1 unit turned over
        # 1 -> -1 (Flip Short) = 2 units turned over
        data['position_change'] = data['signal'].diff().abs().fillna(0)
        data['fees'] = data['position_change'] * self.fee_rate
        
        # Net strategy returns
        data['net_returns'] = data['strategy_returns'] + data['yield_returns'] - data['fees']
        
        # Equity curve
        data['equity'] = self.initial_capital * (1 + data['net_returns']).cumprod()
        data['buy_hold_equity'] = self.initial_capital * (1 + data['returns']).cumprod()
        
        # Compute metrics
        metrics = self._compute_metrics(data['net_returns'], data['equity'])
        metrics['buy_hold_return'] = (data['buy_hold_equity'].iloc[-1] / self.initial_capital) - 1
        
        return {
            'metrics': metrics,
            'data': data
        }
        
    def _compute_metrics(self, returns: pd.Series, equity: pd.Series) -> dict:
        """Compute CAGR, Sharpe, Drawdown, etc."""
        total_return = (equity.iloc[-1] / self.initial_capital) - 1
        
        # Annualized metrics (assuming 4h data = 6 candles/day * 365 days = 2190 candles/year)
        candles_per_year = 6 * 365
        n_years = len(returns) / candles_per_year
        
        cagr = (equity.iloc[-1] / self.initial_capital) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Sharpe Ratio (assuming 0 risk-free rate for simplicity)
        mean_ret = returns.mean() * candles_per_year
        vol = returns.std() * np.sqrt(candles_per_year)
        sharpe = mean_ret / vol if vol > 0 else 0
        
        # Max Drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'volatility': vol
        }

    def plot_results(self, result: dict, title: str = "Backtest Results"):
        """Plot Equity Curve and Drawdown"""
        data = result['data']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Equity Curve
        ax1.plot(data.index, data['equity'], label='Strategy', color='blue')
        ax1.plot(data.index, data['buy_hold_equity'], label='Buy & Hold', color='gray', alpha=0.5, linestyle='--')
        ax1.set_title(f"{title} (Sharpe: {result['metrics']['sharpe_ratio']:.2f})")
        ax1.set_ylabel('Equity ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        rolling_max = data['equity'].cummax()
        drawdown = (data['equity'] - rolling_max) / rolling_max
        ax2.fill_between(data.index, drawdown, 0, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
