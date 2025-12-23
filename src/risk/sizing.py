import pandas as pd
import numpy as np

def apply_vol_targeting(
    signals: pd.Series, 
    prices: pd.Series, 
    target_annual_vol: float = 0.40, 
    window_days: int = 20, 
    candles_per_day: int = 6,
    max_leverage: float = 1.0
) -> pd.Series:
    """
    Apply volatility targeting to position signals.
    
    Formula:
        Scalar = Target_Vol / Realized_Vol
        Position = Signal * min(Scalar, Max_Leverage)
        
    Args:
        signals: Raw strategy signals (-1, 0, 1)
        prices: Price series (close)
        target_annual_vol: Target annualized volatility (e.g., 0.40 for 40%)
        window_days: Lookback window for volatility calculation
        candles_per_day: Number of candles per day (6 for 4h data)
        max_leverage: Maximum allowed position size (default 1.0 = no leverage)
        
    Returns:
        pd.Series: Sized signals (e.g., 0.8, -0.5, 0.0)
    """
    # Calculate returns
    returns = prices.pct_change().fillna(0)
    
    # Calculate Rolling Volatility (Annualized)
    window = window_days * candles_per_day
    rolling_std = returns.rolling(window=window).std()
    
    # Annualize factor
    annualization_factor = np.sqrt(365 * candles_per_day)
    rolling_annual_vol = rolling_std * annualization_factor
    
    # Avoid division by zero
    rolling_annual_vol = rolling_annual_vol.replace(0, np.nan).ffill()
    
    # Calculate Volatility Scalar
    # If Vol is High -> Scalar < 1 -> Reduce Size
    # If Vol is Low -> Scalar > 1 -> Increase Size (capped by max_leverage)
    vol_scalar = target_annual_vol / rolling_annual_vol
    
    # Cap leverage
    vol_scalar = vol_scalar.clip(upper=max_leverage)
    
    # Fill NaNs (start of series) with 1.0 (or 0 if we want to be safe, but 1.0 is standard fallback)
    # Actually, if we don't have vol data yet, we should probably stick to the raw signal or be conservative.
    # Let's fill with 1.0 (no adjustment) for the warm-up period, assuming normal conditions.
    vol_scalar = vol_scalar.fillna(1.0)
    
    # Apply to signals
    sized_signals = signals * vol_scalar
    
    return sized_signals
