import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.loader import load_ohlcv_csv
from src.features.builder import build_features
from src.regimes.inference import rolling_inference
from src.strategies.trend import RegimeTrendStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.robust import RobustTrendStrategy

def get_trades_from_signals(df: pd.DataFrame, signals: pd.Series, fee_rate: float = 0.0005) -> pd.DataFrame:
    """
    Reconstruct trades from a series of position signals.
    signals: Series of -1, 0, 1 (target position)
    """
    trades = []
    
    # Ensure signals are aligned and filled
    pos = signals.fillna(0)
    pos_diff = pos.diff().fillna(0)
    
    # Iterate through changes
    # This is a simple state machine to capture trades
    # Note: This handles simple Long/Short/Flat. 
    # Flips (1 -> -1) are treated as Close Long + Open Short.
    
    current_trade = {}
    
    price = df['close']
    times = df.index
    
    # We need to loop. Vectorizing trade extraction is hard.
    # pos[i] is the target position held FROM t[i] TO t[i+1]
    # The decision is made at t[i] based on data up to t[i]
    # Execution happens at t[i] (assuming close price execution for simplicity, 
    # or we can assume next open, but backtester used close)
    
    # Backtester logic:
    # data['signal'] = signals.shift(1)  <-- signal calculated at t-1 applies to t
    # strategy_returns = pct_change(t) * signal(t)
    # This means we hold position from Open(t) to Close(t) effectively if using Close-to-Close returns
    # Let's align with that.
    
    # Position at time t
    position = pos.shift(1).fillna(0) # This is what we hold during bar t
    
    # Find where position changes
    # If position[t] != position[t-1], a trade happened at t-1 close / t open
    
    # Let's use the raw signals (unshifted) which represent "Target Position after bar t closes"
    # So if signal[t] = 1, we buy at Close[t]
    
    active_trade = None
    
    for i in range(len(df)):
        t = times[i]
        p = price.iloc[i]
        s = pos.iloc[i]
        
        # Check if we need to close or flip
        if active_trade is not None:
            if s != active_trade['type']:
                # Close the trade
                exit_price = p
                entry_price = active_trade['entry_price']
                side = active_trade['type'] # 1 or -1
                
                raw_ret = (exit_price - entry_price) / entry_price * side
                net_ret = raw_ret - (2 * fee_rate) # Entry + Exit fee
                
                trades.append({
                    'entry_time': active_trade['entry_time'],
                    'exit_time': t,
                    'type': 'Long' if side == 1 else 'Short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': net_ret,
                    'raw_return': raw_ret
                })
                active_trade = None
        
        # Check if we need to open
        if active_trade is None and s != 0:
            active_trade = {
                'entry_time': t,
                'entry_price': p,
                'type': s
            }
            
    return pd.DataFrame(trades)

def print_trade_stats(name: str, trades: pd.DataFrame):
    print(f"\n--- Strategy: {name} ---")
    if len(trades) == 0:
        print("No trades executed.")
        return

    n_trades = len(trades)
    win_rate = (trades['return'] > 0).mean()
    avg_ret = trades['return'].mean()
    avg_raw_ret = trades['raw_return'].mean()
    cum_ret = (1 + trades['return']).prod() - 1
    
    print(f"Number of Trades: {n_trades}")
    print(f"Win Rate:         {win_rate*100:.2f}%")
    print(f"Avg Net Return:   {avg_ret*100:.4f}%")
    print(f"Avg Raw Return:   {avg_raw_ret*100:.4f}%")
    print(f"Cum Return (Appx):{cum_ret*100:.2f}%")
    print(f"Avg Duration:     {trades['exit_time'].sub(trades['entry_time']).mean()}")
    
    # Long vs Short
    longs = trades[trades['type'] == 'Long']
    shorts = trades[trades['type'] == 'Short']
    
    print(f"Longs: {len(longs)} | Win: {(longs['return']>0).mean()*100:.1f}% | Avg Raw: {longs['raw_return'].mean()*100:.4f}%")
    print(f"Shorts: {len(shorts)} | Win: {(shorts['return']>0).mean()*100:.1f}% | Avg Raw: {shorts['raw_return'].mean()*100:.4f}%")

def run_debug():
    # 1. Load Data
    data_path = project_root / "data/raw/btc_4h.csv"
    print("Loading data...")
    df = load_ohlcv_csv(data_path)
    
    print("Building features...")
    features = build_features(df)
    features.dropna(inplace=True)
    df = df.loc[features.index]
    
    # 3. Run Inference (Fast settings for debug, but consistent with backtest logic)
    # To save time, we can use the parameters from the failing backtest
    print("Running inference...")
    regimes = rolling_inference(
        features,
        n_components=3,
        window=512,
        covariance_type="full",
        n_pca_components=10,
        refit_interval=10, # Same as run_backtest
        sort_by="rolling_std_medium",
        verbose=True
    )
    
    df = df.loc[regimes.index]
    regime_col = regimes['regime']
    
    # Analyze Regime Distribution
    print("\nRegime Distribution:")
    print(regime_col.value_counts(normalize=True))
    
    # 4. Strategies
    # s1 = MeanReversionStrategy(window=20, z_threshold=2.0, reversion_regime=0)
    # s2 = RegimeTrendStrategy(ema_window=50, trend_regime=1)
    
    # New Robust Strategy
    # Test on Regime 1 (Medium Vol)
    s_robust_r1 = RobustTrendStrategy(fast_span=20, slow_span=50, adx_threshold=20, trend_regime=1)
    # Test on Regime 2 (High Vol)
    s_robust_r2 = RobustTrendStrategy(fast_span=20, slow_span=50, adx_threshold=20, trend_regime=2)
    
    # Generate Signals
    # sig1 = s1.generate_signals(df, regime_col)
    sig_r1 = s_robust_r1.generate_signals(df, regime_col)
    sig_r2 = s_robust_r2.generate_signals(df, regime_col)
    
    # Analyze Trades
    # trades1 = get_trades_from_signals(df, sig1)
    trades_r1 = get_trades_from_signals(df, sig_r1)
    trades_r2 = get_trades_from_signals(df, sig_r2)
    
    # print_trade_stats("Mean Reversion (Regime 0)", trades1)
    print_trade_stats("Robust Trend (Regime 1)", trades_r1)
    print_trade_stats("Robust Trend (Regime 2)", trades_r2)

if __name__ == "__main__":
    run_debug()
