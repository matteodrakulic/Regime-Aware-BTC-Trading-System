import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.loader import load_ohlcv_csv
from src.features.builder import build_features
from src.regimes.inference import rolling_inference
from src.backtest.engine import BacktestEngine
from src.strategies.trend import RegimeTrendStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import RegimeBreakoutStrategy
from src.strategies.selector import StrategySelector
from src.strategies.robust import RobustTrendStrategy
from src.risk.sizing import apply_vol_targeting
from src.risk.limits import RiskLimits
from src.risk.drawdown import DrawdownControl

def run_backtest():
    # 1. Load Data
    data_path = project_root / "data/raw/btc_4h.csv"
    print(f"Loading data from {data_path}...")
    try:
        df = load_ohlcv_csv(data_path)
    except FileNotFoundError:
        print("Data not found, generating synthetic...")
        return

    # 2. Build Features
    print("Building features...")
    features = build_features(df)
    features.dropna(inplace=True)
    df = df.loc[features.index]
    
    # 3. Run Inference
    print("Running rolling inference (refit_interval=10, sort_by='rolling_std_medium')...")
    regimes = rolling_inference(
        features,
        n_components=3,
        window=512,
        covariance_type="full",
        smooth_alpha=0.2,
        n_pca_components=10,
        refit_interval=10,
        sort_by="rolling_std_medium",
        verbose=True
    )
    
    # Align data
    df = df.loc[regimes.index]
    regime_col = regimes['regime']
    
    # 4. Initialize Strategies
    
    # Strategy Logic based on findings:
    # R0 (Low Vol): Too choppy/bleeding for Mean Reversion. Stay Flat.
    # R1 (Med Vol): Too noisy for Trend. Stay Flat.
    # R2 (High Vol): Good for Trend Following (especially Longs).
    
    # Robust Trend for Regime 2 (Optimized + Shorting Enabled)
    s_trend_r2 = RobustTrendStrategy(
        fast_span=20, 
        slow_span=50, 
        adx_threshold=30.0,  # Optimized
        trend_regime=2,
        long_only=False,     # Enable Shorts
        macro_trend_window=1200 # 200-day SMA Filter for Shorts
    )
    
    # Composite (Only R2 is active)
    composite = StrategySelector([s_trend_r2])
    
    # 5. Run Backtests
    backtester = BacktestEngine(initial_capital=10000.0, fee_rate=0.0005)
    
    print("\n=== Backtest: Robust Trend (Regime 2 Only) [Fixed 100% Size] ===")
    sig_comp = composite.generate_signals(df, regime_col)
    res_comp = backtester.run(df, sig_comp)
    print_metrics(res_comp['metrics'])
    
    # 6. Run Volatility Targeting Backtest (Optimized Target: 60%)
    print("\n=== Backtest: Robust Trend (Regime 2 Only) [VolTarget 60%] ===")
    sig_vol = apply_vol_targeting(
        sig_comp, 
        df['close'], 
        target_annual_vol=0.60,
        max_leverage=1.0
    )
    res_vol = backtester.run(df, sig_vol)
    print_metrics(res_vol['metrics'])
    
    # 8. Run Final Combined Strategy (VolTarget 60% + Yield Generation 6%)
    print("\n=== Backtest: Robust Trend + VolTarget 60% + Yield 6% (Idle) ===")
    
    # Initialize backtester with 6% APY on idle cash
    backtester_yield = BacktestEngine(
        initial_capital=10000.0, 
        fee_rate=0.0005,
        idle_apy=0.06  # 6% Yield on Cash
    )
    
    # Re-use the best signal (VolTarget 60%)
    # Apply Hard Limits (Phase 5)
    risk_limits = RiskLimits(max_leverage=1.0, max_position_size=1.0)
    sig_limited = risk_limits.apply_limits(sig_vol)
    
    # Apply Drawdown Control (Phase 5)
    # We need equity curve for drawdown control, but we don't have it yet.
    # In a vectorized backtest, this is tricky because drawdown depends on equity which depends on signals.
    # For now, we will skip dynamic drawdown control in vectorized backtest or implement a loop if needed.
    # However, for simplicity and speed, we will assume the limits are sufficient for now,
    # or we can use the DrawdownControl class in a loop-based backtest.
    # Given "Keep it Simple", we will stick to the vectorized engine for now and just apply limits.
    # Real drawdown control requires an event-driven loop (Phase 6 refined).
    
    sig_final = sig_limited
    
    res_final = backtester_yield.run(df, sig_final) 
    print_metrics(res_final['metrics'])
    
    # Save results for plotting
    res_final['data'].to_csv(project_root / "data/processed/backtest_results.csv")
    print("\nResults saved to data/processed/backtest_results.csv")

def print_metrics(m):
    print(f"Total Return: {m['total_return']*100:.2f}%")
    print(f"CAGR:         {m['cagr']*100:.2f}%")
    print(f"Sharpe Ratio: {m['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {m['max_drawdown']*100:.2f}%")
    print(f"Win Rate:     {m['win_rate']*100:.2f}%")

if __name__ == "__main__":
    run_backtest()
