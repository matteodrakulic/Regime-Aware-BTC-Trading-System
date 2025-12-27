import sys
import pandas as pd
import numpy as np
import itertools
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.loader import load_ohlcv_csv
from src.data.cleaner import clean_ohlcv
from src.features.builder import build_features
from src.regimes.inference import rolling_inference
from src.strategies.robust import RobustTrendStrategy
from src.risk.sizing import apply_vol_targeting
from src.risk.limits import RiskLimits
from src.backtest.engine import BacktestEngine
from src.utils.logging import setup_logger

logger = setup_logger("optimizer")

def run_grid_search():
    # 1. Load and Prep Data (Once)
    data_path = "data/raw/btc_4h.csv"
    logger.info(f"Loading data from {data_path}...")
    try:
        df = load_ohlcv_csv(data_path)
        df = clean_ohlcv(df)
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        return

    # 2. Features & Regimes (Once)
    logger.info("Building features and running regime inference...")
    features = build_features(df)
    features.dropna(inplace=True)
    df = df.loc[features.index]

    # Use standard window/refit for optimization
    regime_df = rolling_inference(
        features, 
        window=1000, 
        refit_interval=100, 
        n_components=3, 
        sort_by='rolling_std_medium'
    )
    df = df.join(regime_df)

    # 3. Generate Raw Signals (Once)
    logger.info("Generating raw strategy signals...")
    strategy = RobustTrendStrategy(
        fast_span=20, 
        slow_span=50, 
        adx_threshold=30.0, 
        trend_regime=2,
        long_only=False
    )
    if 'close' not in df.columns:
        # Restore close if lost (unlikely here but safe)
        df_raw = load_ohlcv_csv(data_path)
        df['close'] = df_raw.loc[df.index, 'close']

    raw_signals = strategy.generate_signals(df, df['regime']).fillna(0.0)

    # 4. Grid Search
    target_vols = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    max_leverages = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    results = []

    logger.info(f"Starting Grid Search over {len(target_vols) * len(max_leverages)} combinations...")
    
    print(f"{'Target Vol':<12} | {'Max Lev':<10} | {'CAGR':<8} | {'Sharpe':<8} | {'Max DD':<8} | {'Calmar':<8}")
    print("-" * 75)

    for tv in target_vols:
        for ml in max_leverages:
            # Apply Vol Targeting
            sig_vol = apply_vol_targeting(
                raw_signals, 
                df['close'], 
                target_annual_vol=tv, 
                window_days=20,
                max_leverage=ml
            )
            
            # Apply Limits
            risk_limits = RiskLimits(max_leverage=ml, max_position_size=ml)
            sig_final = risk_limits.apply_limits(sig_vol)
            
            # Run Backtest
            backtester = BacktestEngine(
                initial_capital=10000.0, 
                fee_rate=0.0005, 
                idle_apy=0.06
            )
            res = backtester.run(df, sig_final)
            m = res['metrics']
            
            # Store result
            results.append({
                'target_vol': tv,
                'max_leverage': ml,
                'cagr': m['cagr'],
                'sharpe': m['sharpe_ratio'],
                'max_dd': m['max_drawdown'],
                'calmar': m['calmar_ratio'],
                'total_ret': m['total_return']
            })
            
            print(f"{tv:<12.2f} | {ml:<10.1f} | {m['cagr']*100:>7.1f}% | {m['sharpe_ratio']:>8.2f} | {m['max_drawdown']*100:>7.1f}% | {m['calmar_ratio']:>8.2f}")

    # 5. Find "Best"
    # Criteria: Maximize Sharpe, subject to Max DD > -30%? 
    # Or just sort by Sharpe
    
    df_res = pd.DataFrame(results)
    
    print("\n" + "="*40)
    print("TOP 5 BY SHARPE RATIO")
    print("="*40)
    print(df_res.sort_values('sharpe', ascending=False).head(5).to_string(index=False))
    
    print("\n" + "="*40)
    print("TOP 5 BY CAGR (High Risk)")
    print("="*40)
    print(df_res.sort_values('cagr', ascending=False).head(5).to_string(index=False))

    print("\n" + "="*40)
    print("TOP 5 BY CALMAR (Risk-Adj Return)")
    print("="*40)
    print(df_res.sort_values('calmar', ascending=False).head(5).to_string(index=False))

if __name__ == "__main__":
    run_grid_search()
