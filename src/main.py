import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.loader import load_ohlcv_csv
from src.data.cleaner import clean_ohlcv
from src.data.fetcher import update_local_ohlcv
from src.features.builder import build_features
from src.regimes.inference import rolling_inference
from src.strategies.robust import RobustTrendStrategy
from src.risk.sizing import apply_vol_targeting
from src.risk.limits import RiskLimits
from src.backtest.engine import BacktestEngine
from src.utils.logging import setup_logger

logger = setup_logger("main")

def load_and_prep_data(filepath: str, update: bool = False):
    if update:
        logger.info("Updating local data from exchange...")
        try:
            # update_local_ohlcv returns the dataframe directly
            df = update_local_ohlcv()
            logger.info("Data updated successfully.")
            
            # Ensure index is set for clean_ohlcv
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df.set_index('timestamp', inplace=True)
                
            # We still clean it just in case
            df = clean_ohlcv(df)
            return df
        except Exception as e:
            logger.error(f"Failed to update data: {e}")
            logger.info("Falling back to local file...")
            
    logger.info(f"Loading data from {filepath}...")
    try:
        df = load_ohlcv_csv(filepath)
        df = clean_ohlcv(df)
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

def run_backtest(args):
    df = load_and_prep_data(args.data)
    
    logger.info("Building features...")
    features = build_features(df)
    features.dropna(inplace=True)
    
    # Align df with features
    df = df.loc[features.index]
    
    logger.info(f"Running regime inference (window={args.window})...")
    regime_df = rolling_inference(
        features, 
        window=args.window, 
        refit_interval=args.refit, 
        n_components=3, 
        sort_by='rolling_std_medium'
    )
    df = df.join(regime_df)
    
    logger.info("Executing strategy (RobustTrendStrategy)...")
    strategy = RobustTrendStrategy(
        fast_span=20, 
        slow_span=50, 
        adx_threshold=30.0, 
        trend_regime=2,
        long_only=False
    )
    
    # Check if close column exists (sanity check)
    if 'close' not in df.columns:
         logger.warning("'close' column missing after join, restoring...")
         # Simple restoration if index aligns (it should)
         df_raw = load_and_prep_data(args.data)
         df['close'] = df_raw.loc[df.index, 'close']

    signals = strategy.generate_signals(df, df['regime'])
    
    # Filter for Regime 2 only (redundant but safe)
    signals = signals.fillna(0.0)
    
    logger.info(f"Applying Volatility Targeting (Target={args.target_vol*100}%)...")
    sig_vol = apply_vol_targeting(
        signals, 
        df['close'], 
        target_annual_vol=args.target_vol, 
        window_days=20,
        max_leverage=1.0
    )
    
    logger.info("Applying Risk Limits...")
    risk_limits = RiskLimits(max_leverage=1.0, max_position_size=1.0)
    sig_final = risk_limits.apply_limits(sig_vol)
    
    logger.info(f"Running Backtest (Capital=${args.capital})...")
    backtester = BacktestEngine(
        initial_capital=args.capital, 
        fee_rate=0.0005, 
        idle_apy=0.06
    )
    results = backtester.run(df, sig_final)
    
    metrics = results['metrics']
    logger.info("Backtest Completed.")
    print("\n" + "="*40)
    print("PERFORMANCE METRICS")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k:<20}: {v:.4f}")
    print("="*40 + "\n")
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results['data'].to_csv(output_path)
        logger.info(f"Results saved to {output_path}")

def run_diagnose(args):
    df = load_and_prep_data(args.data)
    
    logger.info("Building features...")
    features = build_features(df)
    features.dropna(inplace=True)
    
    logger.info(f"Running regime inference (window={args.window})...")
    regime_df = rolling_inference(
        features, 
        window=args.window, 
        refit_interval=args.refit, 
        n_components=3, 
        sort_by='rolling_std_medium'
    )
    
    # Stats
    dist = regime_df['regime'].value_counts(normalize=True).sort_index()
    print("\n" + "="*40)
    print("REGIME DIAGNOSTICS")
    print("="*40)
    print("Regime Distribution:")
    for r, p in dist.items():
        print(f"Regime {int(r)}: {p*100:.2f}%")
    print("-" * 40)
    
    # Transition Matrix (approximate from rolling states)
    # This is a bit hacky since it's rolling, but gives an idea
    states = regime_df['regime'].dropna().astype(int).values
    transitions = np.zeros((3, 3))
    for (s1, s2) in zip(states[:-1], states[1:]):
        transitions[s1, s2] += 1
    
    # Normalize
    trans_mat = transitions / transitions.sum(axis=1, keepdims=True)
    print("Transition Matrix (Empirical):")
    print(trans_mat)
    print("="*40 + "\n")

def run_live(args):
    # For live, we try to update data first
    df = load_and_prep_data(args.data, update=True)
    
    logger.info("Building features...")
    features = build_features(df)
    features.dropna(inplace=True)
    
    # We need to ensure we have enough data for the window
    if len(features) < args.window:
        logger.error(f"Not enough data for inference window. Have {len(features)}, need {args.window}")
        sys.exit(1)
        
    logger.info(f"Running regime inference (window={args.window})...")
    regime_df = rolling_inference(
        features, 
        window=args.window, 
        refit_interval=args.refit, 
        n_components=3, 
        sort_by='rolling_std_medium'
    )
    df = df.join(regime_df)
    
    # Join ADX from features for display
    if 'trend_adx_14' in features.columns:
        df = df.join(features[['trend_adx_14']])
        df.rename(columns={'trend_adx_14': 'adx'}, inplace=True)
    
    # Strategy
    strategy = RobustTrendStrategy(
        fast_span=20, 
        slow_span=50, 
        adx_threshold=30.0, 
        trend_regime=2,
        long_only=False
    )
    
    if 'close' not in df.columns:
         logger.warning("'close' column missing after join, restoring...")
         df_raw = load_and_prep_data(args.data) # Reload from disk/cache
         df['close'] = df_raw.loc[df.index, 'close']

    signals = strategy.generate_signals(df, df['regime'])
    signals = signals.fillna(0.0)
    
    # Vol Target
    sig_vol = apply_vol_targeting(
        signals, 
        df['close'], 
        target_annual_vol=args.target_vol, 
        window_days=20,
        max_leverage=1.0
    )
    
    # Risk Limits
    risk_limits = RiskLimits(max_leverage=1.0, max_position_size=1.0)
    sig_final = risk_limits.apply_limits(sig_vol)
    
    # --- LATEST CANDLE ANALYSIS ---
    last_idx = df.index[-1]
    last_row = df.iloc[-1]
    last_signal = sig_final.iloc[-1]
    
    print("\n" + "="*40)
    print(f"LIVE SIGNAL ANALYSIS: {last_idx}")
    print("="*40)
    print(f"Latest Close:   ${last_row['close']:,.2f}")
    print(f"Regime:         {int(last_row['regime'])} (Prob: {last_row[f'regime_proba_{int(last_row['regime'])}']:.2f})")
    print("-" * 40)
    
    # Debug info
    ema_fast = df['close'].ewm(span=20, adjust=False).mean().iloc[-1]
    ema_slow = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
    # Re-calc ADX locally or assume it's in features if we saved it?
    # build_features returns all features, ADX should be there.
    # Let's check feature columns if needed, but for now just print raw signal.
    
    print(f"EMA20:          {ema_fast:,.2f}")
    print(f"EMA50:          {ema_slow:,.2f}")
    print(f"Trend:          {'BULLISH' if ema_fast > ema_slow else 'BEARISH'}")
    
    if 'adx' in df.columns:
        print(f"ADX:            {df['adx'].iloc[-1]:.2f}")
        
    print("-" * 40)
    
    print(f"Raw Signal:     {signals.iloc[-1]:.2f}")
    print(f"Vol Adjusted:   {sig_vol.iloc[-1]:.2f}")
    print(f"Final Position: {last_signal:.2f} ({last_signal*100:.1f}%)")
    
    if last_signal == 0:
        print("\nACTION: FLAT / CASH (Earn Yield)")
    elif last_signal > 0:
        print(f"\nACTION: LONG BTC ({last_signal*100:.1f}% of Capital)")
    else:
        print(f"\nACTION: SHORT BTC ({abs(last_signal)*100:.1f}% of Capital)")
        
    print("="*40 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Project X Trading Strategy CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Backtest Command
    parser_bt = subparsers.add_parser("backtest", help="Run full strategy backtest")
    parser_bt.add_argument("--data", type=str, default="data/raw/btc_4h.csv", help="Path to OHLCV data")
    parser_bt.add_argument("--capital", type=float, default=10000.0, help="Initial capital")
    parser_bt.add_argument("--target_vol", type=float, default=0.60, help="Target annualized volatility")
    parser_bt.add_argument("--window", type=int, default=1000, help="Regime inference window")
    parser_bt.add_argument("--refit", type=int, default=100, help="Regime refit interval")
    parser_bt.add_argument("--output", type=str, default=None, help="Path to save results CSV")
    
    # Diagnose Command
    parser_diag = subparsers.add_parser("diagnose", help="Run regime diagnostics only")
    parser_diag.add_argument("--data", type=str, default="data/raw/btc_4h.csv", help="Path to OHLCV data")
    parser_diag.add_argument("--window", type=int, default=1000, help="Regime inference window")
    parser_diag.add_argument("--refit", type=int, default=100, help="Regime refit interval")
    
    # Live Command
    parser_live = subparsers.add_parser("live", help="Run live strategy analysis (fetches latest data)")
    parser_live.add_argument("--data", type=str, default="data/raw/btc_4h.csv", help="Path to local cache")
    parser_live.add_argument("--target_vol", type=float, default=0.60, help="Target annualized volatility")
    parser_live.add_argument("--window", type=int, default=1000, help="Regime inference window")
    parser_live.add_argument("--refit", type=int, default=100, help="Regime refit interval")

    args = parser.parse_args()
    
    if args.command == "backtest":
        run_backtest(args)
    elif args.command == "diagnose":
        run_diagnose(args)
    elif args.command == "live":
        run_live(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
