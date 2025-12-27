import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.loader import load_ohlcv_csv
from src.data.cleaner import clean_ohlcv
from src.features.builder import build_features
from src.regimes.inference import rolling_inference
from src.utils.logging import setup_logger

logger = setup_logger("diagnose")

def run_diagnostics():
    # 1. Load Data
    data_path = "data/raw/btc_4h.csv"
    logger.info(f"Loading data from {data_path}...")
    try:
        df = load_ohlcv_csv(data_path)
        df = clean_ohlcv(df)
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        return

    # 2. Features & Regimes
    logger.info("Building features and running regime inference...")
    features = build_features(df)
    features.dropna(inplace=True)
    df = df.loc[features.index]

    regime_df = rolling_inference(
        features, 
        window=1000, 
        refit_interval=100, 
        n_components=3, 
        sort_by='rolling_std_medium'
    )
    df = df.join(regime_df)

    # 3. Calculate Stats for Each Regime
    print("\n" + "="*50)
    print("REGIME PERSONALITY TEST")
    print("="*50)
    print(f"{'Regime':<10} | {'Ann. Volatility':<15} | {'Ann. Mean Return':<15} | {'Count':<10}")
    print("-" * 60)

    for r in [0, 1, 2]:
        subset = df[df['regime'] == r].copy()
        if len(subset) == 0:
            print(f"Regime {r}   | N/A             | N/A             | 0")
            continue
        
        # Calculate Log Returns if not present (build_features might not have simple log_ret)
        subset['log_ret'] = np.log(subset['close'] / subset['close'].shift(1))
        
        # Calculate Volatility (std of returns) and Mean Return
        # 6 * 365 = 2190 candles per year
        vol = subset['log_ret'].std() * np.sqrt(2190) 
        mean_ret = subset['log_ret'].mean() * 2190    
        
        print(f"Regime {r:<4} | {vol*100:>14.1f}% | {mean_ret*100:>14.1f}% | {len(subset):<10}")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    run_diagnostics()
