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
from src.regimes.diagnostics import compute_regime_stats

def run_analysis():
    # 1. Load Data
    data_path = project_root / "data/raw/btc_4h.csv"
    print(f"Loading data from {data_path}...")
    try:
        df = load_ohlcv_csv(data_path)
    except FileNotFoundError:
        print("Data not found, generating synthetic...")
        dates = pd.date_range(start='2020-01-01', periods=5000, freq='4h')
        df = pd.DataFrame({
            'open': 10000 + np.cumsum(np.random.normal(0, 50, 5000)),
            'close': 10000 + np.cumsum(np.random.normal(0, 50, 5000)),
            'high': 0, 'low': 0, 'volume': 1000
        }, index=dates)
        df['high'] = df[['open', 'close']].max(axis=1) + 10
        df['low'] = df[['open', 'close']].min(axis=1) - 10

    # 2. Build Features
    print("Building features...")
    features = build_features(df)
    features.dropna(inplace=True)
    df = df.loc[features.index]
    
    # 3. Run Inference with SORTING
    print("Running rolling inference (refit_interval=10, sort_by='rolling_std_medium')...")
    regimes = rolling_inference(
        features,
        n_components=3,
        window=512,
        covariance_type="full",
        smooth_alpha=0.2,
        n_pca_components=10,
        refit_interval=10, # Speed optimization
        sort_by="rolling_std_medium", # <--- CRITICAL FIX
        verbose=True
    )
    
    df_regimes = regimes.dropna()
    
    if len(df_regimes) == 0:
        print("Error: No regimes inferred.")
        return

    # 4. Compute Stats
    print("\nComputing statistics...")
    stats = compute_regime_stats(features.loc[df_regimes.index], df_regimes)
    
    print("\n=== TRANSITION MATRIX ===")
    print(stats['transition_matrix'])
    
    print("\n=== REGIME DURATIONS ===")
    print(stats['durations'][['mean_duration', 'max_duration', 'count_runs']])
    
    print("\n=== REGIME COUNTS ===")
    print(stats['state_counts'])
    
    print("\n=== FEATURE MEANS PER REGIME ===")
    # Print key features to interpret regimes
    key_features = ['log_return', 'rolling_std_medium', 'trend_rsi_14', 'dist_skew_medium']
    for r, means in stats['per_state_means'].items():
        print(f"\nRegime {r}:")
        for k in key_features:
            if k in means:
                print(f"  {k}: {means[k]:.4f}")

if __name__ == "__main__":
    run_analysis()
