import sys
from pathlib import Path
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.loader import load_ohlcv_csv
from src.features.builder import build_features
from src.regimes.inference import rolling_inference
from src.strategies.robust import RobustTrendStrategy
from src.strategies.backtester import VectorizedBacktester

def optimize():
    # 1. Load Data
    data_path = project_root / "data/raw/btc_4h.csv"
    print("Loading data...")
    df = load_ohlcv_csv(data_path)
    
    print("Building features...")
    features = build_features(df)
    features.dropna(inplace=True)
    df = df.loc[features.index]
    
    # 2. Run Inference (Fixed Context)
    # We use the settings that gave us good regimes
    print("Running inference...")
    regimes = rolling_inference(
        features,
        n_components=3,
        window=512,
        covariance_type="full",
        n_pca_components=10,
        refit_interval=10,
        sort_by="rolling_std_medium",
        verbose=False
    )
    
    df = df.loc[regimes.index]
    regime_col = regimes['regime']
    
    # 3. Define Parameter Grid
    # We want to find the best Trend Logic for Regime 2 (High Vol)
    param_grid = {
        'fast_span': [10, 20, 30],
        'slow_span': [40, 50, 80, 100],
        'adx_threshold': [15, 20, 25, 30]
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Testing {len(combinations)} combinations...")
    
    results = []
    backtester = VectorizedBacktester(initial_capital=10000.0, fee_rate=0.0005)
    
    for params in tqdm(combinations):
        # Skip invalid combinations (fast >= slow)
        if params['fast_span'] >= params['slow_span']:
            continue
            
        # Initialize Strategy
        # Note: We are testing Long-Only for now to find the best trend engine
        strategy = RobustTrendStrategy(
            fast_span=params['fast_span'],
            slow_span=params['slow_span'],
            adx_threshold=params['adx_threshold'],
            trend_regime=2,
            long_only=True
        )
        
        # Run Backtest
        signals = strategy.generate_signals(df, regime_col)
        res = backtester.run(df, signals)
        metrics = res['metrics']
        
        results.append({
            **params,
            'total_return': metrics['total_return'],
            'sharpe': metrics['sharpe_ratio'],
            'drawdown': metrics['max_drawdown'],
            'trades': (signals.diff().abs() > 0).sum() / 2 # Approx trade count
        })
        
    # 4. Analyze Results
    results_df = pd.DataFrame(results)
    
    # Sort by Sharpe Ratio
    top_sharpe = results_df.sort_values('sharpe', ascending=False).head(5)
    
    # Sort by Total Return
    top_return = results_df.sort_values('total_return', ascending=False).head(5)
    
    print("\n=== TOP 5 BY SHARPE RATIO ===")
    print(top_sharpe.to_string(index=False))
    
    print("\n=== TOP 5 BY TOTAL RETURN ===")
    print(top_return.to_string(index=False))
    
    # Save all results
    results_df.to_csv(project_root / "data/processed/optimization_results.csv", index=False)
    print("\nFull results saved to data/processed/optimization_results.csv")

if __name__ == "__main__":
    optimize()
