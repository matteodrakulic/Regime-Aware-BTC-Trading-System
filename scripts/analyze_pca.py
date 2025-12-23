import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path.cwd()
sys.path.append(str(project_root))

from src.data.loader import load_ohlcv_csv
from src.features.builder import build_features

def analyze_pca():
    print("Loading data...")
    data_path = project_root / "data/raw/btc_4h.csv"
    if not data_path.exists():
        print("Data file not found!")
        return

    df = load_ohlcv_csv(data_path)
    print(f"Data loaded: {len(df)} rows")

    print("Building features...")
    features = build_features(df)
    features.dropna(inplace=True)
    print(f"Features shape: {features.shape}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Run PCA
    pca = PCA()
    pca.fit(X_scaled)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print("\nPCA Analysis Results:")
    print("-" * 50)
    print(f"{'Components':<12} | {'Explained Var':<15} | {'Cumulative':<15}")
    print("-" * 50)
    
    thresholds = {0.8: None, 0.9: None, 0.95: None}
    
    for i, (ev, cv) in enumerate(zip(explained_variance, cumulative_variance)):
        n = i + 1
        print(f"{n:<12} | {ev:.4f}          | {cv:.4f}")
        
        for t in thresholds:
            if thresholds[t] is None and cv >= t:
                thresholds[t] = n

    print("-" * 50)
    print("\nRecommendations:")
    for t, n in thresholds.items():
        print(f"For {t*100}% variance: Use {n} components")

if __name__ == "__main__":
    analyze_pca()
