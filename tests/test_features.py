import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from features.volatility import build_volatility_features
from features.trend import build_trend_features
from features.distribution import build_distribution_features
from features.builder import build_features

class TestVolatilityFeatures(unittest.TestCase):
    def setUp(self):
        # Create synthetic data if file doesn't exist, or load sample
        # For reliability, let's create synthetic data
        dates = pd.date_range(start='2023-01-01', periods=200, freq='4h')
        self.df = pd.DataFrame({
            'open': np.random.uniform(100, 200, 200),
            'high': np.random.uniform(200, 210, 200),
            'low': np.random.uniform(90, 100, 200),
            'close': np.random.uniform(100, 200, 200),
            'volume': np.random.uniform(1000, 5000, 200)
        }, index=dates)
        
        # Ensure High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
        self.df['high'] = self.df[['open', 'close', 'high']].max(axis=1) + 1
        self.df['low'] = self.df[['open', 'close', 'low']].min(axis=1) - 1

    def test_build_volatility_features(self):
        features = build_volatility_features(self.df)
        
        # Check columns exist
        expected_cols = [
            'vol_realized_short', 'vol_realized_medium', 'vol_realized_long',
            'vol_parkinson_short', 'vol_parkinson_medium',
            'vol_gk_short', 'vol_gk_medium',
            'vol_of_vol_short', 'vol_of_vol_medium'
        ]
        for col in expected_cols:
            self.assertIn(col, features.columns)
            
        # Check for NaNs at the beginning (expected due to rolling)
        # Check that we have valid values at the end
        self.assertFalse(features.iloc[-1].isna().any())
        
        # Check values are positive (volatility is positive)
        # Note: vol of vol is std dev so it should be positive
        # We start checking from index 100 to ensure all rolling windows are filled (max 32+64=96)
        self.assertTrue((features.iloc[100:] >= 0).all().all())

class TestTrendFeatures(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        dates = pd.date_range(start='2023-01-01', periods=200, freq='4h')
        self.df = pd.DataFrame({
            'close': np.random.uniform(100, 200, 200),
        }, index=dates)

    def test_build_trend_features(self):
        features = build_trend_features(self.df)
        
        # Check columns exist
        expected_cols = [
            'trend_sma_short', 'trend_sma_medium', 'trend_sma_long',
            'trend_ema_short', 'trend_ema_medium', 'trend_ema_long',
            'trend_dist_sma_short', 'trend_dist_sma_medium', 'trend_dist_sma_long',
            'trend_dist_ema_short', 'trend_dist_ema_medium', 'trend_dist_ema_long',
            'trend_macd_line', 'trend_macd_signal', 'trend_macd_hist',
            'trend_rsi_14'
        ]
        
        for col in expected_cols:
            self.assertIn(col, features.columns)
            
        # Check that we have valid values at the end
        self.assertFalse(features.iloc[-1].isna().any())
        
        # Check RSI range
        self.assertTrue((features['trend_rsi_14'].dropna() >= 0).all())
        self.assertTrue((features['trend_rsi_14'].dropna() <= 100).all())

class TestFeatureBuilder(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        dates = pd.date_range(start='2023-01-01', periods=200, freq='4h')
        self.df = pd.DataFrame({
            'open': np.random.uniform(100, 200, 200),
            'high': np.random.uniform(200, 210, 200),
            'low': np.random.uniform(90, 100, 200),
            'close': np.random.uniform(100, 200, 200),
            'volume': np.random.uniform(1000, 5000, 200)
        }, index=dates)
        
        # Ensure High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
        self.df['high'] = self.df[['open', 'close', 'high']].max(axis=1) + 1
        self.df['low'] = self.df[['open', 'close', 'low']].min(axis=1) - 1

    def test_build_features(self):
        features = build_features(self.df)
        
        # Check that we have columns from all categories
        # Returns
        self.assertIn('log_return', features.columns)
        self.assertIn('rolling_std_short', features.columns)
        
        # Volatility
        self.assertIn('vol_realized_short', features.columns)
        self.assertIn('vol_parkinson_short', features.columns)
        
        # Trend
        self.assertIn('trend_sma_short', features.columns)
        self.assertIn('trend_rsi_14', features.columns)
        
        # Distribution
        self.assertIn('dist_skew_short', features.columns)
        self.assertIn('dist_kurt_medium', features.columns)
        
        # Check shape
        # Should have rows equal to df
        self.assertEqual(len(features), len(self.df))
        
        # Check that we have valid values at the end
        self.assertFalse(features.iloc[-1].isna().any())

if __name__ == '__main__':
    unittest.main()
