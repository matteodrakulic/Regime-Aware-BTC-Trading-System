import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from regimes.inference import rolling_inference

class TestRollingInference(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        n = 300
        prices = np.cumsum(np.random.normal(0, 1, n))
        vol = np.abs(np.random.normal(1, 0.2, n))
        for i in range(150, 200):
            vol[i] *= 3.0
        df = pd.DataFrame({
            "open": prices + np.random.normal(0, 0.1, n),
            "high": prices + np.abs(np.random.normal(0, 0.3, n)),
            "low": prices - np.abs(np.random.normal(0, 0.3, n)),
            "close": prices + np.random.normal(0, 0.1, n),
            "volume": vol * 1000
        })
        self.features = df[["close", "volume"]]

    def test_output_shape_and_values(self):
        res = rolling_inference(self.features, n_components=2, window=64, n_iter=25, smooth_alpha=0.2)
        self.assertEqual(len(res), len(self.features))
        self.assertTrue(all(col in res.columns for col in ["regime", "regime_proba_0", "regime_proba_1"]))
        valid_mask = res["regime"].notna()
        self.assertTrue(valid_mask.sum() >= len(self.features) - 100)
        probs = res.loc[valid_mask, ["regime_proba_0", "regime_proba_1"]].values
        sums = probs.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-6)
        self.assertTrue(set(res.loc[valid_mask, "regime"].astype(int).unique()).issubset({0, 1}))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            rolling_inference(np.array([[1,2],[3,4]]), window=10)
        with self.assertRaises(ValueError):
            rolling_inference(self.features, window=1)

if __name__ == '__main__':
    unittest.main()
