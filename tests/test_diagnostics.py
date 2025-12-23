import unittest
import numpy as np
import pandas as pd
import sys
import os
import matplotlib
matplotlib.use("Agg")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from regimes.diagnostics import compute_transition_matrix, compute_state_durations, compute_regime_stats, plot_price_with_regimes, plot_regime_probabilities

class TestDiagnostics(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="H")
        price = np.cumsum(np.random.normal(0, 1, n)) + 20000
        states = np.zeros(n)
        states[50:100] = 1
        states[100:130] = 2
        states[130:160] = 1
        regimes = pd.DataFrame({
            "regime": states
        }, index=dates)
        k = 3
        for i in range(k):
            p = np.zeros(n)
            for t in range(n):
                p[t] = 1.0 if int(states[t]) == i else 0.0
            regimes[f"regime_proba_{i}"] = p
        self.df_price = pd.DataFrame({"close": price}, index=dates)
        self.regimes = regimes
        self.features = pd.DataFrame({"feat1": np.random.normal(0,1,n), "feat2": np.random.normal(0,2,n)}, index=dates)
        self.n_components = k

    def test_transition_matrix_rows_sum_to_one(self):
        tm = compute_transition_matrix(self.regimes["regime"], self.n_components)
        row_sums = tm.values.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums))

    def test_state_durations_positive(self):
        d = compute_state_durations(self.regimes["regime"], self.n_components)
        self.assertTrue((d["mean_duration"].dropna() > 0).all())
        self.assertTrue((d["median_duration"].dropna() > 0).all())
        self.assertTrue((d["max_duration"].dropna() > 0).all())
        self.assertTrue((d["count_runs"] >= 0).all())

    def test_regime_stats_structure(self):
        stats = compute_regime_stats(self.features, self.regimes)
        self.assertIn("transition_matrix", stats)
        self.assertIn("durations", stats)
        self.assertIn("state_counts", stats)
        self.assertIn("per_state_means", stats)
        self.assertEqual(len(stats["per_state_means"]), self.n_components)

    def test_plot_functions(self):
        fig1, ax1 = plot_price_with_regimes(self.df_price, self.regimes)
        self.assertIsNotNone(fig1)
        self.assertTrue(len(ax1) == 2)
        fig2, ax2 = plot_regime_probabilities(self.regimes)
        self.assertIsNotNone(fig2)
        self.assertIsNotNone(ax2)

if __name__ == "__main__":
    unittest.main()
