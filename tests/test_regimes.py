import unittest
import numpy as np
import pandas as pd
import sys
import os
import tempfile

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from regimes.hmm import RegimeHMM

class TestRegimeHMM(unittest.TestCase):
    def setUp(self):
        # Create synthetic data with structure
        # Regime 1: Mean 0, Low Vol
        # Regime 2: Mean 0, High Vol
        # Regime 3: Mean 0.5, Low Vol
        np.random.seed(42)
        
        n_samples = 200
        # Create 2 features
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.random.normal(0, 1, n_samples)
        
        self.X = pd.DataFrame({'f1': X1, 'f2': X2})
        self.hmm = RegimeHMM(n_components=2, n_iter=10)

    def test_fit_and_predict(self):
        # Test fitting
        self.hmm.fit(self.X)
        self.assertTrue(self.hmm._is_fitted)
        self.assertIsNotNone(self.hmm.model)
        
        # Test predict
        states = self.hmm.predict(self.X)
        self.assertEqual(len(states), len(self.X))
        self.assertTrue(set(states).issubset({0, 1}))
        
        # Test predict_proba
        probas = self.hmm.predict_proba(self.X)
        self.assertEqual(probas.shape, (len(self.X), 2))
        # Sum of probabilities should be close to 1
        np.testing.assert_allclose(probas.sum(axis=1), 1.0)

    def test_input_validation(self):
        # Test NaNs
        X_nan = self.X.copy()
        X_nan.iloc[0, 0] = np.nan
        
        with self.assertRaises(ValueError):
            self.hmm.fit(X_nan)
            
        # Test Not Fitted Error
        model_new = RegimeHMM()
        with self.assertRaises(ValueError):
            model_new.predict(self.X)

    def test_persistence(self):
        self.hmm.fit(self.X)
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            save_path = tmp.name
            
        try:
            self.hmm.save(save_path)
            
            loaded_hmm = RegimeHMM.load(save_path)
            
            self.assertTrue(loaded_hmm._is_fitted)
            self.assertEqual(loaded_hmm.n_components, self.hmm.n_components)
            
            # Check if predictions match
            states_orig = self.hmm.predict(self.X)
            states_loaded = loaded_hmm.predict(self.X)
            
            np.testing.assert_array_equal(states_orig, states_loaded)
            
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

if __name__ == '__main__':
    unittest.main()
