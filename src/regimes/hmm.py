"""
Gaussian Hidden Markov Model (HMM) wrapper for regime detection.

This module is responsible for:
1. Defining the HMM architecture
2. Fitting the model to feature matrices
3. Inferring hidden states (regimes)
4. Providing state probabilities
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

class RegimeHMM:
    """
    Wrapper around hmmlearn.hmm.GaussianHMM to handle data scaling, dimensionality reduction, and model persistence.
    """
    
    def __init__(self, n_components: int = 3, covariance_type: str = "full", n_iter: int = 100, tol: float = 1e-2, min_covar: float = 1e-3, n_pca_components: int | None = None, random_state: int = 42):
        """
        Initialize the HMM model.
        
        Args:
            n_components: Number of regimes (hidden states)
            covariance_type: Type of covariance parameters ('full', 'diag', 'spherical', 'tied')
            n_iter: Maximum number of iterations for the EM algorithm
            tol: Convergence threshold
            min_covar: Floor on the diagonal of the covariance matrix to prevent overfitting.
            n_pca_components: Number of PCA components to keep. If None, no PCA is applied.
            random_state: Seed for reproducibility
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.min_covar = min_covar
        self.n_pca_components = n_pca_components
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        if self.n_pca_components is not None:
            self.pca = PCA(n_components=self.n_pca_components, random_state=random_state)
            
        self._is_fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray):
        """
        Fit the HMM to the provided data.
        Data is automatically standardized (Z-score) before fitting.
        
        Args:
            X: Feature matrix (n_samples, n_features). Can be DataFrame or numpy array.
               Must not contain NaNs.
        """
        # Convert to numpy if needed
        data = self._validate_input(X)
        
        # Scale data
        data_scaled = self.scaler.fit_transform(data)
        
        # Apply PCA if configured
        if self.pca is not None:
            data_scaled = self.pca.fit_transform(data_scaled)
        
        # Initialize and fit HMM
        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            min_covar=self.min_covar,
            random_state=self.random_state,
            verbose=False,
            init_params='stmc',
            params='stmc'
        )
        
        self.model.fit(data_scaled)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Infer the most likely sequence of hidden states.
        
        Returns:
            Array of state labels (0 to n_components-1)
        """
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
            
        data = self._validate_input(X)
        data_scaled = self.scaler.transform(data)
        
        if self.pca is not None:
            data_scaled = self.pca.transform(data_scaled)
        
        return self.model.predict(data_scaled)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Compute the posterior probability for each state at each time step.
        
        Returns:
            Array of shape (n_samples, n_components)
        """
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
            
        data = self._validate_input(X)
        data_scaled = self.scaler.transform(data)
        
        if self.pca is not None:
            data_scaled = self.pca.transform(data_scaled)
        
        return self.model.predict_proba(data_scaled)

    def _validate_input(self, X) -> np.ndarray:
        """
        Helper to validate and convert input to numpy array.
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            data = X.values
        else:
            data = X
            
        # Check for NaNs or Infs
        if np.isnan(data).any() or np.isinf(data).any():
            raise ValueError("Input data contains NaNs or Infs. Please handle missing values before passing to HMM.")
            
        return data

    def save(self, filepath: str):
        """
        Save the entire object (including scaler) to disk.
        """
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> 'RegimeHMM':
        """
        Load a fitted object from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)
