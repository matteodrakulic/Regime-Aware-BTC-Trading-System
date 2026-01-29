import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from .hmm import RegimeHMM

def rolling_inference(features: pd.DataFrame, n_components: int = 3, covariance_type: str = "full", n_iter: int = 100, tol: float = 1e-2, min_covar: float = 1e-3, window: int = 512, smooth_alpha: float | None = None, n_pca_components: int | None = 10, on_error: str = "carry", refit_interval: int = 1, sort_by: str | None = None, verbose: bool = True, random_state: int = 42) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        raise ValueError("features must be a pandas DataFrame")
    if window <= 1:
        raise ValueError("window must be > 1")
    idx = features.index
    n = len(features)
    result = pd.DataFrame(index=idx)
    result["regime"] = np.nan
    proba_cols = [f"regime_proba_{i}" for i in range(n_components)]
    for c in proba_cols:
        result[c] = np.nan
    prev_smooth = None
    last_fitted_hmm = None
    last_sorted_map = None
    
    iterator = range(window - 1, n)
    if verbose:
        iterator = tqdm(iterator, desc="Rolling Inference", mininterval=1.0)
        
    for t in iterator:
        window_df = features.iloc[t - window + 1 : t + 1].dropna()
        if len(window_df) < 2:
            continue
        
        # Decide whether to refit
        # NOTE: Refitting on the window ending at t includes data at t.
        # This implies that the model parameters (and scaler stats) are influenced by X_t.
        # While the state inference P(S_t | X_{t-w+1:t}) is a valid filtered probability,
        # the parameter estimation has a 1-step look-ahead.
        # For strict walk-forward, one should fit on t-w:t-1 and predict on t.
        # However, fitting on the current window is common for stability in rolling regimes.
        should_refit = (last_fitted_hmm is None) or ((t - (window - 1)) % refit_interval == 0)
        
        try:
            if should_refit:
                hmm = RegimeHMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter, tol=tol, min_covar=min_covar, n_pca_components=n_pca_components, random_state=random_state)
                
                # Suppress convergence warnings for cleaner CLI output
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Model is not converging")
                    hmm.fit(window_df)
                
                last_fitted_hmm = hmm
                
                # Compute sorting map if requested
                if sort_by is not None and sort_by in window_df.columns:
                    # Predict states on the training window to determine their properties
                    states = hmm.predict(window_df)
                    state_means = []
                    for s in range(n_components):
                        mask = states == s
                        if mask.any():
                            val = window_df.loc[window_df.index[mask], sort_by].mean()
                        else:
                            # Push unused states to the end (infinity)
                            val = np.inf 
                        state_means.append(val)
                    last_sorted_map = np.argsort(state_means)
                else:
                    last_sorted_map = None
            
            # Use the last fitted model (or the one just fitted) to predict for the current window
            probas = last_fitted_hmm.predict_proba(window_df)
            
            # Reorder probabilities based on the sorted map
            if last_sorted_map is not None:
                probas = probas[:, last_sorted_map]
                
            p_t = probas[-1]
            
            if np.isnan(p_t).any():
                if on_error == "carry" and prev_smooth is not None:
                    p_t = prev_smooth
                else:
                    continue
                
            if smooth_alpha is not None and prev_smooth is not None:
                p_t = smooth_alpha * p_t + (1.0 - smooth_alpha) * prev_smooth
            
            # Update prev_smooth only if valid
            if not np.isnan(p_t).any():
                prev_smooth = p_t
                state_t = int(np.argmax(p_t))
                result.iloc[t, result.columns.get_loc("regime")] = state_t
                for i in range(n_components):
                    result.iloc[t, result.columns.get_loc(f"regime_proba_{i}")] = p_t[i]
                    
        except Exception:
            if on_error == "carry" and prev_smooth is not None:
                p_t = prev_smooth
                state_t = int(np.argmax(p_t))
                result.iloc[t, result.columns.get_loc("regime")] = state_t
                for i in range(n_components):
                    result.iloc[t, result.columns.get_loc(f"regime_proba_{i}")] = p_t[i]
            else:
                continue
    return result
