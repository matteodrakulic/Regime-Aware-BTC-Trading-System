import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_transition_matrix(states: pd.Series, n_components: int) -> pd.DataFrame:
    s = states.dropna().astype(int).values
    mat = np.zeros((n_components, n_components), dtype=float)
    for i in range(len(s) - 1):
        mat[s[i], s[i + 1]] += 1.0
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat = mat / row_sums
    idx = [f"r{i}" for i in range(n_components)]
    return pd.DataFrame(mat, index=idx, columns=idx)

def compute_state_durations(states: pd.Series, n_components: int) -> pd.DataFrame:
    s = states.copy()
    durations = {i: [] for i in range(n_components)}
    current_state = None
    current_len = 0
    for x in s:
        if np.isnan(x):
            if current_state is not None and current_len > 0:
                durations[int(current_state)].append(current_len)
            current_state = None
            current_len = 0
            continue
        x = int(x)
        if current_state is None:
            current_state = x
            current_len = 1
        elif x == current_state:
            current_len += 1
        else:
            durations[int(current_state)].append(current_len)
            current_state = x
            current_len = 1
    if current_state is not None and current_len > 0:
        durations[int(current_state)].append(current_len)
    data = {
        "mean_duration": [np.mean(durations[i]) if len(durations[i]) > 0 else np.nan for i in range(n_components)],
        "median_duration": [np.median(durations[i]) if len(durations[i]) > 0 else np.nan for i in range(n_components)],
        "max_duration": [np.max(durations[i]) if len(durations[i]) > 0 else np.nan for i in range(n_components)],
        "count_runs": [len(durations[i]) for i in range(n_components)],
    }
    return pd.DataFrame(data, index=[f"r{i}" for i in range(n_components)])

def compute_regime_stats(features: pd.DataFrame, regimes: pd.DataFrame) -> dict:
    states = regimes["regime"]
    n_components = len([c for c in regimes.columns if c.startswith("regime_proba_")])
    trans = compute_transition_matrix(states, n_components)
    durations = compute_state_durations(states, n_components)
    counts = states.value_counts(dropna=True).sort_index()
    counts = counts.reindex(range(n_components)).fillna(0).astype(int)
    per_state_means = {}
    for i in range(n_components):
        mask = states == i
        if mask.any():
            per_state_means[f"r{i}"] = features.loc[mask].mean(numeric_only=True)
        else:
            per_state_means[f"r{i}"] = pd.Series(dtype=float)
    return {
        "transition_matrix": trans,
        "durations": durations,
        "state_counts": counts,
        "per_state_means": per_state_means,
    }

def plot_price_with_regimes(df_price: pd.DataFrame, regimes: pd.DataFrame, price_col: str = "close", colors: list[str] | None = None, figsize=(12, 6)):
    x = df_price.index
    y = df_price[price_col].values
    states = regimes["regime"].values
    n_components = len([c for c in regimes.columns if c.startswith("regime_proba_")])
    if colors is None:
        colors = plt.cm.tab10(np.arange(n_components))
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(x, y, color="black")
    for i in range(n_components):
        mask = states == i
        ax[0].scatter(x[mask], y[mask], s=10, color=colors[i], label=f"r{i}")
    ax[0].legend(loc="upper left", ncol=min(n_components, 4))
    proba_cols = [f"regime_proba_{i}" for i in range(n_components)]
    if all(c in regimes.columns for c in proba_cols):
        for i in range(n_components):
            ax[1].plot(x, regimes[proba_cols[i]].values, color=colors[i], alpha=0.8, label=f"p(r{i})")
        ax[1].set_ylim(0, 1)
        ax[1].legend(loc="upper left", ncol=min(n_components, 4))
    ax[0].set_ylabel("Price")
    ax[1].set_ylabel("Prob.")
    ax[1].set_xlabel("Time")
    fig.tight_layout()
    return fig, ax

def plot_regime_probabilities(regimes: pd.DataFrame, figsize=(12, 3)):
    n_components = len([c for c in regimes.columns if c.startswith("regime_proba_")])
    x = regimes.index
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = plt.cm.tab10(np.arange(n_components))
    for i in range(n_components):
        ax.plot(x, regimes[f"regime_proba_{i}"].values, color=colors[i], label=f"p(r{i})")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", ncol=min(n_components, 4))
    ax.set_ylabel("Prob.")
    ax.set_xlabel("Time")
    fig.tight_layout()
    return fig, ax
