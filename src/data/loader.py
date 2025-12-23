"""
Data loader for BTC OHLCV data.

Responsibilities:
- Load raw 4h BTC OHLCV data from disk
- Standardize column names
- Parse timestamps
- Return raw pandas DataFrame

No cleaning or validation happens here.
"""

from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def load_ohlcv_csv(
    filepath: str | Path,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)

    df.columns = [c.lower() for c in df.columns]

    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found.")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

    df = df.set_index(timestamp_col)
    df = df.sort_index()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df
