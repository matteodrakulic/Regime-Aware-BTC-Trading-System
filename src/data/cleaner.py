"""
Data cleaning and validation for BTC OHLCV data.

Responsibilities:
- Validate OHLCV integrity
- Check timestamp continuity
- Enforce 4h frequency
- Detect missing or malformed candles
"""

import pandas as pd


EXPECTED_DELTA = pd.Timedelta(hours=4)


def validate_ohlcv(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("OHLCV DataFrame is empty.")

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("High price lower than open/close/low.")

    if (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Low price higher than open/close/high.")

    if (df["volume"] < 0).any():
        raise ValueError("Negative volume detected.")


def validate_time_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Timestamps are not sorted.")

    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps detected.")

    deltas = df.index.to_series().diff().dropna()

    if not (deltas == EXPECTED_DELTA).all():
        raise ValueError(
            "Detected missing or irregular 4h candles in data."
        )


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    validate_time_index(df)
    validate_ohlcv(df)
    return df.copy()
