"""
OHLCV data fetcher for BTC using ccxt (Binance).

Responsibilities:
- Fetch historical 4h BTC/USDT OHLCV data
- Cache data locally as CSV
- Incrementally update existing data
"""

from pathlib import Path
from typing import Optional
import time

import ccxt
import pandas as pd


DATA_PATH = Path("data/raw/btc_4h.csv")
SYMBOL = "BTC/USDT"
TIMEFRAME = "4h"
EXCHANGE_ID = "binance"


def _init_exchange() -> ccxt.Exchange:
    exchange = getattr(ccxt, EXCHANGE_ID)({
        "enableRateLimit": True,
    })
    exchange.load_markets()
    return exchange


def fetch_ohlcv_since(
    exchange: ccxt.Exchange,
    since: Optional[pd.Timestamp] = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch OHLCV candles since a given timestamp.

    Returns DataFrame with columns:
    timestamp, open, high, low, close, volume
    """
    since_ms = None
    if since is not None:
        since_ms = int(since.timestamp() * 1000)

    all_rows = []

    while True:
        ohlcv = exchange.fetch_ohlcv(
            SYMBOL,
            timeframe=TIMEFRAME,
            since=since_ms,
            limit=limit,
        )

        if not ohlcv:
            break

        all_rows.extend(ohlcv)

        # Advance since_ms to last candle + 1 ms
        since_ms = ohlcv[-1][0] + 1

        # Stop if fewer than limit returned (no more data)
        if len(ohlcv) < limit:
            break

        time.sleep(exchange.rateLimit / 1000)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    return df


def update_local_ohlcv() -> pd.DataFrame:
    """
    Fetch BTC 4h OHLCV data and update local CSV cache.

    Returns the full updated DataFrame.
    """
    exchange = _init_exchange()

    if DATA_PATH.exists():
        existing = pd.read_csv(DATA_PATH)
        existing["timestamp"] = pd.to_datetime(
            existing["timestamp"], utc=True
        )
        existing = existing.sort_values("timestamp")

        last_ts = existing["timestamp"].iloc[-1]

        new_data = fetch_ohlcv_since(exchange, since=last_ts)

        if new_data.empty:
            return existing

        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset="timestamp")
        combined = combined.sort_values("timestamp")

    else:
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Fetch 5 years of history (~10k candles for 4h timeframe)
        start_date = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365 * 5)
        print(f"Fetching data starting from {start_date}...")
        combined = fetch_ohlcv_since(exchange, since=start_date)

    combined.to_csv(DATA_PATH, index=False)

    return combined
