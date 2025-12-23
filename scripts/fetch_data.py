from src.data.fetcher import update_local_ohlcv; print('Starting fetch...'); df = update_local_ohlcv(); print(f'Fetched {len(df)} rows')
