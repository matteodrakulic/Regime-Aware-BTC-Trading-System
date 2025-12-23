from src.data.fetcher import update_local_ohlcv

if __name__ == "__main__":
    df = update_local_ohlcv()
    print(f"Data updated. Rows: {len(df)}")
