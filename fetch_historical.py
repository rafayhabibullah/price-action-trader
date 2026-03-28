"""
Fetch 6-12 months of OHLCV data for all assets across all timeframes.
Uses pagination for short timeframes that exceed exchange per-request limits.
"""
import time
import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

from data.cache import CacheManager
from config.settings import CACHE_DIR, CCXT_EXCHANGE, TWELVE_DATA_API_KEY

# ── Assets ────────────────────────────────────────────────────────────────────
CRYPTO_ASSETS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
    "XRP/USDT", "TAO/USDT",
]
STOCK_ASSETS = ["AAPL", "MSFT", "NVDA", "GOOGL"]

DAYS = 365  # 12 months

# ── Bars needed for 365 calendar days ────────────────────────────────────────
CRYPTO_BARS = {
    "1h":  DAYS * 24,        # 8 760
    "4h":  DAYS * 6,         # 2 190
    "1d":  DAYS,             #   365
}
STOCK_BARS = {
    "1h":  252 * 7,          # 1 764
    "4h":  252 * 2,          #   504
    "1d":  252,              #   252
}

TF_MAP_STOCK = {"1h": "1h", "4h": "4h", "1d": "1day"}
EXCHANGE_LIMIT = 1000
STOCK_RATE_SLEEP = 8


def fetch_crypto_paginated(exchange, symbol, timeframe, total_bars, days_back):
    """Fetch up to total_bars candles using pagination."""
    tf_ms = {"1h": 3600_000, "4h": 4 * 3600_000, "1d": 86_400_000}
    ms_per_bar = tf_ms[timeframe]
    since = int((datetime.now(timezone.utc) - timedelta(days=days_back + 2)).timestamp() * 1000)
    all_rows = []

    while True:
        chunk = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=EXCHANGE_LIMIT)
        if not chunk:
            break
        all_rows.extend(chunk)
        if len(chunk) < EXCHANGE_LIMIT:
            break
        since = chunk[-1][0] + ms_per_bar
        time.sleep(0.2)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.set_index("ts").sort_index().drop_duplicates()
    return df.tail(total_bars)


def fetch_stock(symbol, timeframe, outputsize):
    """Fetch stock OHLCV from Twelve Data."""
    import requests
    interval = TF_MAP_STOCK[timeframe]
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={symbol}&interval={interval}&outputsize={outputsize}"
        f"&apikey={TWELVE_DATA_API_KEY}&format=JSON&order=ASC"
    )
    r = requests.get(url, timeout=30)
    data = r.json()
    if "values" not in data:
        raise ValueError(data.get("message", data))
    rows = data["values"]
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    cache = CacheManager(CACHE_DIR)
    exchange_class = getattr(ccxt, CCXT_EXCHANGE)
    exchange = exchange_class()

    print(f"=== Fetching {DAYS}-day crypto data (paginated) ===")
    for asset in CRYPTO_ASSETS:
        for tf, bars_needed in CRYPTO_BARS.items():
            print(f"  {asset} {tf} ({bars_needed} bars)...", end=" ", flush=True)
            try:
                df = fetch_crypto_paginated(exchange, asset, tf, bars_needed, DAYS)
                if not df.empty:
                    cache.write(asset, tf, df)
                    print(f"OK ({len(df)} bars, {str(df.index[0])[:10]} -> {str(df.index[-1])[:10]})")
                else:
                    print("EMPTY")
            except Exception as e:
                print(f"ERROR: {e}")

    print()
    print(f"=== Fetching {DAYS}-day stock data (rate-limited) ===")
    for asset in STOCK_ASSETS:
        for tf, bars_needed in STOCK_BARS.items():
            print(f"  {asset} {tf} ({bars_needed} bars)...", end=" ", flush=True)
            try:
                df = fetch_stock(asset, tf, outputsize=min(bars_needed, 5000))
                if not df.empty:
                    cache.write(asset, tf, df)
                    print(f"OK ({len(df)} bars, {str(df.index[0])[:10]} -> {str(df.index[-1])[:10]})")
                else:
                    print("EMPTY")
            except Exception as e:
                print(f"ERROR: {e}")
            time.sleep(STOCK_RATE_SLEEP)

    print()
    print("Done. Now run walk-forward:")
    print(f"  python main.py walk-forward --universe '{','.join(CRYPTO_ASSETS + STOCK_ASSETS)}' --timeframes '1h,4h,1d'")
