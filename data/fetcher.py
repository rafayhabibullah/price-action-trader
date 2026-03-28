import ccxt
import pandas as pd
from twelvedata import TDClient
from config.settings import CCXT_EXCHANGE, TWELVE_DATA_API_KEY


class CryptoFetcher:
    def __init__(self, exchange=None):
        if exchange is None:
            exchange_class = getattr(ccxt, CCXT_EXCHANGE)
            self.exchange = exchange_class()
        else:
            self.exchange = exchange

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        since: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        kwargs: dict = {"limit": limit}
        if since is not None:
            kwargs["since"] = int(since.timestamp() * 1000)
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, **kwargs)
        if not raw:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df.index = df.index.tz_convert(None)
        return df


class StockFetcher:
    # Twelve Data timeframe mapping: "1h" -> "1h", "1d" -> "1day", "4h" -> "4h"
    TF_MAP = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1day"}

    def __init__(self, client=None):
        if client is None:
            self.client = TDClient(apikey=TWELVE_DATA_API_KEY)
        else:
            self.client = client

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        outputsize: int = 500,
        start_date: str | None = None,
    ) -> pd.DataFrame:
        if timeframe not in self.TF_MAP:
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Supported: {list(self.TF_MAP)}")
        td_tf = self.TF_MAP[timeframe]
        kwargs: dict = {"symbol": symbol, "interval": td_tf, "outputsize": outputsize}
        if start_date:
            kwargs["start_date"] = start_date
        df = self.client.time_series(**kwargs).as_pandas()
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df.sort_index(inplace=True)
        return df
