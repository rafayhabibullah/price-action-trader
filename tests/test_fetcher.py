import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from data.fetcher import CryptoFetcher, StockFetcher

# --- CryptoFetcher ---

def test_crypto_fetcher_returns_dataframe():
    mock_exchange = MagicMock()
    mock_exchange.fetch_ohlcv.return_value = [
        [1704067200000, 42000.0, 42500.0, 41800.0, 42200.0, 100.0],
        [1704070800000, 42200.0, 42800.0, 42100.0, 42600.0, 120.0],
    ]
    fetcher = CryptoFetcher(exchange=mock_exchange)
    df = fetcher.fetch("BTC/USDT", "1h", limit=2)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 2

def test_crypto_fetcher_since_param():
    mock_exchange = MagicMock()
    mock_exchange.fetch_ohlcv.return_value = []
    fetcher = CryptoFetcher(exchange=mock_exchange)
    since = pd.Timestamp("2024-01-01")
    fetcher.fetch("BTC/USDT", "1h", since=since)
    call_kwargs = mock_exchange.fetch_ohlcv.call_args
    assert call_kwargs[1]["since"] == int(since.timestamp() * 1000)

def test_stock_fetcher_invalid_timeframe():
    mock_client = MagicMock()
    fetcher = StockFetcher(client=mock_client)
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        fetcher.fetch("AAPL", "30m")

# --- StockFetcher ---

def test_stock_fetcher_returns_dataframe():
    mock_client = MagicMock()
    mock_client.time_series.return_value.as_pandas.return_value = pd.DataFrame({
        "open": [150.0, 151.0],
        "high": [152.0, 153.0],
        "low":  [149.0, 150.0],
        "close":[151.0, 152.0],
        "volume":[1000000.0, 1100000.0],
    }, index=pd.date_range("2024-01-01", periods=2, freq="1d"))
    fetcher = StockFetcher(client=mock_client)
    df = fetcher.fetch("AAPL", "1d", outputsize=2)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    # Verify TF_MAP translation: "1d" → "1day"
    call_kwargs = mock_client.time_series.call_args[1]
    assert call_kwargs["interval"] == "1day"
