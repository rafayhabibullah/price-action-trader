import pandas as pd
import pytest
from pathlib import Path
from data.cache import CacheManager


@pytest.fixture
def cache(tmp_path):
    return CacheManager(cache_dir=tmp_path)


@pytest.fixture
def sample_ohlcv():
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1h"),
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [101.0, 102.0, 103.0, 104.0, 105.0],
        "low":  [99.0,  100.0, 101.0, 102.0, 103.0],
        "close":[100.5, 101.5, 102.5, 103.5, 104.5],
        "volume":[1000.0, 1100.0, 900.0, 1200.0, 800.0],
    }).set_index("timestamp")


def test_write_and_read(cache, sample_ohlcv):
    cache.write("BTC/USDT", "1h", sample_ohlcv)
    result = cache.read("BTC/USDT", "1h")
    assert result is not None
    assert len(result) == 5
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]


def test_read_missing_returns_none(cache):
    assert cache.read("NONEXISTENT", "1h") is None


def test_last_timestamp(cache, sample_ohlcv):
    cache.write("BTC/USDT", "1h", sample_ohlcv)
    last = cache.last_timestamp("BTC/USDT", "1h")
    assert last == pd.Timestamp("2024-01-01 04:00:00")


def test_append_new_rows(cache, sample_ohlcv):
    cache.write("BTC/USDT", "1h", sample_ohlcv)
    new_rows = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01 05:00:00", periods=2, freq="1h"),
        "open": [105.0, 106.0], "high": [106.0, 107.0],
        "low": [104.0, 105.0], "close": [105.5, 106.5], "volume": [900.0, 950.0],
    }).set_index("timestamp")
    cache.append("BTC/USDT", "1h", new_rows)
    result = cache.read("BTC/USDT", "1h")
    assert len(result) == 7


def test_append_deduplicates_overlapping_rows(cache, sample_ohlcv):
    """Overlapping rows: new values should win (keep='last')."""
    cache.write("BTC/USDT", "1h", sample_ohlcv)
    # Overlap: update the last 2 rows with different close prices
    overlap = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01 03:00:00", periods=2, freq="1h"),
        "open": [103.0, 104.0], "high": [105.0, 106.0],
        "low": [102.0, 103.0], "close": [999.0, 888.0], "volume": [500.0, 600.0],
    }).set_index("timestamp")
    cache.append("BTC/USDT", "1h", overlap)
    result = cache.read("BTC/USDT", "1h")
    assert len(result) == 5  # no extra rows
    assert result.loc[pd.Timestamp("2024-01-01 03:00:00"), "close"] == 999.0
    assert result.loc[pd.Timestamp("2024-01-01 04:00:00"), "close"] == 888.0


def test_last_timestamp_missing_returns_none(cache):
    assert cache.last_timestamp("NONEXISTENT", "1h") is None
