import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from datetime import datetime, timezone
from trading.scanner import Signal, scan_all, _is_candle_closed
from strategies.support_resistance import SupportResistanceStrategy


def _make_trending_df(n=100, start_price=100.0) -> pd.DataFrame:
    """Create uptrending OHLCV data that satisfies EMA50 and volume filters."""
    np.random.seed(42)
    closes = start_price + np.cumsum(np.abs(np.random.randn(n) * 0.5))
    volumes = np.random.uniform(2000, 5000, n)  # volume high enough for filter
    return pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
            "volume": volumes,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )


def test_signal_dataclass():
    sig = Signal(
        asset="BTC/USDT",
        timeframe="1h",
        strategy="SupportResistanceStrategy",
        direction=1,
        entry_price=50000.0,
        sl=49000.0,
        tp=52000.0,
    )
    assert sig.asset == "BTC/USDT"
    assert sig.direction == 1
    assert sig.sl == 49000.0


def test_scan_all_returns_list(monkeypatch):
    df = _make_trending_df(200)
    mock_client = MagicMock()
    mock_client.get_bars.return_value = df
    mock_client.is_market_open.return_value = True

    strategies = [SupportResistanceStrategy()]
    assets_crypto = ["BTC/USDT"]
    assets_stocks = []
    timeframes = ["1h"]

    signals = scan_all(mock_client, strategies, assets_crypto, assets_stocks, timeframes)
    assert isinstance(signals, list)
    for sig in signals:
        assert isinstance(sig, Signal)
        assert sig.asset == "BTC/USDT"
        assert sig.direction in (1, -1)
        assert sig.sl > 0
        assert sig.tp > 0


def test_scan_all_skips_stocks_when_market_closed(monkeypatch):
    df = _make_trending_df(200)
    mock_client = MagicMock()
    mock_client.get_bars.return_value = df
    mock_client.is_market_open.return_value = False

    strategies = [SupportResistanceStrategy()]
    signals = scan_all(
        mock_client, strategies,
        assets_crypto=[],
        assets_stocks=["AAPL"],
        timeframes=["1h"],
    )
    # No stock signals when market closed
    stock_signals = [s for s in signals if s.asset == "AAPL"]
    assert stock_signals == []


def test_scan_all_skips_empty_bars(monkeypatch):
    mock_client = MagicMock()
    mock_client.get_bars.return_value = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"]
    )
    mock_client.is_market_open.return_value = True

    strategies = [SupportResistanceStrategy()]
    signals = scan_all(
        mock_client, strategies,
        assets_crypto=["BTC/USDT"],
        assets_stocks=[],
        timeframes=["1h"],
    )
    assert signals == []


def test_scan_all_filters_tight_sl():
    """Signals with SL distance < 0.1% are filtered out."""
    df = _make_trending_df(200)
    mock_client = MagicMock()
    mock_client.get_bars.return_value = df
    mock_client.is_market_open.return_value = True

    strategies = [SupportResistanceStrategy()]
    signals = scan_all(
        mock_client, strategies,
        assets_crypto=["BTC/USDT"],
        assets_stocks=[],
        timeframes=["1h"],
    )
    for sig in signals:
        sl_dist_pct = abs(sig.entry_price - sig.sl) / sig.entry_price
        assert sl_dist_pct >= 0.001, f"SL too tight: {sl_dist_pct:.5f}"


def test_candle_closure_1h_always_closed():
    """1h candles are always closed when cron runs at :00."""
    for hour in range(24):
        dt = datetime(2024, 1, 1, hour, 0, 0, tzinfo=timezone.utc)
        assert _is_candle_closed("1h", dt) is True


def test_candle_closure_4h():
    """4h candles close at hours 0, 4, 8, 12, 16, 20 UTC."""
    closed_hours = {0, 4, 8, 12, 16, 20}
    for hour in range(24):
        dt = datetime(2024, 1, 1, hour, 0, 0, tzinfo=timezone.utc)
        expected = hour in closed_hours
        assert _is_candle_closed("4h", dt) is expected, f"4h at hour {hour}: expected {expected}"


def test_candle_closure_1d():
    """1d candles close at 00:00 UTC."""
    dt_midnight = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    dt_noon = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert _is_candle_closed("1d", dt_midnight) is True
    assert _is_candle_closed("1d", dt_noon) is False


def test_scan_all_skips_open_4h_candle():
    """Verify that at hour 3, the 4h candle is not considered closed."""
    dt = datetime(2024, 1, 1, 3, 0, 0, tzinfo=timezone.utc)
    assert _is_candle_closed("4h", dt) is False
