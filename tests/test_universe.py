import pytest
from unittest.mock import MagicMock, patch
from data.universe import UniverseManager


def test_crypto_universe_filters_by_volume():
    mock_exchange = MagicMock()
    mock_exchange.fetch_tickers.return_value = {
        "BTC/USDT": {"symbol": "BTC/USDT", "quoteVolume": 5_000_000_000},
        "ETH/USDT": {"symbol": "ETH/USDT", "quoteVolume": 2_000_000_000},
        "SHIB/USDT": {"symbol": "SHIB/USDT", "quoteVolume": 1_000},  # too low
        "DOGE/USDT": {"symbol": "DOGE/USDT", "quoteVolume": None},   # missing
    }
    mgr = UniverseManager(crypto_exchange=mock_exchange)
    symbols = mgr.get_crypto_universe(min_volume_usd=1_000_000)
    assert "BTC/USDT" in symbols
    assert "ETH/USDT" in symbols
    assert "SHIB/USDT" not in symbols
    assert "DOGE/USDT" not in symbols


def test_crypto_universe_only_usdt_pairs():
    mock_exchange = MagicMock()
    mock_exchange.fetch_tickers.return_value = {
        "BTC/USDT": {"symbol": "BTC/USDT", "quoteVolume": 5_000_000_000},
        "BTC/BTC":  {"symbol": "BTC/BTC",  "quoteVolume": 5_000_000_000},
    }
    mgr = UniverseManager(crypto_exchange=mock_exchange)
    symbols = mgr.get_crypto_universe(min_volume_usd=1_000_000)
    assert "BTC/USDT" in symbols
    assert "BTC/BTC" not in symbols


def test_stock_universe_returns_list():
    mgr = UniverseManager(stock_symbols=["AAPL", "MSFT", "SPY"])
    symbols = mgr.get_stock_universe()
    assert symbols == ["AAPL", "MSFT", "SPY"]
