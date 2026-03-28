import pytest
import os


def test_config_reads_alpaca_keys(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    # Re-import to pick up monkeypatched env
    import importlib
    import trading.config as cfg
    importlib.reload(cfg)
    assert cfg.ALPACA_API_KEY == "test-key"
    assert cfg.ALPACA_SECRET_KEY == "test-secret"


def test_config_defaults():
    import importlib
    import trading.config as cfg
    importlib.reload(cfg)
    assert cfg.MAX_CONCURRENT == 15
    assert cfg.RISK_PER_TRADE == 0.02
    assert cfg.MAX_BARS == 100
    assert cfg.CANDLE_HISTORY == 200
    assert "BTC/USDT" in cfg.ASSETS_CRYPTO
    assert "AAPL" in cfg.ASSETS_STOCKS
    assert "1h" in cfg.TIMEFRAMES


def test_to_alpaca_symbol_crypto():
    from trading.symbols import to_alpaca, to_internal, is_crypto

    assert to_alpaca("BTC/USDT") == "BTC/USD"
    assert to_alpaca("ETH/USDT") == "ETH/USD"
    assert to_alpaca("AAPL") == "AAPL"
    assert to_alpaca("SPY") == "SPY"


def test_to_internal_symbol():
    from trading.symbols import to_internal

    assert to_internal("BTC/USD") == "BTC/USDT"
    assert to_internal("ETH/USD") == "ETH/USDT"
    assert to_internal("AAPL") == "AAPL"


def test_is_crypto():
    from trading.symbols import is_crypto

    assert is_crypto("BTC/USDT") is True
    assert is_crypto("ETH/USDT") is True
    assert is_crypto("AAPL") is False
    assert is_crypto("SPY") is False
