# tests/test_market_structure.py
import pandas as pd
import numpy as np
import pytest
from strategies.market_structure import MarketStructureStrategy

@pytest.fixture
def ms_strategy():
    return MarketStructureStrategy(swing_lookback=3)

def make_trending_df(prices):
    return pd.DataFrame({
        "open":  [p * 0.999 for p in prices],
        "high":  [p * 1.005 for p in prices],
        "low":   [p * 0.995 for p in prices],
        "close": prices,
        "volume":[1000.0] * len(prices),
    }, index=pd.date_range("2024-01-01", periods=len(prices), freq="1h"))

def test_returns_required_columns(ms_strategy):
    df = make_trending_df([100.0] * 20)
    result = ms_strategy.generate_signals(df)
    assert {"signal", "sl", "tp"}.issubset(result.columns)

def test_signal_values_valid(ms_strategy):
    prices = [100, 105, 102, 108, 105, 112, 108, 115]
    df = make_trending_df(prices)
    result = ms_strategy.generate_signals(df)
    assert result["signal"].isin([-1, 0, 1]).all()

def test_get_params(ms_strategy):
    p = ms_strategy.get_params()
    assert "swing_lookback" in p

def test_uptrend_generates_buy_signals(ms_strategy):
    # Clear uptrend: HH + HL sequence
    prices = [100, 110, 105, 115, 108, 120, 112, 125]
    df = make_trending_df(prices)
    result = ms_strategy.generate_signals(df)
    buys = (result["signal"] == 1).sum()
    sells = (result["signal"] == -1).sum()
    assert buys >= sells
