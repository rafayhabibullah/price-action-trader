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
    # Clear uptrend with isolated peaks: HH pattern that creates BOS signals
    # Swing highs form at each peak (isolated by 3+ lower bars on each side)
    # swing_lookback=3 -> need local max in ±3 bars
    prices = [
        90, 90, 90, 120, 90, 90, 90,   # swing high at i=3 (120)
        90, 90, 90, 130, 90, 90, 90,   # swing high at i=10 (130) — HH
        90, 90, 90, 140, 90, 90, 90,   # swing high at i=17 (140) — HH
        150,                            # BOS: close 150 > last_sh 140
    ]
    prices_float = [float(p) for p in prices]
    df = make_trending_df(prices_float)
    result = ms_strategy.generate_signals(df)
    buys = (result["signal"] == 1).sum()
    assert buys >= 1, f"Expected at least 1 bullish BOS signal, got {buys}"
