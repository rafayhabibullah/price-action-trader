# tests/test_candlestick_patterns.py
import pandas as pd
import numpy as np
import pytest
from strategies.candlestick_patterns import CandlestickPatternStrategy

@pytest.fixture
def cp_strategy():
    return CandlestickPatternStrategy(min_body_ratio=0.6, confirmation_candles=1)

def make_df(opens, highs, lows, closes):
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": [1000.0] * len(opens),
    }, index=pd.date_range("2024-01-01", periods=len(opens), freq="1h"))

def test_bullish_engulfing_detected(cp_strategy):
    # Bearish candle then larger bullish candle that engulfs it
    df = make_df(
        opens =[100, 102, 99],
        highs =[103, 103, 104],
        lows  =[99,  100, 98],
        closes=[102, 100, 103],  # candle 1 bearish (102->100), candle 2 bullish engulf (99->103)
    )
    result = cp_strategy.generate_signals(df)
    assert result.iloc[2]["signal"] == 1

def test_returns_required_columns(cp_strategy):
    df = make_df([100]*5, [101]*5, [99]*5, [100]*5)
    result = cp_strategy.generate_signals(df)
    assert {"signal", "sl", "tp"}.issubset(result.columns)

def test_signal_values_valid(cp_strategy):
    df = make_df([100]*10, [101]*10, [99]*10, [100]*10)
    result = cp_strategy.generate_signals(df)
    assert result["signal"].isin([-1, 0, 1]).all()

def test_get_params(cp_strategy):
    p = cp_strategy.get_params()
    assert "min_body_ratio" in p
    assert "confirmation_candles" in p

def test_pin_bar_bullish_detected(cp_strategy):
    # Long lower wick = bullish pin bar (hammer)
    # Candle 2: open=100, high=101, low=89, close=100.5
    # lower_wick = min(100, 100.5) - 89 = 11, body = 0.5, total = 12
    # pin_wick_ratio check: lower_wick(11) >= body(0.5) * 2.0 ✓ AND lower_wick(11) >= total(12) * 0.6(7.2) ✓
    df = make_df(
        opens =[100, 100, 100, 100, 100],
        highs =[101, 101, 101, 101, 101],
        lows  =[99,  99,  89,  99,  99],
        closes=[100, 100, 100.5, 100, 100],
    )
    result = cp_strategy.generate_signals(df)
    assert result.iloc[2]["signal"] == 1, "Expected bullish pin bar signal at index 2"
