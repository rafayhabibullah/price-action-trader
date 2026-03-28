# tests/test_fair_value_gaps.py
import pandas as pd
import numpy as np
import pytest
from strategies.fair_value_gaps import FairValueGapStrategy

@pytest.fixture
def fvg_strategy():
    return FairValueGapStrategy(gap_min_pct=0.002, fill_pct=0.5)

@pytest.fixture
def fvg_df():
    """
    Bullish FVG: candle[i].high < candle[i+2].low → gap between them.
    candle[0..2]: base, candle[3]: strong bullish impulse creating a gap,
    candle[4]: normal, candle[5]: price returns to fill the gap area.
    """
    data = {
        "open":  [100, 100, 100, 100, 108, 107, 105],
        "high":  [101, 101, 101, 101, 112, 109, 106],
        "low":   [99,  99,  99,  99,  108, 106, 103],
        "close": [100, 100, 100, 101, 111, 108, 104],
        "volume":[1000] * 7,
    }
    return pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=7, freq="1h"))

def test_returns_required_columns(fvg_strategy, fvg_df):
    result = fvg_strategy.generate_signals(fvg_df)
    assert "signal" in result.columns
    assert "sl" in result.columns
    assert "tp" in result.columns

def test_signal_values_valid(fvg_strategy, fvg_df):
    result = fvg_strategy.generate_signals(fvg_df)
    assert result["signal"].isin([-1, 0, 1]).all()

def test_get_params(fvg_strategy):
    p = fvg_strategy.get_params()
    assert "gap_min_pct" in p
    assert "fill_pct" in p

def test_no_gap_no_signal(fvg_strategy):
    flat = pd.DataFrame({
        "open":  [100.0] * 10, "high": [101.0] * 10,
        "low":   [99.0] * 10,  "close":[100.0] * 10, "volume":[1000.0] * 10,
    }, index=pd.date_range("2024-01-01", periods=10, freq="1h"))
    result = fvg_strategy.generate_signals(flat)
    assert (result["signal"] == 0).all()
