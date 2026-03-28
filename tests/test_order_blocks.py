# tests/test_order_blocks.py
import pandas as pd
import numpy as np
import pytest
from strategies.order_blocks import OrderBlockStrategy

@pytest.fixture
def ob_strategy():
    return OrderBlockStrategy(block_size_pct=0.003, invalidation_pct=0.005)

@pytest.fixture
def ob_df():
    """
    Bullish OB setup:
    - i=2: bearish candle (open=105, close=104) — this is the order block
    - i=3,4,5: bullish impulse (3 consecutive bullish candles)
    - OB formed_at = index[5]
    - i=6: price returns into OB zone (103 <= close <= 106) — signal should fire
    """
    data = {
        "open":  [100, 100, 105, 104, 107, 109, 108, 107],
        "high":  [101, 101, 106, 108, 110, 112, 108, 108],
        "low":   [99,  99,  103, 103, 106, 108, 103.1, 104],
        "close": [100, 100, 104, 107, 109, 111, 104.5, 106],
        "volume":[1000.0]*8,
    }
    return pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=8, freq="1h"))

def test_returns_required_columns(ob_strategy, ob_df):
    result = ob_strategy.generate_signals(ob_df)
    assert "signal" in result.columns
    assert "sl" in result.columns
    assert "tp" in result.columns

def test_signal_values_valid(ob_strategy, ob_df):
    result = ob_strategy.generate_signals(ob_df)
    assert result["signal"].isin([-1, 0, 1]).all()
    # Verify the order block actually fires (not a vacuous pass)
    assert (result["signal"] == 1).any(), "Expected at least one bullish OB signal"

def test_get_params(ob_strategy):
    params = ob_strategy.get_params()
    assert "block_size_pct" in params
    assert "invalidation_pct" in params

def test_no_signal_on_flat_market(ob_strategy):
    flat = pd.DataFrame({
        "open":  [100.0] * 20, "high": [100.5] * 20,
        "low":   [99.5] * 20,  "close":[100.0] * 20, "volume":[1000.0] * 20,
    }, index=pd.date_range("2024-01-01", periods=20, freq="1h"))
    result = ob_strategy.generate_signals(flat)
    assert (result["signal"] == 0).all()
